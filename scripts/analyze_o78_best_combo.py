#!/usr/bin/env python3
"""Recompute v142 complete paired Events; never merge historical best timings."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from analyze_mx8_warp_lut import STAGES, bootstrap_median_ci
from benchmark_o78_best_combo import PREPARATIONS, SYMBOLS, route, timing_contract


def validate_row(row, variant):
    if row['variant'] != variant or row['candidate'] not in (0, 1):
        raise ValueError('variant or paired policy mismatch')
    if not all(row[k] for k in ('bitwise_equal_v67', 'MSE_regression_passed',
            'finite_fp32', 'metadata_exact', 'payload_norm_checked_after_every_call')):
        raise ValueError('numerical acceptance failed')
    if row['mse_vs_v67'] != 0 or row['max_abs_vs_v67'] != 0:
        raise ValueError('output changed')
    for key, value in timing_contract(variant, row['mode'], 100).items():
        if row[key] != value:
            raise ValueError('timing contract changed: '+key)
    resource = row['resources']
    if (resource['kernel_symbol'] != SYMBOLS[row['candidate']] or
            resource['preparation'] != PREPARATIONS[variant] or
            (resource['conversion_host_function'], resource['conversion_selector']) != route(variant, row['candidate'])):
        raise ValueError('actual kernel/conversion route mismatch')
    if set(row['raw_ms']) != set(STAGES[row['mode']]):
        raise ValueError('timing stages changed')
    for stage, raw in row['raw_ms'].items():
        if len(raw) != 200 or any(not 0 < float(x) < float('inf') for x in raw):
            raise ValueError('200 unfiltered positive finite Events required')
        summary = row['stage_summaries'][stage]
        cv = statistics.pstdev(raw)/statistics.mean(raw)*100
        if abs(summary['median_ms']-statistics.median(raw)) > 1e-10 or abs(summary['cv_percent']-cv) > 1e-7:
            raise ValueError('raw Event/summary mismatch')


def analyze(directory):
    rows = list(map(json.loads, (directory/'results.jsonl').read_text().splitlines()))
    variants = {r['variant'] for r in rows}
    if len(variants) != 1 or not variants <= {'o7', 'o8'}:
        raise ValueError('one complete O7/O8 run required')
    variant = variants.pop()
    index = {(r['sample_id'],r['mode'],r['round'],r['candidate']):r for r in rows}
    ids = sorted({r['sample_id'] for r in rows})
    expected = {(s,m,k,p) for s in ids for m in STAGES for k in range(3) for p in (0,1)}
    if len(ids) != 24 or len(rows) != 576 or len(index) != len(rows) or set(index) != expected:
        raise ValueError('exact full24/three-round/four-mode pair required')
    source = json.loads((directory/'source_identity_checked.json').read_text())
    if not source['passed'] or not source['full_v99_source_identity_equal'] or source['variant'] != variant:
        raise ValueError('source identity failed')
    env = json.loads((directory/'environment.json').read_text())
    if any(env['args'][k] != v for k,v in dict(samples=24,rounds=3,warmup=1000,repeats=200,inner=100).items()):
        raise ValueError('measurement counts changed')
    receipt = env['codegen']['output_streaming']
    if (not receipt['worth_runtime_validation'] or receipt['changed_semantics'] or
            receipt['production_default_changed'] or env['production_default_changed'] or
            env['codegen']['new_CUDA_compilation']):
        raise ValueError('binary audit/default gate failed')
    if not all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma']
               for e in receipt['entries'].values()):
        raise ValueError('native two-route INT4 audit failed')
    guards = {}
    for row in rows:
        validate_row(row, variant)
        reference = index[row['sample_id'],'compute_only',0,0]
        if row['mse_vs_paired_fp16'] != reference['mse_vs_paired_fp16']:
            raise ValueError('MSE changed between policies/modes/rounds')
        if row['guard'] != reference['guard'] or row['guard']['invalid_ctas']:
            raise ValueError('guard mismatch/invalid source')
        guards[row['sample_id']] = row['guard']
    stages = []
    for mode, names in STAGES.items():
        for stage in names:
            lat = [[statistics.median(index[s,mode,k,p]['stage_summaries'][stage]['median_ms']
                    for k in range(3)) for s in ids] for p in (0,1)]
            ratios = [statistics.median(index[s,mode,k,0]['stage_summaries'][stage]['median_ms']/
                      index[s,mode,k,1]['stage_summaries'][stage]['median_ms'] for k in range(3)) for s in ids]
            speed = statistics.median(ratios)
            ci = list(bootstrap_median_ci(ratios,10000,.95,20261008))
            stages.append(dict(mode=mode,stage=stage,control_median_ms=statistics.median(lat[0]),
                candidate_median_ms=statistics.median(lat[1]),paired_speedup=speed,
                paired_throughput_change_percent=(speed-1)*100,
                paired_latency_reduction_percent=(1-1/speed)*100,paired_speedup_ci95=ci,
                paired_gain_confirmed=ci[0]>1,
                control_cv_failed=sum(index[s,mode,k,0]['stage_summaries'][stage]['cv_percent']>=3 for s in ids for k in range(3)),
                candidate_cv_failed=sum(index[s,mode,k,1]['stage_summaries'][stage]['cv_percent']>=3 for s in ids for k in range(3)),
                records_per_policy=72))
    errors = [index[s,'compute_only',0,0]['mse_vs_paired_fp16'] for s in ids]
    return dict(scope='v142_v99_with_best_conversion_vs_identical_conversion_v78',
        variant=variant,samples=24,records=len(rows),no_filtering=True,
        original_event_values=sum(len(raw) for r in rows for raw in r['raw_ms'].values()),
        production_default_changed=False,new_CUDA_compilation=False,
        preparation=PREPARATIONS[variant],source_identity=source,
        all_outputs_bitwise_equal=True,output_mse=dict(reference=rows[0]['paired_fp16'],
            median=statistics.median(errors),mean=statistics.mean(errors)),
        stages=stages,guard_summary=guards,formal_extension_sha256=env['extension_sha256'],
        runtime_commit=env['git_commit'],candidate_cubin_sha256=receipt['cubin_sha256'],
        control_cubin_sha256=receipt['baseline_cubin_sha256'],
        results_sha256=hashlib.sha256((directory/'results.jsonl').read_bytes()).hexdigest())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args(); result = analyze(a.input)
    with a.output.open('x') as out:
        json.dump(result,out,indent=2,allow_nan=False); out.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'guard_summary'},indent=2))
