#!/usr/bin/env python3
"""Replay v138 full24 raw Events and exact conversion/MSE proof, no filtering."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

from analyze_mx8_warp_lut import STAGES,bootstrap_median_ci


def analyze(directory):
    rows=list(map(json.loads,(directory/'results.jsonl').read_text().splitlines()))
    index={(r['sample_id'],r['mode'],r['round'],r['candidate']):r for r in rows}
    ids=sorted({r['sample_id'] for r in rows});rounds=sorted({r['round'] for r in rows})
    expected={(s,m,k,p) for s in ids for m in STAGES for k in rounds for p in (0,1)}
    if len(ids)!=24 or rounds!=list(range(3)) or len(index)!=len(rows) or set(index)!=expected:
        raise ValueError('exact full24/three-round/four-mode paired evidence required')
    source=json.loads((directory/'source_identity_checked.json').read_text())
    if not source['passed'] or not source['full_v99_source_identity_equal'] or source['samples']!=24 or source['variant']!='o8':
        raise ValueError('source identity gate failed')
    env=json.loads((directory/'environment.json').read_text())
    if any(env['args'][k]!=v for k,v in dict(samples=24,rounds=3,warmup=1000,repeats=200,inner=100).items()):
        raise ValueError('timing counts drift')
    for row in rows:
        if row['variant']!='o8' or not all(row[k] for k in (
                'bitwise_equal_v67','MSE_regression_passed','finite_fp32','metadata_exact',
                'gemm_cufunction_identical_between_policies','activation_preparation_identical',
                'payload_norm_checked_after_every_call')):
            raise ValueError('numerical/GEMM/activation identity acceptance failed')
        if row['mse_vs_v67']!=0 or row['max_abs_vs_v67']!=0:
            raise ValueError('new/old output differs')
        if row['mse_vs_paired_fp16']!=index[row['sample_id'],'compute_only',0,0]['mse_vs_paired_fp16']:
            raise ValueError('paired FP16 MSE changed')
        if (row['modified_stage']!='weight_payload_decode_and_exact_square_sum_only' or
            row['activation_preparation_implementation']!='v123_packed_FP6' or
            set(row['raw_ms'])!=set(STAGES[row['mode']])):
            raise ValueError('timing/optimization scope drift')
        for stage,raw in row['raw_ms'].items():
            if len(raw)!=200 or any(not 0<float(x)<float('inf') for x in raw):
                raise ValueError('200 unfiltered finite Event values required')
            summary=row['stage_summaries'][stage]
            cv=statistics.pstdev(raw)/statistics.mean(raw)*100
            if abs(summary['median_ms']-statistics.median(raw))>1e-10 or abs(summary['cv_percent']-cv)>1e-7:
                raise ValueError('raw Event/summary mismatch')
    records=[]
    for mode,stages in STAGES.items():
        for stage in stages:
            values=[[statistics.median(index[s,mode,k,p]['stage_summaries'][stage]['median_ms']
                for k in rounds) for s in ids] for p in (0,1)]
            ratios=[statistics.median(index[s,mode,k,0]['stage_summaries'][stage]['median_ms']/
                index[s,mode,k,1]['stage_summaries'][stage]['median_ms'] for k in rounds) for s in ids]
            speed=statistics.median(ratios)
            records.append(dict(mode=mode,stage=stage,
                control_median_ms=statistics.median(values[0]),candidate_median_ms=statistics.median(values[1]),
                paired_speedup=speed,paired_throughput_change_percent=(speed-1)*100,
                paired_latency_reduction_percent=(1-1/speed)*100,
                paired_speedup_ci95=list(bootstrap_median_ci(ratios,10000,.95,20261008)),
                control_cv_failed=sum(index[s,mode,k,0]['stage_summaries'][stage]['cv_percent']>=3 for s in ids for k in rounds),
                candidate_cv_failed=sum(index[s,mode,k,1]['stage_summaries'][stage]['cv_percent']>=3 for s in ids for k in rounds),
                records_per_policy=72))
    if {r['resources']['kernel_symbol'] for r in rows}!={'adangel_roof_o78_eight_chain_candidate'}:
        raise ValueError('identical actual v78 GEMM required')
    errors=[index[s,'compute_only',0,0]['mse_vs_paired_fp16'] for s in ids]
    snapshots=list(map(json.loads,(directory/'gpu_snapshots.jsonl').read_text().splitlines()))
    clocks=[int(x.group(1)) for r in snapshots for x in re.finditer(r'(\d+)\s+MHz',r['gpu'])]
    if len(snapshots)!=72 or len(clocks)!=72:raise ValueError('incomplete GPU snapshots')
    receipt=env['codegen']['conversion_swar']
    if not receipt['audit']['worth_runtime_validation'] or receipt['GEMM_modified']:
        raise ValueError('resource audit/default drift')
    return dict(scope='v138_HiF4_conversion_identical_v123_activation_v78_GEMM',
        samples=24,records=len(rows),original_event_values=sum(len(v) for r in rows for v in r['raw_ms'].values()),
        source_identity=source,GEMM_modified=False,production_default_changed=False,no_filtering=True,
        all_outputs_bitwise_equal=True,output_mse_vs_O6=dict(median=statistics.median(errors),mean=statistics.mean(errors)),
        stages=records,gpu_sampled_sm_clock_MHz=dict(min=min(clocks),max=max(clocks)),
        conversion_compile_audit=receipt['audit'],GEMM_speedup_claim=False,
        all_real_integer_path=all(r['guard']['fallback_ctas']==r['guard']['invalid_ctas']==0 for r in rows),
        results_sha256=hashlib.sha256((directory/'results.jsonl').read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=analyze(a.input)
    with a.output.open('x') as out:json.dump(result,out,indent=2,allow_nan=False);out.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='conversion_compile_audit'},indent=2))


if __name__=='__main__':main()
