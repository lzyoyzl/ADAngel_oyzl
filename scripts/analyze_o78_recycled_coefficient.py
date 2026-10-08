#!/usr/bin/env python3
"""Recompute all full24 v141 pairs from raw Events; no trimming or new run."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from benchmark_a100_o1 import stats
from benchmark_o78_coefficient_probe import summarize
from benchmark_o78_recycled_coefficient import timing_contract


def read(path):return json.loads(path.read_text())
def jsonl(path):return [json.loads(line) for line in path.read_text().splitlines()]


def analyze(directory):
    env=read(directory/'environment.json'); rows=jsonl(directory/'results.jsonl')
    modes=('conversion_only','compute_only','cold','steady_state') if env['args']['full_modes'] else ('compute_only',)
    ids={f'layer_{layer:02d}_{proj}_proj' for layer in (0,6,12,18,24,31) for proj in ('q','k','v','o')}
    expected={(s,v,m,r,p) for s in ids for v in ('o7','o8') for m in modes for r in range(3) for p in (0,1)}
    by_key={(r['sample_id'],r['variant'],r['mode'],r['round'],r['candidate']):r for r in rows}
    if len(by_key)!=len(rows) or set(by_key)!=expected:
        raise ValueError('all24 x2 variants x3 rounds x2 policies required; no partial analysis')
    if env['production_default_changed'] or not env['no_filtering']:
        raise ValueError('scope or filtering drift')
    if any(env['args'][k]!=v for k,v in dict(samples=24,rounds=3,warmup=1000,repeats=200,inner=100).items()):
        raise ValueError('declared measurement protocol differs')
    mse_values={};guards={}; event_values=0
    for key,r in by_key.items():
        if not all(r[k] for k in ('finite_fp32','metadata_exact','bitwise_equal_v67','MSE_regression_passed')):
            raise ValueError('numerical/metadata regression')
        if r['mse_vs_v67']!=0 or r['max_abs_vs_v67']!=0:
            raise ValueError('output differs from frozen fullK reference')
        if r['paired_fp16']!={'o7':'o5','o8':'o6'}[r['variant']]:raise ValueError('wrong MSE reference')
        identity=(r['sample_id'],r['variant'])
        if mse_values.setdefault(identity,r['mse_vs_paired_fp16'])!=r['mse_vs_paired_fp16']:
            raise ValueError('MSE differs between modes/policies/rounds')
        if not math.isfinite(r['mse_vs_paired_fp16']) or r['mse_vs_paired_fp16']<0:
            raise ValueError('invalid MSE')
        if guards.setdefault(identity,r['guard'])!=r['guard'] or r['guard']['invalid_ctas']:
            raise ValueError('guard changed or invalid source')
        if r['guard']['integer_ctas']+r['guard']['fallback_ctas']!=2048:
            raise ValueError('missing output tiles')
        for name,value in timing_contract(r['mode'],100).items():
            if r[name]!=value:raise ValueError('timing definition changed: '+name)
        for stage,times in r['raw_ms'].items():
            if len(times)!=200 or not all(math.isfinite(t) and t>0 for t in times):
                raise ValueError('missing/invalid raw Event values')
            actual=stats(times); stored=r['stage_summaries'][stage]
            if set(actual)!=set(stored) or any(not math.isclose(actual[k],stored[k],rel_tol=1e-12,abs_tol=1e-12) for k in actual):
                raise ValueError('raw statistics mismatch')
            event_values+=len(times)
        selected='gemm' if r['mode']=='compute_only' else 'total'
        if r['summary']!=r['stage_summaries'][selected]:raise ValueError('selected latency mismatch')
        mate=by_key[(*key[:-1],1-key[-1])]
        if r['execution_order']!=mate['execution_order'] or set(r['execution_order'])!={0,1}:
            raise ValueError('paired execution order mismatch')
    recomputed=summarize(rows,modes,('o7','o8'))
    if recomputed!=read(directory/'summary.json')['records']:raise ValueError('paired summary mismatch')
    provenance=jsonl(directory/'source_provenance.jsonl')
    if len(provenance)!=48 or {(r['sample_id'],r['variant']) for r in provenance}!=set(mse_values):
        raise ValueError('missing original FP16 source identities')
    table=[]
    for variant in ('o7','o8'):
        for mode in modes:
            old,new=[r for r in recomputed if r['variant']==variant and r['mode']==mode]
            rounds=[]
            for ri in range(3):
                ratios=[by_key[s,variant,mode,ri,0]['summary']['median_ms']/
                        by_key[s,variant,mode,ri,1]['summary']['median_ms'] for s in sorted(ids)]
                rounds.append(statistics.median(ratios))
            table.append(dict(variant=variant,mode=mode,old_median_ms=old['median_ms'],
                new_median_ms=new['median_ms'],paired_speedup=new['paired_speedup'],
                paired_throughput_change_percent=(new['paired_speedup']-1)*100,
                paired_latency_reduction_percent=(1-1/new['paired_speedup'])*100,
                paired_speedup_ci95=new['paired_speedup_ci95'],round_paired_speedups=rounds,
                old_cv_failed=old['selected_cv_failed_records'],new_cv_failed=new['selected_cv_failed_records'],
                old_any_stage_cv_failed=old['any_stage_cv_failed_records'],new_any_stage_cv_failed=new['any_stage_cv_failed_records'],
                old_median_cv_percent=statistics.median(by_key[s,variant,mode,ri,0]['summary']['cv_percent']
                    for s in ids for ri in range(3)),
                new_median_cv_percent=statistics.median(by_key[s,variant,mode,ri,1]['summary']['cv_percent']
                    for s in ids for ri in range(3)),
                records_each=72,median_mse=new['median_mse'],mean_mse=new['mean_mse'],
                output_bitwise_unchanged=True,
                median_speedup_ci_above_one=new['paired_speedup_ci95'][0]>1,
                all_record_CV_below3=old['selected_cv_failed_records']==new['selected_cv_failed_records']==0))
    return dict(scope='v141_full24_unfiltered_paired_replay_not_cross_run_speedup',
        records=len(rows),raw_event_values_including_duplicate_compute_total=event_values,
        samples=24,variants=['o7','o8'],modes=list(modes),table=table,
        guard_summary=[dict(sample_id=s,variant=v,**g) for (s,v),g in sorted(guards.items())],
        production_default_changed=False,conversion_modified=False,new_conversion_gain_claim=False,
        no_filtering=True,extension_sha256=env['extension_sha256'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=analyze(a.input)
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT):p.error('fresh repository output required')
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='guard_summary'},indent=2))
