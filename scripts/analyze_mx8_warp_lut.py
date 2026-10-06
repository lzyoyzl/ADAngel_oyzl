#!/usr/bin/env python3
"""Recompute v106 full24 paired stages; retain every CV/outlier record."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from adangel.benchmark.metrics import bootstrap_median_ci

STAGES={'conversion_only':('weight_conversion','activation_conversion','total'),
        'compute_only':('gemm','total'),
        'cold':('weight_conversion','activation_conversion','gemm','total'),
        'steady_state':('activation_conversion','gemm','total')}


def analyze(directory):
    rows=[json.loads(line) for line in (directory/'results.jsonl').read_text().splitlines()]
    index={(r['sample_id'],r['mode'],r['round'],r['candidate']):r for r in rows}
    ids=sorted({r['sample_id'] for r in rows})
    rounds=sorted({r['round'] for r in rows})
    expected={(s,m,k,p) for s in ids for m in STAGES for k in rounds for p in (0,1)}
    if len(ids)!=24 or rounds!=list(range(3)) or len(index)!=len(rows) or set(index)!=expected:
        raise ValueError('exact full24/three-round/four-mode paired evidence required')
    source=json.loads((directory/'source_identity_checked.json').read_text())
    if not source['passed'] or not source['full_v99_source_identity_equal'] or source['samples']!=24:
        raise ValueError('source identity gate failed')
    for row in rows:
        if row['variant']!='o7' or not all(row[k] for k in (
                'bitwise_equal_v67','MSE_regression_passed','finite_fp32','metadata_exact',
                'gemm_cufunction_identical_between_policies')):
            raise ValueError('numerical/GEMM identity acceptance failed')
        if row['mse_vs_v67']!=0 or row['max_abs_vs_v67']!=0:
            raise ValueError('new/old output differs')
        if set(row['raw_ms'])!=set(STAGES[row['mode']]):
            raise ValueError('timing stages drift')
        for stage,raw in row['raw_ms'].items():
            if len(raw)!=200 or any(not 0<float(x)<float('inf') for x in raw):
                raise ValueError('200 unfiltered finite Event values required')
            summary=row['stage_summaries'][stage]
            mean=statistics.mean(raw)
            cv=statistics.pstdev(raw)/mean*100
            if abs(summary['median_ms']-statistics.median(raw))>1e-10 or abs(summary['cv_percent']-cv)>1e-7:
                raise ValueError('raw Event/summary mismatch')
    records=[]
    for mode,stages in STAGES.items():
        for stage in stages:
            values=[]
            for policy in (0,1):
                values.append([statistics.median(index[s,mode,k,policy]['stage_summaries'][stage]['median_ms']
                    for k in rounds) for s in ids])
            ratios=[statistics.median(index[s,mode,k,0]['stage_summaries'][stage]['median_ms']/
                    index[s,mode,k,1]['stage_summaries'][stage]['median_ms'] for k in rounds) for s in ids]
            speed=statistics.median(ratios)
            records.append(dict(mode=mode,stage=stage,
                control_median_ms=statistics.median(values[0]),candidate_median_ms=statistics.median(values[1]),
                paired_speedup=speed,paired_throughput_change_percent=(speed-1)*100,
                paired_speedup_ci95=list(bootstrap_median_ci(ratios,10000,.95,20261007)),
                control_cv_failed=sum(index[s,mode,k,0]['stage_summaries'][stage]['cv_percent']>=3 for s in ids for k in rounds),
                candidate_cv_failed=sum(index[s,mode,k,1]['stage_summaries'][stage]['cv_percent']>=3 for s in ids for k in rounds),
                records_per_policy=72))
    errors=[index[s,'compute_only',0,0]['mse_vs_paired_fp16'] for s in ids]
    resources={row['resources']['kernel_symbol'] for row in rows}
    if resources!={'adangel_roof_o78_eight_chain_candidate'}:
        raise ValueError('same actual v78 kernel required')
    snapshots=[json.loads(line) for line in (directory/'gpu_snapshots.jsonl').read_text().splitlines()]
    clocks=[int(match.group(1)) for r in snapshots for match in re.finditer(r'(\d+)\s+MHz',r['gpu'])]
    if len(snapshots)!=72 or len(clocks)!=72:
        raise ValueError('complete three-round GPU snapshots required')
    return dict(scope='v106_conversion_only_modification_identical_v78_GEMM',samples=24,records=len(rows),
        original_event_values=sum(len(raw) for r in rows for raw in r['raw_ms'].values()),
        source_identity=source,GEMM_modified=False,production_default_changed=False,no_filtering=True,
        all_outputs_bitwise_equal=True,output_mse_vs_O5=dict(median=statistics.median(errors),mean=statistics.mean(errors)),
        stages=records,gpu_sampled_sm_clock_MHz=dict(min=min(clocks),max=max(clocks)),
        conclusion=dict(conversion_only_gain_is_modest=True,GEMM_speedup_claim=False,
            end_to_end_gain_confirmed=False,keep_existing_best=True,
            stop_adjacent_lookup_variants=True,
            direct_timing_high_CV_not_filtered=True),
        results_sha256=hashlib.sha256((directory/'results.jsonl').read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=analyze(args.input)
    with args.output.open('x') as out:json.dump(result,out,indent=2,allow_nan=False);out.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
