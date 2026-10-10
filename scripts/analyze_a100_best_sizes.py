#!/usr/bin/env python3
"""Unfiltered size-run analysis; no rewriting timings or selecting fastest rounds."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics as st


def analyze(directory):
    complete = json.loads((directory/'completion.json').read_text())
    rows = [json.loads(x) for x in (directory/'results.jsonl').read_text().splitlines()]
    assert complete['passed'] and len(rows) == complete['records']
    assert len({(r['sample_id'],r['size'],r['variant'],r['mode'],r['round']) for r in rows}) == len(rows)
    ids = sorted({r['sample_id'] for r in rows})
    out = []
    for size in (512,1024):
        for variant in ('o1','o3','o5','o6','o7','o8'):
            for mode in ('conversion_only','compute_only','cold','steady_state'):
                subset = [r for r in rows if (r['size'],r['variant'],r['mode']) == (size,variant,mode)]
                stages = {}
                for stage in subset[0]['stages']:
                    med = [st.median(r['stages'][stage]['median_ms'] for r in subset if r['sample_id']==sid) for sid in ids]
                    mean = [st.mean(r['stages'][stage]['mean_ms'] for r in subset if r['sample_id']==sid) for sid in ids]
                    stages[stage] = dict(median_ms=st.median(med), mean_ms=st.mean(mean),
                        mean_definition='mean_of_all_unfiltered_event_samples_across_rounds_and_traces')
                primary = 'gemm' if mode=='compute_only' else 'total'
                errors = [next(r['mse'] for r in subset if r['sample_id']==sid) for sid in ids]
                adjacent, outliers = 0, 0
                for row in subset:
                    times = row['raw_ms'][primary]
                    bins = sorted({round(t*1e6) for t in times})  # rounded nanoseconds
                    adjacent += len(bins)==2 and abs(bins[1]-bins[0]-1024)<=2
                    outliers += max(times)>1.2*st.median(times)
                out.append(dict(size=size,variant=variant,mode=mode,samples=len(ids),records=len(subset),stages=stages,
                    reference=subset[0]['reference'], mse_median=st.median(errors),mse_mean=st.mean(errors),
                    primary_cv_failed=sum(r['stages'][primary]['cv_percent']>=3 for r in subset),
                    any_stage_cv_failed=sum(any(v['cv_percent']>=3 for v in r['stages'].values()) for r in subset),
                    median_primary_cv=st.median(r['stages'][primary]['cv_percent'] for r in subset),
                    only_two_adjacent_1024ns_bins=adjacent,large_outlier_records=outliers,
                    fallback_tiles_max=max(r['kernel'].get('fallback_tiles',0) for r in subset),
                    kernel=subset[0]['kernel']))
    paired = []
    for size in (512,1024):
        for mode in ('conversion_only','compute_only','cold','steady_state'):
            stage = 'gemm' if mode=='compute_only' else 'total'
            for faster,base in [('o3','o1'),('o7','o5'),('o8','o6')]:
                ratios = []
                for sid in ids:
                    lats = {v: st.median(r['stages'][stage]['median_ms'] for r in rows
                        if (r['size'],r['mode'],r['sample_id'],r['variant'])==(size,mode,sid,v)) for v in (faster,base)}
                    ratios.append(lats[base]/lats[faster])
                paired.append(dict(size=size,mode=mode,variant=faster,baseline=base,
                                   median_paired_speedup=st.median(ratios),mean_paired_speedup=st.mean(ratios)))
    example = next(r for r in rows if r['size']==512 and r['variant']=='o3' and r['mode']=='compute_only')
    return dict(passed=True,records=len(rows),samples=len(ids),no_filtering=True,
        aggregation='median(24 trace medians of 3 round medians); means include every event observation',
        results_sha256=hashlib.sha256((directory/'results.jsonl').read_bytes()).hexdigest(),
        tables=out,paired=paired,
        event_quantization_example=dict(sample_id=example['sample_id'],variant='o3',size=512,
            statistics=example['stages']['gemm'], histogram_us=sorted(Counter(round(t*1000,3) for t in example['raw_ms']['gemm']).items())))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('fresh output required')
    result=analyze(args.input)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for row in result['tables']:
        stage='gemm' if row['mode']=='compute_only' else 'total'
        print(row['size'],row['variant'],row['mode'],row['stages'][stage],
              'CV flags',row['primary_cv_failed'],'/',row['records'])


if __name__=='__main__':main()
