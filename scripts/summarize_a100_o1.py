#!/usr/bin/env python3
"""Summarize complete paired O1 measurements without dropping timing outliers."""
import argparse
import json
from pathlib import Path
import statistics
import numpy as np

from benchmark_a100_o1 import stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    records=[json.loads(line) for line in (args.input/'results.jsonl').read_text().splitlines() if line]
    groups={}
    for r in records:
        key=(r['sample_id'],r['mode'],r['implementation'])
        groups.setdefault(key,[]).append(r)
    all_samples=sorted({r['sample_id'] for r in records})
    modes=sorted({r['mode'] for r in records})
    impls=sorted({r['implementation'] for r in records})
    summary=dict(source_run=str(args.input),sample_count=len(all_samples),modes={},
                 mse={},cv_failures=[],policy='All raw measurements retained; summaries across sample medians')
    rng=np.random.default_rng(20260913)
    for mode in modes:
        per_mode={}
        for impl in impls:
            rows=[groups[(s,mode,impl)] for s in all_samples]
            stages=rows[0][0]['timings_ms']
            stage_stats={}
            for stage in stages:
                pooled=[stats([v for r in rs for v in r['timings_ms'][stage]]) for rs in rows]
                medians=[s['median_ms'] for s in pooled]
                stage_stats[stage]=dict(median_ms=statistics.median(medians),mean_ms=statistics.fmean(medians),
                    p5_ms=float(np.percentile(medians,5)),p95_ms=float(np.percentile(medians,95)),
                    max_cv_percent=max(s['cv_percent'] for s in pooled))
                for sample,s in zip(all_samples,pooled):
                    if s['cv_percent']>=3:
                        summary['cv_failures'].append(dict(sample_id=sample,mode=mode,implementation=impl,stage=stage,cv_percent=s['cv_percent']))
            pairs={}
            for ref in ('o0','baseline'):
                if ref not in impls: continue
                metric='gemm' if mode=='compute_only' else 'total'
                ratios=[]
                for sample,rs in zip(all_samples,rows):
                    reference={r['round']:r['summary'][metric]['median_ms'] for r in groups[(sample,mode,ref)]}
                    ratios.append(statistics.median(reference[r['round']]/r['summary'][metric]['median_ms'] for r in rs))
                bootstrap=np.median(np.asarray(ratios)[rng.integers(0,len(ratios),(10000,len(ratios)))],axis=1)
                pairs[ref]=dict(paired_speedup_median=statistics.median(ratios),
                               bootstrap95=list(map(float,np.percentile(bootstrap,[2.5,97.5]))),per_sample=ratios)
            per_mode[impl]=dict(stages=stage_stats,paired_speedups=pairs)
            if mode=='compute_only':
                per_mode[impl]['logical_tops']=2*4096**3/(stage_stats['gemm']['median_ms']*1e9)
                mses=[rs[-1]['mse_vs_o0'] for rs in rows]
                summary['mse'][impl]=dict(median=statistics.median(mses),mean=statistics.fmean(mses),
                    bitwise_equal_baseline=all(r['bitwise_equal_baseline'] for rs in rows for r in rs) if impl!='o0' else None)
        summary['modes'][mode]=per_mode
    args.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
