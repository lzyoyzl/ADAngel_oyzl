#!/usr/bin/env python3
"""Recompute complete v92 Event evidence; retain first run and any retry."""
import argparse
import hashlib
import json
from pathlib import Path

from analyze_grouped_cta_retest import ROOT, summarize


def read_run(directory):
    env=json.loads((directory/'environment.json').read_text()); args=env['args']
    if args['samples']!=24 or env['production_default_changed']:
        raise ValueError('all24 independent-candidate evidence required')
    validation=json.loads((directory/'validation.json').read_text())
    if not validation['passed']: raise ValueError('GPU correctness gate did not pass')
    rows=[json.loads(line) for line in (directory/'results.jsonl').read_text().splitlines()]
    summaries=summarize(rows,args['rounds'],args['repeats'])
    if {r['mode'] for r in rows}!={'compute_only'}:
        raise ValueError('this report covers cached GEMM only')
    provenance=json.loads((directory/'input_provenance.json').read_text()) if (directory/'input_provenance.json').exists() else env
    symbols=sorted({r.get('kernel',r.get('resources',{}))['kernel_symbol'] for r in rows})
    identity=dict(extension_sha256=env['extension_sha256'],
        raw_manifest_sha256=provenance['raw_manifest_sha256'],
        prepared_manifest_sha256=provenance['prepared_manifest_sha256'],symbols=symbols,
        counts={name:args[name] for name in ('samples','rounds','warmup','repeats','inner')})
    return dict(run=str(directory),records=len(rows),summary=summaries,identity=identity,
        hashes={name:hashlib.sha256((directory/name).read_bytes()).hexdigest()
            for name in ('environment.json','validation.json','results.jsonl','summary.json')},
        no_filtering=True,new_conversion_or_E2E_result=False)


def compare(first,retry):
    if first['identity']!=retry['identity']: raise ValueError('retry input/kernel/count identity changed')
    key=lambda x:(x['variant'],x['policy'])
    old={key(x):x for x in first['summary']}; new={key(x):x for x in retry['summary']}
    if set(old)!=set(new): raise ValueError('retry coverage drift')
    for k in old:
        if (old[k]['median_mse'],old[k]['mean_mse'])!=(new[k]['median_mse'],new[k]['mean_mse']):
            raise ValueError('retry MSE drift')
    return [dict(variant=k[0],policy=k[1],first=old[k],retry=new[k]) for k in sorted(old)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--o3',type=Path,required=True)
    p.add_argument('--o78',type=Path,required=True)
    p.add_argument('--o3-retry',type=Path)
    p.add_argument('--o78-retry',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT):
        p.error('fresh repository output required')
    runs={kind:read_run(getattr(args,kind)) for kind in ('o3','o78')}
    result=dict(runs=runs,no_filtering=True,production_default_changed=False,
        scope='complete all24 paired Event/correctness evidence; not NCU or a causal stall decomposition')
    retries={}
    for kind in ('o3','o78'):
        directory=getattr(args,kind+'_retry')
        if directory:
            retry=read_run(directory);retries[kind+'_retry']=retry
            result[kind+'_retry']=retry
            result[kind+'_repeat_comparison']=compare(runs[kind],retry)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for kind,run in list(runs.items())+list(retries.items()):
        for row in run['summary']:
            print(kind,row['variant'],row['policy'],'ms=',row['median_ms'],
                'paired_gain_percent=',100*(row['paired_speedup']-1),
                'CI=',row['paired_speedup_ci95'],'CV_failed=',row['selected_cv_failed_records'],
                '/',row['records'],flush=True)


if __name__=='__main__':main()
