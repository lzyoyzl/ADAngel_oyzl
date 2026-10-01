#!/usr/bin/env python3
"""Profile selected producer/control launches; these are not Event measurements."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from analyze_roof_scale_ncu import analyze

ROOT=Path(__file__).resolve().parents[1]


def analyze_profile(raw,source,variant,policy,expected_resources):
    tune=54 if variant=='o3' else 59
    symbol='adangel_roof_producer_o3' if variant=='o3' else 'adangel_roof_producer_o78'
    row=analyze(raw,source,tune,variant,True,True,expected_symbol=symbol)
    metrics=list(csv.DictReader(io.StringIO(raw)))[1]
    threads=int(metrics['launch__block_size'].replace(',',''))
    if threads!=expected_resources['threads'] or row['registers_per_thread']!=expected_resources['registers_per_thread']:
        raise ValueError('NCU launch identity/resources do not match requested probe')
    row.update(threads=threads,producer_mode=policy,reference_math_tune=row.pop('tune'),
               probe_resources=expected_resources)
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--cubins',type=Path,required=True)
    p.add_argument('--runs-prefix',type=Path,required=True)
    p.add_argument('--variants',nargs='+',choices=['o3','o7','o8'],default=['o3','o7'])
    p.add_argument('--candidate',type=int,choices=[1,2],default=1)
    a=p.parse_args()
    if (not a.directory.resolve().is_relative_to(ROOT) or a.directory.exists()
            or not a.runs_prefix.resolve().is_relative_to(ROOT) or len(set(a.variants))!=len(a.variants)):
        p.error('fresh repository report directory and valid run prefix required')
    ncu='/usr/local/cuda-12.8/bin/ncu';a.directory.mkdir(parents=True)
    receipts=[]
    def run(cmd,output):
        with output.open('w') as f:
            subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    for variant in a.variants:
        rows=[];sources=[]
        for policy in (0,a.candidate):
            prefix=a.directory/f'ncu_{variant}_p{policy}'
            run_dir=Path(str(a.runs_prefix)+f'_{variant}_p{policy}')
            if run_dir.exists(): raise ValueError('profiling run already exists')
            symbol='adangel_roof_producer_o3' if variant=='o3' else 'adangel_roof_producer_o78'
            # Sample0/variant0/round0: [0,candidate], 50 warmups and 1 sample each.
            skip=50 if policy==0 else 101
            cmd=[ncu,'--set','full','--cache-control','all','--clock-control','none',
                '--kernel-name-base','function','--kernel-name',symbol,
                '--launch-skip',str(skip),'--launch-count','1','-o',str(prefix),
                sys.executable,'scripts/benchmark_roof_producer_probe.py','--cubins',str(a.cubins),
                '--output',str(run_dir),'--variants',variant,'--samples','1','--rounds','1',
                '--warmup','50','--repeats','1','--policies','0',str(a.candidate)]
            receipts.append(dict(variant=variant,producer_mode=policy,command=cmd,run_dir=str(run_dir)))
            (a.directory/'commands.json').write_text(json.dumps(receipts,indent=2)+'\n')
            run(cmd,Path(str(prefix)+'.log'))
            texts=[]
            for suffix,page in (('raw','raw'),('source_sass','source')):
                export=[ncu,'--import',str(prefix)+'.ncu-rep','--csv','--page',page]
                if page=='source': export+=['--print-source','sass']
                path=Path(str(prefix)+f'_{suffix}.csv');run(export,path)
                payload=path.read_bytes();texts.append(payload.decode('utf-8-sig'))
                sources.append(dict(file=str(path),sha256=hashlib.sha256(payload).hexdigest()))
            measured=[json.loads(s) for s in (run_dir/'results.jsonl').read_text().splitlines()]
            chosen=[r for r in measured if r['producer_mode']==policy]
            if len(chosen)!=1: raise ValueError('missing or duplicate profiling result')
            row=analyze_profile(*texts,variant,policy,chosen[0]['probe_resources']);rows.append(row)
            print(variant,policy,'captured',flush=True)
        (a.directory/f'ncu_{variant}_analysis.json').write_text(json.dumps(dict(
            scope='one-launch diagnostic, not Event timing, O8 inference or production acceptance',
            rows=rows,sources=sources),indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
