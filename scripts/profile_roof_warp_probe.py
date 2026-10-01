#!/usr/bin/env python3
"""Capture explicit control/candidate launches; never reuse NCU time as Event time."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--cubins',type=Path,required=True)
    p.add_argument('--runs-prefix',type=Path,required=True)
    p.add_argument('--variants',nargs='+',choices=['o3','o7','o8'],default=['o3','o7'])
    a=p.parse_args()
    if (not a.directory.resolve().is_relative_to(ROOT) or a.directory.exists()
            or not a.runs_prefix.resolve().is_relative_to(ROOT) or len(set(a.variants))!=len(a.variants)):
        p.error('fresh repository report directory and valid repository run prefix required')
    ncu='/usr/local/cuda-12.8/bin/ncu'
    a.directory.mkdir(parents=True)
    receipts=[]
    def run(cmd,output):
        with output.open('w') as f:
            subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    for variant in a.variants:
        for geometry in (0,1):
            prefix=a.directory/f'ncu_{variant}_g{geometry}'
            run_dir=Path(str(a.runs_prefix)+f'_{variant}_g{geometry}')
            if run_dir.exists(): raise ValueError('refusing to reuse a profiling run')
            symbol='adangel_roof_warp_o3' if variant=='o3' else 'adangel_roof_warp_o78'
            # The one-variant, sample0/round0 launch order is always [0,1].
            # Each policy has 50 warmup launches, then its single Event sample.
            skip=50 if geometry==0 else 101
            cmd=[ncu,'--set','full','--cache-control','all','--clock-control','none',
                '--kernel-name-base','function','--kernel-name',symbol,
                '--launch-skip',str(skip),'--launch-count','1','-o',str(prefix),
                sys.executable,'scripts/benchmark_roof_warp_probe.py','--cubins',str(a.cubins),
                '--output',str(run_dir),'--variants',variant,'--samples','1',
                '--rounds','1','--warmup','50','--repeats','1']
            record=dict(variant=variant,warp_geometry=geometry,command=cmd,run_dir=str(run_dir),
                expected_profile_threads=128 if geometry==0 else 256)
            receipts.append(record)
            (a.directory/'commands.json').write_text(json.dumps(receipts,indent=2)+'\n')
            run(cmd,Path(str(prefix)+'.log'))
            for suffix,page in (('raw','raw'),('source_sass','source')):
                export=[ncu,'--import',str(prefix)+'.ncu-rep','--csv','--page',page]
                if page=='source': export+=['--print-source','sass']
                run(export,Path(str(prefix)+f'_{suffix}.csv'))
            print(variant,geometry,'captured',flush=True)
        run([sys.executable,'scripts/analyze_roof_warp_ncu.py','--directory',str(a.directory),
            '--variant',variant],a.directory/f'ncu_{variant}_analysis.json')


if __name__=='__main__': main()
