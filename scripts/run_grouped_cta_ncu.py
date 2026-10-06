#!/usr/bin/env python3
"""v89 paired NCU diagnosis of validated grouped/control kernels, not speedups."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys

from analyze_roof_scale_ncu import analyze
from probe_grouped_cta_codegen import ROOT, CONFIG
from run_eight_chain_ncu import normalized_counts


def analyze_capture(raw,source,receipt):
    v=receipt['variant'];cfg=CONFIG['o3' if v=='o3' else 'o78']
    symbol=cfg['control'] if receipt['policy']==0 else cfg['symbol']
    if (receipt['expected_kernel']!=symbol or not receipt['numerical_checks_passed']
            or not receipt['bitwise_previous_best'] or receipt['guard']['integer_ctas']!=2048
            or receipt['guard']['fallback_ctas'] or receipt['guard']['invalid_ctas']):
        raise ValueError('profile identity, guard or output mismatch')
    expected_sha=receipt['gemm_codegen']['baseline_cubin_sha256'] if receipt['policy']==0 else receipt['gemm_codegen']['cubin_sha256']
    if receipt['gemm_binary_sha256']!=expected_sha:raise ValueError('actual CUBIN drift')
    result=analyze(raw,source,54 if v=='o3' else 59,v,True,True,
        expected_symbol=symbol,fullk_integer=True)
    metric=list(csv.DictReader(io.StringIO(raw)))[1]
    if int(metric['launch__block_size'].replace(',',''))!=128 or int(metric['launch__grid_size'].replace(',',''))!=2048:
        raise ValueError('launch work changed')
    counts=Counter()
    for row in csv.DictReader(io.StringIO(source.split('\n',1)[1])):
        match=re.match(r'\s*(?:@!?U?P(?:T|\d+)\s+)?([A-Z][A-Z0-9_]*)',row['Source'])
        if not match:raise ValueError('unknown source opcode')
        counts[match[1]]+=1
    expected=receipt['gemm_codegen']['entries'][symbol]
    if counts!=normalized_counts(expected['opcode_counts']) or sum(counts.values())!=expected['instructions']:
        raise ValueError('profile static SASS fingerprint mismatch')
    # Preserve selected raw memory metrics rather than inferring hit rate from
    # duration or calling excessive wavefronts DRAM bytes/bank conflicts.
    memory={key:value for key,value in metric.items() if key and key.startswith(('lts__','dram__','l1tex__'))}
    result.update(policy=receipt['policy'],static_fingerprint_verified=True,
        mse_vs_paired_fp16=receipt['mse_vs_paired_fp16'],mse_vs_previous_best=receipt['mse_vs_previous_best'],
        memory_metrics_raw=memory)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variants',nargs='+',choices=('o3','o7','o8'),default=['o3'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--analyze-only',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    if not out.is_relative_to(ROOT) or (out.exists() and not a.analyze_only) or len(set(a.variants))!=len(a.variants):
        p.error('fresh repository output required for collection')
    ncu='/usr/local/cuda-12.8/bin/ncu';commands=[]
    def run(cmd,name):
        commands.append(dict(command=cmd,output=name))
        (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        with (out/name).open('w') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    if not a.analyze_only:
        out.mkdir(parents=True);run([ncu,'--version'],'ncu_version.txt')
        for v in a.variants:
            cfg=CONFIG['o3' if v=='o3' else 'o78']
            for policy in (0,1):
                tag=f'{v}_{policy}';prefix=out/tag
                symbol=cfg['control'] if policy==0 else cfg['symbol']
                run([ncu,'--set','full','--cache-control','all','--clock-control','none',
                    '--kernel-name-base','function','--kernel-name',symbol,'--launch-skip',str(51 if policy==0 else 50),
                    '--launch-count','1','-o',str(prefix),sys.executable,'scripts/profile_grouped_cta_kernel.py',
                    '--variant',v,'--policy',str(policy),'--output',str(prefix)],tag+'.log')
                run([ncu,'--import',str(prefix)+'.ncu-rep','--csv','--page','raw'],tag+'_raw.csv')
                run([ncu,'--import',str(prefix)+'.ncu-rep','--csv','--page','source','--print-source','sass'],tag+'_source_sass.csv')
                print(tag,'captured',flush=True)
    rows=[];hashes={}
    for v in a.variants:
        for policy in (0,1):
            tag=f'{v}_{policy}';paths=[out/(tag+'_raw.csv'),out/(tag+'_source_sass.csv'),out/tag/'receipt.json']
            rows.append(analyze_capture(paths[0].read_text(encoding='utf-8-sig'),paths[1].read_text(encoding='utf-8-sig'),
                json.loads(paths[2].read_text())))
            hashes.update({str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths})
    (out/'analysis.json').write_text(json.dumps(dict(scope='v89_paired_NCU_diagnostic_not_Event_speedup',
        rows=rows,input_sha256=hashes,new_performance_result=False,production_default_changed=False),indent=2,allow_nan=False)+'\n')
    print('GROUPED CTA NCU IDENTITY/WORK CHECKS PASSED',flush=True)


if __name__=='__main__':main()
