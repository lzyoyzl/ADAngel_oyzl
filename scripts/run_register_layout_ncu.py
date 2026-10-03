#!/usr/bin/env python3
"""v86: why v85 failed. Two fixed kernels, full NCU, never Event speedups."""
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
from profile_register_layout_kernel import ROOT,SYMBOLS
from run_eight_chain_ncu import normalized_counts


def analyze_capture(raw,source,receipt):
    symbol=SYMBOLS[receipt['policy']]
    if receipt['expected_kernel']!=symbol or receipt['variant']!='o7':raise ValueError('wrong identity')
    if not all(receipt[k] for k in ('numerical_checks_passed','bitwise_previous_fullk','packed_candidate_verified')):
        raise ValueError('numerical acceptance missing')
    guard=receipt['guard']
    if guard['integer_ctas']!=2048 or guard['fallback_ctas'] or guard['invalid_ctas']:raise ValueError('integer work required')
    result=analyze(raw,source,59,'o7',True,True,expected_symbol=symbol,fullk_integer=True)
    metrics=list(csv.DictReader(io.StringIO(raw)))[1]
    if int(metrics['launch__block_size'].replace(',',''))!=128 or int(metrics['launch__grid_size'].replace(',',''))!=2048:
        raise ValueError('wrong launch')
    if result['registers_per_thread']!=receipt['resources']['registers_per_thread']:raise ValueError('wrong registers')
    actual=Counter()
    for row in csv.DictReader(io.StringIO(source.split('\n',1)[1])):
        match=re.match(r'\s*(?:@!?U?P(?:T|\d+)\s+)?([A-Z][A-Z0-9_]*)',row['Source'])
        if not match:raise ValueError('unknown opcode')
        actual[match[1]]+=1
    expected=receipt['gemm_codegen']['entries'][symbol]
    if actual!=normalized_counts(expected['opcode_counts']) or sum(actual.values())!=expected['instructions']:
        raise ValueError('profile does not match audited cubin')
    result.update(policy=receipt['policy'],static_fingerprint_verified=True,
        mse_vs_paired_fp16=receipt['mse_vs_paired_fp16'],mse_vs_previous_fullk=receipt['mse_vs_previous_fullk'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--analyze-only',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    if not out.is_relative_to(ROOT) or (out.exists() and not a.analyze_only):p.error('fresh repository output required')
    if not a.analyze_only:
        out.mkdir(parents=True);commands=[];ncu='/usr/local/cuda-12.8/bin/ncu'
        def run(cmd,name):
            commands.append(dict(command=cmd,output=name))
            (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
            with (out/name).open('w') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        run([ncu,'--version'],'ncu_version.txt')
        for policy,symbol in enumerate(SYMBOLS):
            name=f'o7_p{policy}';prefix=out/name
            run([ncu,'--set','full','--cache-control','all','--clock-control','none',
                '--kernel-name-base','function','--kernel-name',symbol,'--launch-skip','50','--launch-count','1',
                '-o',str(prefix),sys.executable,'scripts/profile_register_layout_kernel.py',
                '--policy',str(policy),'--output',str(prefix)],name+'.log')
            for suffix,page in (('raw','raw'),('source_sass','source')):
                cmd=[ncu,'--import',str(prefix)+'.ncu-rep','--csv','--page',page]
                if page=='source':cmd+=['--print-source','sass']
                run(cmd,f'{name}_{suffix}.csv')
            print(name,'captured',flush=True)
    rows=[];hashes={}
    for policy in (0,1):
        prefix=f'o7_p{policy}'
        paths=[out/f'{prefix}_{suffix}.csv' for suffix in ('raw','source_sass')]
        receipt=out/prefix/'receipt.json'
        rows.append(analyze_capture(*(f.read_text(encoding='utf-8-sig') for f in paths),json.loads(receipt.read_text())))
        hashes.update({str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [*paths,receipt]})
    (out/'analysis.json').write_text(json.dumps(dict(scope='existing_layout_NCU_diagnostic_not_new_speedup',
        rows=rows,input_sha256=hashes,new_performance_result=False,production_default_changed=False),indent=2,allow_nan=False)+'\n')
    print('REGISTER LAYOUT NCU IDENTITY/WORK CHECKS PASSED',flush=True)


if __name__=='__main__':main()
