#!/usr/bin/env python3
"""Measure native MMA issue capacity at 3 CTA/SM; not O3/O7/O8 performance."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

from analyze_o3_mma_lowering import SASS_FUNCTION, SASS_INSTRUCTION, split_sections
from benchmark_a100_o1 import stats

ROOT=Path(__file__).resolve().parents[1]
SYMBOLS=('adangel_capacity_signed','adangel_capacity_unsigned','adangel_capacity_merged')


def audit(sass):
    sections=split_sections(sass,SASS_FUNCTION)
    result={}
    for mode,symbol in enumerate(SYMBOLS):
        ops=[(int(m[1],16),m[2],m[3]) for m in SASS_INSTRUCTION.finditer(sections[symbol])]
        counts=Counter(op for _,op,_ in ops)
        signed=counts['IMMA.16864.S4.S4'];unsigned=counts['IMMA.16864.U4.S4']
        if (signed,unsigned)!=((64,0),(0,64),(32,32))[mode]:raise ValueError('wrong native MMA work')
        if any(counts[op] for op in counts if op.startswith(('LDL','STL','LDSM')) or '.S8' in op or '.U8' in op):
            raise ValueError('unexpected local/fragment work or INT8 lowering')
        loops=[]
        for pc,opcode,args in ops:
            if opcode!='BRA':continue
            target=re.search(r'0x([0-9a-fA-F]+)',args)
            if target and int(target[1],16)<pc:
                selected=[op for address,op,_ in ops if int(target[1],16)<=address<=pc]
                if sum(op.startswith('IMMA.') for op in selected)==64:
                    loops.append(dict(begin=hex(int(target[1],16)),end=hex(pc),instructions=len(selected)))
        if len(loops)!=1:raise ValueError('one runtime group loop containing exactly64 MMA required')
        result[symbol]=dict(opcode_counts=dict(sorted(counts.items())),loop=loops[0],
                           native_signed=signed,native_unsigned=unsigned)
    return result


def summarize(rows):
    if {r['mode'] for r in rows}!={0,1,2} or len(rows)!=3:raise ValueError('all three capacity modes required')
    result=[]
    for r in rows:
        if (not r['checksum_passed'] or r['validation_checks']!=24 or r['blocks']!=2048
                or r['threads']!=128 or r['groups']!=256 or r['mma_per_warp_group']!=64
                or r['active_ctas_per_sm']!=3 or len(r['raw_ms'])!=200 or min(r['raw_ms'])<=0):
            raise ValueError('diagnostic scope/work/validation mismatch')
        s=stats(r['raw_ms'])
        instructions=r['blocks']*(r['threads']//32)*r['groups']*r['mma_per_warp_group']
        operations=instructions*2*16*8*64
        result.append(dict(mode=r['mode'],symbol=r['symbol'],statistics=s,
            dynamic_mma_work=instructions,physical_int4_ops=operations,
            measured_physical_tops=operations/s['median_ms']/1e9,
            normalized_32_group_ms=s['median_ms']*32/r['groups'],
            cv_passed=s['cv_percent']<3))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh project output required')
    out.mkdir(parents=True);cuda=Path('/usr/local/cuda-12.8/bin')
    source=ROOT/'csrc/sm80/mma_issue_capacity_probe.cu'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    if 'release 12.8' not in version:raise ValueError('CUDA12.8 required')
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    receipt=dict(scope='native_MMA_capacity_diagnostic_not_GEMM_or_MSE_or_attainable_kernel_peak',
        production_default_changed=False,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={str(s.relative_to(ROOT)):digest(s) for s in (source,Path(__file__))},nvcc=version,commands=[])
    def run(cmd,name):
        receipt['commands'].append(cmd)
        with (out/name).open('w') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    binary=out/'mma_issue_capacity'
    run([str(cuda/'nvcc'),'-O3','-std=c++17','-arch=sm_80','-lineinfo',str(source),'-o',str(binary),'-Xptxas=-v'],'build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(binary)],'kernel.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(binary)],'resources.txt')
    receipt.update(binary_sha256=digest(binary),entries=audit((out/'kernel.sass').read_text()))
    (out/'codegen.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print((out/'build.log').read_text());print('exact native MMA loop audit passed',flush=True)
    if a.run:
        run([str(binary)],'results.jsonl')
        rows=[json.loads(line) for line in (out/'results.jsonl').read_text().splitlines()]
        summary=dict(scope=receipt['scope'],records=summarize(rows),production_default_changed=False,
            no_filtering=True,original_experiment_MSE_measured=False,
            note='256-group instruction diagnostic normalized by8; excludes real payload supply, scale, accumulator and output work. Not a new strict lower bound or a speedup over any GEMM.')
        (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        (out/'codegen.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
