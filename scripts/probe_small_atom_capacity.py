#!/usr/bin/env python3
"""v102: new INT4 atom capacity gate, not real GEMM/MSE or a tile sweep.

Compare equal physical MAC work and32 logical partial registers, at3 CTA/SM.
m8n8k32 exposes16 source chains but needs4x MMA instructions. Observe all
result registers in both modes. No candidate GEMM without >=10% paired
capacity gain. v82 had fewer checksum observations, so do not mix its timing
with this matched control. No input/scale/FP32/full-K-accumulator work here.
"""
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
SYMBOLS=('adangel_capacity_large_observed','adangel_capacity_small_observed')


def physical_work(row):
    return row['blocks']*(row['threads']//32)*row['groups']*row['mma_per_warp_group']*2*__import__('math').prod(row['shape'])


def audit(sass):
    sections=split_sections(sass,SASS_FUNCTION)
    result={}
    for mode,symbol in enumerate(SYMBOLS):
        ops=[(int(m[1],16),m[2],m[3]) for m in SASS_INSTRUCTION.finditer(sections[symbol])]
        expected=256 if mode else 64
        wanted='IMMA.8832' if mode else'IMMA.16864'
        loops=[]
        for pc,op,args in ops:
            if op!='BRA':continue
            match=re.search(r'0x([0-9a-fA-F]+)',args)
            if match and int(match[1],16)<pc:
                low=int(match[1],16)
                selected=[o for address,o,_ in ops if low<=address<=pc]
                if sum(o.startswith('IMMA.') for o in selected)==expected:
                    loops.append((low,pc,selected))
        if len(loops)!=1:raise ValueError('exact runtime group loop missing: '+symbol)
        low,high,selected=loops[0];counts=Counter(selected)
        if counts[wanted+'.S4.S4']!=expected//2 or counts[wanted+'.U4.S4']!=expected//2:
            raise ValueError('wrong native INT4 SASS shape/work: '+symbol)
        if any(op.startswith('IMMA.') and not op.startswith(wanted+'.') for op in counts):
            raise ValueError('unexpected MMA lowering')
        if any(op.startswith(('LDL','STL','LDG','LDS','STS','STG')) for op in selected):
            raise ValueError('capacity loop contains memory')
        result[symbol]=dict(begin_pc=hex(low),end_pc=hex(high),instructions=len(selected),
            opcode_counts=dict(sorted(counts.items())),hot_memory_instructions=0,
            source_chains=16 if mode else 8,logical_partial_registers=32,
            static_native_mma_count=expected,programmed_chains_not_hardware_concurrency=True)
    return result


def summarize(rows):
    import numpy as np
    if len(rows)!=2 or {r['mode'] for r in rows}!={0,1}:raise ValueError('two matched records required')
    rows=sorted(rows,key=lambda r:r['mode']);items=[]
    for mode,r in enumerate(rows):
        if (r['shape']!=([16,8,64],[8,8,32])[mode] or r['blocks']!=2048 or r['threads']!=128
                or r['groups']!=256 or r['mma_per_warp_group']!=(64,256)[mode]
                or r['logical_partial_registers']!=32 or r['source_chains']!=(8,16)[mode]
                or r['active_ctas_per_sm']!=3 or r['warmup']!=50 or r['validation_checks']!=24
                or not r['checksum_passed'] or len(r['raw_ms'])!=200 or min(r['raw_ms'])<=0):
            raise ValueError('capacity diagnostic contract drift')
        s=stats(r['raw_ms']);work=physical_work(r)
        items.append(dict(mode=mode,shape=r['shape'],statistics=s,physical_int4_ops=work,
            measured_physical_tops=work/s['median_ms']/1e9,
            normalized_32_group_ms=s['median_ms']/8,cv_passed=s['cv_percent']<3))
    if items[0]['physical_int4_ops']!=items[1]['physical_int4_ops']:raise ValueError('unequal physical work')
    ratios=np.array(rows[0]['raw_ms'])/np.array(rows[1]['raw_ms'])
    estimate=float(np.median(ratios))
    rng=np.random.default_rng(102);boot=np.median(ratios[rng.integers(0,200,size=(10000,200))],axis=1)
    ci=[float(v) for v in np.percentile(boot,[2.5,97.5])]
    return dict(records=items,paired_capacity_ratio=estimate,paired_bootstrap_95_ci=ci,
        full_GEMM_worth_implementing=ci[0]>1.10,
        gate='Require >=10% lower-CI capacity gain before a real GEMM implementation; no retry-to-pass.',
        scope='matched_MMA_capacity_only_not_GEMM_MSE_attainable_peak_or_real_case_speedup',
        no_filtering=True,production_default_changed=False,new_experiment_MSE_measured=False,
        note='Both modes use the same32 affine C seeds to prevent common starts, include64 checksum additions/group and observe every D register. No real payload/shared loads, group factors, full-K accumulator or FP32 output. Source chain count is not measured hardware concurrency. Do not compare absolute timing with v82, whose checksum work differs.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--run',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh project output required')
    cuda=Path('/usr/local/cuda-12.8/bin')
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    if 'release12.8' not in version.replace(' ',''):raise ValueError('CUDA12.8 required')
    source=ROOT/'csrc/sm80/mma_small_atom_capacity_probe.cu'
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    out.mkdir(parents=True)
    receipt=dict(scope='v102_small_INT4_atom_capacity_only',production_default_changed=False,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={str(f.relative_to(ROOT)):sha(f) for f in (source,Path(__file__))},nvcc=version,commands=[])
    def run(cmd,name):
        receipt['commands'].append(cmd)
        with (out/name).open('w') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    binary=out/'small_atom_capacity'
    flags=[str(cuda/'nvcc'),'-O3','-std=c++17','-arch=sm_80','-lineinfo',str(source)]
    run(flags+['-o',str(binary),'-Xptxas=-v'],'build.log')
    run(flags+['-ptx','-o',str(out/'kernel.ptx')],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(binary)],'kernel.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(binary)],'resources.txt')
    receipt.update(binary_sha256=sha(binary),entries=audit((out/'kernel.sass').read_text()))
    (out/'codegen.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print((out/'build.log').read_text());print('native INT4 matched-work capacity audit passed',flush=True)
    if a.run:
        query=['nvidia-smi','--query-gpu=name,clocks.current.sm,temperature.gpu,power.draw','--format=csv,noheader']
        run(query,'gpu_before.txt');run([str(binary)],'results.jsonl');run(query,'gpu_after.txt')
        rows=[json.loads(s) for s in (out/'results.jsonl').read_text().splitlines()]
        summary=summarize(rows)
        (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        receipt['artifact_sha256']={f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='codegen.json'}
        (out/'codegen.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
