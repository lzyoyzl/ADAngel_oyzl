#!/usr/bin/env python3
"""Compile one four-MMA/four-scale-warp candidate after full24 INT16 proof.

No parameter sweep. Preserve original INT4 operand reuse; unlike v98 helpers
perform only weighted integer accumulation. No launch or performance claim.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from probe_o78_eight_chain_codegen import generated_header as eight_header, MERGED
from probe_eight_warp_fullk_codegen import generated_header as warp_header, generated_fallback

ROOT=Path(__file__).resolve().parents[1]
SYMBOL='adangel_roof_o78_partial_handoff_candidate'
CONTROL='adangel_roof_o78_eight_chain_candidate'
LIMITS=dict(max_registers=128, max_hot_local=0, minimum_active_ctas=2,
            max_weighted_static_ratio=1.75, required_mma_each=32, required_ldsm=16)


def producer_header():
    text=eight_header((ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
    text=text[text.index('__device__ __forceinline__ void body'):text.index('  // Reuse the 64 INT32')]
    text=text.replace('void body(', 'void producer(')
    text=text.replace('auto& s=*reinterpret_cast<Storage*>(buf);',
                      'auto& h=*reinterpret_cast<Shared*>(buf);auto& s=h.matrix;')
    text='\n'.join(l for l in text.split('\n') if not any(t in l for t in (
        'auto coords=','auto acc=','cute::size<1>(acc)','cute::size(acc)')))
    text=text.replace('__syncthreads();', 'wait<0,128>();')
    old=MERGED[MERGED.rindex('      o1_static_for<0,2>'):]
    if text.count(old)!=1:raise ValueError('original scale-update boundary drift')
    new='''      constexpr int buffer=decltype(nb)::value;
      wait<2+2*buffer,256>();
      o1_static_for<0,4>([&](auto q) {
        const uint4 v=make_uint4(pack(partial(q*8),partial(q*8+1)),
            pack(partial(q*8+2),partial(q*8+3)),
            pack(partial(q*8+4),partial(q*8+5)),
            pack(partial(q*8+6),partial(q*8+7)));
        h.partial[buffer][q][threadIdx.x]=v;
      });
      h.factors[buffer][threadIdx.x]=threadIdx.x<64 ?
          s.activation_factors[slot][threadIdx.x] :
          s.weight_factors[slot][buffer*64+threadIdx.x-64];
      arrive<1+2*buffer>();
'''
    return '// Generated from the exact v78 four-warp MMA/payload schedule.\n'+text.replace(old,new)+'}\n'


def inspect_live(text):
    block=next(b for b in re.split(r'(?=^//-+ \.text\.)',text,flags=re.M)
        if re.match(r'//-+ \.text\.'+SYMBOL+r'\s',b))
    regs=int(re.search(r'SHI_REGISTERS=(\d+)',block)[1])
    ins=[];labels={};pending=[]
    for line in block.splitlines():
        lab=re.match(r'\s*(\.L_[A-Za-z0-9_]+):',line)
        if lab:pending.append(lab[1])
        m=re.search(r'/\*([0-9a-fA-F]+)\*/\s*(.*?)\s*// \|\s*(\d+)\s*\|',line)
        if m:
            pc=int(m[1],16)
            for label in pending:labels[label]=pc
            pending=[];ins.append((pc,m[2].strip(),int(m[3])))
    loops=[]
    for pc,op,_ in ins:
        m=re.search(r'\bBRA\s+`\((\.L_[A-Za-z0-9_]+)\)',op)
        if not m or labels.get(m[1],pc)>=pc:continue
        region=[x for x in ins if labels[m[1]]<=x[0]<=pc]
        counts=Counter(re.sub(r'^@!?[A-Z0-9]+\s+','',x[1]).split()[0] for x in region)
        count=lambda s:sum(v for k,v in counts.items() if k.startswith(s))
        if count('IMMA') not in (32,64) and count('BAR')<4:continue
        kind=('producer' if count('IMMA')==64 else ('fp32_fallback' if count('I2F') else
             'integer_fallback')) if count('IMMA') else 'consumer'
        loops.append(dict(kind=kind,begin_pc=hex(region[0][0]),end_pc=hex(pc),
            static_instructions=len(region),max_live=max(x[2] for x in region),
            opcode_counts=dict(sorted(counts.items()))))
    return dict(registers=regs,loops=loops)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-gate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh project output required')
    data=json.loads((args.data_gate/'summary.json').read_text())
    if not data['investment_gate_passed'] or data['records']!=48:raise ValueError('full24 data gate failed')
    prior_dir=ROOT/'reports/o378_roof_v78_codegen'
    prior=json.loads((prior_dir/'codegen.json').read_text())
    for path,digest in prior['sources'].items():
        if sha(ROOT/path)!=digest:raise ValueError('old source drift: '+path)
    if sha(prior_dir/'o78_eight_chain.cubin')!=prior['cubin_sha256']:raise ValueError('old binary drift')
    cuda=Path('/usr/local/cuda-12.8/bin');cutlass=ROOT/'third_party/cutlass-src'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    commit=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'release 12.8' not in version or commit!='db1c288993354c88e551c40c19a8fb93a774a241':
        raise ValueError('pinned toolchain required')
    out.mkdir(parents=True)
    headers={'o78_eight_chain_generated.cuh':eight_header((ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()),
        'o78_eight_warp_fullk_generated.cuh':warp_header('o78'),
        'o78_eight_warp_fullk_fallback_generated.cuh':generated_fallback('o78'),
        'o78_partial_handoff_producer_generated.cuh':producer_header()}
    for name,body in headers.items():(out/name).write_text(body)
    sources=set(prior['sources'])|{'scripts/probe_partial_handoff_codegen.py',
        'scripts/probe_eight_warp_fullk_codegen.py','scripts/compare_a100_codegen.py',
        'csrc/sm80/roof_o78_eight_warp_fullk_probe.cu','csrc/sm80/o78_warp_geometry_probe.cuh',
        'csrc/sm80/o78_partial_handoff_probe.cuh','csrc/sm80/roof_o78_partial_handoff_probe.cu'}
    r=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(sources)},headers={n:sha(out/n) for n in headers},
        data_gate_sha256=sha(args.data_gate/'summary.json'),limits=LIMITS,
        shared_bytes=51712,threads=256,stages=2,cta_tile=[64,128,128],
        no_candidate_launched=True,no_performance_or_MSE_measured=True,production_default_changed=False,
        nvcc=version,cutlass_commit=commit,commands=[])
    def run(cmd,name):
        r['commands'].append(cmd)
        with (out/name).open('w') as f:
            subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,
                env={**os.environ,'TMPDIR':str(ROOT/'tmp')})
    flags=[str(cuda/'nvcc'),'-O3','-std=c++17','--expt-relaxed-constexpr','-lineinfo','-arch=sm_80',
        '-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out),
        str(ROOT/'csrc/sm80/roof_o78_partial_handoff_probe.cu')]
    cubin=out/'o78_partial_handoff.cubin'
    run(flags+['-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(flags+['-ptx','-o',str(out/'o78_partial_handoff.ptx')],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],'o78_partial_handoff.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    sass=(out/'o78_partial_handoff.sass').read_text();live=inspect_live((out/'liveness.txt').read_text())
    control=compare((prior_dir/'o78_eight_chain.sass').read_text(),sass,'^'+CONTROL+'$')
    roles={l['kind']:l for l in live['loops']}
    if len(roles)!=4 or len(live['loops'])!=4:raise ValueError('four exact producer/consumer/fallback loops required')
    producer,consumer=roles['producer'],roles['consumer']
    c=producer['opcode_counts'];ratio=(producer['static_instructions']+consumer['static_instructions'])/383
    hot_local=sum(v for l in (producer,consumer) for op,v in l['opcode_counts'].items() if op.startswith(('LDL','STL')))
    checks=dict(control_unchanged=control['passed'],register_budget=live['registers']<=128,
        no_hot_local=hot_local==0,weighted_instruction_budget=ratio<=LIMITS['max_weighted_static_ratio'],
        same_four_warp_native_mma=c.get('IMMA.16864.S4.S4')==c.get('IMMA.16864.U4.S4')==32,
        same_operand_reuse=c.get('LDSM.16.M88.4')==16,
        helper_has_no_mma=not any(op.startswith('IMMA') for op in consumer['opcode_counts']))
    r.update(liveness=live,control_comparison=control,cubin_sha256=sha(cubin),
        compile_gate=dict(passed=all(checks.values()),checks=checks,
            weighted_instruction_ratio=ratio,hot_local_instructions=hot_local),
        artifact_sha256={n:sha(out/n) for n in ('build.log','resources.txt','liveness.txt',
            'o78_partial_handoff.sass','o78_partial_handoff.ptx')})
    (out/'codegen.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r['compile_gate'],indent=2),flush=True)


if __name__=='__main__':main()
