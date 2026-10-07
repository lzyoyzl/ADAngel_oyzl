#!/usr/bin/env python3
"""v132: cache a bounded O7 coefficient table in WEIGHT conversion.

Not v76's per-CTA scalar table generation, nor a table-size/layout sweep.
Fixed10 powers cover the measured O7 Af range; GPU checks every used row.
Compile gate includes enlarged supply/resources. No candidate GPU execution.
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
from inspect_o78_register_liveness import analyze
from probe_o78_eight_chain_codegen import generated_header as eight_header
from probe_roof_fullk_integer_codegen import static_entries

ROOT=Path(__file__).resolve().parents[1]
CONTROL='adangel_roof_o78_eight_chain_candidate'
SYMBOL='adangel_roof_o7_cached_coeff_candidate'
STEM='o7_cached_coeff'
SHARED=43520
LIMITS=dict(max_work_ratio=1.02,max_allocated_gpr=168,max_hot_local=1,
    min_imad_reduction=.20,min_runtime_cta=3,min_opportunity_fraction=.95,
    signed_mma=32,unsigned_mma=32,ldsm=16,copies=12,hot_barriers=1)
OLD_COPY='''  if(threadIdx.x<32) copy16(s.weight_factors[slot]+first,
      wf+group*n+blockIdx.x*128+first);'''
NEW_COPY='''  // Table lives after original32*N factors, preserving the old fallback ABI.
  // It was built in weight conversion; never count that work as free Cold.
  o1_static_for<0,3>([&](auto chunk) {
    const unsigned off=first+chunk*512;
    if(off<1280) copy16(s.coefficients[slot]+off,
        wf+32*n+(group*n+blockIdx.x*128)*10+off);
  });'''
OLD_COEFF='''            const int coefficient=s.activation_factors[slot][cute::get<0>(coord)]*
                                  s.weight_factors[slot][cute::get<1>(coord)];'''
NEW_COEFF='''            const unsigned f=static_cast<unsigned>(s.activation_factors[slot][cute::get<0>(coord)]);
            const unsigned d=31-__clz(f),column=cute::get<1>(coord);
            const int coefficient=static_cast<int>(s.coefficients[slot][
                (column/8)*80+d*8+column%8]);'''


def generated_header():
    text=eight_header((ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
    changes=(('  int weight_factors[2][128];','  uint32_t coefficients[2][1280];'),
        ('static_assert(sizeof(Storage)==34304);\nstatic_assert(sizeof(Storage)==sizeof(C::Storage));',
         'static_assert(sizeof(Storage)==43520);'),(OLD_COPY,NEW_COPY),(OLD_COEFF,NEW_COEFF))
    for old,new in changes:
        if text.count(old)!=1:raise ValueError('v78 source drift: '+old[:50])
        text=text.replace(old,new)
    return text.replace('o78_eight_chain_experiment','o7_cached_coeff_experiment')


def table_offset(group,column,d,n):
    if n<=0 or n%128 or not (0<=group<32 and 0<=column<n and 0<=d<10):
        raise ValueError('aligned N and in-range table coordinate required')
    tile,local=divmod(column,128)
    return 32*n+(group*n+tile*128)*10+(local//8)*80+d*8+local%8


def analyze_candidate(text):
    block=next(b for b in re.split(r'(?=^//-+ \.text\.)',text,flags=re.M)
        if re.match(r'//-+ \.text\.'+re.escape(SYMBOL)+r'\s',b))
    allocated=int(re.search(r'SHI_REGISTERS=(\d+)',block)[1])
    ops=[];labels={};pending=[]
    for line in block.splitlines():
        label=re.match(r'\s*(\.L_[A-Za-z0-9_]+):',line)
        if label:pending.append(label[1])
        ins=re.search(r'/\*([0-9a-fA-F]+)\*/\s*(.*?)\s*// \|\s*(\d+)\s*\|',line)
        if ins:
            pc=int(ins[1],16)
            for name in pending:labels[name]=pc
            pending.clear();ops.append((pc,ins[2].strip(),int(ins[3])))
    loops=[]
    for pc,inst,live in ops:
        branch=re.search(r'\bBRA\s+`\((\.L_[A-Za-z0-9_]+)\)',inst)
        if not branch or labels[branch[1]]>=pc:continue
        region=[x for x in ops if labels[branch[1]]<=x[0]<=pc]
        counts=Counter(re.sub(r'^@!?[A-Z0-9]+\s+','',x[1]).split()[0] for x in region)
        if sum(v for op,v in counts.items() if op.startswith('IMMA.'))!=64:continue
        kind=('fp32_fallback' if counts.get('I2F',0) else
              'cached_coeff_integer' if any(op.startswith('FLO') for op in counts) else 'integer_fallback')
        loops.append(dict(kind=kind,begin_pc=hex(region[0][0]),end_pc=hex(region[-1][0]),
            static_instructions=len(region),max_live_gpr=max(x[2] for x in region),
            opcode_counts=dict(sorted(counts.items()))))
    if len(loops)!=3 or {x['kind'] for x in loops}!={'fp32_fallback','integer_fallback','cached_coeff_integer'}:
        raise ValueError('expected exact3 complete loops: '+str(loops))
    return dict(symbol=SYMBOL,allocated_gpr=allocated,loops=loops,
        function_max_live_gpr=max(x[2] for x in ops),cta_threads=128,
        interpretation='static_binary_cost_not_dynamic_work_or_runtime_safety')


def cost_gate(old,new):
    a=next(x for x in old['loops'] if x['kind']=='integer')
    b=next(x for x in new['loops'] if x['kind']=='cached_coeff_integer')
    count=lambda x,p:sum(v for op,v in x['opcode_counts'].items() if op.startswith(p))
    ratio=b['static_instructions']/a['static_instructions']
    reduction=1-count(b,'IMAD')/count(a,'IMAD')
    checks=dict(allocation=new['allocated_gpr']<=168,hot_local=count(b,'LDL')+count(b,'STL')<=1,
        work=ratio<=1.02,integer_service=reduction>=.20,
        same_native_mma=count(b,'IMMA.16864.S4.S4')==count(b,'IMMA.16864.U4.S4')==32,
        supply=count(b,'LDSM.')==16 and count(b,'LDGSTS')==12,
        no_table_build_in_gemm=count(b,'STS')==0,hot_barrier=count(b,'BAR')==1)
    return dict(passed=all(checks.values()),checks=checks,limits=LIMITS,
        old_static=a['static_instructions'],new_static=b['static_instructions'],work_ratio=ratio,
        old_imad=count(a,'IMAD'),new_imad=count(b,'IMAD'),imad_reduction=reduction,
        scope='compile_investment_gate_excludes_startup_guard_not_measured_speedup')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=Path('reports/o378_roof_v78_codegen'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();base=args.baseline.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh repository output required')
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    prior=json.loads((base/'codegen.json').read_text())
    for name,digest in prior['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('v78 source drift: '+name)
    if sha(base/'o78_eight_chain.cubin')!=prior['cubin_sha256']:raise ValueError('v78 cubin drift')
    cuda=Path('/usr/local/cuda-12.8/bin');cutlass=ROOT/'third_party/cutlass-src'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    commit=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'release 12.8' not in version or commit!='db1c288993354c88e551c40c19a8fb93a774a241':p.error('pinned toolchain required')
    out.mkdir(parents=True)
    (out/'o78_eight_chain_generated.cuh').write_bytes((base/'o78_eight_chain_generated.cuh').read_bytes())
    (out/(STEM+'_generated.cuh')).write_text(generated_header())
    sources=set(prior['sources'])|{'scripts/probe_o7_cached_coeff_codegen.py',
        'csrc/sm80/roof_o7_cached_coeff_probe.cu','scripts/inspect_o78_register_liveness.py',
        'scripts/compare_a100_codegen.py','scripts/probe_roof_fullk_integer_codegen.py'}
    r=dict(scope='O7_cached_coefficient_compile_gate_not_GPU_acceptance',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(sources)},commands=[],limits=LIMITS,
        nvcc=version,cutlass_commit=commit,baseline_cubin_sha256=prior['cubin_sha256'],
        production_default_changed=False,new_candidate_GPU_executed=False,changed_semantics=False,
        cta_tile=[64,128,128],threads=128,stages=2,shared_bytes=SHARED,
        table_rows=10,extra_weight_bytes_4096=32*4096*10*4,
        g128_input_bytes_old=16384+256+512,g128_input_bytes_new=16384+256+5120,
        weight_conversion_must_include_table_build=True,table_range_guard_in_gemm=True)
    def run(cmd,name):
        r['commands'].append(cmd);(out/'progress.json').write_text(json.dumps(r,indent=2)+'\n')
        with (out/name).open('w') as f:
            subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,TMPDIR=str(ROOT/'tmp')),stdout=f,stderr=subprocess.STDOUT,check=True)
    flags=[str(cuda/'nvcc'),'-O3','-std=c++17','--expt-relaxed-constexpr','-lineinfo','-arch=sm_80',
        '-DADANGEL_FULLK_INTEGER=1','-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out)]
    src=str(ROOT/'csrc/sm80/roof_o7_cached_coeff_probe.cu');cubin=out/(STEM+'.cubin')
    run(flags+[src,'-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(flags+[src,'-ptx','-o',str(out/(STEM+'.ptx'))],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],STEM+'.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    sass=(out/(STEM+'.sass')).read_text();ptx=(out/(STEM+'.ptx')).read_text();ltext=(out/'liveness.txt').read_text()
    es=static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})
    live={CONTROL:analyze(ltext,CONTROL),SYMBOL:analyze_candidate(ltext)}
    control=compare((base/'o78_eight_chain.sass').read_text(),sass,'^'+CONTROL+'$')
    gate=cost_gate(live[CONTROL],live[SYMBOL]);gate['control_encoding_unchanged']=control['passed'];gate['passed'] &= control['passed']
    for s in (CONTROL,SYMBOL):
        entry=next(x for x in re.split(r'(?=\.visible \.entry )',ptx) if x.startswith('.visible .entry '+s+'('))
        assert all(x in entry for x in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32'))
    r.update(entries=es,liveness=live,control_comparison=control,cost_gate=gate,
        cubin_sha256=sha(cubin),artifact_sha256={f.name:sha(f) for f in out.iterdir()
            if f.is_file() and f.name not in ('progress.json','codegen.json')})
    (out/'codegen.json').write_text(json.dumps(r,indent=2)+'\n')
    assert control['passed'] and all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma'] and e['all_copies_bypass_l1'] for e in es.values())
    print(json.dumps(dict(cost_gate=gate,liveness=live),indent=2),flush=True)
    print('PASSED: table validation, resource/safety and full24 paired test required' if gate['passed'] else
        'FAILED: stop cached-table route, no GPU timing or neighboring table scan',flush=True)


if __name__=='__main__':main()
