#!/usr/bin/env python3
"""v125: split low/high factor-weighted updates, not merged MMA or DP2A.

Keep O3 v89 guard, buffers, grouped CTA, copy stages and epilogue. Exact modulo
2**32 mad.lo prevents C++ signed-overflow UB in separate cancelling terms; the
old guard still proves the reconstructed group and each prefix fit INT32.
Only compiler/source evidence here; no GPU launch or performance claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from inspect_eight_chain_schedule import instructions
from inspect_o78_register_liveness import analyze
from probe_grouped_cta_codegen import checked as checked_grouped, generated_headers
from probe_roof_fullk_integer_codegen import static_entries

ROOT=Path(__file__).resolve().parents[1]
CONTROL='adangel_roof_o3_grouped_cta_candidate'
SYMBOL='adangel_roof_o3_split_weighted_candidate'
STEM='o3_split_weighted'
LIMITS=dict(max_work_ratio=1.08,max_allocated_gpr=168,max_hot_local=0,
            chains=32,mma_per_chain=2,min_chain_peak=8,mma_each=32,
            ldsm=16,copies=9,barriers=1)
BEGIN='      // Eight independent chains: two M atoms x four N atoms.\n'
END='    });\n  }\n  auto final_value='
BODY='''      // Eight independent 2-MMA chains: four low and four high. The two
      // M atoms are streamed through the same32 logical partial registers.
      // Neither signed-high result becomes C for an unsigned-low MMA.
      o1_static_for<0,2>([&](auto mi) {
        auto pl=cute::make_tensor<int>(cute::make_shape(cute::_4{},cute::_4{}));
        auto ph=cute::make_fragment_like(pl);cute::clear(pl);cute::clear(ph);
        o1_static_for<0,4>([&](auto ni) {
          auto l=pl(cute::_,ni),h=ph(cute::_,ni);
          cute::gemm(LA{},l,a00(cute::_,mi,cute::_0{}),b00(cute::_,ni,cute::_0{}),l);
          cute::gemm(HA{},h,h00(cute::_,mi,cute::_0{}),b00(cute::_,ni,cute::_0{}),h);
        });
        o1_static_for<0,4>([&](auto ni) {
          auto l=pl(cute::_,ni),h=ph(cute::_,ni);
          cute::gemm(LA{},l,a01(cute::_,mi,cute::_0{}),b01(cute::_,ni,cute::_0{}),l);
          cute::gemm(HA{},h,h01(cute::_,mi,cute::_0{}),b01(cute::_,ni,cute::_0{}),h);
        });
        o1_static_for<0,4>([&](auto ni) {
          auto full_ni=nb*cute::_4{}+ni;
          o1_static_for<0,4>([&](auto vi) {
            const auto coord=coords(vi,mi,full_ni);
            const int coefficient=s.factor[slot][cute::get<1>(coord)];
            const uint32_t high_coefficient=uint32_t(coefficient)*16u;
            auto& target=acc(vi,mi,full_ni);
            // PTX low32 integer arithmetic is intentional, NOT saturation.
            // Individual terms may wrap; their modular sum equals the old
            // guarded, representable INT32 prefix. No new range assumption.
            asm volatile("mad.lo.s32 %0,%1,%2,%0;"
                : "+r"(target) : "r"(pl(vi,ni)),"r"(coefficient));
            asm volatile("mad.lo.s32 %0,%1,%2,%0;"
                : "+r"(target) : "r"(ph(vi,ni)),"r"(high_coefficient));
          });
        });
      });
'''


def generated_header():
    source=generated_headers('o3')[0]
    if source.count(BEGIN)!=1 or source.count(END)!=1:
        raise ValueError('v89 source boundary drift')
    begin,end=source.index(BEGIN),source.index(END)
    return (source[:begin]+BODY+source[end:]).replace(
        'o3_grouped_cta_experiment','o3_split_weighted_experiment')


def split_chain_trace(sass,symbol,liveness):
    """Track actual SASS C fragments. No latency/concurrency simulation."""
    loop=next(x for x in liveness['loops'] if x['kind']=='integer')
    tags,chains,active,peak={},[],set(),0
    for pc,inst in instructions(sass,symbol,loop):
        op=inst.split()[0]
        mma=re.match(r'IMMA\.16864\.(S4|U4)\.S4\s+R(\d+),.*?,.*?,\s*(RZ|R\d+)$',inst)
        if mma:
            kind,d,c=mma[1],int(mma[2]),mma[3]
            if c=='RZ':
                cid=len(chains);chains.append(dict(id=cid,kind=kind,mma=[]));active.add(cid)
            else:
                source=[tags.get(int(c[1:])+i,frozenset()) for i in range(4)]
                if any(len(x)!=1 for x in source) or len(set(source))!=1:
                    raise ValueError(f'unresolved C fragment at {pc:x}: {source}')
                cid=next(iter(source[0]))
                if cid not in active or chains[cid]['kind']!=kind:
                    raise ValueError('mixed signedness or completed chain reused')
            chain=chains[cid];chain['mma'].append(dict(pc=hex(pc),destination=d,c=c))
            for i in range(4):tags[d+i]=frozenset({cid})
            peak=max(peak,len(active))
            if len(chain['mma'])==2:active.remove(cid)
            elif len(chain['mma'])>2:raise ValueError('more than two MMA per chain')
            continue
        dest=re.match(r'\S+\s+R(\d+)(?:\.[a-zA-Z0-9]+)?\s*,(.*)',inst)
        if not dest:continue
        d,rest=int(dest[1]),dest[2]
        width=(int(op.rsplit('.',1)[1]) if op.startswith('LDSM.') else
               4 if op.startswith(('LDS','LDG','LDL')) and '.128' in op else
               2 if (op.startswith(('LDS','LDG','LDL')) and '.64' in op) or
                    op=='CS2R' or op.startswith('IMAD.WIDE') else 1)
        clear=op.startswith(('LDG','LDS','LDL','LDSM','S2R','CS2R'))
        if not clear and op.split('.')[0] not in {
                'MOV','IMAD','IADD3','SHF','LOP3','LEA','PRMT','SEL','I2F','FMUL',
                'FADD','FFMA','FSEL','FLO'}:
            raise ValueError('unhandled writer: '+inst)
        values=frozenset() if clear else frozenset().union(
            *(tags.get(int(r),frozenset()) for r in re.findall(r'\bR(\d+)\b',rest)))
        for i in range(width):tags[d+i]=values
    if active or len(chains)!=32 or any(len(c['mma'])!=2 for c in chains):
        raise ValueError('expected32 complete homogeneous two-MMA chains')
    return dict(chains=chains,chain_count=len(chains),mma_per_chain=2,
                peak_started_not_finished=peak,
                scope='static programmed dataflow, not hardware inflight count')


def cost_gate(old,new,chain):
    a=next(x for x in old['loops'] if x['kind']=='integer')
    b=next(x for x in new['loops'] if x['kind']=='integer')
    count=lambda p:sum(v for op,v in b['opcode_counts'].items() if op.startswith(p))
    ratio=b['static_instructions']/a['static_instructions']
    checks=dict(work=ratio<=LIMITS['max_work_ratio'],
        allocation=new['allocated_gpr']<=LIMITS['max_allocated_gpr'],
        hot_local=count('LDL')+count('STL')==LIMITS['max_hot_local'],
        shorter_independent_chains=chain['chain_count']==32 and chain['mma_per_chain']==2
            and chain['peak_started_not_finished']>=LIMITS['min_chain_peak'],
        native_math=count('IMMA.16864.S4.S4')==count('IMMA.16864.U4.S4')==32,
        same_supply=count('LDSM.')==16 and count('LDGSTS')==9,
        same_barrier=count('BAR')==1)
    return dict(passed=all(checks.values()),checks=checks,limits=LIMITS,
        old_static=a['static_instructions'],new_static=b['static_instructions'],
        work_ratio=ratio,scope='predeclared_compile_gate_not_measured_speedup')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=Path('reports/o378_roof_v89_o3_codegen'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();baseline=args.baseline.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh repository output required')
    prior=checked_grouped(baseline,'o3');sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    cuda=Path('/usr/local/cuda-12.8/bin');cutlass=ROOT/'third_party/cutlass-src'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    commit=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'release 12.8' not in version or commit!='db1c288993354c88e551c40c19a8fb93a774a241':
        p.error('pinned CUDA12.8/CUTLASS required')
    out.mkdir(parents=True)
    for f in baseline.iterdir():
        if f.suffix in ('.cu','.cuh'):(out/f.name).write_bytes(f.read_bytes())
    header=out/(STEM+'_generated.cuh');header.write_text(generated_header())
    sources=set(prior['sources'])|{'csrc/sm80/roof_o3_split_weighted_probe.cu',
        'scripts/probe_split_weighted_codegen.py','scripts/inspect_eight_chain_schedule.py',
        'scripts/inspect_o78_register_liveness.py','scripts/probe_roof_fullk_integer_codegen.py',
        'scripts/compare_a100_codegen.py'}
    r=dict(scope='O3_split_weighted_compile_gate_not_GPU_acceptance',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(sources)},commands=[],limits=LIMITS,
        nvcc=version,cutlass_commit=commit,production_default_changed=False,changed_semantics=False,
        cta_tile=[64,128,128],threads=128,stages=3,shared_bytes=50688,partial_registers=32,
        conversion_changed=False,baseline_cubin_sha256=prior['cubin_sha256'])
    def run(cmd,name):
        r['commands'].append(cmd);(out/'progress.json').write_text(json.dumps(r,indent=2)+'\n')
        with (out/name).open('w') as f:
            subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,TMPDIR=str(ROOT/'tmp')),
                stdout=f,stderr=subprocess.STDOUT,check=True)
    flags=[str(cuda/'nvcc'),'-std=c++17','--expt-relaxed-constexpr','-arch=sm_80',
        '-DADANGEL_FULLK_INTEGER=1','-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out)]
    src=str(ROOT/'csrc/sm80/roof_o3_split_weighted_probe.cu');cubin=out/(STEM+'.cubin')
    run(flags+['-O3','-lineinfo',src,'-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(flags+['-O3','-lineinfo',src,'-ptx','-o',str(out/(STEM+'.ptx'))],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],STEM+'.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    sass=(out/(STEM+'.sass')).read_text();ptx=(out/(STEM+'.ptx')).read_text()
    es=static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})
    live={s:analyze((out/'liveness.txt').read_text(),s) for s in (CONTROL,SYMBOL)}
    for s in (CONTROL,SYMBOL):
        body=next(b for b in re.split(r'(?=\.visible \.entry )',ptx) if b.startswith('.visible .entry '+s+'('))
        if not all(x in body for x in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32')):
            raise ValueError('same-entry PTX native math/supply missing')
    control=compare((baseline/'o3_grouped_cta.sass').read_text(),sass,'^'+CONTROL+'$')
    chain=split_chain_trace(sass,SYMBOL,live[SYMBOL])
    gate=cost_gate(live[CONTROL],live[SYMBOL],chain)
    gate['control_encoding_unchanged']=control['passed'];gate['passed'] &= control['passed']
    r.update(entries=es,liveness=live,chains=chain,cost_gate=gate,control_comparison=control,
        cubin_sha256=sha(cubin),artifact_sha256={f.name:sha(f) for f in out.iterdir()
            if f.is_file() and f.name not in ('codegen.json','progress.json')})
    (out/'codegen.json').write_text(json.dumps(r,indent=2)+'\n')
    if not control['passed'] or not all(e['native_u4_s4'] and e['native_s4_s4'] and
            not e['int8_mma'] and e['all_copies_bypass_l1'] for e in es.values()):
        raise ValueError('native INT4/control audit failed')
    print(json.dumps(dict(cost_gate=gate,liveness=live,
                         chains={k:v for k,v in chain.items() if k!='chains'}),indent=2),flush=True)
    print('PASSED: numerical/resource/full24 validation next' if gate['passed'] else
          'FAILED: no candidate GPU execution or adjacent schedule scan',flush=True)


if __name__=='__main__':main()
