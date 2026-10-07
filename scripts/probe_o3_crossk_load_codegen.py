#!/usr/bin/env python3
"""v128: next-K A fragment loads before the old-K last N64 weighted updates.

Reuse the SAME A registers once all current-group MMA have consumed them.
Cache old-group coefficients before the transition barrier; the old stage
may subsequently be overwritten while other warps finish register updates.
Not v49/v96/v97 cross-N scheduling or v42 global-copy relocation. No GPU run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from inspect_eight_chain_schedule import instructions, trace
from inspect_o78_register_liveness import analyze
from probe_grouped_cta_codegen import checked as checked_grouped, generated_headers
from probe_roof_fullk_integer_codegen import static_entries

ROOT=Path(__file__).resolve().parents[1]
CONTROL='adangel_roof_o3_grouped_cta_candidate'
SYMBOL='adangel_roof_o3_crossk_load_candidate'
STEM='o3_crossk_load'
LIMITS=dict(max_work_ratio=1.05,max_allocated_gpr=168,max_hot_local=1,
    min_actual_cta=3,mma_each=32,ldsm=16,copies=9,barriers=1,
    next_a_loads_after_last_mma=8,min_weighted_updates_after_first_next_load=8)
LOOP='''  for(int group=0;group<groups;++group) {
    const int slot=group%3;
    if(group+2<groups) asm volatile("cp.async.wait_group 1;" ::: "memory");
    else asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads(); // Completed inputs/scales and all previous readers before slot reuse.
    if(group+2<groups) prefetch(s,(group+2)%3,group+2,a,w,ws,m,n,k);
    load_a(slot,a00,a01,h00,h01);
'''
NEW_LOOP='''  // Prime exactly one A fragment. Two copy groups were committed above.
  asm volatile("cp.async.wait_group 1;" ::: "memory");
  __syncthreads();
  load_a(0,a00,a01,h00,h01);
  for(int group=0;group<groups;++group) {
    const int slot=group%3;
    if(group+2<groups) prefetch(s,(group+2)%3,group+2,a,w,ws,m,n,k);
'''
UPDATE='''      o1_static_for<0,2>([&](auto mi) {
        o1_static_for<0,4>([&](auto ni) {
          auto full_ni=nb*cute::_4{}+ni;
          o1_static_for<0,4>([&](auto vi) {
            const auto coord=coords(vi,mi,full_ni);
            const int coefficient=s.factor[slot][cute::get<1>(coord)];
            acc(vi,mi,full_ni)+=partial(vi,mi,ni)*coefficient;
          });
        });
      });
'''
NEXT='''        // Read every old-stage factor BEFORE releasing the old stage.
        // CuTe coordinates supply the mapping; no guessed lane/column map.
        auto factors=cute::make_fragment_like<int>(partial);
        o1_static_for<0,2>([&](auto mi) {
          o1_static_for<0,4>([&](auto ni) {
            auto full_ni=nb*cute::_4{}+ni;
            o1_static_for<0,4>([&](auto vi) {
              const auto coord=coords(vi,mi,full_ni);
              factors(vi,mi,ni)=s.factor[slot][cute::get<1>(coord)];
            });
          });
        });
        if(group+1<groups) {
          if(group+3<groups) asm volatile("cp.async.wait_group 1;" ::: "memory");
          else asm volatile("cp.async.wait_group 0;" ::: "memory");
          // All warps have finished current-stage reads. All following old
          // work is register-only, even if another warp starts the next copy.
          __syncthreads();
          load_a((group+1)%3,a00,a01,h00,h01);
        }
'''


def generated_header():
    source=generated_headers('o3')[0]
    if source.count(LOOP)!=1 or source.count(UPDATE)!=1:
        raise ValueError('v89 loop/update source boundary drift')
    cached=UPDATE.replace('const auto coord=coords(vi,mi,full_ni);\n            ', '')
    cached=cached.replace('s.factor[slot][cute::get<1>(coord)]','factors(vi,mi,ni)')
    replacement=('      if constexpr(decltype(nb)::value==1) {\n'+NEXT+cached+
                 '      } else {\n'+UPDATE+'      }\n')
    return source.replace(LOOP,NEW_LOOP).replace(UPDATE,replacement).replace(
        'o3_grouped_cta_experiment','o3_crossk_load_experiment')


def ring_schedule(groups=32):
    """CPU ordering oracle; not proof of GPU compilation or concurrent safety."""
    if groups!=32:raise ValueError('the existing guarded K4096 path only')
    events=[('copy',0,0),('copy',1,1),('wait',0,1),('barrier',0),('load_a',0,0)]
    for g in range(groups):
        if g+2<groups:events.append(('copy',g+2,(g+2)%3))
        events.extend([('mma_first_n',g),('update_first_n',g),('mma_last_n',g),
                       ('cache_factor',g,g%3)])
        if g+1<groups:
            events.extend([('wait',g+1,1 if g+3<groups else 0),
                           ('barrier',g+1),('load_a',g+1,(g+1)%3)])
        events.append(('update_last_n',g))
    return events


def overlap_evidence(sass,symbol,live):
    chain=trace(sass,symbol,live)
    loop=next(x for x in live['loops'] if x['kind']=='integer')
    ops=instructions(sass,symbol,loop)
    last=max(int(c['end_pc'],16) for c in chain['chains'])
    loads=[x for x in chain['loads'] if int(x['pc'],16)>last]
    first=int(loads[0]['pc'],16) if loads else int(loop['end_pc'],16)+1
    # Identify real final-N partial sources, not arbitrary address IMAD.
    partials=set()
    for c in chain['chains'][8:]:
        partials.update(range(c['mma'][-1]['destination'],c['mma'][-1]['destination']+4))
    updates=[]
    for pc,ins in ops:
        if pc<=last:continue
        op=ins.split()[0]
        regs=[int(r) for r in re.findall(r'\bR(\d+)\b',ins)]
        if op=='IMAD' and len(regs)==4 and any(r in partials for r in regs[1:3]):
            if pc>first:updates.append(dict(pc=hex(pc),instruction=ins))
        # No tag propagation needed: only plain final weighted updates accepted.
        if re.match(r'\S+\s+R\d+(?:\.[\w]+)?\s*,',ins):
            width=int(op.rsplit('.',1)[1]) if op.startswith('LDSM.') else (
                4 if op.startswith(('LDS','LDG','LDL')) and '.128' in op else
                2 if (op.startswith(('LDS','LDG','LDL')) and '.64' in op) or
                     op=='CS2R' or op.startswith('IMAD.WIDE') else 1)
            partials.difference_update(range(regs[0],regs[0]+width))
    return dict(next_loads=loads,weighted_updates_after_first_next_load=updates,
        last_mma_pc=hex(last),chains=chain,
        scope='static load-before-weighted-update evidence, not measured overlap/latency')


def cost_gate(old,new,evidence):
    a=next(x for x in old['loops'] if x['kind']=='integer')
    b=next(x for x in new['loops'] if x['kind']=='integer')
    count=lambda p:sum(v for op,v in b['opcode_counts'].items() if op.startswith(p))
    ratio=b['static_instructions']/a['static_instructions']
    checks=dict(work=ratio<=LIMITS['max_work_ratio'],
        allocation=new['allocated_gpr']<=LIMITS['max_allocated_gpr'],
        hot_local=count('LDL')+count('STL')<=LIMITS['max_hot_local'],
        native_math=count('IMMA.16864.S4.S4')==count('IMMA.16864.U4.S4')==32,
        same_supply=count('LDSM.')==16 and count('LDGSTS')==9,
        same_barrier=count('BAR')==1,
        next_A_really_moved=len(evidence['next_loads'])==8,
        useful_tail_work=len(evidence['weighted_updates_after_first_next_load'])>=8)
    return dict(passed=all(checks.values()),checks=checks,limits=LIMITS,
        old_static=a['static_instructions'],new_static=b['static_instructions'],work_ratio=ratio,
        scope='predeclared_compile_gate_not_measured_speedup')


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
    (out/(STEM+'_generated.cuh')).write_text(generated_header())
    sources=set(prior['sources'])|{'csrc/sm80/roof_o3_crossk_load_probe.cu',
        'scripts/probe_o3_crossk_load_codegen.py','scripts/inspect_eight_chain_schedule.py',
        'scripts/inspect_o78_register_liveness.py','scripts/probe_roof_fullk_integer_codegen.py',
        'scripts/compare_a100_codegen.py'}
    r=dict(scope='O3_cross_K_load_compile_gate_not_GPU_acceptance',
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
    src=str(ROOT/'csrc/sm80/roof_o3_crossk_load_probe.cu');cubin=out/(STEM+'.cubin')
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
    overlap=overlap_evidence(sass,SYMBOL,live[SYMBOL])
    gate=cost_gate(live[CONTROL],live[SYMBOL],overlap)
    gate['control_encoding_unchanged']=control['passed'];gate['passed'] &= control['passed']
    r.update(entries=es,liveness=live,overlap=overlap,cost_gate=gate,control_comparison=control,
        cubin_sha256=sha(cubin),artifact_sha256={f.name:sha(f) for f in out.iterdir()
            if f.is_file() and f.name not in ('codegen.json','progress.json')})
    (out/'codegen.json').write_text(json.dumps(r,indent=2)+'\n')
    if not control['passed'] or not all(e['native_u4_s4'] and e['native_s4_s4'] and
            not e['int8_mma'] and e['all_copies_bypass_l1'] for e in es.values()):
        raise ValueError('native INT4/control audit failed')
    print(json.dumps(dict(cost_gate=gate,liveness=live,
        next_A_loads=len(overlap['next_loads']),
        later_weighted_updates=len(overlap['weighted_updates_after_first_next_load'])),indent=2),flush=True)
    print('PASSED: actual resources, race/memory/numerical and full24 validation next' if gate['passed'] else
          'FAILED: no candidate GPU execution or adjacent schedule scan',flush=True)


if __name__=='__main__':main()
