#!/usr/bin/env python3
"""v134: one address-rematerialization repair of v128, same cost gate.

Cache four CuTe-derived shared addresses before the loop. One own-lane shuffle
per address preserves its value but makes the prologue result opaque to NVVM,
so it can remain live instead of re-expanding the lane/layout expressions.
The shift/order/guard, stage lifetime and all MMA/scale work are unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from inspect_o78_register_liveness import analyze
from probe_o3_crossk_load_codegen import (CONTROL, LIMITS, cost_gate,
    generated_header as crossk_header, overlap_evidence)
from probe_grouped_cta_codegen import checked as checked_grouped
from probe_roof_fullk_integer_codegen import static_entries

ROOT=Path(__file__).resolve().parents[1]
SYMBOL='adangel_roof_o3_cached_a_iterator_candidate'
STEM='o3_cached_a_iterator'
OLD='''  auto load_a=[&](int slot,auto& al0,auto& al1,auto& ah0,auto& ah1) {
    auto dl0=lc.retile_D(al0),dl1=lc.retile_D(al1);
    auto dh0=hc.retile_D(ah0),dh1=hc.retile_D(ah1);
    cute::copy(LCopy{},lc.partition_S(tile_a(low(slot),cute::_0{})),dl0);
    cute::copy(LCopy{},lc.partition_S(tile_a(low(slot),cute::_1{})),dl1);
    cute::copy(SCopy{},hc.partition_S(tile_a(high(slot),cute::_0{})),dh0);
    cute::copy(SCopy{},hc.partition_S(tile_a(high(slot),cute::_1{})),dh1);
  };
'''
NEW='''  // Derive all four source addresses from the SAME CuTe copy partition.
  // Stage and sign-plane translations are verified on the host for every lane.
  auto a_addresses=cute::make_tensor<uint32_t>(cute::make_shape(cute::_2{},cute::_2{}));
  o1_static_for<0,2>([&](auto half) {
    auto src=cute::recast<cute::uint128_t>(lc.partition_S(tile_a(low(0),half)));
    static_assert(decltype(cute::size(src))::value==2);
    o1_static_for<0,2>([&](auto mi) {
      uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(&src(mi)));
      // Every thread participates, and selects itself. Not a lane permutation.
      a_addresses(mi,half)=__shfl_sync(0xffffffffu,address,threadIdx.x&31);
    });
  });
  auto load_words=[&](int slot,auto half,auto plane,auto& destination,auto const& copier) {
    auto d=cute::recast<uint32_t>(copier.retile_D(destination));
    static_assert(decltype(cute::size(d))::value==8);
    o1_static_for<0,2>([&](auto mi) {
      const uint32_t address=a_addresses(mi,half)+uint32_t(slot)*4096u+uint32_t(plane)*12288u;
      // Identical public PTX operation to SM75_U32x4_LDSM_N::copy.
      asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];"
          : "=r"(d(4*mi)),"=r"(d(4*mi+1)),"=r"(d(4*mi+2)),"=r"(d(4*mi+3))
          : "r"(address));
    });
  };
  auto load_a=[&](int slot,auto& al0,auto& al1,auto& ah0,auto& ah1) {
    load_words(slot,cute::_0{},cute::_0{},al0,lc);
    load_words(slot,cute::_1{},cute::_0{},al1,lc);
    load_words(slot,cute::_0{},cute::_1{},ah0,hc);
    load_words(slot,cute::_1{},cute::_1{},ah1,hc);
  };
'''


def generated_header():
    source=crossk_header()
    if source.count(OLD)!=1:raise ValueError('v128 A-copy boundary drift')
    return source.replace(OLD,NEW).replace('o3_crossk_load_experiment','o3_cached_a_iterator_experiment')


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
    sources=set(prior['sources'])|{'csrc/sm80/roof_o3_cached_a_iterator_probe.cu',
        'csrc/sm80/verify_o3_cached_a_iterator.cpp','scripts/probe_o3_cached_a_iterator_codegen.py',
        'scripts/probe_o3_crossk_load_codegen.py','scripts/inspect_eight_chain_schedule.py',
        'scripts/inspect_o78_register_liveness.py','scripts/probe_roof_fullk_integer_codegen.py',
        'scripts/compare_a100_codegen.py'}
    r=dict(scope='O3_crossK_cached_A_iterator_compile_repair_not_GPU_acceptance',
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
    verifier=out/'verify_a_iterator'
    run(flags+[str(ROOT/'csrc/sm80/verify_o3_cached_a_iterator.cpp'),'-o',str(verifier)],'host_build.log')
    run([str(verifier)],'host_layout.json') # CPU layout checks, never launches CUDA.
    layout=json.loads((out/'host_layout.json').read_text())
    if layout!={'passed':True,'source_addresses':3072,'destination_words':12288}:
        raise ValueError('incomplete CuTe layout verification')
    src=str(ROOT/'csrc/sm80/roof_o3_cached_a_iterator_probe.cu');cubin=out/(STEM+'.cubin')
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
    hot=next(x for x in live[SYMBOL]['loops'] if x['kind']=='integer')
    gate['checks']['no_hot_shuffles']=not any(op.startswith('SHFL') for op in hot['opcode_counts'])
    gate['passed'] &= gate['checks']['no_hot_shuffles']
    r.update(entries=es,liveness=live,overlap=overlap,cost_gate=gate,control_comparison=control,
        host_layout=layout,cubin_sha256=sha(cubin),artifact_sha256={f.name:sha(f) for f in out.iterdir()
            if f.is_file() and f.name not in ('codegen.json','progress.json')})
    (out/'codegen.json').write_text(json.dumps(r,indent=2)+'\n')
    if not control['passed'] or not all(e['native_u4_s4'] and e['native_s4_s4'] and
            not e['int8_mma'] and e['all_copies_bypass_l1'] for e in es.values()):
        raise ValueError('native INT4/control audit failed')
    print(json.dumps(dict(cost_gate=gate,liveness=live,host_layout=layout,
        next_A_loads=len(overlap['next_loads']),
        later_weighted_updates=len(overlap['weighted_updates_after_first_next_load'])),indent=2),flush=True)
    print('PASSED: actual resources, safety, numerical and full24 validation next' if gate['passed'] else
          'FAILED: stop one address repair; no GPU execution or adjacent iterator scan',flush=True)


if __name__=='__main__':main()
