#!/usr/bin/env python3
"""v107: fixed-budget operand-residency axis transposition, compile gate first.

Not v48's full-N128/two-M partial window, v94's four-chain/A reload path,
v88's traversal permutation, or a tile/warp/stage scan. Full B is retained;
M32 A tiles stream through the same 32 partial slots/eight MMA chains.
Expected LDSM16 and native MMA64/G128 are unchanged. No claimed speedup.
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
from probe_o78_eight_chain_codegen import ROOT,generated_header as eight_header
from probe_roof_fullk_integer_codegen import static_entries

BASELINE='reports/o378_roof_v78_codegen'
CONTROL='adangel_roof_o78_eight_chain_candidate'
SYMBOL='adangel_roof_o78_residency_axis_candidate'
STEM='o78_residency_axis'
SHARED=34304

SETUP='''  // Same warp ownership, but A is M32 and B is full N128.
  // TiledMMA's permutation tile is fixed: use M32 for A partition/copy.
  using SliceMma=cute::TiledMMA<cute::MMA_Atom<cute::SM80_16x8x64_S32U4S4S32_TN>,
      cute::Layout<cute::Shape<cute::_2,cute::_2,cute::_1>>,
      cute::Tile<cute::_32,cute::_128,cute::_64>>;
  using SliceHighMma=cute::TiledMMA<cute::MMA_Atom<cute::SM80_16x8x64_S32S4S4S32_TN>,
      cute::Layout<cute::Shape<cute::_2,cute::_2,cute::_1>>,
      cute::Tile<cute::_32,cute::_128,cute::_64>>;
  SliceMma slice_mma;SliceHighMma slice_high_mma;
  auto atr=slice_mma.get_slice(threadIdx.x);
  auto aht=slice_high_mma.get_slice(threadIdx.x);
  auto tile_a=[&](auto t,auto mb,auto half) {return cute::local_tile(t,
      cute::make_shape(cute::_32{},cute::_64{}),cute::make_coord(mb,half));};
  auto a0=atr.partition_fragment_A(tile_a(low(0),cute::_0{},cute::_0{}));auto a1=cute::make_fragment_like(a0);
  auto h0=aht.partition_fragment_A(tile_a(high(0),cute::_0{},cute::_0{}));auto h1=cute::make_fragment_like(h0);
  using LCopy=cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,cutlass::uint4b_t>;
  using SCopy=cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,cutlass::int4b_t>;
  auto lc=cute::make_tiled_copy_A(LCopy{},slice_mma).get_slice(threadIdx.x);
  auto hc=cute::make_tiled_copy_A(SCopy{},slice_high_mma).get_slice(threadIdx.x);
  auto ld0=lc.retile_D(a0),ld1=lc.retile_D(a1);
  auto hd0=hc.retile_D(h0),hd1=hc.retile_D(h1);
  auto tile_b=[&](int slot,auto half) {return cute::local_tile(weight(slot),
      cute::make_shape(cute::_128{},cute::_64{}),cute::make_coord(cute::_0{},half));};
  auto b0=thr.partition_fragment_B(tile_b(0,cute::_0{}));auto b1=cute::make_fragment_like(b0);
  auto bc=cute::make_tiled_copy_B(SCopy{},mma).get_slice(threadIdx.x);
  auto bd0=bc.retile_D(b0),bd1=bc.retile_D(b1);
  static_assert(decltype(cute::size<1>(a0))::value==1);
  static_assert(decltype(cute::size<1>(b0))::value==8);
  static_assert(decltype(cute::size<1>(acc))::value==2);
  static_assert(decltype(cute::size(acc))::value==64);
  using LA=cute::MMA_Atom<cute::SM80_16x8x64_S32U4S4S32_TN>;
  using HA=cute::MMA_Atom<cute::SM80_16x8x64_S32S4S4S32_TN>;
'''

MATH='''    // B spans both old N64 windows; no reload across M32 A tiles.
    cute::copy(SCopy{},bc.partition_S(tile_b(slot,cute::_0{})),bd0);
    cute::copy(SCopy{},bc.partition_S(tile_b(slot,cute::_1{})),bd1);
    o1_static_for<0,2>([&](auto mb) {
      cute::copy(LCopy{},lc.partition_S(tile_a(low(slot),mb,cute::_0{})),ld0);
      cute::copy(SCopy{},hc.partition_S(tile_a(high(slot),mb,cute::_0{})),hd0);
      cute::copy(LCopy{},lc.partition_S(tile_a(low(slot),mb,cute::_1{})),ld1);
      cute::copy(SCopy{},hc.partition_S(tile_a(high(slot),mb,cute::_1{})),hd1);
      auto partial=cute::make_tensor<int>(cute::make_shape(cute::_4{},cute::_8{}));
      cute::clear(partial);
      o1_static_for<0,8>([&](auto ni) {
        auto p=partial(cute::_,ni);
        cute::gemm(HA{},p,h0(cute::_,cute::_0{},cute::_0{}),b0(cute::_,ni,cute::_0{}),p);
      });
      o1_static_for<0,8>([&](auto ni) {
        auto p=partial(cute::_,ni);
        cute::gemm(HA{},p,h1(cute::_,cute::_0{},cute::_0{}),b1(cute::_,ni,cute::_0{}),p);
      });
      o1_static_for<0,32>([&](auto i) { partial(i)*=16; });
      o1_static_for<0,8>([&](auto ni) {
        auto p=partial(cute::_,ni);
        cute::gemm(LA{},p,a0(cute::_,cute::_0{},cute::_0{}),b0(cute::_,ni,cute::_0{}),p);
      });
      o1_static_for<0,8>([&](auto ni) {
        auto p=partial(cute::_,ni);
        cute::gemm(LA{},p,a1(cute::_,cute::_0{},cute::_0{}),b1(cute::_,ni,cute::_0{}),p);
      });
      o1_static_for<0,8>([&](auto ni) {
        o1_static_for<0,4>([&](auto vi) {
          const auto coord=coords(vi,mb,ni);
          const int coefficient=s.activation_factors[slot][cute::get<0>(coord)]*
                                s.weight_factors[slot][cute::get<1>(coord)];
          acc(vi,mb,ni)+=partial(vi,ni)*coefficient;
        });
      });
    });
'''


def generated_header():
    text=eight_header((ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
    start='  auto tile_a=[&]';end='  constexpr int Groups=32;'
    if text.count(start)!=1 or text.count(end)!=1:raise ValueError('setup boundary drift')
    text=text[:text.index(start)]+SETUP+text[text.index(end):]
    start='    cute::copy(LCopy{},lc.partition_S(tile_a(low(slot),cute::_0{})),ld0);'
    end='  }\n  // Reuse the 64 INT32'
    if text.count(start)!=1 or text.count(end)!=1:raise ValueError('group boundary drift')
    text=text[:text.index(start)]+MATH+text[text.index(end):]
    return text.replace('o78_eight_chain_experiment','o78_residency_axis_experiment')


def worth_runtime(live,old):
    loop=next(x for x in live['loops'] if x['kind']=='integer')
    prior=next(x for x in old['loops'] if x['kind']=='integer')
    ops=loop['opcode_counts']
    structural=(live['allocated_gpr']<=old['allocated_gpr'] and
        not any(v for op,v in ops.items() if op.startswith(('LDL','STL'))) and
        ops.get('IMMA.16864.S4.S4')==ops.get('IMMA.16864.U4.S4')==32 and
        ops.get('LDSM.16.M88.4')==16)
    # One predeclared potential gate, not a measured speedup or occupancy proof.
    useful=(loop['static_instructions']<=.95*prior['static_instructions'] or
            loop['max_live_gpr']<=prior['max_live_gpr']-16 or live['allocated_gpr']<=128)
    return structural and useful


def checked(directory):
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    r=json.loads((directory/'codegen.json').read_text())
    for path,digest in r['sources'].items():
        if sha(ROOT/path)!=digest:raise ValueError('source drift: '+path)
    for name,digest in r['artifact_sha256'].items():
        if sha(directory/name)!=digest:raise ValueError('artifact drift: '+name)
    if (directory/(STEM+'_generated.cuh')).read_text()!=generated_header():
        raise ValueError('residency-axis generated body drift')
    if r['production_default_changed'] or r['changed_semantics'] or not r['coordinates']['passed']:
        raise ValueError('default/math/mapping drift')
    if not r['control_comparison']['passed']:
        raise ValueError('old encoded kernel drift')
    if not all(x['native_u4_s4'] and x['native_s4_s4'] and not x['int8_mma']
            and x['all_copies_bypass_l1'] for x in r['entries'].values()):
        raise ValueError('native INT4/copy audit failed')
    if r['worth_runtime_validation']!=worth_runtime(r['liveness'][SYMBOL],r['liveness'][CONTROL]):
        raise ValueError('predeclared potential gate drift')
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();baseline=ROOT/BASELINE
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh repository output required')
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    prior=json.loads((baseline/'codegen.json').read_text())
    for path,digest in prior['sources'].items():
        if sha(ROOT/path)!=digest:raise ValueError('v78 source drift: '+path)
    if sha(baseline/'o78_eight_chain.cubin')!=prior['cubin_sha256']:raise ValueError('v78 binary drift')
    cuda=Path('/usr/local/cuda-12.8/bin');cutlass=ROOT/'third_party/cutlass-src'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    commit=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'release 12.8' not in version or commit!='db1c288993354c88e551c40c19a8fb93a774a241':
        p.error('pinned CUDA/CUTLASS required')
    out.mkdir(parents=True)
    for f in baseline.iterdir():
        if f.suffix in ('.cu','.cuh'):(out/f.name).write_text(f.read_text())
    (out/(STEM+'_generated.cuh')).write_text(generated_header())
    sources=set(prior['sources'])|{'scripts/probe_residency_axis_codegen.py',
        'csrc/sm80/roof_o78_residency_axis_probe.cu','tests/cuda/validate_residency_axis_coordinates.cu'}
    r=dict(scope='v107_fixed_budget_operand_residency_axis_compile_gate',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(sources)},commands=[],nvcc=version,cutlass_commit=commit,
        cta_tile=[64,128,128],threads=128,stages=2,shared_bytes=SHARED,
        partial_registers=32,independent_chains=8,logical_A_registers=16,logical_B_registers=32,
        old_logical_A_registers=32,old_logical_B_registers=16,
        intended_LDSM_per_group=16,baseline_LDSM_per_group=16,
        global_payload_copy_bytes_changed=False,changed_semantics=False,production_default_changed=False,
        baseline_cubin_sha256=prior['cubin_sha256'])
    def run(cmd,name):
        r['commands'].append(cmd)
        with (out/name).open('w') as log:
            subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,TMPDIR=str(ROOT/'tmp')),
                stdout=log,stderr=subprocess.STDOUT,check=True)
    common=[str(cuda/'nvcc'),'-std=c++17','--expt-relaxed-constexpr','-arch=sm_80',
        '-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out)]
    coordinate=out/'validate_coordinates'
    run(common+['-O2',str(ROOT/'tests/cuda/validate_residency_axis_coordinates.cu'),
        '-o',str(coordinate)],'coordinates_build.log')
    run([str(coordinate)],'coordinates.log')
    r['coordinates']=json.loads((out/'coordinates.log').read_text())
    if not r['coordinates']['passed'] or r['coordinates']['gpu_execution']:
        raise ValueError('host-only CuTe mapping proof required')
    flags=common+['-O3','-lineinfo',str(ROOT/'csrc/sm80/roof_o78_residency_axis_probe.cu')]
    cubin=out/(STEM+'.cubin')
    run(flags+['-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(flags+['-ptx','-o',str(out/(STEM+'.ptx'))],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],STEM+'.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    sass=(out/(STEM+'.sass')).read_text();ptx=(out/(STEM+'.ptx')).read_text();symbols={CONTROL,SYMBOL}
    es=static_entries(sass,'^(?:'+'|'.join(sorted(symbols))+')$',symbols)
    for symbol in symbols:
        body=next(b for b in re.split(r'(?=\.visible \.entry )',ptx) if b.startswith('.visible .entry '+symbol+'('))
        if not all(x in body for x in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32')):
            raise ValueError('same-entry native PTX math/copy missing')
    live=(out/'liveness.txt').read_text()
    r.update(entries=es,liveness={s:analyze(live,s) for s in sorted(symbols)},cubin_sha256=sha(cubin),
        control_comparison=compare((baseline/'o78_eight_chain.sass').read_text(),sass,'^'+CONTROL+'$'))
    r['worth_runtime_validation']=worth_runtime(r['liveness'][SYMBOL],r['liveness'][CONTROL])
    r['artifact_sha256']={f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='codegen.json'}
    (out/'codegen.json').write_text(json.dumps(r,indent=2)+'\n');checked(out)
    print(json.dumps(dict(runtime_justified=r['worth_runtime_validation'],coordinates=r['coordinates'],
        liveness=r['liveness']),indent=2),flush=True)


if __name__=='__main__':main()
