#!/usr/bin/env python3
"""v94: trade extra shared A reads for four naturally resident N128 CTAs.

One fixed O7/O8 candidate. Keep the global cp.async pipeline, B reuse, all
G128 factors, guard and epilogue. One M atom/four partial chains and one
overwritten A fragment replace the32-partial/four-A-fragment live window.
Unlike the old N64 CTA, global payload/factor bytes are not duplicated.
The necessary cost is24 LDSM/group versus16. No register-cap sweep.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from inspect_o78_register_liveness import analyze
from probe_o78_eight_chain_codegen import ROOT, MERGED, generated_header as eight_header
from probe_roof_fullk_integer_codegen import static_entries

BASELINE = 'reports/o378_roof_v78_codegen'
CONTROL = 'adangel_roof_o78_eight_chain_candidate'
SYMBOL = 'adangel_roof_o78_operand_stream_candidate'
STEM = 'o78_operand_stream'
SHARED = 34304

ATOM_SETUP = '''  // A single warp atom; derive its M tile from CuTe C coordinates.
  using LA=cute::MMA_Atom<cute::SM80_16x8x64_S32U4S4S32_TN>;
  using HA=cute::MMA_Atom<cute::SM80_16x8x64_S32S4S4S32_TN>;
  auto atom_low=cute::make_tiled_mma(LA{});
  auto atom_high=cute::make_tiled_mma(HA{});
  auto lt=atom_low.get_slice(threadIdx.x%32);
  auto at=atom_high.get_slice(threadIdx.x%32);
  using LCopy=cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,cutlass::uint4b_t>;
  using SCopy=cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,cutlass::int4b_t>;
  auto ac_low=cute::make_tiled_copy_A(LCopy{},atom_low).get_slice(threadIdx.x%32);
  auto ac_high=cute::make_tiled_copy_A(SCopy{},atom_high).get_slice(threadIdx.x%32);
  auto atom_tile_a=[&](auto tensor,int atom_m,auto half) {
    return cute::local_tile(tensor,cute::make_shape(cute::_16{},cute::_64{}),
                           cute::make_coord(atom_m,half));
  };
'''

ATOM_MATH = '''      // Keep the original B0/B1 fragments across both M atoms.
      o1_static_for<0,2>([&](auto mi) {
        const int atom_m=cute::get<0>(coords(cute::_0{},mi,nb*cute::_4{}))/16;
        auto partial=cute::make_tensor<int>(cute::make_shape(cute::_4{},cute::_4{}));
        cute::clear(partial);
        auto ar=at.partition_fragment_A(atom_tile_a(high(slot),atom_m,cute::_0{}));
        // Same-width signed/unsigned view, not conversion or new storage.
        auto al=cute::recast<cutlass::uint4b_t>(ar);
        static_assert(decltype(cute::size(ar))::value==32);
        auto hd=ac_high.retile_D(ar);auto ld=ac_low.retile_D(al);
        cute::copy(SCopy{},ac_high.partition_S(atom_tile_a(high(slot),atom_m,cute::_0{})),hd);
        o1_static_for<0,4>([&](auto ni) {
          auto p=partial(cute::_,ni);
          cute::gemm(HA{},p,ar(cute::_,cute::_0{},cute::_0{}),b0(cute::_,ni,cute::_0{}),p);
        });
        cute::copy(SCopy{},ac_high.partition_S(atom_tile_a(high(slot),atom_m,cute::_1{})),hd);
        o1_static_for<0,4>([&](auto ni) {
          auto p=partial(cute::_,ni);
          cute::gemm(HA{},p,ar(cute::_,cute::_0{},cute::_0{}),b1(cute::_,ni,cute::_0{}),p);
        });
        o1_static_for<0,16>([&](auto i) { partial(i)*=16; });
        cute::copy(LCopy{},ac_low.partition_S(atom_tile_a(low(slot),atom_m,cute::_0{})),ld);
        o1_static_for<0,4>([&](auto ni) {
          auto p=partial(cute::_,ni);
          cute::gemm(LA{},p,al(cute::_,cute::_0{},cute::_0{}),b0(cute::_,ni,cute::_0{}),p);
        });
        cute::copy(LCopy{},ac_low.partition_S(atom_tile_a(low(slot),atom_m,cute::_1{})),ld);
        o1_static_for<0,4>([&](auto ni) {
          auto p=partial(cute::_,ni);
          cute::gemm(LA{},p,al(cute::_,cute::_0{},cute::_0{}),b1(cute::_,ni,cute::_0{}),p);
        });
        o1_static_for<0,4>([&](auto ni) {
          auto full_ni=nb*cute::_4{}+ni;
          o1_static_for<0,4>([&](auto vi) {
            const auto coord=coords(vi,mi,full_ni);
            const int coefficient=s.activation_factors[slot][cute::get<0>(coord)]*
                                  s.weight_factors[slot][cute::get<1>(coord)];
            acc(vi,mi,full_ni)+=partial(vi,ni)*coefficient;
          });
        });
      });
'''


def generated_header():
    text=eight_header((ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
    start='  auto a0=thr.partition_fragment_A('
    end='  using SliceMma='
    assert text.count(start)==text.count(end)==1
    lo,hi=text.index(start),text.index(end)
    text=text[:lo]+ATOM_SETUP+text[hi:]
    old=('  using LA=cute::MMA_Atom<cute::SM80_16x8x64_S32U4S4S32_TN>;\n'
         '  using HA=cute::MMA_Atom<cute::SM80_16x8x64_S32S4S4S32_TN>;\n')
    # Remove the old later declarations, keeping the new atom setup.
    assert text.count(old)==2
    lo=text.index(old,text.index('  using SliceMma='))
    text=text[:lo]+text[lo+len(old):]
    first='    cute::copy(LCopy{},lc.partition_S(tile_a(low(slot),cute::_0{})),ld0);\n'
    last='    // Preserve N64 operand reuse;'
    lo,hi=text.index(first),text.index(last)
    text=text[:lo]+text[hi:]
    assert text.count(MERGED)==1
    text=text.replace(MERGED,ATOM_MATH)
    return text.replace('o78_eight_chain_experiment','o78_operand_stream_experiment')


def worth_runtime(live):
    loop=next(x for x in live['loops'] if x['kind']=='integer')
    counts=loop['opcode_counts']
    return (live['allocated_gpr']<=128
        and not any(v for op,v in counts.items() if op.split('.')[0] in ('LDL','STL'))
        and counts.get('IMMA.16864.S4.S4')==counts.get('IMMA.16864.U4.S4')==32
        and counts.get('LDSM.16.M88.4')==24)


def checked(directory):
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    r=json.loads((directory/'codegen.json').read_text())
    for name,digest in r['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('operand-stream source drift: '+name)
    for name,digest in r['artifact_sha256'].items():
        if sha(directory/name)!=digest:raise ValueError('operand-stream artifact drift: '+name)
    if (directory/(STEM+'_generated.cuh')).read_text()!=generated_header():
        raise ValueError('generated operand stream drift')
    if not r['control_comparison']['passed'] or r['production_default_changed'] or r['changed_semantics']:
        raise ValueError('control/default/math drift')
    if not all(x['native_u4_s4'] and x['native_s4_s4'] and not x['int8_mma']
               and x['all_copies_bypass_l1'] for x in r['entries'].values()):
        raise ValueError('same-entry native INT4/copy failed')
    if r['worth_runtime_validation']!=worth_runtime(r['liveness'][SYMBOL]):
        raise ValueError('compile gate drift')
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();baseline=ROOT/BASELINE
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh repository output required')
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    prior=json.loads((baseline/'codegen.json').read_text())
    for name,digest in prior['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('v78 source drift: '+name)
    if sha(baseline/'o78_eight_chain.cubin')!=prior['cubin_sha256']:raise ValueError('v78 cubin drift')
    cuda=Path('/usr/local/cuda-12.8/bin');cutlass=ROOT/'third_party/cutlass-src'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    commit=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'release 12.8' not in version or commit!='db1c288993354c88e551c40c19a8fb93a774a241':
        p.error('pinned CUDA/CUTLASS required')
    out.mkdir(parents=True)
    for f in baseline.iterdir():
        if f.suffix in ('.cu','.cuh'):(out/f.name).write_text(f.read_text())
    (out/(STEM+'_generated.cuh')).write_text(generated_header())
    sources=set(prior['sources'])|{'scripts/probe_operand_stream_codegen.py',
        'csrc/sm80/roof_o78_operand_stream_probe.cu'}
    r=dict(scope='v94_A_operand_lifetime_four_CTA_compile_gate',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(sources)},commands=[],nvcc=version,cutlass_commit=commit,
        cta_tile=[64,128,128],threads=128,stages=2,shared_bytes=SHARED,
        partial_registers=16,independent_chains=4,logical_A_registers=4,
        intended_LDSM_per_group=24,baseline_LDSM_per_group=16,
        global_payload_copy_bytes_changed=False,changed_semantics=False,production_default_changed=False,
        baseline_cubin_sha256=prior['cubin_sha256'])
    def run(cmd,name):
        r['commands'].append(cmd)
        with (out/name).open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    flags=[str(cuda/'nvcc'),'-O3','-std=c++17','--expt-relaxed-constexpr','-lineinfo','-arch=sm_80',
        '-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out),
        str(ROOT/'csrc/sm80/roof_o78_operand_stream_probe.cu')]
    cubin=out/(STEM+'.cubin')
    run(flags+['-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(flags+['-ptx','-o',str(out/(STEM+'.ptx'))],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],STEM+'.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    sass=(out/(STEM+'.sass')).read_text();ptx=(out/(STEM+'.ptx')).read_text();symbols={CONTROL,SYMBOL}
    entries=static_entries(sass,'^(?:'+'|'.join(sorted(symbols))+')$',symbols)
    for symbol in symbols:
        body=next(b for b in re.split(r'(?=\.visible \.entry )',ptx) if b.startswith('.visible .entry '+symbol+'('))
        if not all(x in body for x in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32')):
            raise ValueError('same-entry native PTX math/copy missing')
    live=(out/'liveness.txt').read_text()
    r.update(entries=entries,liveness={s:analyze(live,s) for s in sorted(symbols)},cubin_sha256=sha(cubin),
        control_comparison=compare((baseline/'o78_eight_chain.sass').read_text(),sass,'^'+CONTROL+'$'))
    r['worth_runtime_validation']=worth_runtime(r['liveness'][SYMBOL])
    r['artifact_sha256']={f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='codegen.json'}
    (out/'codegen.json').write_text(json.dumps(r,indent=2)+'\n');checked(out)
    print(json.dumps(dict(runtime_justified=r['worth_runtime_validation'],liveness=r['liveness']),indent=2),flush=True)


if __name__=='__main__':main()
