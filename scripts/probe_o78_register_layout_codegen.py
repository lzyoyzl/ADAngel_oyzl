#!/usr/bin/env python3
"""One v85 operand-feed candidate: lane-vector LDS instead of LDSM.

This does not remove online layout preparation: an isolated timer includes
the extra exact A/W permutations in each corresponding conversion stage.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from inspect_o78_register_liveness import analyze
from probe_o78_eight_chain_codegen import generated_header as eight_header, SYMBOL as CONTROL
from probe_roof_fullk_integer_codegen import static_entries

ROOT = Path(__file__).resolve().parents[1]
SYMBOL = 'adangel_roof_o78_register_layout_candidate'
OLD_A = '''    cute::copy(LCopy{},lc.partition_S(tile_a(low(slot),cute::_0{})),ld0);
    cute::copy(SCopy{},hc.partition_S(tile_a(high(slot),cute::_0{})),hd0);
    cute::copy(LCopy{},lc.partition_S(tile_a(low(slot),cute::_1{})),ld1);
    cute::copy(SCopy{},hc.partition_S(tile_a(high(slot),cute::_1{})),hd1);
'''
NEW_A = '''    // Warp/lane coordinates come from the same CuTe MMA partition, not
    // an assumed warp-to-M/N assignment. Payload already has register order.
    auto load_a=[&](auto half,auto& av,auto& hv) {
      o1_static_for<0,2>([&](auto mi) {
        const int atom=cute::get<0>(coords(cute::_0{},mi,cute::_0{}))/16;
        const int off=atom*1024+half*512+(threadIdx.x&31)*16;
        auto ar=cute::recast<uint32_t>(av(cute::_,mi,cute::_0{}));
        auto hr=cute::recast<uint32_t>(hv(cute::_,mi,cute::_0{}));
        const uint4 la=load128(s.low[slot]+off),ha=load128(s.high[slot]+off);
        ar(0)=la.x;ar(1)=la.y;ar(2)=la.z;ar(3)=la.w;
        hr(0)=ha.x;hr(1)=ha.y;hr(2)=ha.z;hr(3)=ha.w;
      });
    };
    load_a(cute::_0{},a0,h0);load_a(cute::_1{},a1,h1);
'''
OLD_B = '''      cute::copy(SCopy{},bc.partition_S(tile_b(slot,nb,cute::_0{})),bd0);
      cute::copy(SCopy{},bc.partition_S(tile_b(slot,nb,cute::_1{})),bd1);
'''
NEW_B = '''      auto load_b=[&](auto half,auto& bv) {
        auto br=cute::recast<uint32_t>(bv);
        o1_static_for<0,2>([&](auto pair) {
          const int atom=cute::get<1>(coords(cute::_0{},cute::_0{},nb*cute::_4{}+pair*cute::_2{}))/16;
          const uint4 data=load128(s.weight[slot]+atom*1024+half*512+(threadIdx.x&31)*16);
          br(pair*4+0)=data.x;br(pair*4+1)=data.y;br(pair*4+2)=data.z;br(pair*4+3)=data.w;
        });
      };
      load_b(cute::_0{},b0);load_b(cute::_1{},b1);
'''
LOAD_HELPER = '''__device__ __forceinline__ uint4 load128(const void* p) {
  uint4 v;const uint32_t address=uint32_t(__cvta_generic_to_shared(p));
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
      : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "r"(address));
  return v;
}
'''


def generated_header(source):
    text = eight_header(source)
    replacements = ((OLD_A, NEW_A), (OLD_B, NEW_B),
        ('s.low[slot]+la(row,col)', 's.low[slot]+off'),
        ('s.high[slot]+la(row,col)', 's.high[slot]+off'),
        ('s.weight[slot]+lb(row,col)', 's.weight[slot]+off'),
        ('__device__ __forceinline__ void body(', LOAD_HELPER + '\n__device__ __forceinline__ void body('))
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError('v78 source boundary drift: ' + old)
        text = text.replace(old, new)
    return text.replace('o78_eight_chain_experiment', 'o78_register_layout_experiment')


ONLINE = '''#include "roof_o78_row_fused_prepare.cu"
#include "o78_register_pack.cuh"
namespace {
struct RegisterLayoutOnline:RowFusedOnline {
  using RowFusedOnline::RowFusedOnline;
  void weight(bool layout) {
    RowFusedOnline::weight(true);
    if(layout) o78_register_pack::pack<true><<<dim3(n/16,32),32,0,stream>>>(
      reinterpret_cast<const uint8_t*>(v[1]),reinterpret_cast<uint8_t*>(v[1])+size_t(n)*2048,n);
  }
  void activation(bool layout) {
    RowFusedOnline::activation(true);
    if(layout) o78_register_pack::pack<false><<<dim3(m/16,32),32,0,stream>>>(
      reinterpret_cast<const uint8_t*>(v[0]),reinterpret_cast<uint8_t*>(v[0])+size_t(m)*4096,m);
  }
};
}
'''


def generated_driver(source):
    marker = 'extern "C" int roof_o78_row_fused_benchmark('
    if source.count(marker) != 1:
        raise ValueError('v73 timer boundary drift')
    body = source[source.index(marker):]
    if body.count('RowFusedOnline x(') != 1:
        raise ValueError('unexpected timer construction')
    return ONLINE + body.replace('roof_o78_row_fused_benchmark(', 'roof_o78_register_layout_benchmark(').replace(
        'RowFusedOnline x(', 'RegisterLayoutOnline x(')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('reports/o378_roof_v78_codegen'))
    parser.add_argument('--preparation', type=Path, default=Path('reports/o378_roof_v73_codegen'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):
        parser.error('fresh repository output required')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    prior = json.loads((args.baseline / 'codegen.json').read_text())
    preparation = json.loads((args.preparation / 'build.json').read_text())
    for receipt in (prior, preparation):
        for path, value in receipt['sources'].items():
            if sha(ROOT / path) != value:
                raise ValueError('baseline source drift: ' + path)
    if sha(args.baseline / 'o78_eight_chain.cubin') != prior['cubin_sha256']:
        raise ValueError('v78 cubin drift')
    cuda=Path('/usr/local/cuda-12.8/bin'); cutlass=ROOT/'third_party/cutlass-src'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    commit=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'release 12.8' not in version or commit!='db1c288993354c88e551c40c19a8fb93a774a241':
        parser.error('pinned CUDA/CUTLASS required')
    out.mkdir(parents=True)
    source=(ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    (out/'o78_eight_chain_generated.cuh').write_text(eight_header(source))
    header=out/'o78_register_layout_generated.cuh';header.write_text(generated_header(source))
    driver=out/'register_layout_driver.cu'
    driver.write_text(generated_driver((ROOT/'csrc/sm80/roof_o78_row_fused_prepare.cu').read_text()))
    sources=set(prior['sources'])|set(preparation['sources'])|{
        'csrc/sm80/o78_register_pack.cuh','csrc/sm80/roof_o78_register_layout_probe.cu',
        'scripts/probe_o78_register_layout_codegen.py'}
    receipt=dict(scope='register_layout_compile_not_performance_acceptance',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(sources)},generated_header_sha256=sha(header),
        generated_driver_sha256=sha(driver),baseline_cubin_sha256=prior['cubin_sha256'],
        production_default_changed=False,cta_tile=[64,128,128],threads=128,stages=2,shared_bytes=34304,
        extra_payload_bytes_4096=25165824,nvcc=version,cutlass_commit=commit,commands=[])
    def run(command, filename):
        receipt['commands'].append(command)
        with (out/filename).open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    compiler=[str(cuda/'nvcc'),'-O3','-std=c++17','--expt-relaxed-constexpr','-lineinfo','-arch=sm_80',
        '-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out)]
    src=str(ROOT/'csrc/sm80/roof_o78_register_layout_probe.cu')
    cubin=out/'o78_register_layout.cubin'
    run(compiler+[src,'-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(compiler+[src,'-ptx','-o',str(out/'o78_register_layout.ptx')],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],'o78_register_layout.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    sass=(out/'o78_register_layout.sass').read_text()
    entries=static_entries(sass,'^adangel_roof_o78_(?:eight_chain_candidate|register_layout_candidate)$',{CONTROL,SYMBOL})
    receipt.update(cubin_sha256=sha(cubin),entries=entries,
        control_comparison=compare((args.baseline/'o78_eight_chain.sass').read_text(),sass,'^'+CONTROL+'$'),
        liveness={s:analyze((out/'liveness.txt').read_text(),s) for s in (CONTROL,SYMBOL)},
        artifact_sha256={p:sha(out/p) for p in ('build.log','o78_register_layout.sass',
            'o78_register_layout.ptx','resources.txt','liveness.txt')})
    (out/'codegen.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if not receipt['control_comparison']['passed'] or not all(
        e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma'] and e['all_copies_bypass_l1']
        for e in entries.values()): raise ValueError('same-entry native INT4/control audit failed')
    loop=next(l for l in receipt['liveness'][SYMBOL]['loops'] if l['kind']=='integer')
    if any(op.startswith('LDSM') for op in loop['opcode_counts']):
        raise ValueError('integer path still uses LDSM')
    lib=out/'libo78_gpu_prepare.so'
    run(compiler+[str(driver),'-shared','-Xcompiler=-fPIC','-Xptxas=-v','-lcuda','-o',str(lib)],'driver_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(lib)],'prepare.sass')
    (out/'build.json').write_text(json.dumps(dict(scope='v73_with_explicit_online_operand_repack',
        source_commit=receipt['source_commit'],sources=receipt['sources'],driver_sha256=sha(lib),
        generated_driver_sha256=sha(driver),nvcc=version),indent=2)+'\n')
    print((out/'build.log').read_text())
    print(json.dumps(receipt['liveness'][SYMBOL],indent=2))


if __name__=='__main__':main()
