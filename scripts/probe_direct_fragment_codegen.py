#!/usr/bin/env python3
"""v112: cache-fed register fragments, not v85 LDS substitution/v94 rescan.

Reuse the proven v85 packing and v94 single-A-atom math, but remove payload
shared staging/slot states/hot barriers. Cooperative full-K metadata cache;
ordinary public ld.global.ca feeds native two-route INT4 MMA. One fixed
128-thread/4-CTA register budget, no parameter/cache/partial/stage scan.
Logical payload reads increase3x, extra online packing remains chargeable.
Compile gates are not real performance/MSE/safety acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from compare_a100_codegen import compare
from probe_cooperative_reuse_codegen import analyze,once
from probe_o78_eight_chain_codegen import ROOT
from probe_roof_fullk_integer_codegen import static_entries

CONTROL='adangel_roof_o78_eight_chain_candidate'
SYMBOL='adangel_roof_o78_direct_fragment_candidate'
STEM='o78_direct_fragment'
BASE=ROOT/'reports/o378_roof_v78_codegen'
FROZEN=ROOT/'docs/evidence/a100_o378_roof_v94/reports/o378_roof_v94_codegen'
SHARED=24576

CONFIG='''// v112: fixed public cache-fed experiment; no production dispatch.
#pragma once
namespace o78_direct_fragment_experiment {
using C=O78::O3AmpereConfig<64,128,128,false,2,true,2>;
template<class Scale> struct alignas(128) Storage {
  alignas(16) Scale activation_factors[32][64];
  alignas(16) Scale weight_factors[32][128];
};
static_assert(sizeof(Storage<int>)==24576 && sizeof(Storage<float>)==24576);
__device__ __forceinline__ uint4 load128(const void* p) {
  uint4 v;
  asm volatile("ld.global.ca.v4.b32 {%0,%1,%2,%3}, [%4];"
      : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p) : "memory");
  return v;
}
template<class Scale>
__device__ __forceinline__ void cache_metadata(Storage<Scale>& s,
    const Scale* af,const Scale* wf,unsigned m,unsigned n) {
  for(unsigned off=threadIdx.x*4;off<32*64;off+=128*4) {
    auto v=load128(af+(off/64)*m+blockIdx.y*64+off%64);
    *reinterpret_cast<uint4*>(&s.activation_factors[0][0]+off)=v;
  }
  for(unsigned off=threadIdx.x*4;off<32*128;off+=128*4) {
    auto v=load128(wf+(off/128)*n+blockIdx.x*128+off%128);
    *reinterpret_cast<uint4*>(&s.weight_factors[0][0]+off)=v;
  }
  __syncthreads(); // Single publish barrier; cache is read-only thereafter.
}
'''


def header(source):
    body=source[source.index('__device__ __forceinline__ void body('):]
    body=body.replace('o78_operand_stream_experiment','o78_direct_fragment_experiment')
    body=once(body,'__device__ __forceinline__ void body(',
        'template<bool Integer>\n__device__ __forceinline__ void body(')
    body=once(body,'const int32_t* af,const int32_t* wf,',
        'const std::conditional_t<Integer,int,float>* af,\n'
        '    const std::conditional_t<Integer,int,float>* wf,')
    body=once(body,'reinterpret_cast<Storage*>(buf)',
        'reinterpret_cast<Storage<std::conditional_t<Integer,int,float>>*>(buf)')
    body=once(body,'auto acc=cute::make_fragment_like<int>(thr.make_fragment_C(coords));',
        'auto acc=cute::make_fragment_like<std::conditional_t<Integer,int,float>>(thr.make_fragment_C(coords));')
    # Fragment construction inspects dtype/layout only. No null pointer load.
    for field in ('low','high','weight'):
        body=once(body,'static_cast<void*>(s.'+field+'[slot])','static_cast<void*>(nullptr)')
    body=once(body,'  prefetch(s,0,0,a,w,af,wf,m,n,k);',
        '  cache_metadata(s,af,wf,m,n);')
    old='''    const int slot=group%2;
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads(); // All prior readers finish before a slot is reused.
    if(group+1<Groups) prefetch(s,1-slot,group+1,a,w,af,wf,m,n,k);
'''
    body=once(body,old,'    const int slot=0; // Type-only dummy fragment view.\n')
    old='''      cute::copy(SCopy{},bc.partition_S(tile_b(slot,nb,cute::_0{})),bd0);
      cute::copy(SCopy{},bc.partition_S(tile_b(slot,nb,cute::_1{})),bd1);
'''
    new='''      auto load_b=[&](auto half,auto& bv) {
        auto br=cute::recast<uint32_t>(bv);
        o1_static_for<0,2>([&](auto pair) {
          const int off=o78_register_layout_mapping::w_offset(
              cute::get<1>(coords(cute::_0{},cute::_0{},nb*cute::_4{}+pair*cute::_2{})),half,threadIdx.x&31);
          const uint4 v=load128(w+size_t(group)*n*64+blockIdx.x*128*64+off);
          br(pair*4+0)=v.x;br(pair*4+1)=v.y;br(pair*4+2)=v.z;br(pair*4+3)=v.w;
        });
      };
      load_b(cute::_0{},b0);load_b(cute::_1{},b1);
'''
    body=once(body,old,new)
    marker='        auto hd=ac_high.retile_D(ar);auto ld=ac_low.retile_D(al);'
    helper='''        auto load_a=[&](bool high_plane,int half) {
          const int off=o78_register_layout_mapping::a_offset(atom_m*16,half,threadIdx.x&31);
          const auto* src=a+size_t(group)*m*64+blockIdx.y*64*64+off;
          if(high_plane)src+=size_t(m)*(k/2);
          const uint4 v=load128(src);auto words=cute::recast<uint32_t>(ar);
          words(0)=v.x;words(1)=v.y;words(2)=v.z;words(3)=v.w;
        };'''
    body=once(body,marker,helper)
    for kind,copy,view,plane in (('high','SCopy','ac_high','true'),('low','LCopy','ac_low','false')):
        for half in (0,1):
            dst='hd' if kind=='high' else 'ld'
            old=f'        cute::copy({copy}{{}},{view}.partition_S(atom_tile_a({kind}(slot),atom_m,cute::_{half}{{}})),{dst});'
            body=once(body,old,f'        load_a({plane},{half});')
    old='''            const int coefficient=s.activation_factors[slot][cute::get<0>(coord)]*
                                  s.weight_factors[slot][cute::get<1>(coord)];
            acc(vi,mi,full_ni)+=partial(vi,ni)*coefficient;'''
    new='''            if constexpr(Integer) {
              const int coefficient=s.activation_factors[group][cute::get<0>(coord)]*
                                    s.weight_factors[group][cute::get<1>(coord)];
              acc(vi,mi,full_ni)+=partial(vi,ni)*coefficient;
            } else {
              const float scale=__fmul_rn(s.activation_factors[group][cute::get<0>(coord)],
                                       s.weight_factors[group][cute::get<1>(coord)]);
              acc(vi,mi,full_ni)=__fmaf_rn(float(partial(vi,ni)),scale,acc(vi,mi,full_ni));
            }'''
    body=once(body,old,new)
    start=body.index('  // Reuse the 64 INT32 register slots')
    end=body.index('  o1_static_for<0,decltype(cute::size(acc))::value/2>',start)
    body=body[:start]+'  if constexpr(Integer) {\n'+body[start:end]+'  }\n'+body[end:]
    body=body.replace('__int_as_float(acc(i+cute::_1{}))','output_value(i+cute::_1{})')
    body=body.replace('__int_as_float(acc(i))','output_value(i)')
    body=body.replace('__int_as_float(acc(i+cute::_1{}))','output_value(i+cute::_1{})')
    store='  o1_static_for<0,decltype(cute::size(acc))::value/2>'
    helper='''  auto output_value=[&](auto i) {
    if constexpr(Integer)return __int_as_float(acc(i));
    else return acc(i);
  };
'''
    body=once(body,store,helper+store)
    return CONFIG+body


def cost_model():
    old=2048*32*(64*128+128*64+(64+128)*4)
    payload=2048*32*4*24*32*16
    metadata=2048*(64+128)*32*4
    return dict(cta_tile=[64,128,128],threads=128,target_ctas_per_sm=4,
        source_chains=4,acc_per_thread=64,partial_per_thread=16,shared_bytes=SHARED,
        payload_vector_reads_per_warp=24,metadata_publish_barriers_per_CTA=1,
        old_logical_read_bytes=old,new_payload_read_bytes=payload,
        new_metadata_read_bytes=metadata,new_logical_read_ratio=(payload+metadata)/old,
        extra_packed_allocation_4096_bytes=25165824,online_pack_read_write_bytes=50331648,
        interpretation='logical_access_model_not_physical_DRAM_traffic_or_predicted_speedup')


def compile_gate(old,new):
    ops=new['loop']['opcode_counts']
    local=sum(v for k,v in ops.items() if k.startswith(('LDL','STL')))
    checks=dict(four_CTA_register_budget=new['allocated_gpr']<=128,
        no_hot_local=local==0,native_INT4_64=ops.get('IMMA.16864.S4.S4')==ops.get('IMMA.16864.U4.S4')==32,
        no_hot_payload_shared_path=not any(k.startswith(('LDSM','LDGSTS','BAR.','DEPBAR','LDGDEPBAR','STS')) for k in ops),
        fewer_instructions=new['loop']['static_instructions']<old['loop']['static_instructions'])
    return dict(passed=all(checks.values()),checks=checks,
        runtime_gate='actual>=4 CTAs/SM, full GPU correctness/fallback/safety, then full24 paired Event/MSE',
        conversion_caveat='online A/W fragment packing charged in conversion/cold/steady; no free permutation')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh repository output required')
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    prior=json.loads((BASE/'codegen.json').read_text());old94=json.loads((FROZEN/'codegen.json').read_text())
    src=FROZEN/'o78_operand_stream_generated.cuh'
    if sha(src)!=old94['artifact_sha256'][src.name]:raise ValueError('v94 frozen source drift')
    for name,digest in prior['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('v78 source drift: '+name)
    if sha(BASE/'o78_eight_chain.cubin')!=prior['cubin_sha256']:raise ValueError('v78 binary drift')
    cuda=Path('/usr/local/cuda-12.8/bin');cutlass=ROOT/'third_party/cutlass-src'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    commit=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'release 12.8' not in version or commit!='db1c288993354c88e551c40c19a8fb93a774a241':p.error('pinned toolchain required')
    extensions=list((ROOT/'python/adangel').glob('_sm80*.so'))
    if len(extensions)!=1:raise ValueError('one unchanged production extension required')
    out.mkdir(parents=True)
    (out/'o78_eight_chain_generated.cuh').write_text((BASE/'o78_eight_chain_generated.cuh').read_text())
    (out/(STEM+'_generated.cuh')).write_text(header(src.read_text()))
    own={'scripts/probe_direct_fragment_codegen.py','csrc/sm80/roof_o78_direct_fragment_probe.cu',
        'csrc/sm80/o78_register_layout_mapping.cuh','csrc/sm80/verify_o78_register_layout.cu',
        str(src.relative_to(ROOT))}
    r=dict(scope='v112_cache_direct_fragment_compile_gate_not_runtime_acceptance',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(set(prior['sources'])|own)},
        work_model=cost_model(),shared_bytes=SHARED,commands=[],nvcc=version,cutlass_commit=commit,
        extension_sha256_before=sha(extensions[0]),candidate_gpu_launched=False,
        new_GEMM_measured=False,new_MSE_measured=False,production_default_changed=False,
        old_failures_not_relaxed=['v85_shared_LDS','v94_shared_A_stream','v109_private_cp_async'])
    def run(cmd,name):
        r['commands'].append(cmd)
        with (out/name).open('w') as log:
            subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,TMPDIR=str(ROOT/'tmp')),
                stdout=log,stderr=subprocess.STDOUT,check=True)
    common=[str(cuda/'nvcc'),'-std=c++17','--expt-relaxed-constexpr','-arch=sm_80',
        '-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out)]
    verifier=out/'verify_coordinates'
    run(common+['-O2',str(ROOT/'csrc/sm80/verify_o78_register_layout.cu'),'-o',str(verifier)],'mapping_build.log')
    run([str(verifier)],'mapping.json');r['mapping']=json.loads((out/'mapping.json').read_text())
    if not r['mapping']['passed'] or r['mapping']['gpu_execution']:raise ValueError('host mapping failed')
    flags=common+['-O3','-lineinfo',str(ROOT/'csrc/sm80/roof_o78_direct_fragment_probe.cu')]
    cubin=out/(STEM+'.cubin')
    run(flags+['-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(flags+['-ptx','-o',str(out/(STEM+'.ptx'))],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],STEM+'.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    sass=(out/(STEM+'.sass')).read_text();live=(out/'liveness.txt').read_text()
    r['control_comparison']=compare((BASE/'o78_eight_chain.sass').read_text(),sass,'^'+CONTROL+'$')
    if not r['control_comparison']['passed']:raise ValueError('old v78 machine words changed')
    r['entries']=static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})
    if not all(e['native_s4_s4'] and e['native_u4_s4'] and not e['int8_mma'] for e in r['entries'].values()):
        raise ValueError('same-entry native INT4 audit failed')
    r['liveness']={s:analyze(live,s) for s in (CONTROL,SYMBOL)}
    r['compile_gate']=compile_gate(r['liveness'][CONTROL],r['liveness'][SYMBOL])
    r['extension_sha256_after']=sha(extensions[0])
    if r['extension_sha256_before']!=r['extension_sha256_after']:raise ValueError('extension changed')
    r['artifact_sha256']={f.name:sha(f) for f in out.iterdir() if f.is_file()}
    (out/'codegen.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(dict(compile_gate=r['compile_gate'],liveness=r['liveness'],mapping=r['mapping']),indent=2),flush=True)


if __name__=='__main__':main()
