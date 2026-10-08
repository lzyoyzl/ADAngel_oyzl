#!/usr/bin/env python3
"""v138: packed HiF4 micro8/micro4 decode, identical v123 A and v78 GEMM.

One candidate, no performance screening or production-default change.
Reuse the checked v123 library entries, not a stale activation baseline.
"""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from analyze_o78_row_fused_codegen import entries
from compare_a100_codegen import compare
import probe_nv6_swar_codegen as nv6

ROOT=Path(__file__).resolve().parents[1]
BASELINE=ROOT/'reports/o378_roof_v123_codegen_r2'


def scalar(code,e8,e4):
    magnitude=round((code&7)*(2**(e8+e4))/4)
    return -magnitude if code&8 else magnitude


def reference(word,e8,e4):
    values=[scalar((word>>(4*j))&15,e8,(e4>>(j//4))&1) for j in range(8)]
    return sum((q&15)<<(4*j) for j,q in enumerate(values)),sum(q*q for q in values)


def swar(word,e8,e4):
    if any(isinstance(x,bool) or not isinstance(x,int) for x in (word,e8,e4)) or not (
            0<=word<=0xffffffff and 0<=e8<=1 and 0<=e4<=3):
        raise ValueError('uint32 word, micro8 bit and micro4 pair required')
    lsb=0x11111111
    magnitude=word&0x77777777
    half=((word>>1)&0x33333333)+((word&(word>>1))&lsb)
    quarter=((word>>2)&lsb)+(((word>>1)&(word|(word>>2)))&lsb)
    mask=(e4&1)*0xffff+((e4>>1)&1)*0xffff0000
    hi,lo=(magnitude,half) if e8 else (half,quarter)
    rounded=(hi&mask)|(lo&~mask)
    sign=(word>>3)&lsb
    packed=((rounded^(sign*7))+sign)^(sign<<3)
    return packed,sum(((rounded>>(4*j))&7)**2 for j in range(8))


def generated_header():
    old=(ROOT/'csrc/sm80/roof_row_fused_conversion.cuh').read_text()
    start='  // Micro8 is shared by eight elements; micro4 by four.'
    end='  const unsigned dest=dst_group*(128/Elements)+lane;'
    if old.count(start)!=1 or old.count(end)!=1:raise ValueError('v73 decoder boundary drift')
    a,b=old.index(start),old.index(end)
    body='''  if constexpr(Format==Kind::Hif4) {
    // One metadata-byte load per sharing level for this 16-element vector.
    const unsigned e8=(micro8[src_group*2+lane/4]>>(2*(lane%4)))&3u;
    const unsigned e4=(micro4[src_group*4+lane/2]>>(4*(lane%2)))&15u;
    #pragma unroll
    for(int j=0;j<2;++j) {
      const uint2 v=hif4_swar_probe::packed_word(words[j],(e8>>j)&1u,(e4>>(2*j))&3u);
      low[j]=v.x;square+=v.y;
    }
  } else {
'''+old[a:b]+'  }\n'
    new=(old[:a]+body+old[b:]).replace('namespace row_fused_probe','namespace hif4_row_swar_probe')
    new=new.replace('adangel_sm80_row_conversion_metadata','adangel_sm80_hif4_swar_metadata')
    return new.replace('#include "roof_vector_conversion_impl.cuh"',
        '#include "roof_vector_conversion_impl.cuh"\n#include "hif4_swar_conversion.cuh"')


def generated_host():
    old=(ROOT/'csrc/sm80/roof_o78_row_fused_prepare.cu').read_text()
    boundary='extern "C" int roof_o78_row_fused_benchmark('
    if old.count(boundary)!=1:raise ValueError('v73 Event boundary drift')
    benchmark=old[old.index(boundary):].replace('roof_o78_row_fused_benchmark','roof_o78_hif4_swar_benchmark')
    benchmark=benchmark.replace('RowFusedOnline x(','Hif4SwarOnline x(')
    return '''// Generated v138; retain every original v123 library entry.
#include "nv6_swar_prepare.cu"
#include "hif4_swar_generated.cuh"
namespace {
struct Hif4SwarOnline:Nv6SwarOnline {
  using Nv6SwarOnline::Nv6SwarOnline;
  void activation(bool) {Nv6SwarOnline::activation(true);}
  void weight(bool candidate) {
    if(!candidate || variant!=8){RowFusedOnline::weight(true);return;}
    hif4_row_swar_probe::adangel_sm80_hif4_swar_metadata<Kind::Hif4,16>
      <<<n,256,0,stream>>>(
        reinterpret_cast<const uint8_t*>(sw[0]),reinterpret_cast<const uint8_t*>(sw[1]),
        reinterpret_cast<const float*>(sw[2]),reinterpret_cast<const uint8_t*>(sw[3]),
        reinterpret_cast<const uint8_t*>(sw[4]),reinterpret_cast<uint8_t*>(v[1]),
        reinterpret_cast<float*>(v[3]),reinterpret_cast<uint32_t*>(v[17]),n,32,w_mult,
        reinterpret_cast<int32_t*>(v[5]),reinterpret_cast<float*>(v[7]),
        reinterpret_cast<uint64_t*>(v[9]),reinterpret_cast<int32_t*>(v[11]),
        reinterpret_cast<uint32_t*>(v[13]));
  }
};
__global__ void adangel_hif4_swar_exhaustive(uint4* out) {
  const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
  const unsigned lo=i&65535u,hi=(i&65536u)?(lo^65535u):lo;
  const unsigned e8=(i>>17)&1u,e4=(i>>18)&3u,word=lo|(hi<<16);
  unsigned packed=0,square=0;
  #pragma unroll
  for(int j=0;j<8;++j) {
    const unsigned c=(word>>(4*j))&15u;
    const int mag=int(vector_probe::rne((c&7u)<<(e8+((e4>>(j/4))&1u)),2));
    const int q=(c&8u)?-mag:mag;
    packed|=(unsigned(q)&15u)<<(4*j);square+=unsigned(q*q);
  }
  const uint2 v=hif4_swar_probe::packed_word(word,e8,e4);
  out[i]=make_uint4(packed,v.x,square,v.y);
}
}
extern "C" int roof_hif4_swar_exhaustive(uint4* out,void* stream) {
  if(!out)return 1;
  adangel_hif4_swar_exhaustive<<<4096,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(out);
  return int(cudaGetLastError());
}
// No kernel launch: inspect actual conversion residency before timing.
extern "C" int roof_hif4_swar_resources(int* values) {
  if(!values)return 1;
  cudaFuncAttributes old{},candidate{};int old_blocks=0,new_blocks=0;
  cudaError_t err=cudaFuncGetAttributes(&old,
      row_fused_probe::adangel_sm80_row_conversion_metadata<Kind::Hif4,16>);
  if(err!=cudaSuccess)return int(err);
  err=cudaFuncGetAttributes(&candidate,
      hif4_row_swar_probe::adangel_sm80_hif4_swar_metadata<Kind::Hif4,16>);
  if(err!=cudaSuccess)return int(err);
  err=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&old_blocks,
      row_fused_probe::adangel_sm80_row_conversion_metadata<Kind::Hif4,16>,256,0);
  if(err!=cudaSuccess)return int(err);
  err=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&new_blocks,
      hif4_row_swar_probe::adangel_sm80_hif4_swar_metadata<Kind::Hif4,16>,256,0);
  if(err!=cudaSuccess)return int(err);
  values[0]=old.numRegs;values[1]=old.localSizeBytes;values[2]=old.sharedSizeBytes;
  values[3]=old.maxThreadsPerBlock;values[4]=old_blocks;
  values[5]=candidate.numRegs;values[6]=candidate.localSizeBytes;values[7]=candidate.sharedSizeBytes;
  values[8]=candidate.maxThreadsPerBlock;values[9]=new_blocks;
  return 0;
}
'''+benchmark


def audit(directory):
    before=(BASELINE/'prepare.sass').read_text();after=(directory/'prepare.sass').read_text()
    old,new=entries(before),entries(after)
    pattern='|'.join('^'+re.escape(s)+'$' for s in old)
    raw_comparison=compare(before,after,pattern)
    # nvcc embeds the TU name/hash in this anonymous test-kernel symbol.
    # Compare all its actual instruction words, not just the stable kernels;
    # normalize ONLY this verified one-to-one symbol, never instruction data.
    probe_suffix='adangel_nv6_swar_exhaustiveEP5uint4'
    lhs=[s for s in old if s.endswith(probe_suffix)]
    rhs=[s for s in new if s.endswith(probe_suffix)]
    if len(lhs)!=1 or len(rhs)!=1:raise ValueError('unique retained FP6 test entry required')
    comparison=compare(before,after.replace(rhs[0],lhs[0]),pattern)
    def find(rows,name):
        found=[(s,e) for s,e in rows.items() if name in s and 'GroupedSourceKindE2E' in s]
        if len(found)!=1:raise ValueError('one HiF4 entry required: '+name)
        return found[0]
    old_symbol,old_entry=find(old,'adangel_sm80_row_conversion_metadata')
    symbol,new_entry=find(new,'adangel_sm80_hif4_swar_metadata')
    resources={s:dict(registers=int(r),stack=int(stack),shared=int(shared),local=int(local))
        for s,r,stack,shared,local in re.findall(
            r' Function (\S+):\s+REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)',
            (directory/'resources.txt').read_text())}
    a,b=resources[old_symbol],resources[symbol];ops=new_entry['opcode_counts']
    local_count=sum(c for op,c in ops.items() if op.startswith(('LDL','STL')))
    barriers=sum(c for op,c in ops.items() if op.startswith('BAR.SYNC'))
    dots=sum(c for op,c in ops.items() if op.startswith('IDP.4A.U8.U8'))
    improvement=1-new_entry['instructions']/old_entry['instructions']
    lib=ct.CDLL(str((directory/'libo78_gpu_prepare.so').resolve()))
    fn=lib.roof_hif4_swar_resources;fn.argtypes=[ct.POINTER(ct.c_int)];fn.restype=ct.c_int
    values=(ct.c_int*10)();error=fn(values)
    if error:raise RuntimeError('CUDA resource query failed: '+str(error))
    runtime={name:dict(zip(('registers','local','shared','max_threads','active_blocks_per_sm'),
                         list(values)[offset:offset+5])) for name,offset in (('control',0),('candidate',5))}
    if any(runtime[name][key]!=resources[sym][key] for name,sym in
            (('control',old_symbol),('candidate',symbol)) for key in ('registers','local','shared')):
        raise ValueError('SASS/runtime resource mismatch')
    residency=runtime['candidate']['active_blocks_per_sm']>=runtime['control']['active_blocks_per_sm']
    gate=(comparison['passed'] and residency and b['stack']==b['local']==local_count==0 and
          b['shared']==256 and barriers==1 and improvement>=.05 and dots==4)
    return dict(scope='HiF4_conversion_compile_gate_not_latency',old_controls=comparison,
        raw_symbol_comparison=raw_comparison,anonymous_probe_symbol_mapping={rhs[0]:lhs[0]},
        control=dict(symbol=old_symbol,**a,**old_entry),candidate=dict(symbol=symbol,**b,**new_entry),
        static_instruction_reduction_fraction=improvement,local_instructions=local_count,
        scalar_DP4A_instructions=dots,cta_barriers=barriers,runtime_resources=runtime,
        residency_not_worse=residency,activation_encoding_identical=True,worth_runtime_validation=gate)


def checked(directory):
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    receipt=json.loads((directory/'build.json').read_text())
    for name,digest in receipt['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('source drift: '+name)
    for name,digest in receipt['artifact_sha256'].items():
        if sha(directory/name)!=digest:raise ValueError('artifact drift: '+name)
    for name,source in generated_files().items():
        if (directory/name).read_text()!=source:raise ValueError('generated source drift: '+name)
    if receipt['production_default_changed'] or receipt['GEMM_modified']:raise ValueError('scope drift')
    return receipt


def generated_files():
    return {'nv6_swar_generated.cuh':nv6.generated_header(),'nv6_swar_prepare.cu':nv6.generated_host(),
        'hif4_swar_generated.cuh':generated_header(),'hif4_swar_prepare.cu':generated_host()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh repository output required')
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    original=nv6.checked(BASELINE)
    cuda=Path('/usr/local/cuda-12.8/bin')
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    if 'release 12.8' not in version:p.error('CUDA12.8 required')
    out.mkdir(parents=True)
    names=set(original['sources'])|{'csrc/sm80/hif4_swar_conversion.cuh','scripts/probe_hif4_swar_codegen.py'}
    receipt=dict(scope='v138_HiF4_packed_micro8_micro4_decode_DP4A_norm',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(names)},nvcc=version,commands=[],GEMM_modified=False,
        production_default_changed=False,quantization='HiF4_G128_Q4_F0_RNE_unchanged',
        activation_implementation='v123_packed_FP6_both_policies',
        control_build_sha256=sha(BASELINE/'build.json'),no_small_performance_screen=True)
    for name,source in generated_files().items():(out/name).write_text(source)
    lib=out/'libo78_gpu_prepare.so'
    def run(command,name):
        receipt['commands'].append(command)
        with (out/name).open('w') as log:
            subprocess.run(command,cwd=ROOT,env=dict(os.environ,TMPDIR=str(ROOT/'tmp')),
                stdout=log,stderr=subprocess.STDOUT,check=True)
    run([str(cuda/'nvcc'),'-O3','-std=c++17','-lineinfo','-arch=sm_80','-shared',
        '-Xcompiler=-fPIC','-Xptxas=-v','-I'+str(ROOT/'csrc/sm80'),
        str(out/'hif4_swar_prepare.cu'),'-lcuda','-o',str(lib)],'build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(lib)],'prepare.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(lib)],'resources.txt')
    receipt['audit']=audit(out);receipt['driver_sha256']=sha(lib)
    receipt['artifact_sha256']={f.name:sha(f) for f in out.iterdir() if f.is_file()}
    (out/'build.json').write_text(json.dumps(receipt,indent=2)+'\n')
    checked(out)
    print(json.dumps({k:v for k,v in receipt['audit'].items() if k not in ('control','candidate','old_controls')},indent=2))


if __name__=='__main__':main()
