#!/usr/bin/env python3
"""v118: one exact packed-word NVFP4 weight conversion, compile gate first.

Not v34 scalar integer decode, v53 vector loads, v73 row metadata fusion,
or v106 MXFP8 warp lookup. Keep v73 geometry/metadata and identical GEMM.
The candidate replaces sixteen scalar nibble lookups/repacking/squaring
with two eight-nibble Boolean decoders and exact bit-plane square sums.
This can only improve conversion/Cold, not Compute-only GEMM peak.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from analyze_o78_row_fused_codegen import entries
from compare_a100_codegen import compare

ROOT=Path(__file__).resolve().parents[1]
BASELINE=ROOT/'reports/o378_roof_v73_codegen'
MASK=0x11111111


def reference(word):
    packed=square=0
    for i in range(8):
        code=(word>>(4*i))&15
        q=round((-.5 if code&8 else .5)*[0,1,2,3,4,6,8,12][code&7])
        packed|=(q&15)<<(4*i)
        square+=q*q
    return packed,square


def truth(lut,a,b,c):
    """Independent CPU truth-table evaluator on the nibble-LSB positions."""
    out=0
    for i in range(8):
        if (lut>>i)&1:
            out|=(a if i&4 else ~a)&(b if i&2 else ~b)&(c if i&1 else ~c)
    return out&MASK


def swar(word):
    if not 0<=word<=0xffffffff:raise ValueError('uint32 packed word required')
    x0=word&MASK;x1=(word>>1)&MASK;x2=(word>>2)&MASK;sign=(word>>3)&MASK
    m0=truth(0x24,x2,x1,x0);m1=truth(0xb8,x2,x1,x0);m2=truth(0xc0,x2,x1,x0)
    magnitude=m0|(m1<<1)|(m2<<2)
    low=((magnitude^(sign*7))+sign)&0x77777777
    packed=low|((sign&(x1|x2))<<3)
    square=m0.bit_count()+sum(weight*truth(lut,x2,x1,x0).bit_count()
        for lut,weight in ((0x98,4),(0x20,8),(0x40,16),(0x80,32)))
    return packed,square


def generated_header():
    old=(ROOT/'csrc/sm80/roof_row_fused_conversion.cuh').read_text()
    start='  // Micro8 is shared by eight elements; micro4 by four.'
    end='  const unsigned dest=dst_group*(128/Elements)+lane;'
    if old.count(start)!=1 or old.count(end)!=1:raise ValueError('v73 decoder boundary drift')
    a,b=old.index(start),old.index(end)
    if a>=b:raise ValueError('invalid v73 decoder range')
    body='''  if constexpr(Format==Kind::Nv4) {
    #pragma unroll
    for(int w=0;w<2;++w) {
      const uint2 result=nv4_swar_probe::packed_word(words[w]);
      low[w]=result.x;square+=result.y;
    }
  } else {
'''+old[a:b]+'  }\n'
    new=(old[:a]+body+old[b:]).replace('namespace row_fused_probe','namespace nv4_row_swar_probe')
    new=new.replace('adangel_sm80_row_conversion_metadata','adangel_sm80_row_swar_metadata')
    return new.replace('#include "roof_vector_conversion_impl.cuh"',
        '#include "roof_vector_conversion_impl.cuh"\n#include "nv4_swar_conversion.cuh"')


def generated_host():
    old=(ROOT/'csrc/sm80/roof_o78_row_fused_prepare.cu').read_text()
    boundary='extern "C" int roof_o78_row_fused_benchmark('
    if old.count(boundary)!=1:raise ValueError('v73 Event wrapper boundary drift')
    benchmark=old[old.index(boundary):].replace('roof_o78_row_fused_benchmark','roof_o78_nv4_swar_benchmark')
    benchmark=benchmark.replace('RowFusedOnline x(','Nv4SwarOnline x(')
    prefix='''// Generated v118; all old library/host entries remain unchanged.
#include "roof_o78_row_fused_prepare.cu"
#include "nv4_swar_generated.cuh"
namespace {
struct Nv4SwarOnline:RowFusedOnline {
  using RowFusedOnline::RowFusedOnline;
  void activation(bool) { RowFusedOnline::activation(true); }
  void weight(bool candidate) {
    if(!candidate || variant!=7) {RowFusedOnline::weight(true);return;}
    nv4_row_swar_probe::adangel_sm80_row_swar_metadata<Kind::Nv4,16>
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
__global__ void adangel_nv4_swar_exhaustive(uint4* out) {
  const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
  const unsigned lo=i&65535u;
  const unsigned hi=i&65536u?(lo^65535u):lo;
  const unsigned word=lo|(hi<<16);
  unsigned packed=0,square=0;
  #pragma unroll
  for(int j=0;j<8;++j) {
    const int q=vector_probe::nv4((word>>(4*j))&15u);
    packed|=(unsigned(q)&15u)<<(4*j);square+=unsigned(q*q);
  }
  const uint2 candidate=nv4_swar_probe::packed_word(word);
  out[i]=make_uint4(packed,candidate.x,square,candidate.y);
}
}
extern "C" int roof_nv4_swar_exhaustive(uint4* out,void* stream) {
  if(!out)return 1;
  adangel_nv4_swar_exhaustive<<<512,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(out);
  return int(cudaGetLastError());
}
'''
    return prefix+benchmark


def audit(directory):
    before=(BASELINE/'prepare.sass').read_text();after=(directory/'prepare.sass').read_text()
    old,new=entries(before),entries(after)
    comparison=compare(before,after,'|'.join('^'+re.escape(s)+'$' for s in old))
    find=lambda rows,name:next((s,e) for s,e in rows.items()
        if name in s and 'GroupedSourceKindE0E' in s)
    old_symbol,old_entry=find(old,'adangel_sm80_row_conversion_metadata')
    symbol,new_entry=find(new,'adangel_sm80_row_swar_metadata')
    resources={s:dict(registers=int(r),stack=int(stack),shared=int(shared),local=int(local))
        for s,r,stack,shared,local in re.findall(
        r' Function (\S+):\s+REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)',
        (directory/'resources.txt').read_text())}
    a,b=resources[old_symbol],resources[symbol];ops=new_entry['opcode_counts']
    local_count=sum(count for op,count in ops.items() if op.startswith(('LDL','STL')))
    barriers=sum(count for op,count in ops.items() if op.startswith('BAR.SYNC'))
    improvement=1-new_entry['instructions']/old_entry['instructions']
    # Static potential, not a latency claim. No register/local/sync tradeoff.
    gate=(comparison['passed'] and b['registers']<=a['registers'] and
        b['stack']==b['local']==local_count==0 and b['shared']==256 and barriers==1 and
        improvement>=.05 and ops.get('POPC',0)==10)
    return dict(scope='static_weight_conversion_gate_not_latency_or_MSE',old_controls=comparison,
        control=dict(symbol=old_symbol,**a,**old_entry),candidate=dict(symbol=symbol,**b,**new_entry),
        static_instruction_reduction_fraction=improvement,local_instructions=local_count,
        cta_barriers=barriers,worth_runtime_validation=gate)


def checked(directory):
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    receipt=json.loads((directory/'build.json').read_text())
    for name,digest in receipt['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('source drift: '+name)
    for name,digest in receipt['artifact_sha256'].items():
        if sha(directory/name)!=digest:raise ValueError('artifact drift: '+name)
    if (directory/'nv4_swar_generated.cuh').read_text()!=generated_header():raise ValueError('decoder drift')
    if (directory/'nv4_swar_prepare.cu').read_text()!=generated_host():raise ValueError('Event wrapper drift')
    if receipt['production_default_changed'] or receipt['GEMM_modified']:raise ValueError('scope drift')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh repository output required')
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    original=json.loads((BASELINE/'build.json').read_text())
    for name,digest in original['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('v73 source drift: '+name)
    if sha(BASELINE/'libo78_gpu_prepare.so')!=original['driver_sha256']:raise ValueError('v73 library drift')
    cuda=Path('/usr/local/cuda-12.8/bin')
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    if 'release 12.8' not in version:p.error('CUDA12.8 required')
    out.mkdir(parents=True)
    names=set(original['sources'])|{'csrc/sm80/nv4_swar_conversion.cuh','scripts/probe_nv4_swar_codegen.py',
        'scripts/analyze_o78_row_fused_codegen.py','scripts/compare_a100_codegen.py'}
    receipt=dict(scope='v118_NVFP4_weight_packed_Boolean_conversion_exact_norm',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(names)},nvcc=version,commands=[],GEMM_modified=False,
        production_default_changed=False,quantization='E2M1_to_Q4_F0_RNE_unchanged',
        metadata_and_Event_timing='verbatim_v73',no_small_performance_screen=True)
    (out/'nv4_swar_generated.cuh').write_text(generated_header())
    (out/'nv4_swar_prepare.cu').write_text(generated_host())
    lib=out/'libo78_gpu_prepare.so'
    def run(command,name):
        receipt['commands'].append(command)
        with (out/name).open('w') as log:
            subprocess.run(command,cwd=ROOT,env=dict(os.environ,TMPDIR=str(ROOT/'tmp')),
                stdout=log,stderr=subprocess.STDOUT,check=True)
    run([str(cuda/'nvcc'),'-O3','-std=c++17','-lineinfo','-arch=sm_80','-shared',
        '-Xcompiler=-fPIC','-Xptxas=-v','-I'+str(ROOT/'csrc/sm80'),
        str(out/'nv4_swar_prepare.cu'),'-lcuda','-o',str(lib)],'build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(lib)],'prepare.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(lib)],'resources.txt')
    receipt['audit']=audit(out);receipt['driver_sha256']=sha(lib)
    receipt['artifact_sha256']={f.name:sha(f) for f in out.iterdir() if f.is_file()}
    (out/'build.json').write_text(json.dumps(receipt,indent=2)+'\n')
    checked(out);print(json.dumps(receipt['audit'],indent=2),flush=True)


if __name__=='__main__':main()
