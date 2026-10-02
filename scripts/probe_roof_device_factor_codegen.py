#!/usr/bin/env python3
"""v61 compile gate: CTA-uniform device selection removes per-launch host decision.

Not a production API. A caller must enqueue v60 preparation on the same stream
before this kernel, and reject invalid status before exposing any output.
Unsafe INT32 columns use the exact old FP32 path at N128 CTA granularity.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from probe_roof_factor_async_codegen import generated_header
from probe_roof_fullk_integer_codegen import static_entries
from compare_a100_codegen import compare

ROOT=Path(__file__).resolve().parents[1]
PATTERN=r'^adangel_roof_(?:device_factor_o3|fullk_integer_o78)$'
SYMBOLS={'adangel_roof_device_factor_o3','adangel_roof_fullk_integer_o78'}
OLD='''void adangel_roof_fullk_integer_o3(const uint8_t* a,const uint8_t* w,const float* as,
    const uint8_t* ws,float* y,int m,int n,int k) {
#if ADANGEL_FULLK_INTEGER==0
  O3::o3_body<64,128,128,false,false,false,2,false,true,false,true,true,true,false,
      false,true,6,false,false,false,false,3>(a,w,as,ws,y,m,n,k);
#else
  o3_fullk_integer_experiment::body(a,w,as,ws,y,m,n,k);
#endif
}'''
NEW='''void adangel_roof_device_factor_o3(const uint8_t* a,const uint8_t* w,const float* as,
    const uint8_t* ws,const int32_t* meta,const uint32_t* status,float* y,int m,int n,int k) {
  // v60 stores one flag per N128 tile. The load/branch is CTA-uniform.
  // Invalid input never reaches either arithmetic path. Host rejects it after
  // stream completion, before exposing y (invalid tiles deliberately do not write).
  const uint32_t flag=status[blockIdx.x];
  if(flag&6u) return;
  if(flag&1u) {
    O3::o3_body<64,128,128,false,false,false,2,false,true,false,true,true,true,false,
        false,true,6,false,false,false,false,3>(a,w,as,ws,y,m,n,k);
  } else {
    o3_fullk_integer_experiment::body(a,w,as,reinterpret_cast<const uint8_t*>(meta),y,m,n,k);
  }
}'''


def wrapper(source):
    inc='#include "o3_fullk_integer_probe.cuh"'
    if source.count(OLD)!=1 or source.count(inc)!=1:
        raise ValueError('unexpected canonical wrapper')
    return source.replace(OLD,NEW).replace(inc,'#include "o3_factor_async_generated.cuh"')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh repository output required')
    cuda=Path('/usr/local/cuda-12.8/bin');cutlass=ROOT/'third_party/cutlass-src'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    commit=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'release 12.8' not in version or commit!='db1c288993354c88e551c40c19a8fb93a774a241':
        p.error('pinned toolkit/CUTLASS required')
    out.mkdir(parents=True)
    header=ROOT/'csrc/sm80/o3_fullk_integer_probe.cuh';src=ROOT/'csrc/sm80/roof_fullk_integer_probe.cu'
    (out/'o3_factor_async_generated.cuh').write_text(generated_header(header.read_text()))
    (out/'device_factor.cu').write_text(wrapper(src.read_text()))
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    result=dict(scope='compile_only_no_runtime_performance_or_MSE_claim',nvcc=version,cutlass_commit=commit,
        sources={str(p.relative_to(ROOT)):digest(p) for p in (header,src)},commands=[],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    def run(cmd,name):
        result['commands'].append(cmd)
        with (out/name).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    base=[str(cuda/'nvcc'),'-O3','-std=c++17','--expt-relaxed-constexpr','-lineinfo','-arch=sm_80',
          '-DADANGEL_FULLK_INTEGER=1','-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),str(out/'device_factor.cu')]
    cubin=out/'device_factor.cubin'
    run(base+['-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(base+['-ptx','-o',str(out/'device_factor.ptx')],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],'device_factor.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    sass=(out/'device_factor.sass').read_text();entries=static_entries(sass,PATTERN,SYMBOLS)
    assert all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma'] and
               e['all_copies_bypass_l1'] for e in entries.values())
    blocks=re.split(r'(?=\.visible \.entry )',(out/'device_factor.ptx').read_text())
    block=next(b for b in blocks if b.startswith('.visible .entry adangel_roof_device_factor_o3('))
    assert all(x in block for x in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32'))
    sentinel=compare((ROOT/'docs/evidence/a100_o378_roof_v59/reports/o378_roof_v59/factor_async_1.sass').read_text(),
                     sass,r'^adangel_roof_fullk_integer_o78$')
    assert sentinel['passed']
    result.update(cubin_sha256=digest(cubin),entries=entries,o78_sentinel=sentinel)
    (out/'codegen.json').write_text(json.dumps(result,indent=2)+'\n')
    print('same-entry native INT4 audit passed; resource/runtime gates still required')


if __name__=='__main__':main()
