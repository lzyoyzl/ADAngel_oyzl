#!/usr/bin/env python3
"""v66 one exact FP6 conversion candidate, isolated from GEMM/default/5090.

Replace only the E2M3 -> Q6 decoder; packing, RNE, scales and vector16 layout
are unchanged. Builds both controls locally on the server from committed code.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from compare_a100_codegen import instructions

ROOT=Path(__file__).resolve().parents[1]
OLD='''__device__ __forceinline__ int nv6(unsigned c) {
  const unsigned e=(c&31)>>3,m=c&7;
  const int q=e>=2?int((8+m)<<(e-2)):int(rne(e?8+m:m,1));
  return c&32?-q:q;
}'''
NEW='''__device__ __forceinline__ int nv6(unsigned c) {
  // For e=0/1 the magnitude is RNE((c&31)/2). For e=2/3 it is
  // (8+mantissa) << (e&1). All intermediates are nonnegative and bounded.
  const unsigned x=c&31u;
  const unsigned small=(x>>1)+unsigned((x&3u)==3u);
  const unsigned large=(8u+(x&7u))<<((x>>3)&1u);
  unsigned magnitude;
  asm("{ .reg .pred p; setp.ne.u32 p,%1,0; selp.u32 %0,%2,%3,p; }"
      : "=r"(magnitude) : "r"(x&16u),"r"(large),"r"(small));
  const int sign=-int((c>>5)&1u);
  return (int(magnitude)^sign)-sign;
}'''


def transform(header):
    if header.count(OLD)!=1:raise ValueError('canonical nv6 decoder changed')
    return header.replace(OLD,NEW)


def audit_payloads(sasses,resources):
    all_words=[instructions(s,r'adangel_sm80_') for s in sasses]
    targets=[s for s in all_words[0] if 'vector_fixed_conversion' in s and 'E3ELi' in s]
    controls=[s for s in all_words[0] if s not in targets]
    same={s:all_words[0][s]==all_words[1].get(s) for s in controls}
    rows=[]
    for policy,(sass,resource) in enumerate(zip(sasses,resources)):
        for block in re.split(r'(?=Function\s*:\s*)',sass):
            if not block.startswith('Function'):continue
            symbol=block.splitlines()[0].split(':',1)[1].strip()
            if symbol not in targets:continue
            match=re.search(r'Function\s+(?:\:\s*)?'+re.escape(symbol)+r'\s*:\s*([^\n]+)',resource)
            fields={k:int(v) for k,v in re.findall(r'(REG|LOCAL|STACK):(\d+)',match[0] if match else '')}
            rows.append(dict(policy=policy,symbol=symbol,resources=fields,
                instructions=len(all_words[policy][symbol])//2,
                static_BRA=len(re.findall(r'\bBRA(?:\.|\s)',block)),
                local_memory_sass=bool(re.search(r'\b(?:LDL|STL)(?:\.|\s)',block))))
    passed=(len(targets)==2 and len(controls)==18 and all(same.values()) and len(rows)==4
        and set(all_words[0])==set(all_words[1]) and all(set(r['resources'])=={'REG','LOCAL','STACK'}
        and r['resources']['LOCAL']==0 and r['resources']['STACK']==0 and not r['local_memory_sass'] for r in rows))
    return dict(passed=passed,preserved_entries=same,targets=rows,
        scope='conversion SASS/resources only; not a GEMM audit or performance result')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh repository output required')
    cuda=Path('/usr/local/cuda-12.8/bin')
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    if 'release 12.8' not in version:raise RuntimeError('CUDA12.8 required')
    out.mkdir(parents=True)
    header=ROOT/'csrc/sm80/roof_vector_conversion_impl.cuh'
    wrapper=ROOT/'csrc/sm80/roof_vector_conversion_probe.cu'
    body=wrapper.read_text().replace('#include "roof_vector_conversion_impl.cuh"','#include "generated.cuh"')
    records=[];sasses=[];resources=[]
    for policy in (0,1):
        target=out/f'policy_{policy}';target.mkdir()
        (target/'generated.cuh').write_text(header.read_text() if policy==0 else transform(header.read_text()))
        (target/'probe.cu').write_text(body)
        lib=target/'libconversion.so'
        cmd=[str(cuda/'nvcc'),'-std=c++17','-O3','-lineinfo','-gencode=arch=compute_80,code=sm_80',
            '-Xcompiler=-fPIC','-shared','-Xptxas=-v','-Iinclude','-Icsrc/sm80',str(target/'probe.cu'),
            'csrc/sm80/roof_fused_conversion.cu','csrc/sm80/roof_integer_conversion.cu','-o',str(lib)]
        with (target/'build.log').open('w') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        for suffix,mode in (('sass','--dump-sass'),('resources.txt','--dump-resource-usage')):
            with (target/f'conversion.{suffix}').open('w') as f:
                subprocess.run([str(cuda/'cuobjdump'),mode,str(lib)],stdout=f,stderr=subprocess.STDOUT,check=True)
        sasses.append((target/'conversion.sass').read_text());resources.append((target/'conversion.resources.txt').read_text())
        records.append(dict(policy=policy,command=cmd,library=str(lib),
            sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),
            generated_sha256=hashlib.sha256((target/'generated.cuh').read_bytes()).hexdigest()))
    result=audit_payloads(sasses,resources)
    result.update(builds=records,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        nvcc=version,sources=[dict(file=str(f.relative_to(ROOT)),sha256=hashlib.sha256(f.read_bytes()).hexdigest())
                            for f in (Path(__file__),header,wrapper)])
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
