#!/usr/bin/env python3
"""Build and statically inspect isolated v73 row-fused preparation library."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh repository output required')
    cuda=Path('/usr/local/cuda-12.8')
    version=subprocess.check_output([str(cuda/'bin/nvcc'),'--version'],text=True)
    if 'release 12.8' not in version:p.error('CUDA 12.8 required')
    out.mkdir(parents=True)
    names=('roof_o78_row_fused_prepare.cu','roof_row_fused_conversion.cuh','roof_o78_fused_prepare.cu','roof_vector_norm_conversion.cuh','roof_o78_gpu_prepare.cu',
           'roof_vector_conversion_impl.cuh','roof_fused_conversion_api.h','roof_producer_warp_driver.cpp')
    sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
    receipt=dict(scope='independent_row_fused_conversion_factor_metadata_no_GEMM_or_default_change',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={str(Path('csrc/sm80')/n):sha(ROOT/'csrc/sm80'/n) for n in names},nvcc=version,commands=[])
    lib=out/'libo78_gpu_prepare.so'
    cmd=[str(cuda/'bin/nvcc'),'-O3','-std=c++17','-lineinfo','-arch=sm_80','-shared','-Xcompiler=-fPIC',
         '-Xptxas=-v',str(ROOT/'csrc/sm80/roof_o78_row_fused_prepare.cu'),'-lcuda','-o',str(lib)]
    receipt['commands'].append(cmd)
    with (out/'build.log').open('w') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    for flag,name in (('--dump-sass','prepare.sass'),('--dump-resource-usage','resources.txt')):
        with (out/name).open('w') as f:subprocess.run([str(cuda/'bin/cuobjdump'),flag,str(lib)],stdout=f,check=True)
    symbols=(out/'prepare.sass').read_text()
    assert 'adangel_sm80_vector_fixed_conversion_with_norm' in symbols
    assert 'adangel_o78_factor_from_squares' in symbols
    assert 'adangel_sm80_row_conversion_metadata' in symbols
    receipt['driver_sha256']=sha(lib)
    (out/'build.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(lib)

if __name__=='__main__':main()
