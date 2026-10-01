#!/usr/bin/env python3
"""Finite synthetic core checks for v42; not quantizer or performance validation."""
import argparse
import json
from pathlib import Path
import subprocess

from benchmark_roof_prefetch_probe import Driver, ROOT
from benchmark_a100_roof_candidates import group_major_scales
from roof_reduction_validation import reference_fp64
from roof_payload_validation import verify_grouped_payload
from benchmark_a100_o1 import command


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cubins',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT):
        p.error('fresh output inside repository required')
    import torch
    from adangel import _sm80 as native
    from adangel.quantization.arbitrary_bits import split_int8_to_packed_int4
    from adangel.trace.storage import sha256_file
    from validate_a100_split_grouped import pack_q4
    torch.set_num_threads(4);torch.cuda.init()
    assert torch.cuda.get_device_capability()==(8,0)
    a.output.mkdir(parents=True)
    library=(a.output/'libroof_probe_driver.so').resolve()
    with (a.output/'driver_build.log').open('w') as log:
        subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC','-I/usr/local/cuda-12.8/include',
            str(ROOT/'csrc/sm80/roof_producer_warp_driver.cpp'),'-lcuda','-o',str(library)],
            stdout=log,stderr=subprocess.STDOUT,check=True)
    cubins={i:(a.cubins/f'prefetch_{i}.cubin').resolve() for i in (0,1,2)}
    codegen=json.loads((a.cubins/'codegen.json').read_text())
    audit=json.loads((a.cubins/'audit.json').read_text())
    assert audit['passed'] and all(audit['control_encoded_sass_matches_best'].values())
    for source in audit['sources']:
        assert sha256_file(Path(source['file']))==source['sha256']
    for i,path in cubins.items():
        assert sha256_file(path)==codegen['variants'][str(i)]['cubin_sha256']
    checks=[]
    for variant in ('o3','o7','o8'):
        for m,n,k in ((64,128,128),(64,128,256),(64,128,384),(128,256,640),(64,128,4096)):
            for pattern in ('random','zero','extrema','zero_scale'):
                torch.manual_seed(20261001+k)
                low,high=(-32,32) if variant=='o8' else (-128,128)
                act=torch.randint(low,high,(m,k),device='cuda',dtype=torch.int8)
                weight=torch.randint(-8,8,(n,k),device='cuda',dtype=torch.int8)
                if pattern=='zero': act.zero_();weight.zero_()
                if pattern=='extrema':
                    act[:,::2]=low;act[:,1::2]=high-1
                    weight[:,::2]=-8;weight[:,1::2]=7
                g=k//128
                if variant=='o3':
                    asc=torch.linspace(.001,.03,m,device='cuda')
                    wsc=((torch.arange(n*g,device='cuda').reshape(n,g)*7)%13+116).byte()
                    if pattern=='zero_scale': asc.zero_()
                    tune=54
                else:
                    def scales(rows,factor):
                        row=torch.arange(rows,device='cuda')[:,None]
                        group=torch.arange(g,device='cuda')[None,:]
                        x=(1+(row*factor+group*29)%113/128)*torch.exp2(((row+3*group)%7-10).float())
                        return group_major_scales(x)
                    asc,wsc=scales(m,13),scales(n,17)
                    if pattern=='zero_scale': asc[:,::2]=0;wsc[:,1::2]=0
                    tune=59
                values=(split_int8_to_packed_int4(act),asc,pack_q4(weight),wsc)
                semantic=reference_fp64(variant,values)
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    best=native._benchmark_roof_candidate(variant,tune,*values,0,1)
                    verify_grouped_payload(best,tune,values[0],values[2],values[3])
                    driver=Driver(library,cubins,variant,best['kernel']['shared_memory_bytes'])
                    try:
                        for policy in (0,1,2):
                            ws=best['converted_weight_scale'] if variant=='o3' else wsc
                            y,_=driver.run(policy,best,asc,ws,0,1)
                            torch.testing.assert_close(y.double(),semantic,rtol=1e-3,atol=1e-3)
                            checks.append(dict(variant=variant,shape=[m,n,k],pattern=pattern,policy=policy,
                                bitwise_equal_best=True,finite_fp32=True,probe_resources=driver.resources[policy],
                                max_abs_error_fp64=(y.double()-semantic).abs().max().item()))
                    finally: driver.close()
                stream.synchronize()
        print(variant,'checks complete',flush=True)
    result=dict(passed=True,count=len(checks),checks=checks,
        git_commit=command('git','rev-parse','HEAD'),extension_sha256=sha256_file(Path(native.__file__)),
        cubins={str(i):sha256_file(path) for i,path in cubins.items()},
        scope='finite prepared-core checks, nondefault stream, not conversion or full experimental acceptance')
    (a.output/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('checks:',len(checks),flush=True)


if __name__=='__main__': main()
