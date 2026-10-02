#!/usr/bin/env python3
"""Full-K synthetic semantic checks and exact old-path fallback, nondefault stream."""
import argparse
import json
from pathlib import Path
import subprocess
from o3_fullk_probe import Driver,ROOT
from roof_reduction_validation import reference_fp64
from roof_payload_validation import verify_grouped_payload
from benchmark_a100_o1 import command


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cubins',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT):
        p.error('fresh repository output required')
    import torch
    from adangel import _sm80 as native
    from adangel.quantization.arbitrary_bits import split_int8_to_packed_int4
    from adangel.trace.storage import sha256_file
    from validate_a100_split_grouped import pack_q4
    torch.set_num_threads(4);torch.cuda.init()
    assert torch.cuda.get_device_capability()==(8,0)
    codegen=json.loads((args.cubins/'codegen.json').read_text())
    audit=json.loads((args.cubins/'audit.json').read_text())
    assert audit['passed'] and codegen['native_int4_entries'] and codegen['host_stream_mapping_passed']
    for source in audit['sources']:
        assert sha256_file(Path(source['file']))==source['sha256']
    cubins={i:(args.cubins/f'fullk_integer_{i}.cubin').resolve() for i in (0,1)}
    for i,path in cubins.items(): assert sha256_file(path)==codegen['variants'][str(i)]['cubin_sha256']
    args.output.mkdir(parents=True)
    library=(args.output/'libroof_probe_driver.so').resolve()
    with (args.output/'driver_build.log').open('w') as log:
        subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC','-I/usr/local/cuda-12.8/include',
            str(ROOT/'csrc/sm80/roof_producer_warp_driver.cpp'),'-lcuda','-o',str(library)],
            stdout=log,stderr=subprocess.STDOUT,check=True)
    checks=[]; cache_checks=[]
    for m,n,k in ((64,128,4096),(128,256,4096),(64,128,256)):
        for pattern in ('random','zero','extrema','zero_scale','wide_safe','unsafe'):
            torch.manual_seed(20261002+k)
            act=torch.randint(-128,128,(m,k),device='cuda',dtype=torch.int8)
            weight=torch.randint(-8,8,(n,k),device='cuda',dtype=torch.int8)
            if pattern=='zero': act.zero_();weight.zero_()
            if pattern in ('extrema','wide_safe','unsafe'):
                act.fill_(-128);weight.fill_(-8)
                if pattern=='extrema':
                    act[::2,1::2]=127;weight[::2,1::2]=7
            g=k//128
            asc=torch.linspace(.001,.03,m,device='cuda')
            wsc=((torch.arange(n*g,device='cuda').reshape(n,g)*7)%5+116).byte()
            if pattern=='zero_scale': asc.zero_()
            if pattern in ('wide_safe','unsafe'):
                wsc.fill_(116);wsc[:,-1]=128 if pattern=='wide_safe' else 131
            values=(split_int8_to_packed_int4(act),asc,pack_q4(weight),wsc)
            semantic=reference_fp64('o3',values)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                best=native._benchmark_roof_candidate('o3',54,*values,0,1)
                verify_grouped_payload(best,54,values[0],values[2],values[3])
                ws=best['converted_weight_scale']
                driver=Driver(library,cubins,'o3',50688)
                try:
                    for policy in (0,1):
                        y,_=driver.run(policy,best,asc,ws,0,1)
                        expected=policy if k==4096 and pattern!='unsafe' else 0
                        assert driver.last_policy==expected
                        torch.testing.assert_close(y.double(),semantic,rtol=1e-3,atol=1e-3)
                        if expected==0:
                            assert torch.equal(y.view(torch.int32),best['output'].view(torch.int32))
                        checks.append(dict(shape=[m,n,k],pattern=pattern,policy=policy,executed_policy=expected,
                            finite_fp32=True,max_abs_error_fp64=(y.double()-semantic).abs().max().item(),
                            max_abs_difference_best=(y-best['output']).abs().max().item(),
                            resources=driver.resources[expected],guard=driver.guard_metadata))
                    if pattern=='wide_safe' and k==4096:
                        # Same tensor/version cache: safe -> unsafe -> invalid.
                        ws[-1].fill_(131)
                        altered=native._benchmark_roof_candidate('o3',54,*values[:3],ws.T.contiguous(),0,1)
                        y,_=driver.run(1,best,asc,ws,0,1)
                        assert driver.last_policy==0
                        assert torch.equal(y.view(torch.int32),altered['output'].view(torch.int32))
                        cache_checks.append(dict(shape=[m,n,k],mutation_forces_fallback=True))
                    ws.fill_(255)
                    try: driver.run(1,best,asc,ws,0,1)
                    except ValueError as e: assert 'code 255' in str(e)
                    else: raise AssertionError('invalid mutated scale must fail before launch')
                finally: driver.close()
            stream.synchronize()
    result=dict(passed=True,count=len(checks),checks=checks,cache_checks=cache_checks,
        git_commit=command('git','rev-parse','HEAD'),extension_sha256=sha256_file(Path(native.__file__)),
        cubins={str(i):sha256_file(path) for i,path in cubins.items()},
        scope='isolated compute-core correctness; no performance or E2E claim')
    (args.output/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('passed checks:',len(checks),flush=True)


if __name__=='__main__': main()
