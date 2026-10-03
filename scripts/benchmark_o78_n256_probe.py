#!/usr/bin/env python3
"""Four-sample cached N256 screen versus v78; no conversion/E2E/default claim."""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import benchmark_o78_eight_chain_probe as eight
import benchmark_o78_coefficient_probe as paired
from probe_o78_n256_codegen import ROOT, SYMBOL, generated_header, loops

base=eight.base


def checked(directory):
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    r=json.loads((directory/'codegen.json').read_text())
    for p,sha in r['sources'].items():
        if digest(ROOT/p)!=sha:raise ValueError('compiled source drift: '+p)
    if (digest(directory/'o78_n256.cubin')!=r['cubin_sha256'] or not r['control_comparison']['passed']
            or (directory/'o78_n256_generated.cuh').read_text()!=generated_header((ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())):
        raise ValueError('N256 binary/body/control drift')
    for p,sha in r['artifact_sha256'].items():
        if digest(directory/p)!=sha:raise ValueError('compiled artifact drift')
    integer=next(x for x in loops((directory/'o78_n256.sass').read_text(),SYMBOL,128) if x['kind']=='integer')
    counts=integer['opcodes']
    if (counts.get('LDSM.16.M88.4')!=24 or any(op.startswith(('LDL','STL')) for op in counts)
            or not r['entries'][SYMBOL]['native_u4_s4'] or not r['entries'][SYMBOL]['native_s4_s4']
            or r['entries'][SYMBOL]['int8_mma']):raise ValueError('candidate codegen gate not met')
    return r


class Driver(eight.Driver):
    def __init__(self, library, baseline, candidate):
        proof=checked(candidate)
        # Reuse unchanged v73 GPU preparation and v78/v67 numerical controls.
        super().__init__(library,baseline,Path('reports/o378_roof_v78_codegen'))
        self.handles[0]=self.handles[1]
        self.resources[0]=dict(self.resources[1])
        try:
            h=ct.c_void_p()
            self.check(self.lib.roof_probe_open(str((candidate/'o78_n256.cubin').resolve()).encode(),SYMBOL.encode(),51712,ct.byref(h)))
            self.handles[1]=h
            values=(ct.c_int*4)();self.check(self.lib.roof_probe_resources(h,values))
            if list(values)!=[255,0,128,2]:raise ValueError('unexpected N256 resources')
            self.resources[1]=dict(registers_per_thread=255,local_size_bytes=0,threads=128,active_blocks_per_sm=2,
                shared_memory_bytes=51712,cta_tile=[64,256,128],pipeline_stages=2,kernel_symbol=SYMBOL)
            path=candidate/'libn256_timing.so'
            command=['g++','-O3','-std=c++17','-shared','-fPIC','-I/usr/local/cuda-12.8/include',
                str(ROOT/'csrc/sm80/roof_o78_n256_driver.cpp'),'-L/usr/local/cuda-12.8/lib64/stubs','-lcuda','-o',str(path)]
            sources=('csrc/sm80/roof_o78_n256_driver.cpp','csrc/sm80/roof_producer_warp_driver.cpp')
            hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
            receipt=candidate/'timing_driver.json'
            if not path.exists():
                with (candidate/'driver_build.log').open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
                receipt.write_text(json.dumps(dict(command=command,sources=hashes,
                    binary_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2)+'\n')
            saved=json.loads(receipt.read_text())
            if saved['sources']!=hashes or saved['binary_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError('cached driver drift')
            self.timing=ct.CDLL(str(path.resolve()))
            self.timing.roof_probe_error.restype=ct.c_char_p
            self.timing.roof_o78_n256_benchmark.argtypes=[ct.c_void_p,ct.POINTER(ct.c_uint64)]+[ct.c_int]*6+[ct.c_void_p,ct.POINTER(ct.c_float)]
            self.timing.roof_o78_n256_benchmark.restype=ct.c_int
            self.codegen=dict(previous=self.codegen,n256=proof,timing_driver=saved)
        except Exception:self.close();raise

    def run(self,case,policy,mode,warmup,repeats,inner):
        import torch
        if mode!='compute_only' or policy not in self.handles:raise ValueError('cached N256 screen only')
        if np.any(case.oracle['status_flat']>1):raise ValueError('invalid source rejected')
        tile=256 if policy==1 else 128
        if case.n%tile:raise ValueError('N not aligned to candidate CTA')
        times=(ct.c_float*repeats)()
        result=self.timing.roof_o78_n256_benchmark(self.handles[policy],case.state_pointers,
            case.m,case.n,4096,tile,warmup,repeats,torch.cuda.current_stream().cuda_stream,times)
        if result:raise RuntimeError(self.timing.roof_probe_error().decode())
        values=list(times)
        return case.state['y'],dict(gemm=values,total=values.copy())


def validate(driver):
    import torch
    from adangel import _sm80 as native
    from adangel.quantization import mixed_formats as mf
    from roof_reduction_validation import reference_fp64
    checks=[]
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for pattern in ('random','zero','alternating','wide_scale'):
            m,n,k=128,512,4096
            torch.manual_seed(20261003)
            a=(torch.randn(m,k,device='cuda')*.4).half();w=(torch.randn(n,k,device='cuda')*.1).half()
            if pattern=='zero':a.zero_();w.zero_()
            if pattern=='alternating':a[:,::2]=-8;a[:,1::2]=7;w[:,::2]=-7;w[:,1::2]=6
            for variant,(wf,af) in mf.VARIANTS.items():
                ws,acs=mf.quantize_source(w,wf),mf.quantize_source(a,af)
                if pattern=='wide_scale':
                    if variant=='o7':acs['scale'].fill_(127);acs['scale'][:64,-1]=159
                    else:ws['scale'].fill_(1);ws['scale'][:128,-1]=192
                old=native._benchmark_mixed(variant,'compute_only',ws,acs,0,1,2,'64x128x256','group_major',59,5)
                case=paired.fused.Case(variant,ws,acs,old);guard=driver.prepare(case)
                semantic=reference_fp64(variant,(*old['converted_activation'],*old['converted_weight']))
                expected,_=driver.run(case,2,'compute_only',0,1,2);expected=expected.clone()
                for policy in (0,1):
                    y,_=driver.run(case,policy,'compute_only',0,1,2)
                    assert y.dtype==torch.float32 and bool(torch.isfinite(y).all())
                    torch.testing.assert_close(y.double(),semantic,rtol=1e-3,atol=1e-3)
                    torch.testing.assert_close(y,expected,rtol=1e-3,atol=1e-3)
                    exact=torch.equal(y.view(torch.int32),expected.view(torch.int32))
                    if not guard['fallback_ctas']:assert exact
                    checks.append(dict(pattern=pattern,variant=variant,policy=policy,shape=[m,n,k],
                        finite_fp32=True,semantic_tolerance_passed=True,bitwise_equal_v67=exact,nondefault_stream=True,**guard))
        stream.synchronize()
    return dict(passed=True,count=len(checks),checks=checks,scope='compute_only_small_MN_full_K_not_all_shapes_or_four_modes')


def contract(mode,inner):
    r=eight.timing_contract(mode,inner)
    r.update(comparison='v78_N128_vs_v83_N256_cached_same_native_driver',timing_scope='cached_GEMM_only',
        wider_tile_falls_back_if_either_N128_guard_rejects=True)
    return r


if __name__=='__main__':
    if '--full-modes' in sys.argv:raise SystemExit('No conversion/E2E claims in this cached screen')
    # Keep original strict bitwise/MSE checks and unfiltered paired statistics.
    # Mixed fallback may fail that strict performance gate; do not relax it.
    paired.validate=validate
    paired.main(driver_cls=Driver,default_gpu_build=Path('reports/o378_roof_v73_codegen'),
        labels=('v78_N128_same_v73_preparation','v83_N256_same_v73_preparation'),experiment='n256_reuse',
        banner='N256 REUSE',contract=contract,description=__doc__)
