#!/usr/bin/env python3
"""v135: one runtime reconsideration of frozen v115, NOT a new kernel.

The original static investment gate stays failed. Its 3 hot local reads and
5.26% instruction increase were a heuristic, not a measurement of performance.
The user accepts small spill subject to correctness/speed/safety. This single
full24 comparison resolves an unmeasured dependency-shortening candidate. No
adjacent tuning, no single-INT8 route, and no production default change.
"""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import benchmark_o3_eight_chain_probe as protocol
from benchmark_o3_grouped_cta import validate as original_validate
from benchmark_o78_grouped_cta import full_sample_args
from compare_a100_codegen import compare
from probe_o3_dp2a_codegen import CONTROL, SYMBOL, ROOT, dot_liveness, gate, pack_metadata_reference
from roof_full_pipeline_probe import MODES, Pipeline as FullPipeline, build as full_build, stage_contract

CUBIN_SHA='cf22985184630cb8c5011c6be06df0337e3934ab973574f0aa56423d4318d782'
PACK='adangel_roof_o3_dp2a_pack_metadata'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def reviewed(receipt):
    """Finite pre-runtime exception; do not mutate/rename the historical gate."""
    live=receipt['liveness'];ops=live['loop']['opcode_counts']
    requirements=dict(
        unchanged_frozen_cubin=receipt['cubin_sha256']==CUBIN_SHA,
        original_gate_still_failed=receipt['compile_gate']['passed'] is False,
        registers=live['allocated_gpr']==168,
        exactly_three_local_reads=ops.get('LDL')==3,
        no_local_stores=not any(v for k,v in ops.items() if k.startswith('STL')),
        fixed_old_workload=live['loop']['static_instructions']==340,
        native_two_int4=ops.get('IMMA.16864.S4.S4')==ops.get('IMMA.16864.U4.S4')==32,
        scalar_dot_not_int8_tensor=ops.get('IDP.2A.LO.S16.U8')==64 and not any(
            v for k,v in ops.items() if k.startswith('IMMA') and ('S8' in k or 'U8' in k)),
        unchanged_control=receipt['control_comparison']['passed'],
    )
    if not all(requirements.values()):raise ValueError('fixed pre-runtime review failed: '+str(requirements))
    return dict(checks=requirements,original_gate=receipt['compile_gate'],
        purpose='one full24 measurement of previously unexecuted v115; not a new optimization',
        limits='no resource scan; no default change; negative result closes runtime direction')


def checked(codegen):
    r=json.loads((codegen/'analysis.json').read_text())
    for name,digest in r['source_hashes'].items():
        if sha(ROOT/name)!=digest:raise ValueError('audited CUDA/source drift: '+name)
    for name,digest in r['artifact_sha256'].items():
        if sha(codegen/name)!=digest:raise ValueError('frozen artifact drift: '+name)
    old=ROOT/'reports/o378_roof_v89_o3_codegen'
    base=json.loads((old/'codegen.json').read_text())
    if sha(old/'o3_grouped_cta.cubin')!=r['baseline_cubin_sha256']:raise ValueError('v89 cubin drift')
    live=dot_liveness((codegen/'liveness.txt').read_text())
    if live!=r['liveness'] or gate(base['liveness'][CONTROL],live)!=r['compile_gate']:
        raise ValueError('original static audit replay mismatch')
    if compare((old/'o3_grouped_cta.sass').read_text(),(codegen/'o3_dp2a.sass').read_text(),'^'+CONTROL+'$')!=r['control_comparison']:
        raise ValueError('control encoding mismatch')
    reviewed(r)
    return r


def build(output,codegen):
    receipt=checked(codegen)
    built=full_build(output,ROOT/'reports/o378_roof_v59',ROOT/'reports/o378_roof_v61',
                     ROOT/'runs/o378_roof_v60_screen/build')
    driver=ROOT/'csrc/sm80/roof_o3_dp2a_driver.cpp';lib=output/'libdp2a_driver.so'
    command=['g++','-O3','-std=c++17','-shared','-fPIC','-I/usr/local/cuda-12.8/include',
             str(driver),'-lcuda','-o',str(lib)]
    with (output/'dp2a_driver_build.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    sources=[Path(__file__),driver,ROOT/'scripts/benchmark_o3_eight_chain_probe.py',
             ROOT/'scripts/benchmark_o3_grouped_cta.py',ROOT/'csrc/sm80/roof_full_pipeline_driver.cpp']
    (output/'dp2a_build.json').write_text(json.dumps(dict(
        codegen=receipt,review=reviewed(receipt),runtime_sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
        command=command,driver_sha256=sha(lib),no_GEMM_recompile=True,
        candidate_weight_conversion_includes_metadata_pack=True,production_default_changed=False),indent=2)+'\n')
    return (lib,*built[1:],ROOT/'reports/o378_roof_v89_o3_codegen/o3_grouped_cta.cubin',codegen/'o3_dp2a.cubin')


def timing_check(r,mode,inner,repeats,policy):
    # Both sides are guarded full-K kernels. The legacy helper assumed two
    # weight kernels; candidate packing is a real third operation, not free.
    assert policy==1 and r['kernel']['kernel_symbol'] in (CONTROL,SYMBOL)
    candidate=r['kernel']['kernel_symbol']==SYMBOL
    assert r['stage_timing_inner_repeats']==stage_contract(mode,inner)
    assert r['weight_cached']==(mode not in ('conversion_only','cold'))
    assert r['activation_prepared']==(mode=='compute_only')
    assert r['kernel']['weight_conversion_kernels']==2+int(candidate)
    assert r['kernel']['activation_conversion_kernels']==1
    assert set(r['timings_ms'])==set(stage_contract(mode,inner))
    assert all(len(v)==repeats and all(t>0 for t in v) for v in r['timings_ms'].values())
    if mode=='conversion_only':
        for total,w,a in zip(r['timings_ms']['total'],r['timings_ms']['weight_conversion'],r['timings_ms']['activation_conversion']):
            assert abs(total-w-a)<1e-5*max(total,1e-6)


class Pipeline(FullPipeline):
    def __init__(self,*built):
        self.pair={};self.pack=None
        super().__init__(*built[:-2])
        try:
            for policy,(cubin,symbol) in enumerate(zip(built[-2:],(CONTROL,SYMBOL))):
                handle=ct.c_void_p()
                self.check(self.lib.roof_probe_open(str(cubin.resolve()).encode(),symbol.encode(),50688,ct.byref(handle)))
                self.pair[policy]=handle
                values=(ct.c_int*4)();self.check(self.lib.roof_probe_resources(handle,values))
                if values[2]!=128 or values[3]<3:raise ValueError('actual three-CTA capacity required before launch')
                self.resources['v89' if policy==0 else 'frozen_v115']=dict(registers_per_thread=values[0],
                    local_size_bytes=values[1],threads=values[2],active_blocks_per_sm=values[3],
                    shared_memory_bytes=50688,kernel_symbol=symbol)
            self.pack=ct.c_void_p()
            # Opener verifies maximum block capacity; actual pack launch is128.
            self.check(self.lib.roof_four_open_conversion(str(built[-1].resolve()).encode(),PACK.encode(),ct.byref(self.pack)))
            self.lib.roof_dp2a_benchmark.argtypes=([ct.c_void_p]*6+[ct.c_uint64]*11+[ct.c_int]*8+
                [ct.c_void_p,ct.POINTER(ct.c_float),ct.POINTER(ct.c_int)])
        except Exception:self.close();raise

    def close(self):
        for handle in getattr(self,'pair',{}).values():self.check(self.lib.roof_probe_close(handle))
        self.pair={}
        if getattr(self,'pack',None):self.check(self.lib.roof_probe_close(self.pack));self.pack=None
        super().close()

    def run_four(self,policy,mode,a,asc,w,ws,warmup,repeats,inner=100):
        import numpy as np
        import torch
        assert policy in (0,1) and mode in MODES and warmup>=0 and repeats>=1 and inner>=1
        assert a.ndim==w.ndim==2
        m,k=a.shape;n=w.shape[0]
        assert k==4096 and m%64==0 and n%128==0
        assert a.dtype==torch.int8 and w.dtype==ws.dtype==torch.uint8 and asc.dtype==torch.float32
        assert tuple(w.shape)==(n,k//2) and tuple(ws.shape)==(n,32) and tuple(asc.shape)==(m,)
        assert all(t.is_cuda and t.device==a.device and t.is_contiguous() for t in (a,asc,w,ws))
        if bool(((ws==0)|(ws==255)).any()):raise ValueError('normal UE8M0 required')
        if not bool(torch.isfinite(asc).all()):raise ValueError('finite row scale required')
        pa=torch.empty((2,32,m,64),device=a.device,dtype=torch.uint8)
        pw=torch.empty((32,n,64),device=a.device,dtype=torch.uint8)
        gws=torch.empty((32,n),device=a.device,dtype=torch.uint8)
        meta=torch.empty((33,n),device=a.device,dtype=torch.int32)
        joined=torch.empty(65*n+n//128,device=a.device,dtype=torch.int32)
        status=torch.empty(n//128,device=a.device,dtype=torch.int32)
        y=torch.empty((m,n),device=a.device,dtype=torch.float32)
        times=(ct.c_float*(4*repeats))();verdict=ct.c_int(-1)
        self.check(self.lib.roof_dp2a_benchmark(self.prep,self.pair[0],self.pair[1],
            self.conversion['activation'],self.conversion['weight'],self.pack,
            a.data_ptr(),w.data_ptr(),asc.data_ptr(),ws.data_ptr(),pa.data_ptr(),pw.data_ptr(),gws.data_ptr(),
            meta.data_ptr(),joined.data_ptr(),status.data_ptr(),y.data_ptr(),m,n,k,warmup,repeats,inner,
            MODES.index(mode),policy,torch.cuda.current_stream().cuda_stream,times,ct.byref(verdict)))
        # Independent packing check and coverage are outside all measured events.
        if policy:
            expected=pack_metadata_reference(meta.cpu().numpy())
            if not np.array_equal(expected,joined.cpu().numpy()):raise AssertionError('GPU metadata pack mismatch')
            selected=joined[65*n:].bool() & (status==0)
            coverage=float(selected.float().mean())
        else:coverage=0.0
        if not bool(torch.isfinite(y).all()):raise AssertionError('nonfinite output')
        counts=stage_contract(mode,inner)
        raw={s:list(times[j*repeats:(j+1)*repeats]) for j,s in enumerate(
            ('weight_conversion','activation_conversion','gemm','total')) if s in counts}
        return dict(output=y,timings_ms=raw,stage_timing_inner_repeats=counts,status=verdict.value,
            packed_activation_g128_major=pa,packed_weight_g128_major=pw,converted_weight_scale=gws,
            weight_cached=mode not in ('cold','conversion_only'),activation_prepared=mode=='compute_only',
            total_timing='sum_of_batched_stage_samples' if mode=='conversion_only' else 'single_execution_cuda_event',
            kernel=dict(gemm_tune='v89_grouped_eight' if policy==0 else 'v115_dp2a_runtime_v135',
                kernel_symbol=CONTROL if policy==0 else SYMBOL,cta_tile=[64,128,128],pipeline_stages=3,
                cta_order_group_m=8,partial_registers=32,two_native_int4=True,int8_tensor_core=False,
                scalar_recomposition='PRMT+DP2A' if policy else 'high16+low',
                dp2a_cta_fraction=coverage,conversion_candidate=2,weight_conversion_kernels=2+policy,
                activation_conversion_kernels=1,guard_preparation_charged_to='weight_conversion',
                guard_cached=mode in ('compute_only','steady_state'),extra_pack_charged_to='weight_conversion' if policy else None,
                production_default_changed=False))


def validate(driver):
    import torch
    r=original_validate(driver)
    from roof_reduction_validation import reference_fp64
    extra=[]
    m,n=128,384
    a=((torch.arange(m*4096,device='cuda')*13)%256-128).to(torch.int8).reshape(m,4096)
    w=(torch.arange(n*2048,device='cuda')%256).byte().reshape(n,2048)
    asc=torch.linspace(0,.01,m,device='cuda')
    # CTA0: DP2A; CTA1: exact integer v89; CTA2: original floating fallback.
    ws=((torch.arange(n*32,device='cuda')%4)+116).byte().reshape(n,32)
    ws[128:256,-1]=120;ws[256:,-1]=131
    old=driver.run_four(0,'compute_only',a,asc,w,ws,0,1,2)
    pa=old['packed_activation_g128_major'].permute(0,2,1,3).contiguous().reshape(2,m,2048)
    pw=old['packed_weight_g128_major'].permute(1,0,2).contiguous().reshape(n,2048)
    ref=reference_fp64('o3',(pa,asc,pw,ws))
    for mode in MODES:
        new=driver.run_four(1,mode,a,asc,w,ws,0,2,2)
        assert torch.equal(old['output'].view(torch.int32),new['output'].view(torch.int32))
        torch.testing.assert_close(new['output'].double(),ref,rtol=1e-3,atol=1e-3)
        assert abs(new['kernel']['dp2a_cta_fraction']-1/3)<1e-7
        extra.append(dict(mode=mode,three_paths_in_one_launch=True,bitwise_best=True,
            semantic_tolerance_passed=True,all_int8_codes=True,all_e2m1_codes=True,pack_cpu_exact=True))
    r['dp2a_extra']=extra
    return r


def main():
    full_sample_args()
    for name,value in (('--warmup','1000'),('--repeats','200'),('--rounds','3'),('--inner','100'),
                       ('--codegen','reports/o378_roof_v115_o3_dp2a_codegen')):
        if name not in sys.argv:sys.argv.extend((name,value))
    protocol.build=build;protocol.Pipeline=Pipeline;protocol.validate=validate;protocol.timing_check=timing_check
    protocol.main()
    out=Path(sys.argv[sys.argv.index('--output')+1]);env=json.loads((out/'environment.json').read_text())
    env.update(scope='v135 full24 first runtime test of frozen v115, versus best O3 v89',
        control=CONTROL,candidate=SYMBOL,no_small_performance_screen=True,
        historical_static_gate_passed=False,runtime_review='small spill accepted, no frozen gate overwritten',
        extra_candidate_metadata_pack_charged_to='weight_conversion',single_int8_route=False)
    (out/'environment.json').write_text(json.dumps(env,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
