"""v62 isolated four-mode harness. Exact existing conversion2 plus v61/old54."""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import re
import subprocess
from roof_device_factor_probe import Pipeline as DevicePipeline,build as device_build,ROOT

MODES=('conversion_only','compute_only','cold','steady_state')


def stage_contract(mode,inner):
    if mode not in MODES or inner<1:raise ValueError('invalid timing configuration')
    result={}
    if mode in ('conversion_only','cold'):result['weight_conversion']=inner
    if mode!='compute_only':result['activation_conversion']=inner
    if mode!='conversion_only':result['gemm']=1
    result['total']=inner if mode=='conversion_only' else 1
    return result


def build(directory,gemm_dir,device_dir,prep_dir):
    # v61 build performs source/hash checks for unchanged guard and GEMM cubins.
    _,prep,cubins,dev=device_build(directory,gemm_dir,device_dir,prep_dir)
    cuda=Path('/usr/local/cuda-12.8/bin');source=ROOT/'csrc/sm80/roof_o3_conversion.cu'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    if 'release 12.8' not in version:raise ValueError('CUDA12.8 required')
    conv=directory/'conversion.cubin';lib=directory/'libfull_pipeline_driver.so'
    driver=ROOT/'csrc/sm80/roof_full_pipeline_driver.cpp'
    commands=[([str(cuda/'nvcc'),'-O3','-std=c++17','-lineinfo','-arch=sm_80','-cubin',str(source),
                '-o',str(conv),'-Xptxas=-v'],'conversion_build.log'),
        ([str(cuda/'cuobjdump'),'--dump-sass',str(conv)],'conversion.sass'),
        ([str(cuda/'cuobjdump'),'--dump-resource-usage',str(conv)],'conversion_resources.txt'),
        (['g++','-O3','-std=c++17','-shared','-fPIC','-I/usr/local/cuda-12.8/include',str(driver),'-lcuda','-o',str(lib)],'full_driver_build.log')]
    for cmd,name in commands:
        with (directory/name).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    symbols={}
    for symbol in re.findall(r'Function\s*:\s*(\S+)',(directory/'conversion.sass').read_text()):
        match=re.search(r'adangel_sm80_o3_tiled_conversionILb([01])ELb1EE',symbol)
        if match:
            kind='activation' if match[1]=='1' else 'weight'
            if kind in symbols:raise ValueError('duplicate vector conversion')
            symbols[kind]=symbol
    if set(symbols)!={'activation','weight'}:raise ValueError('vector conversion symbols not found')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    old=json.loads((directory/'build.json').read_text())
    result=dict(core=old,nvcc=version,commands=commands,conversion_symbols=symbols,
        conversion_sha256=digest(conv),driver_sha256=digest(lib),
        sources={str(p.relative_to(ROOT)):digest(p) for p in
            (source,ROOT/'csrc/sm80/roof_o3_conversion_api.h',driver)})
    (directory/'four_build.json').write_text(json.dumps(result,indent=2)+'\n')
    return lib,prep,cubins,dev,conv,symbols


class Pipeline(DevicePipeline):
    def __init__(self,library,prep,cubins,device,conversion,symbols):
        super().__init__(library,prep,cubins,device)
        self.conversion={}
        self.lib.roof_four_open_conversion.argtypes=[ct.c_char_p,ct.c_char_p,ct.POINTER(ct.c_void_p)]
        try:
            for kind,symbol in symbols.items():
                handle=ct.c_void_p()
                self.check(self.lib.roof_four_open_conversion(str(conversion).encode(),symbol.encode(),ct.byref(handle)))
                self.conversion[kind]=handle
        except Exception:self.close();raise
        self.lib.roof_four_benchmark.argtypes=([ct.c_void_p]*5+[ct.c_uint64]*10+[ct.c_int]*8+
            [ct.c_void_p,ct.POINTER(ct.c_float),ct.POINTER(ct.c_int)])

    def close(self):
        for h in getattr(self,'conversion',{}).values():self.check(self.lib.roof_probe_close(h))
        self.conversion={};super().close()

    def run_four(self,policy,mode,a,asc,w,ws,warmup,repeats,inner=100):
        import torch
        assert policy in (0,1) and mode in MODES and warmup>=0 and repeats>=1 and inner>=1
        assert a.ndim==2 and w.ndim==2
        m,k=a.shape;n=w.shape[0]
        assert k==4096 and m%64==0 and n%128==0
        assert a.dtype==torch.int8 and w.dtype==ws.dtype==torch.uint8 and asc.dtype==torch.float32
        assert tuple(w.shape)==(n,k//2) and tuple(ws.shape)==(n,32) and tuple(asc.shape)==(m,)
        assert all(t.is_cuda and t.device==a.device and t.is_contiguous() for t in (a,w,ws,asc))
        # Same normal-scale preflight for both implementations, outside all timed regions.
        if bool(((ws==0)|(ws==255)).any()):raise ValueError('normal UE8M0 codes 1..254 required')
        if not bool(torch.isfinite(asc).all()):raise ValueError('finite row scales required')
        pa=torch.empty((2,32,m,64),device=a.device,dtype=torch.uint8)
        pw=torch.empty((32,n,64),device=a.device,dtype=torch.uint8)
        gws=torch.empty((32,n),device=a.device,dtype=torch.uint8)
        y=torch.empty((m,n),device=a.device,dtype=torch.float32)
        meta=torch.empty((33,n),device=a.device,dtype=torch.int32)
        status=torch.empty((n//128,),device=a.device,dtype=torch.int32)
        times=(ct.c_float*(4*repeats))();verdict=ct.c_int(-1)
        self.check(self.lib.roof_four_benchmark(self.prep,self.handles[0],self.device,
            self.conversion['activation'],self.conversion['weight'],a.data_ptr(),w.data_ptr(),asc.data_ptr(),
            ws.data_ptr(),pa.data_ptr(),pw.data_ptr(),gws.data_ptr(),meta.data_ptr(),status.data_ptr(),y.data_ptr(),
            m,n,k,warmup,repeats,inner,MODES.index(mode),policy,torch.cuda.current_stream().cuda_stream,times,ct.byref(verdict)))
        counts=stage_contract(mode,inner)
        stages=('weight_conversion','activation_conversion','gemm','total')
        raw={s:list(times[j*repeats:(j+1)*repeats]) for j,s in enumerate(stages) if s in counts}
        if not bool(torch.isfinite(y).all()):raise AssertionError('nonfinite output')
        return dict(output=y,timings_ms=raw,stage_timing_inner_repeats=counts,status=verdict.value,
            packed_activation_g128_major=pa,packed_weight_g128_major=pw,converted_weight_scale=gws,
            weight_cached=mode not in ('cold','conversion_only'),activation_prepared=mode=='compute_only',
            total_timing='sum_of_batched_stage_samples' if mode=='conversion_only' else 'single_execution_cuda_event',
            kernel=dict(gemm_tune=54 if policy==0 else 'v61_device_fullk',conversion_candidate=2,
                weight_conversion_kernels=1+policy,activation_conversion_kernels=1,
                guard_preparation_charged_to='weight_conversion' if policy else None,
                guard_cached=bool(policy and mode in ('compute_only','steady_state')),
                production_default_changed=False))
