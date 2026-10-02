"""Internal device-select probe; immutable inputs throughout each timed batch."""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
from roof_gpu_factor_probe import Pipeline as OldPipeline, ROOT
from probe_roof_factor_async_codegen import checked_cubins

MODES=('control_compute','cached_v59_compute','device_cached_compute','prepare_device_compute')


def build(directory,gemm_dir,device_dir,prep_dir):
    directory=directory.resolve()
    if directory.exists() or not directory.is_relative_to(ROOT):
        raise ValueError('fresh repository build directory required')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    cubins=checked_cubins(gemm_dir)
    prep=prep_dir/'factor_prepare.cubin';pb=json.loads((prep_dir/'build.json').read_text())
    assert digest(prep)==pb['preparation_cubin_sha256']
    # Ensure old preparation source/ABI still matches the audited GPU binary.
    for p,h in pb['sources'].items():assert digest(ROOT/p)==h
    dev=device_dir/'device_factor.cubin';db=json.loads((device_dir/'codegen.json').read_text())
    assert digest(dev)==db['cubin_sha256'] and db['o78_sentinel']['passed']
    for p,h in db['sources'].items():assert digest(ROOT/p)==h
    from probe_roof_device_factor_codegen import wrapper,generated_header
    assert (device_dir/'device_factor.cu').read_text()==wrapper((ROOT/'csrc/sm80/roof_fullk_integer_probe.cu').read_text())
    assert (device_dir/'o3_factor_async_generated.cuh').read_text()==generated_header((ROOT/'csrc/sm80/o3_fullk_integer_probe.cuh').read_text())
    directory.mkdir(parents=True);lib=directory/'libdevice_factor_driver.so'
    src=ROOT/'csrc/sm80/roof_device_factor_driver.cpp'
    cmd=['g++','-O3','-std=c++17','-shared','-fPIC','-I/usr/local/cuda-12.8/include',str(src),'-lcuda','-o',str(lib)]
    with (directory/'driver_build.log').open('w') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    (directory/'build.json').write_text(json.dumps(dict(command=cmd,driver_sha256=digest(lib),
        sources={str(p.relative_to(ROOT)):digest(p) for p in (src,ROOT/'csrc/sm80/roof_gpu_factor_driver.cpp',
            ROOT/'csrc/sm80/roof_producer_warp_driver.cpp')},preparation_cubin_sha256=digest(prep),
        device_cubin_sha256=digest(dev),gemm_cubins={str(i):digest(p) for i,p in cubins.items()}),indent=2)+'\n')
    return lib,prep,cubins,dev


class Pipeline(OldPipeline):
    def __init__(self,library,prep,cubins,device):
        super().__init__(library,prep,cubins)
        self.device=ct.c_void_p()
        try:
            self.check(self.lib.roof_probe_open(str(device).encode(),b'adangel_roof_device_factor_o3',50688,ct.byref(self.device)))
            values=(ct.c_int*4)();self.check(self.lib.roof_probe_resources(self.device,values))
            self.resources['device']=dict(registers_per_thread=values[0],local_size_bytes=values[1],
                threads=values[2],active_blocks_per_sm=values[3],shared_memory_bytes=50688)
            assert values[2]==128
        except Exception:self.close();raise
        self.lib.roof_device_factor_benchmark.argtypes=([ct.c_void_p]*4+[ct.c_uint64]*7+
            [ct.c_int]*6+[ct.c_void_p,ct.POINTER(ct.c_float),ct.POINTER(ct.c_int)])

    def close(self):
        if getattr(self,'device',None):
            self.check(self.lib.roof_probe_close(self.device));self.device=None
        super().close()

    def run_mode(self,mode,best,asc,ws,warmup,repeats,inner=100):
        import torch
        mode=MODES.index(mode)
        a=best['packed_activation_g128_major'];w=best['packed_weight_g128_major']
        assert a.ndim==4 and a.shape[0]==2 and a.shape[1]==32 and a.shape[3]==64
        m=a.shape[2];n=w.shape[1]
        assert a.dtype==w.dtype==ws.dtype==torch.uint8 and asc.dtype==torch.float32
        assert tuple(w.shape)==(32,n,64) and tuple(ws.shape)==(32,n) and tuple(asc.shape)==(m,)
        assert all(t.is_cuda and t.device==a.device and t.is_contiguous() for t in (a,w,ws,asc))
        y=torch.empty((m,n),device=a.device,dtype=torch.float32)
        meta=torch.empty((33,n),device=a.device,dtype=torch.int32)
        status=torch.empty((n//128,),device=a.device,dtype=torch.int32)
        times=(ct.c_float*repeats)();verdict=ct.c_int(-1)
        self.check(self.lib.roof_device_factor_benchmark(self.prep,self.handles[0],self.handles[1],self.device,
            a.data_ptr(),w.data_ptr(),asc.data_ptr(),ws.data_ptr(),meta.data_ptr(),status.data_ptr(),y.data_ptr(),
            m,n,4096,warmup,repeats,mode,torch.cuda.current_stream().cuda_stream,times,ct.byref(verdict)))
        return dict(output=y,raw_ms=list(times),metadata=meta,status=verdict.value,block_status=status,
            executed_policy='device_per_N128' if mode>=2 else ('cached' if mode==1 else 'control'))
