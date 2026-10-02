"""v60 internal guard/preparation harness; no production dispatch change."""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
from o3_fullk_probe import Driver,ROOT
from probe_roof_factor_async_codegen import checked_cubins

MODES=('control_compute','cached_compute','prepare_decide_compute','prepare_gpu_only')


def fast_guard_flag(codes):
    """Independent Python mirror of the saturating GPU decision, not the oracle."""
    if len(codes)!=32 or any(not 0<=v<=255 for v in codes):
        raise ValueError('32 byte codes required')
    h=min(codes)
    return (int(sum(1<<min(v-h,14) for v in codes)>16383)
            | (2 if 255 in codes else 0) | (4 if 0 in codes else 0))


def build(directory,gemm_dir):
    directory=directory.resolve()
    if directory.exists() or not directory.is_relative_to(ROOT):
        raise ValueError('fresh repository build directory required')
    cubins=checked_cubins(gemm_dir)
    directory.mkdir(parents=True)
    cuda=Path('/usr/local/cuda-12.8/bin');source=ROOT/'csrc/sm80/roof_factor_prepare_probe.cu'
    driver=ROOT/'csrc/sm80/roof_gpu_factor_driver.cpp'
    prep=directory/'factor_prepare.cubin';lib=directory/'libgpu_factor_driver.so'
    commands=[([str(cuda/'nvcc'),'-O3','-std=c++17','-lineinfo','-arch=sm_80','-cubin',
                str(source),'-o',str(prep),'-Xptxas=-v'],'build.log'),
              ([str(cuda/'cuobjdump'),'--dump-sass',str(prep)],'prepare.sass'),
              ([str(cuda/'cuobjdump'),'--dump-resource-usage',str(prep)],'resources.txt'),
              (['g++','-O3','-std=c++17','-shared','-fPIC','-I/usr/local/cuda-12.8/include',
                str(driver),'-lcuda','-o',str(lib)],'driver_build.log')]
    for cmd,name in commands:
        with (directory/name).open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    result=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                sources={str(p.relative_to(ROOT)):digest(p) for p in
                         (source,driver,ROOT/'csrc/sm80/roof_producer_warp_driver.cpp')},
                gemm_cubins={str(i):digest(p) for i,p in cubins.items()},
                preparation_cubin_sha256=digest(prep),driver_sha256=digest(lib),commands=commands,
                scope='preparation_only_new_code; v59_control_and_candidate_GEMM_unchanged')
    (directory/'build.json').write_text(json.dumps(result,indent=2)+'\n')
    return lib,prep,cubins


class Pipeline(Driver):
    def __init__(self,library,prep,cubins):
        super().__init__(library,cubins,'o3',50688,factor_metadata=True)
        self.prep=ct.c_void_p()
        try:
            self.check(self.lib.roof_probe_open(str(prep).encode(),b'adangel_roof_factor_prepare',0,ct.byref(self.prep)))
        except Exception:
            self.close();raise
        self.lib.roof_gpu_factor_benchmark.argtypes=([ct.c_void_p]*3+[ct.c_uint64]*7+
            [ct.c_int]*7+[ct.c_void_p,ct.POINTER(ct.c_float),ct.POINTER(ct.c_int)])

    def close(self):
        if getattr(self,'prep',None):
            self.check(self.lib.roof_probe_close(self.prep));self.prep=None
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
        status=torch.empty(((n+127)//128,),device=a.device,dtype=torch.int32)
        times=(ct.c_float*repeats)();verdict=ct.c_int(-1)
        self.check(self.lib.roof_gpu_factor_benchmark(self.prep,self.handles[0],self.handles[1],
            a.data_ptr(),w.data_ptr(),asc.data_ptr(),ws.data_ptr(),meta.data_ptr(),status.data_ptr(),y.data_ptr(),
            m,n,4096,warmup,repeats,inner,mode,torch.cuda.current_stream().cuda_stream,times,ct.byref(verdict)))
        return dict(output=y if mode!=3 else None,raw_ms=list(times),metadata=meta,status=verdict.value,
                    executed_policy=1 if mode!=0 and verdict.value==0 else 0)
