"""Isolated full-K driver and exact conservative INT32 guard; not production API."""
import ctypes as ct
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
PARTIAL_BOUND = 128 * 128 * 8
INT32_MAX = (1 << 31) - 1


def guard_columns(columns):
    """Python integers avoid overflow while proving every signed prefix safe."""
    if not columns or any(not c for c in columns):
        raise ValueError('nonempty scale columns required')
    if any(not isinstance(v, int) or not 0 <= v < 255 for c in columns for v in c):
        raise ValueError('invalid UE8M0 code 255 or out-of-range code')
    anchors = [min(c) for c in columns]
    bounds = [PARTIAL_BOUND * sum(1 << (v-h) for v in c)
              for c, h in zip(columns, anchors)]
    normal = all(h >= 1 for h in anchors)
    full = all(len(c) == 32 for c in columns)
    return anchors, dict(safe=normal and full and max(bounds) <= INT32_MAX,
        max_abs_prefix_bound=max(bounds), columns=len(columns),
        normal_scales=normal, full_k4096=full,
        max_exponent_difference=max(max(c)-h for c,h in zip(columns,anchors)),
        proof='128*128*8*sum(2**(code-anchor)); bounds every product and prefix')


class Driver:
    def __init__(self, library, cubins, variant, smem):
        assert variant == 'o3'
        self.lib = ct.CDLL(str(library))
        self.lib.roof_probe_error.restype = ct.c_char_p
        self.lib.roof_probe_open.argtypes = [ct.c_char_p,ct.c_char_p,ct.c_uint,ct.POINTER(ct.c_void_p)]
        self.lib.roof_probe_close.argtypes = [ct.c_void_p]
        self.lib.roof_probe_resources.argtypes = [ct.c_void_p,ct.POINTER(ct.c_int)]
        self.lib.roof_probe_benchmark.argtypes = [ct.c_void_p]+[ct.c_uint64]*5+[ct.c_int]*5+[ct.c_void_p,ct.POINTER(ct.c_float)]
        self.handles = {}; self.resources = {}; self.guard_cache = {}
        self.guard_metadata = None; self.last_policy = None
        try:
            for policy,path in cubins.items():
                handle = ct.c_void_p()
                self.check(self.lib.roof_probe_open(str(path).encode(),b'adangel_roof_fullk_integer_o3',50688,ct.byref(handle)))
                self.handles[policy] = handle
                values = (ct.c_int*4)()
                self.check(self.lib.roof_probe_resources(handle,values))
                assert values[2] == 128
                self.resources[policy] = dict(registers_per_thread=values[0],local_size_bytes=values[1],
                    threads=values[2],active_blocks_per_sm=values[3],shared_memory_bytes=50688,
                    cta_tile=[64,128,128],pipeline_stages=3,group_fusion=32 if policy else 1,
                    accumulator_dtype='int32' if policy else 'float32')
        except Exception:
            self.close(); raise

    def check(self, code):
        if code: raise RuntimeError(self.lib.roof_probe_error().decode())

    def close(self):
        for handle in self.handles.values(): self.check(self.lib.roof_probe_close(handle))
        self.handles.clear()

    def run(self, policy, best, activation_scale, weight_scale, warmup, repeats):
        import torch
        a=best['packed_activation_g128_major']; w=best['packed_weight_g128_major']
        _,g,m,kbytes=a.shape; n=w.shape[1]; k=g*128
        assert policy in (0,1) and kbytes==64 and a.is_contiguous() and w.is_contiguous()
        assert a.dtype==w.dtype==weight_scale.dtype==torch.uint8
        assert activation_scale.dtype==torch.float32 and tuple(activation_scale.shape)==(m,)
        assert activation_scale.is_contiguous() and weight_scale.is_contiguous()
        assert tuple(weight_scale.shape)==(g,n) and tuple(w.shape)==(g,n,64)
        assert tuple(best['output'].shape)==(m,n)
        assert all(t.is_cuda and t.device==a.device for t in (w,activation_scale,weight_scale,best['output']))
        key=(weight_scale.data_ptr(),weight_scale._version,tuple(weight_scale.shape),str(weight_scale.device))
        if key not in self.guard_cache:
            start=time.perf_counter()
            columns=weight_scale.T.cpu().tolist()
            anchors,guard=guard_columns(columns)
            if not guard['normal_scales']:
                raise ValueError('isolated probe requires normal UE8M0 codes 1..254')
            meta=None
            if guard['safe']:
                anchor=torch.tensor(anchors,dtype=torch.uint8,device=a.device).reshape(1,n)
                meta=torch.cat((weight_scale,anchor),dim=0)
            # Metadata copy completion is deliberately included in preparation wall time.
            torch.cuda.current_stream().synchronize()
            guard.update(preparation_wall_ms=(time.perf_counter()-start)*1000,
                scope='host guard and metadata preparation excluded from compute-only; not E2E')
            self.guard_cache[key]=(weight_scale,meta,guard) # Strong ref prevents allocator address reuse.
        _,meta,self.guard_metadata=self.guard_cache[key]
        self.last_policy=policy if self.guard_metadata['safe'] else 0
        ws=meta if self.last_policy==1 else weight_scale
        y=torch.empty_like(best['output']); times=(ct.c_float*repeats)()
        self.check(self.lib.roof_probe_benchmark(self.handles[self.last_policy],a.data_ptr(),w.data_ptr(),
            activation_scale.data_ptr(),ws.data_ptr(),y.data_ptr(),m,n,k,warmup,repeats,
            torch.cuda.current_stream().cuda_stream,times))
        if y.dtype!=torch.float32 or not torch.isfinite(y).all():
            raise AssertionError('nonfinite or non-FP32 probe output')
        return y,list(times)
