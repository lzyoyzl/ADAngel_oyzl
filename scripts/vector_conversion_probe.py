"""Host contract for isolated v53 conversion library; no production dispatch."""
import ctypes as ct
from pathlib import Path


class VectorConversionProbe:
    def __init__(self, path):
        self.path = Path(path).resolve(strict=True)
        self.lib = ct.CDLL(str(self.path))
        self.lib.vector_conversion_error.restype = ct.c_char_p
        self.lib.vector_conversion_benchmark.argtypes = ([ct.c_int] * 2 + [ct.c_void_p] * 7
                                                        + [ct.c_int] * 5 + [ct.c_void_p, ct.POINTER(ct.c_float)])
        self.lib.vector_conversion_benchmark.restype = ct.c_int

    def __call__(self, source, implementation, warmup, repeats, inner):
        import torch
        from adangel.quantization import mixed_formats as mf
        mf.validate_source(source)
        if implementation not in (0, 1, 2) or warmup < 0 or repeats < 1 or inner < 1:
            raise ValueError('invalid probe policy/counts')
        x = source['payload']
        if not x.is_cuda or torch.cuda.get_device_capability(x.device) != (8, 0):
            raise ValueError('SM80 CUDA input required')
        fmt = source['format']; rows, k = source['shape']; groups = k // 128
        if rows*k > 2147483647 or groups > 65535:
            raise ValueError('index/grid range exceeded')
        weight = mf.FORMATS[fmt][1] == 4
        if fmt == 'mxfp8_e4m3_g128' and int(source['scale'].max()) > 252:
            raise ValueError('MXFP8 effective scale overflow')
        if 'tensor_scale' in source:
            bound = mf.decode_scalar(int(source['scale'].max()), 'e4m3') * float(source['tensor_scale'][0])
            if bound > torch.finfo(torch.float32).max:
                raise ValueError('effective scale overflow')
        with torch.cuda.device(x.device):
            packed = torch.empty((groups, rows, 64) if weight else (2, groups, rows, 64),
                                 dtype=torch.uint8, device=x.device)
            physical_scale = torch.empty((groups, rows), dtype=torch.float32, device=x.device)
            times = (ct.c_float * repeats)()
            ptr = lambda key: source[key].data_ptr() if key in source else None
            code = self.lib.vector_conversion_benchmark(list(mf.FORMATS).index(fmt), implementation,
                ptr('payload'), ptr('scale'), ptr('tensor_scale'), ptr('micro8'), ptr('micro4'),
                packed.data_ptr(), physical_scale.data_ptr(), rows, k, warmup, repeats, inner,
                torch.cuda.current_stream(x.device).cuda_stream, times)
            if code:
                raise RuntimeError(self.lib.vector_conversion_error().decode())
        return dict(packed=packed, scale=physical_scale.T, inner_repeats=inner,
                    timings_ms=list(times), conversion_kernel_count=1)
