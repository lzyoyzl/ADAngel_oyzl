#!/usr/bin/env python3
"""v69 fused vector conversion/group norms; reuse v68 paired timing and proof."""
import ctypes as ct
import numpy as np
import benchmark_o78_fullk_gpu_prepare as base

OriginalCase, OriginalDriver = base.Case, base.Driver
original_timing_contract = base.timing_contract


class Case(OriginalCase):
    def __init__(self, *args, **kwargs):
        import torch
        from adangel.quantization import mixed_formats as mf
        super().__init__(*args, **kwargs)
        self.group_squares_reference = {}
        for name, src, rows in (("asq", self.asrc, self.m), ("wsq", self.wsrc, self.n)):
            q, _ = mf.to_fixed_reference(src)
            self.group_squares_reference[name] = q.reshape(rows,32,128).long().square().sum(-1).cpu().numpy().astype(np.uint32)
            self.state[name] = torch.empty((rows,32),dtype=torch.uint32,device=q.device)
        self.state_pointers = (ct.c_uint64 * 18)(*(self.state[key].data_ptr() for key in (*base.STATE_NAMES,"asq","wsq")))


class Driver(OriginalDriver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for old, new in (("roof_o78_gpu_prepare","roof_o78_fused_prepare"),
                         ("roof_o78_gpu_benchmark","roof_o78_fused_benchmark")):
            function = getattr(self.lib,new)
            function.argtypes = getattr(self.lib,old).argtypes
            function.restype = ct.c_int
            setattr(self.lib,old,function)

    def prepare(self, case):
        result = super().prepare(case)
        for name, reference in case.group_squares_reference.items():
            assert np.array_equal(case.state[name].cpu().numpy(),reference),name
        # Implementation identity belongs to timing_contract(); validation
        # combines both dictionaries as keyword arguments, so keep keys disjoint.
        result.update(group_squares_exact=True)
        return result


def timing_contract(mode, inner):
    result = original_timing_contract(mode,inner)
    result.update(preparation_implementation="fused_conversion_group_squares_then_metadata",
                  square_scratch_bytes_for_4096=1048576, payload_reread_for_metadata=False)
    return result


if __name__ == '__main__':
    base.Case, base.Driver, base.timing_contract = Case, Driver, timing_contract
    base.main()
