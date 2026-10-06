#!/usr/bin/env python3
"""v95 diagnostic: query both rejected cubins without launching their kernels."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

from benchmark_o78_fullk_gpu_prepare import checked_gpu_build
from probe_slot_pipeline_codegen import ROOT, CONFIG, checked


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT):
        p.error('fresh repository output required')
    import torch
    if torch.cuda.get_device_capability()!=(8,0):raise RuntimeError('A100 required')
    anchor=torch.empty(1,device='cuda') # establish the actual current driver context
    library,_=checked_gpu_build(ROOT/'reports/o378_roof_v73_codegen')
    lib=ct.CDLL(str(library))
    lib.roof_probe_open.argtypes=[ct.c_char_p,ct.c_char_p,ct.c_uint,ct.POINTER(ct.c_void_p)]
    lib.roof_probe_resources.argtypes=[ct.c_void_p,ct.POINTER(ct.c_int)]
    lib.roof_probe_close.argtypes=[ct.c_void_p]
    rows={}
    for kind,cfg in CONFIG.items():
        directory=ROOT/f'reports/o378_roof_v95_{kind}_codegen';receipt=checked(directory,kind)
        handle=ct.c_void_p()
        result=lib.roof_probe_open(str(directory/(cfg['stem']+'.cubin')).encode(),
                                  cfg['symbol'].encode(),cfg['shared'],ct.byref(handle))
        if result:raise RuntimeError(f'CUfunction load failed: {result}')
        try:
            values=(ct.c_int*4)();result=lib.roof_probe_resources(handle,values)
            if result:raise RuntimeError(f'resource query failed: {result}')
            rows[kind]=dict(registers=values[0],local_bytes=values[1],threads=values[2],
                active_blocks_per_sm=values[3],shared_bytes=cfg['shared'],symbol=cfg['symbol'],
                cubin_sha256=receipt['cubin_sha256'],compile_gate=receipt['worth_runtime_validation'])
        finally:
            if lib.roof_probe_close(handle):raise RuntimeError('CUfunction unload failed')
    r=dict(kind='CUDA_driver_resource_query_no_candidate_kernel_launch',kernels=rows,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),gpu=torch.cuda.get_device_name())
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))


if __name__=='__main__':main()
