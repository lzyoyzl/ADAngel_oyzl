#!/usr/bin/env python3
"""v85 lane-vector LDS vs v78 LDSM, with exact online repacking charged."""
import ctypes as ct
import hashlib
import json
from pathlib import Path

import numpy as np
import benchmark_o78_eight_chain_probe as eight
from benchmark_o78_coefficient_probe import main as paired_main
from probe_o78_register_layout_codegen import ROOT, SYMBOL, generated_header, generated_driver


def register_order_indices(weight):
    """Independent SM80 atom oracle, checked against CuTe by GPU packing/GEMM.

    A packs a16x128 tile. W packs a32x128 tile with two N warps; adjacent
    fragments within one warp are 16 columns apart, not adjacent N8 atoms.
    """
    result=[]
    for warp in range(2 if weight else 1):
        for half in range(2):
            for lane in range(32):
                for value in range(32):
                    row=warp*8+lane//4+(16*(value//16) if weight else 8*((value//8)%2))
                    k=half*64+(lane%4)*8+value%8+32*((value//8)%2 if weight else value//16)
                    result.append(row*128+k)
    return np.asarray(result,dtype=np.int64)


def packed_reference(natural,weight):
    original=np.asarray(natural,dtype=np.uint8)
    rows=original.shape[-2]
    tile=32 if weight else 16
    if rows%tile or original.shape[-1]!=64:raise ValueError('natural G128 packed rows required')
    planes=1 if weight else 2
    values=np.stack((original&15,original>>4),axis=-1).reshape(planes,32,rows//tile,tile*128)
    ordered=values[...,register_order_indices(weight)]
    return (ordered[...,::2]|(ordered[...,1::2]<<4)).reshape(-1)


def checked(directory):
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    r=json.loads((directory/'codegen.json').read_text())
    if sha(directory/'o78_register_layout.cubin')!=r['cubin_sha256']:raise ValueError('cubin drift')
    for path,value in r['sources'].items():
        if sha(ROOT/path)!=value:raise ValueError('source drift: '+path)
    for path,value in r['artifact_sha256'].items():
        if sha(directory/path)!=value:raise ValueError('artifact drift: '+path)
    if (directory/'o78_register_layout_generated.cuh').read_text()!=generated_header(
        (ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()):raise ValueError('body drift')
    if (directory/'register_layout_driver.cu').read_text()!=generated_driver(
        (ROOT/'csrc/sm80/roof_o78_row_fused_prepare.cu').read_text()):raise ValueError('timer drift')
    if not r['control_comparison']['passed'] or r['production_default_changed']:raise ValueError('control/default changed')
    return r


class Driver(eight.Driver):
    def __init__(self,library,baseline,candidate):
        receipt=checked(candidate)
        super().__init__(library,baseline,ROOT/'reports/o378_roof_v78_codegen')
        self.handles[0]=self.handles[1];self.resources[0]=dict(self.resources[1])
        try:
            handle=ct.c_void_p()
            self.check(self.lib.roof_probe_open(str((candidate/'o78_register_layout.cubin').resolve()).encode(),
                SYMBOL.encode(),34304,ct.byref(handle)))
            self.handles[1]=handle
            values=(ct.c_int*4)();self.check(self.lib.roof_probe_resources(handle,values))
            self.resources[1]=dict(registers_per_thread=values[0],local_size_bytes=values[1],threads=values[2],
                active_blocks_per_sm=values[3],shared_memory_bytes=34304,cta_tile=[64,128,128],pipeline_stages=2,
                kernel_symbol=SYMBOL,operand_layout='lane_vector_16x64',integer_operand_load='LDS128',
                fallback_operand_layout='original_G128_major')
            function=self.lib.roof_o78_register_layout_benchmark
            function.argtypes=self.lib.roof_o78_gpu_benchmark.argtypes;function.restype=ct.c_int
            self.codegen=dict(previous=self.codegen,register_layout=receipt)
        except Exception:
            self.close();raise

    def prepare(self,case):
        import torch
        if not hasattr(case,'register_layout_storage'):
            case.register_layout_storage={}
            for name in ('a','w'):
                old=case.state[name]
                storage=torch.empty(2*old.numel(),dtype=old.dtype,device=old.device)
                case.register_layout_storage[name]=storage
                case.state[name]=storage[:old.numel()].view_as(old)
            case.state_pointers=(ct.c_uint64*18)(*(case.state[key].data_ptr()
                for key in (*eight.base.STATE_NAMES,'asq','wsq')))
        return super().prepare(case)

    def run(self,case,policy,mode,warmup,repeats,inner):
        import torch
        if policy not in self.handles or mode not in eight.base.MODES:raise ValueError('invalid policy/mode')
        if not hasattr(case,'register_layout_storage'):raise ValueError('prepare must allocate dual payload first')
        if np.any(case.oracle['status_flat']==2):raise ValueError('invalid source must not launch GEMM')
        values=(ct.c_float*(4*repeats))()
        self.check(self.lib.roof_o78_register_layout_benchmark(self.handles[policy],int(case.variant[1:]),int(policy==1),
            eight.base.MODES.index(mode),case.a_source,case.w_source,case.state_pointers,case.m,case.n,
            case.a_multiplier,case.w_multiplier,warmup,repeats,inner,torch.cuda.current_stream().cuda_stream,values))
        if policy==1 and not getattr(case,'register_layout_verified',False):
            # Outside every measured interval; validate all nibbles, not only Y.
            for name in ('a','w'):
                old=case.state[name]
                actual=case.register_layout_storage[name][old.numel():].cpu().numpy()
                assert np.array_equal(actual,packed_reference(old.cpu().numpy(),name=='w')),name
            case.register_layout_verified=True
        return case.state['y'],eight.base.normalize_timings(mode,np.ctypeslib.as_array(values).reshape(4,repeats),repeats)


def timing_contract(mode,inner):
    result=eight.timing_contract(mode,inner)
    result.update(comparison='v78_LDSM_vs_v85_lane_vector_LDS128',
        online_repack_charged_in_corresponding_conversion_stage=True,
        preparation_implementation='v73_plus_explicit_optional_lane_repack',
        preparation_launches_by_candidate={'0':{'weight':1,'activation':2},'1':{'weight':2,'activation':3}},
        candidate_extra_weight_preparation_launches=1,candidate_extra_activation_preparation_launches=1,
        additional_payload_bytes_4096=25165824)
    result.pop('weight_preparation_launches');result.pop('activation_preparation_launches')
    return result


if __name__=='__main__':
    paired_main(driver_cls=Driver,default_gpu_build=Path('reports/o378_roof_v85_codegen'),
        labels=('v78_LDSM_same_v73_preparation','v85_register_layout_with_online_repack'),
        experiment='register_layout',banner='REGISTER LAYOUT',contract=timing_contract,description=__doc__)
