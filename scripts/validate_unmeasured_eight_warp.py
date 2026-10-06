#!/usr/bin/env python3
"""v104: resolve a missing runtime datum, not repeat a completed optimization.

Reuse exact v98 cubins (never previously launched), best controls and existing
source/guard semantics. Preserve the FAILED v98 heuristic in all evidence.
First query real128/256-thread occupancy. Runtime performance is independent
evidence, not retroactive relaxation of the v98 compile gate. No geometry scan.
"""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

import numpy as np

from probe_eight_warp_fullk_codegen import CONFIG, checked
from benchmark_a100_o1 import stats

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'csrc/sm80/roof_latency_validation_driver.cpp'


def resources_gate(resources):
    expected={(kind,p) for kind in ('o3','o78') for p in (0,1)}
    if set(resources)!=expected:raise ValueError('four exact resources required')
    for (kind,policy),r in resources.items():
        if r['threads']!=(128,256)[policy]:raise ValueError('wrong launch geometry')
        # Entry-local bytes are disclosed, not an automatic rejection: the
        # accepted control itself has non-hot local accesses. The immutable
        # codegen receipt separately audits the candidate's hot-loop local.
        if r['local_size_bytes']<0 or r['active_blocks_per_sm']<2: return False
        if policy and r['active_blocks_per_sm']*r['threads']//32<16:return False
    for kind in ('o3','o78'):
        old,new=(resources[kind,p] for p in (0,1))
        if new['active_blocks_per_sm']*new['threads']<=old['active_blocks_per_sm']*old['threads']:
            return False
    return True


def model_rationale():
    # v90 necessary capacity, applying v98 LDSM workload multiplier.
    # Not a predicted runtime, not a hard resource or accuracy gate.
    return dict(v98_original_compile_gate_changed=False,
        v98_original_heuristic_failed=True,previous_candidate_GPU_launches=0,
        v90_issue_active_percent={'o3':43.131528,'o7':46.924698},
        shared_capacity_estimate_ms={'o3':.172251*1.5,'o7':.182576*1.5},
        original_mma_necessary_ms=.220347,
        no_new_optimization_or_parameter_scan=True,
        purpose='one_missing_full24_runtime_comparison_not_static_speedup_claim')


class Driver:
    def __init__(self,output,directories):
        output.mkdir(parents=True)
        self.receipts={k:checked(directories[k],k) for k in ('o3','o78')}
        library=output/'liblatency_validation.so'
        command=['g++','-std=c++17','-O2','-shared','-fPIC',str(SOURCE),
            '-I/usr/local/cuda-12.8/include','-L/usr/local/cuda-12.8/lib64/stubs',
            '-lcuda','-o',str(library)]
        temporary=ROOT/'tmp';temporary.mkdir(exist_ok=True)
        result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              env={**os.environ,'TMPDIR':str(temporary)})
        (output/'host_compile.log').write_text(result.stdout)
        if result.returncode:raise RuntimeError('cached cubin host driver compile failed')
        self.lib=ct.CDLL(str(library.resolve()))
        self.lib.roof_latency_error.restype=ct.c_char_p
        self.lib.roof_latency_open.argtypes=[ct.c_char_p,ct.c_char_p,ct.c_int,ct.c_int,ct.c_uint,
                                             ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_int)]
        self.lib.roof_latency_close.argtypes=[ct.c_void_p]
        self.lib.roof_latency_benchmark.argtypes=[ct.c_void_p,ct.POINTER(ct.c_uint64)]+[ct.c_int]*6+[
            ct.c_void_p,ct.POINTER(ct.c_float)]
        self.handles={};self.resources={}
        try:
            for kind,cfg in CONFIG.items():
                path=directories[kind]/(cfg['stem']+'.cubin')
                for policy,symbol in enumerate((cfg['control'],cfg['symbol'])):
                    h=ct.c_void_p();r=(ct.c_int*4)()
                    self.check(self.lib.roof_latency_open(str(path.resolve()).encode(),symbol.encode(),
                        3 if kind=='o3' else 78,(128,256)[policy],cfg['shared'],ct.byref(h),r))
                    self.handles[kind,policy]=h
                    self.resources[kind,policy]=dict(registers_per_thread=r[0],local_size_bytes=r[1],
                        threads=r[2],active_blocks_per_sm=r[3],active_warps_per_sm=r[3]*r[2]//32,
                        shared_memory_bytes=cfg['shared'],kernel_symbol=symbol,
                        cubin_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            self.receipt=dict(host_command=command,compiler_temporary_directory=str(temporary),
                driver_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE,Path(__file__))},
                resources={f'{k}_{p}':r for (k,p),r in self.resources.items()},original_codegen=self.receipts,
                rationale=model_rationale(),capacity_gate=resources_gate(self.resources))
            (output/'resources.json').write_text(json.dumps(self.receipt,indent=2)+'\n')
        except Exception:self.close();raise

    def check(self,code):
        if code:raise RuntimeError(self.lib.roof_latency_error().decode())

    def close(self):
        for h in self.handles.values():self.check(self.lib.roof_latency_close(h))
        self.handles={}

    def run(self,kind,policy,tensors,m,n,warmup,repeats):
        import torch
        pointers=(ct.c_uint64*len(tensors))(*(t.data_ptr() for t in tensors))
        values=(ct.c_float*repeats)()
        self.check(self.lib.roof_latency_benchmark(self.handles[kind,policy],pointers,len(tensors),
            m,n,4096,warmup,repeats,torch.cuda.current_stream().cuda_stream,values))
        return tensors[-1],list(values)


def o3_case(x,native):
    import torch
    from o3_fullk_probe import guard_columns
    from adangel.quantization.mxfp4 import mxfp4_to_q4_packed
    if not torch.equal(x.W_q4,mxfp4_to_q4_packed(x.W_mxfp4_g128)):
        raise ValueError('O3 prepared Q4 differs from current mapping')
    base=native.benchmark('o3','compute_only',x.A_int8,x.A_scale,x.W_mxfp4_g128,x.W_scale_g128,0,1,100,'production')
    columns=x.W_scale_g128.cpu().tolist()
    anchors,guard=guard_columns(columns)
    if not guard['normal_scales']:raise ValueError('O3 requires normal scales for cached proof')
    # The existing conservative bound is stronger than every actual partial.
    # Identical cached metadata/route for both exact kernels; not online timing.
    n=len(columns)
    meta=torch.zeros((33,n),dtype=torch.int32,device=x.A_int8.device)
    if guard['safe']:
        meta.copy_(torch.tensor([[1<<(columns[col][g]-anchors[col]) for col in range(n)]
            for g in range(32)]+[anchors],dtype=torch.int32,device=meta.device))
    status=torch.full((n//128,),0 if guard['safe'] else 1,device=meta.device,dtype=torch.uint32)
    output=torch.empty_like(base['output'])
    tensors=(base['packed_activation_g128_major'],base['packed_weight_g128_major'],x.A_scale,
        x.W_scale_g128.T.contiguous(),meta,status,output)
    return tensors,guard,base


def summarize(rows):
    ids=sorted({r['sample_id'] for r in rows});rounds=sorted({r['round'] for r in rows})
    index={(r['sample_id'],r['variant'],r['round'],r['policy']):r for r in rows}
    if len(ids)!=24 or len(index)!=len(rows) or set(index)!={
            (s,v,r,p) for s in ids for v in ('o3','o7','o8') for r in rounds for p in (0,1)}:
        raise ValueError('complete full24 paired records required')
    from adangel.benchmark.metrics import bootstrap_median_ci
    output=[]
    for variant in ('o3','o7','o8'):
        ratios=[statistics.median(index[s,variant,r,0]['summary']['median_ms']/
                                  index[s,variant,r,1]['summary']['median_ms'] for r in rounds) for s in ids]
        for policy in (0,1):
            selected=[index[s,variant,r,policy] for s in ids for r in rounds]
            if not all(row['bitwise_equal_control'] and row['finite_fp32'] for row in selected):
                raise ValueError('numerical acceptance failed')
            mse=[index[s,variant,rounds[0],policy]['mse_vs_reference'] for s in ids]
            output.append(dict(variant=variant,policy=policy,samples=24,records=len(selected),
                median_ms=statistics.median(statistics.median(index[s,variant,r,policy]['summary']['median_ms']
                    for r in rounds) for s in ids),paired_speedup=1 if policy==0 else statistics.median(ratios),
                paired_ci95=[1,1] if policy==0 else list(bootstrap_median_ci(ratios,10000,.95,20261007)),
                median_MSE=statistics.median(mse),mean_MSE=statistics.mean(mse),
                cv_failed=sum(row['summary']['cv_percent']>=3 for row in selected)))
    return dict(scope='exact_unmeasured_v98_full24_cached_GEMM_not_conversion_E2E',
        records=len(rows),summary=output,production_default_changed=False,
        original_v98_compile_gate_changed=False,no_filtering=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--o3-codegen',type=Path,default=Path('reports/o378_roof_v98_o3_codegen'))
    p.add_argument('--o78-codegen',type=Path,default=Path('reports/o378_roof_v98_o78_codegen'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resources-only',action='store_true')
    p.add_argument('--data',type=Path,default=Path('data/prepared/llama2_7b_prefill_o0_o4'))
    p.add_argument('--raw-data',type=Path,default=Path('data/raw/llama2_7b_prefill'))
    p.add_argument('--trace-config',type=Path,default=Path('configs/trace/llama2_7b_prefill.yaml'))
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--warmup',type=int,default=50)
    p.add_argument('--repeats',type=int,default=200)
    args=p.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT) or args.rounds<1 or args.warmup<0 or args.repeats<1:
        p.error('fresh project output and valid counts required')
    import torch
    torch.cuda.init();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    context_anchor=torch.empty(1,device='cuda') # Driver context is lazy until a real allocation.
    if torch.cuda.get_device_capability()!=(8,0):raise ValueError('A100 SM80 required')
    driver=Driver(args.output/'build',{'o3':args.o3_codegen,'o78':args.o78_codegen})
    try:
        if args.resources_only:
            print(json.dumps(driver.receipt['resources'],indent=2),flush=True)
            print('CAPACITY',driver.receipt['capacity_gate'],flush=True)
            return
        if not driver.receipt['capacity_gate']:raise ValueError('no active warp capacity gain')
        raise NotImplementedError('Full24 protocol added after real occupancy evidence; no candidate launch here')
    finally:driver.close()


if __name__=='__main__':main()
