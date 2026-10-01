#!/usr/bin/env python3
"""Internal partial-recomposition cubin screen; not a production adapter or four-mode result.

All three versions share a native CUDA Driver Event loop. Compare them only
within this run, never against previously recorded extension launch timings.
Original trace preparation and bitwise best-kernel references are reused.
"""
import argparse
import ctypes as ct
import json
from pathlib import Path
import statistics
import subprocess
import time

from benchmark_a100_o1 import command, stats
from benchmark_a100_roof_trace import measurement_order
from benchmark_a100_mixed import validate_fp16_result
from benchmark_a100_mixed_trace import inspect_inputs, inspect_raw_inputs, mse, source_identity, verify_raw_prepared
from roof_payload_validation import verify_grouped_payload

ROOT=Path(__file__).resolve().parents[1]


class Driver:
    def __init__(self, library, cubins, variant, smem):
        self.lib=ct.CDLL(str(library))
        self.lib.roof_probe_error.restype=ct.c_char_p
        self.lib.roof_probe_open.argtypes=[ct.c_char_p,ct.c_char_p,ct.c_uint,ct.POINTER(ct.c_void_p)]
        self.lib.roof_probe_close.argtypes=[ct.c_void_p]
        self.lib.roof_probe_benchmark.argtypes=[ct.c_void_p]+[ct.c_uint64]*5+[ct.c_int]*5+[ct.c_void_p,ct.POINTER(ct.c_float)]
        self.handles={}
        try:
            for size,path in cubins.items():
                handle=ct.c_void_p()
                self.check(self.lib.roof_probe_open(str(path).encode(),
                    ('adangel_roof_recompose_o3' if variant=='o3' else 'adangel_roof_recompose_o78').encode(),smem,ct.byref(handle)))
                self.handles[size]=handle
        except Exception:
            self.close();raise

    def check(self, code):
        if code: raise RuntimeError(self.lib.roof_probe_error().decode())

    def close(self):
        for h in self.handles.values(): self.check(self.lib.roof_probe_close(h))
        self.handles.clear()

    def run(self,size,best,activation_scale,weight_scale,warmup,repeats):
        import torch
        a=best['packed_activation_g128_major'];w=best['packed_weight_g128_major']
        _,g,m,kbytes=a.shape;n=w.shape[1];k=g*128
        assert kbytes==64 and a.is_contiguous() and w.is_contiguous()
        assert a.dtype==w.dtype==torch.uint8 and activation_scale.dtype==torch.float32
        assert all(t.is_cuda and t.device==a.device for t in (w,activation_scale,weight_scale,best['output']))
        y=torch.empty_like(best['output']);times=(ct.c_float*repeats)()
        self.check(self.lib.roof_probe_benchmark(self.handles[size],a.data_ptr(),w.data_ptr(),
            activation_scale.data_ptr(),weight_scale.data_ptr(),y.data_ptr(),m,n,k,warmup,repeats,
            torch.cuda.current_stream().cuda_stream,times))
        if y.dtype!=torch.float32 or not torch.isfinite(y).all() or not torch.equal(y.view(torch.int32),best['output'].view(torch.int32)):
            raise AssertionError('Recomposition probe does not equal current best bitwise')
        return y,list(times)


def summary(rows):
    from compare_roof_trace_candidates import metrics
    out=[]
    for variant in sorted({r['variant'] for r in rows}):
        selected=[r for r in rows if r['variant']==variant]
        index={(r['sample_id'],r['round'],r['recompose_policy']):r for r in selected}
        if len(index)!=len(selected): raise ValueError('duplicate probe record')
        ids=sorted({r['sample_id'] for r in selected})
        rounds=sorted({r['round'] for r in selected})
        expected={(sid,r,size) for sid in ids for r in rounds for size in (0,1,2)}
        if set(index)!=expected or rounds!=list(range(len(rounds))):
            raise ValueError('incomplete probe coverage')
        for sid in ids:
            group=[r for r in selected if r['sample_id']==sid]
            if (any(not r['bitwise_equal_current_best'] or r['mse_vs_current_best']!=0 for r in group)
                    or len({(r['mse_vs_o0'],r['mse_vs_paired_fp16']) for r in group})!=1):
                raise ValueError('bitwise or MSE regression')
        for size in (0,1,2):
            chosen=[r for r in selected if r['recompose_policy']==size]
            speeds=[statistics.median(index[sid,r,0]['summary']['median_ms']/index[sid,r,size]['summary']['median_ms'] for r in rounds) for sid in ids]
            latencies=[statistics.median(index[sid,r,size]['summary']['median_ms'] for r in rounds) for sid in ids]
            errors=[index[sid,0,size]['mse_vs_paired_fp16'] for sid in ids]
            out.append(dict(variant=variant,recompose_policy=size,records=len(chosen),samples=len(ids),
                median_ms=statistics.median(latencies),paired_speedup=statistics.median(speeds),
                paired_speedup_ci95=list(metrics.bootstrap_median_ci(speeds,10000,.95,20261001)) if len(ids)>1 else None,
                cv_failed_records=sum(r['summary']['cv_percent']>=3 for r in chosen),
                median_mse=statistics.median(errors),mean_mse=statistics.mean(errors)))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cubins',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('data/prepared/llama2_7b_prefill_o0_o4'))
    p.add_argument('--raw-data',type=Path,default=Path('data/raw/llama2_7b_prefill'))
    p.add_argument('--trace-config',type=Path,default=Path('configs/trace/llama2_7b_prefill.yaml'))
    p.add_argument('--samples',type=int,default=24)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--warmup',type=int,default=50)
    p.add_argument('--repeats',type=int,default=200)
    p.add_argument('--variants',nargs='+',choices=['o3','o7','o8'],default=['o3','o7','o8'])
    args=p.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT) or not 1<=args.samples<=24 or min(args.rounds,args.repeats)<1 or args.warmup<0 or len(set(args.variants))!=len(args.variants):
        p.error('fresh repository output, valid coverage and distinct variants required')
    import torch
    from adangel import _sm80 as native
    from adangel.quantization import mixed_formats as mf
    from adangel.trace.storage import load_prepared, sha256_file
    from adangel.trace.prepare import _load_and_validate_raw
    assert torch.cuda.get_device_capability()==(8,0)
    torch.cuda.init();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    manifest,mh=inspect_inputs(args.data)
    raw_manifest,rh=inspect_raw_inputs(args.raw_data,manifest,args.trace_config)
    raw_entries={r['sample_id']:r for r in raw_manifest['samples']}
    codegen=json.loads((args.cubins/'codegen.json').read_text())
    assert codegen['all_probe_copies_bypass_l1'] and codegen['native_int4_entries']
    cubins={s:(args.cubins/f'recompose_{s}.cubin').resolve() for s in (0,1,2)}
    for s,path in cubins.items(): assert sha256_file(path)==codegen['variants'][str(s)]['cubin_sha256']
    args.output.mkdir(parents=True)
    def save(name,value): (args.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    def append(name,value):
        with (args.output/name).open('a') as f: f.write(json.dumps(value,allow_nan=False)+'\n')
    library=(args.output/'libroof_probe_driver.so').resolve()
    build=['g++','-O3','-std=c++17','-shared','-fPIC','-I/usr/local/cuda-12.8/include',
        str(ROOT/'csrc/sm80/roof_probe_driver.cpp'),'-lcuda','-o',str(library)]
    with (args.output/'driver_build.log').open('w') as log: subprocess.run(build,stdout=log,stderr=subprocess.STDOUT,check=True)
    save('environment.json',dict(git_commit=command('git','rev-parse','HEAD'),
        extension_sha256=sha256_file(Path(native.__file__)),cubins={str(s):sha256_file(path) for s,path in cubins.items()},
        driver_sha256=sha256_file(library),driver_build_command=build,
        torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
        prepared_manifest_sha256=mh,raw_manifest_sha256=rh,
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        scope='internal_cubin_compute_only_screen_not_extension_or_four_mode_acceptance',
        timing='single_launch_native_driver_cuda_events_allocation_outside_intervals',
        policy='unlocked shared GPU; cyclic paired order; no filtering'))
    rows=[]
    for si,entry in enumerate(manifest['samples'][:args.samples]):
        path=args.data/entry['file'];re=raw_entries[entry['sample_id']];rp=args.raw_data/re['file']
        assert sha256_file(path)==entry['sha256'] and sha256_file(rp)==re['sha256']
        x=load_prepared(path,device='cuda')
        raw=_load_and_validate_raw(rp,re['layer'],re['projection'])
        operands=(raw['activation_fp16'],raw['weight_fp16']);verify_raw_prepared(x,operands)
        assert x.sample_id==entry['sample_id'] and list(x.shape)==entry['shape']
        o0=native.benchmark_o0(x.A_int8,x.A_scale,x.W_mxfp4,x.W_scale,'compute_only',0,1,100)['output']
        for vi,variant in enumerate(args.variants):
            if variant=='o3':
                base=native.benchmark('o3','compute_only',x.A_int8,x.A_scale,x.W_mxfp4_g128,x.W_scale_g128,0,1,100,'production')
                values=(base['converted_activation'],x.A_scale,base['converted_weight'],x.W_scale_g128)
                paired=o0;ref='o0';tune=54
            else:
                wf,af=mf.VARIANTS[variant]
                wsrc=mf.quantize_source(operands[1].cuda(),wf);asrc=mf.quantize_source(operands[0].cuda(),af)
                append('source_provenance.jsonl',dict(sample_id=x.sample_id,variant=variant,
                    raw_prepared_replay_bitwise=True,weight=source_identity(wsrc),activation=source_identity(asrc)))
                ref=mf.PAIRED_BASELINE[variant]
                fp=native._benchmark_mixed(ref,'compute_only',wsrc,asrc,0,1,100,'64x128x256','row_major')
                validate_fp16_result(fp,wsrc,asrc);paired=fp['output']
                base=native._benchmark_mixed(variant,'compute_only',wsrc,asrc,0,1,100,'64x128x256','group_major')
                values=(*base['converted_activation'],*base['converted_weight']);tune=59
            best=native._benchmark_roof_candidate(variant,tune,*values,0,1)
            payload=verify_grouped_payload(best,tune,values[0],values[2],values[3])
            scale=best['converted_weight_scale'] if variant=='o3' else values[3]
            driver=Driver(library,cubins,variant,best['kernel']['shared_memory_bytes'])
            try:
                for r in range(args.rounds):
                    append('gpu_snapshots.jsonl',dict(sample_id=x.sample_id,variant=variant,round=r,time=time.time(),
                        gpu=command('nvidia-smi','--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu','--format=csv')))
                    order=measurement_order([0,1,2],si,vi,r)
                    for size in order:
                        y,times=driver.run(size,best,values[1],scale,args.warmup,args.repeats)
                        row=dict(sample_id=x.sample_id,variant=variant,round=r,recompose_policy=size,execution_order=order,
                            raw_ms=times,summary=stats(times),bitwise_equal_current_best=True,reference_tune=tune,
                            mse_vs_current_best=0.0,mse_vs_paired_fp16=mse(y,paired),mse_vs_o0=mse(y,o0),paired_fp16=ref,
                            reference_kernel=dict(best['kernel']),**payload,
                            probe_symbol='adangel_roof_recompose_o3' if variant=='o3' else 'adangel_roof_recompose_o78')
                        rows.append(row);append('results.jsonl',row)
                print(x.sample_id,variant,'complete',flush=True)
            finally: driver.close()
    assert len(rows)==args.samples*len(args.variants)*args.rounds*3
    save('summary.json',dict(correctness_passed=True,no_filtering=True,production_default_changed=False,
        scope='internal_cubin_compute_only_screen',records=summary(rows)))


if __name__=='__main__': main()
