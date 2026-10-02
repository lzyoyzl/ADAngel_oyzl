#!/usr/bin/env python3
"""Internal guarded/integer streaming full-K integer alignment screen; not a production adapter or four-mode result.

Both versions share a native CUDA Driver Event loop. Compare them only
within this run, never against previously recorded extension launch timings.
Original trace preparation is reused; candidate FP32 reassociation is explicitly tested.
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


from o3_fullk_probe import Driver


def summary(rows,policies=(0,1)):
    from compare_roof_trace_candidates import metrics
    import math
    if (not 2<=len(policies)<=3 or policies[0]!=0 or len(set(policies))!=len(policies)
            or not set(policies)<={0,1} or {r['variant'] for r in rows}!={'o3'}):
        raise ValueError('O3 only; explicit control and full-K candidate required')
    index={(r['sample_id'],r['round'],r['fullk_integer']):r for r in rows}
    ids=sorted({r['sample_id'] for r in rows});rounds=sorted({r['round'] for r in rows})
    if len(index)!=len(rows): raise ValueError('duplicate probe record')
    if set(index)!={(sid,r,s) for sid in ids for r in rounds for s in policies} or rounds!=list(range(len(rounds))):
        raise ValueError('incomplete probe coverage')
    for sid in ids:
        for size in policies:
            group=[index[sid,r,size] for r in rounds]
            if len({r['mse_vs_o0'] for r in group})!=1:
                raise ValueError('nondeterministic MSE across rounds')
            for row in group:
                if not row['finite_fp32'] or not row['output_close_current_best']:
                    raise ValueError('output correctness failed')
                if size==0 and (not row['bitwise_equal_current_best'] or row['mse_vs_current_best']!=0):
                    raise ValueError('control must match bitwise')
                if not math.isclose(row['mse_vs_o0'],row['current_best_mse_vs_o0'],rel_tol=1e-5,abs_tol=1e-12):
                    raise ValueError('MSE regression beyond allowed tolerance')
    out=[]
    for size in policies:
        chosen=[r for r in rows if r['fullk_integer']==size]
        speeds=[statistics.median(index[sid,r,0]['summary']['median_ms']/index[sid,r,size]['summary']['median_ms'] for r in rounds) for sid in ids]
        latencies=[statistics.median(index[sid,r,size]['summary']['median_ms'] for r in rounds) for sid in ids]
        errors=[index[sid,0,size]['mse_vs_o0'] for sid in ids]
        out.append(dict(variant='o3',fullk_integer=size,records=len(chosen),samples=len(ids),
            median_ms=statistics.median(latencies),paired_speedup=statistics.median(speeds),
            paired_speedup_ci95=list(metrics.bootstrap_median_ci(speeds,10000,.95,20261001)) if len(ids)>1 else None,
            cv_failed_records=sum(r['summary']['cv_percent']>=3 for r in chosen),
            median_mse=statistics.median(errors),mean_mse=statistics.mean(errors),
            max_output_difference=max(r['max_abs_difference_current_best'] for r in chosen),
            max_mse_vs_current_best=max(r['mse_vs_current_best'] for r in chosen),
            bitwise_equal_records=sum(r['bitwise_equal_current_best'] for r in chosen)))
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
    p.add_argument('--variants',nargs='+',choices=['o3'],default=['o3'])
    p.add_argument('--policies',nargs='+',type=int,choices=[0,1],default=[0,1])
    p.add_argument('--factor-async',action='store_true',help='v59 int32 factor metadata; isolated candidate only')
    args=p.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT) or not 1<=args.samples<=24 or min(args.rounds,args.repeats)<1 or args.warmup<0 or len(set(args.variants))!=len(args.variants) or not 2<=len(args.policies)<=3 or args.policies[0]!=0 or len(set(args.policies))!=len(args.policies):
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
    if args.factor_async:
        from probe_roof_factor_async_codegen import checked_cubins
        cubins=checked_cubins(args.cubins)
    else:
        assert codegen['host_stream_mapping_passed']
        assert codegen['all_probe_copies_bypass_l1'] and codegen['native_int4_entries']
        audit=json.loads((args.cubins/'audit.json').read_text())
        assert audit['passed'] and all(audit['control_encoded_sass_matches_best'].values())
        for source in audit['sources']:
            assert sha256_file(Path(source['file']))==source['sha256']
        cubins={s:(args.cubins/f'fullk_integer_{s}.cubin').resolve() for s in args.policies}
    for s,path in cubins.items(): assert sha256_file(path)==codegen['variants'][str(s)]['cubin_sha256']
    args.output.mkdir(parents=True)
    def save(name,value): (args.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    def append(name,value):
        with (args.output/name).open('a') as f: f.write(json.dumps(value,allow_nan=False)+'\n')
    library=(args.output/'libroof_probe_driver.so').resolve()
    build=['g++','-O3','-std=c++17','-shared','-fPIC','-I/usr/local/cuda-12.8/include',
        str(ROOT/'csrc/sm80/roof_producer_warp_driver.cpp'),'-lcuda','-o',str(library)]
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
            driver=Driver(library,cubins,variant,best['kernel']['shared_memory_bytes'],factor_metadata=args.factor_async)
            try:
                for r in range(args.rounds):
                    append('gpu_snapshots.jsonl',dict(sample_id=x.sample_id,variant=variant,round=r,time=time.time(),
                        gpu=command('nvidia-smi','--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu','--format=csv')))
                    order=measurement_order(args.policies,si,vi,r)
                    for size in order:
                        y,times=driver.run(size,best,values[1],scale,args.warmup,args.repeats)
                        row=dict(sample_id=x.sample_id,variant=variant,round=r,fullk_integer=size,execution_order=order,
                            factor_async=args.factor_async,
                            raw_ms=times,summary=stats(times),
                            finite_fp32=bool(torch.isfinite(y).all()),
                            output_close_current_best=bool(torch.allclose(y,best['output'],rtol=1e-3,atol=1e-3)),
                            bitwise_equal_current_best=bool(torch.equal(y.view(torch.int32),best['output'].view(torch.int32))),
                            max_abs_difference_current_best=(y-best['output']).abs().max().item(),
                            current_best_mse_vs_o0=mse(best['output'],o0),reference_tune=tune,
                            mse_vs_current_best=mse(y,best['output']),mse_vs_paired_fp16=mse(y,paired),mse_vs_o0=mse(y,o0),paired_fp16=ref,
                            reference_kernel=dict(best['kernel']),probe_resources=driver.resources[driver.last_policy],**payload,
                            executed_policy=driver.last_policy,guard=driver.guard_metadata,
                            probe_symbol='adangel_roof_fullk_integer_o3')
                        rows.append(row);append('results.jsonl',row)
                print(x.sample_id,variant,'complete',flush=True)
            finally: driver.close()
    assert len(rows)==args.samples*len(args.variants)*args.rounds*len(args.policies)
    save('summary.json',dict(correctness_passed=True,no_filtering=True,production_default_changed=False,
        factor_async=args.factor_async,
        scope='internal_cubin_compute_only_screen_guard_outside_interval_not_end_to_end',policies=args.policies,records=summary(rows,args.policies)))


if __name__=='__main__': main()
