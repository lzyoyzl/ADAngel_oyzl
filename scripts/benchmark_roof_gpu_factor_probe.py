#!/usr/bin/env python3
"""GPU exact guard/factor preparation and prepared-payload core timing, NOT full Cold."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from roof_gpu_factor_probe import ROOT,MODES,Pipeline,build
from o3_fullk_probe import guard_columns
from probe_roof_factor_async_codegen import checked_cubins
from benchmark_a100_o1 import stats,command
from benchmark_a100_roof_trace import measurement_order
from benchmark_a100_mixed_trace import inspect_inputs,inspect_raw_inputs,verify_raw_prepared,mse


def validate(driver):
    import torch
    from adangel import _sm80 as native
    from adangel.quantization.arbitrary_bits import split_int8_to_packed_int4
    from validate_a100_split_grouped import pack_q4
    from roof_reduction_validation import reference_fp64
    checks=[];rejections=0
    for m,n in ((64,128),(128,256)):
        for pattern in ('random','zero','extrema','zero_scale','wide_safe','unsafe'):
            torch.manual_seed(60)
            a=torch.randint(-128,128,(m,4096),device='cuda',dtype=torch.int8)
            w=torch.randint(-8,8,(n,4096),device='cuda',dtype=torch.int8)
            if pattern=='zero':a.zero_();w.zero_()
            if pattern in ('extrema','wide_safe','unsafe'):a.fill_(-128);w.fill_(-8)
            asc=torch.linspace(.001,.03,m,device='cuda')
            if pattern=='zero_scale':asc.zero_()
            wsc=((torch.arange(n*32,device='cuda').reshape(n,32)*7)%5+116).byte()
            if pattern in ('wide_safe','unsafe'):
                wsc.fill_(116);wsc[:,-1]=128 if pattern=='wide_safe' else 131
            values=(split_int8_to_packed_int4(a),asc,pack_q4(w),wsc)
            ref=reference_fp64('o3',values)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                best=native._benchmark_roof_candidate('o3',54,*values,0,1)
                ws=best['converted_weight_scale'];columns=ws.T.cpu().tolist()
                anchors,guard=guard_columns(columns)
                expected=torch.tensor([[1<<min(c[g]-h,14) for c,h in zip(columns,anchors)]
                                       for g in range(32)]+[anchors],device='cuda',dtype=torch.int32)
                for mode in MODES:
                    r=driver.run_mode(mode,best,asc,ws,0,1,2)
                    if mode!='control_compute':
                        assert r['status']==(0 if guard['safe'] else 1)
                        assert torch.equal(r['metadata'],expected)
                    if r['output'] is not None:
                        torch.testing.assert_close(r['output'].double(),ref,rtol=1e-3,atol=1e-3)
                        assert torch.isfinite(r['output']).all()
                        if mode=='control_compute' or not guard['safe']:
                            assert torch.equal(r['output'],best['output'])
                    checks.append(dict(shape=[m,n,4096],pattern=pattern,mode=mode,
                        guard_safe=guard['safe'],status=r['status'],metadata_exact=mode!='control_compute',
                        finite_fp32=r['output'] is not None))
                if pattern=='random':
                    for code in (255,0):
                        ws.fill_(code)
                        for mode in MODES[1:]:
                            try:driver.run_mode(mode,best,asc,ws,0,1,2)
                            except RuntimeError as e:
                                assert ('code 255' if code==255 else 'normal UE8M0') in str(e)
                                rejections+=1
                            else:raise AssertionError('invalid scale must reject')
            stream.synchronize()
    return dict(passed=True,count=len(checks),checks=checks,rejections=rejections,
                scope='small_MN_full_K4096; nondefault_stream; same_GPU_guard_in_all_modes')


def summarize(rows):
    from compare_roof_trace_candidates import metrics
    index={(r['sample_id'],r['round'],r['mode']):r for r in rows}
    ids=sorted({r['sample_id'] for r in rows});rounds=sorted({r['round'] for r in rows})
    if len(index)!=len(rows) or set(index)!={(s,r,m) for s in ids for r in rounds for m in MODES}:
        raise ValueError('incomplete or duplicate records')
    out=[]
    for mode in MODES:
        chosen=[r for r in rows if r['mode']==mode]
        if any(not r['metadata_exact'] or r['guard_status']!=0 for r in chosen if mode!='control_compute'):
            raise ValueError('guard/metadata regression')
        if any(not r['output_bitwise_best'] for r in chosen if mode!='prepare_gpu_only'):
            raise ValueError('real trace output regression')
        ms=[statistics.median(index[s,r,mode]['summary']['median_ms'] for r in rounds) for s in ids]
        record=dict(mode=mode,samples=len(ids),records=len(chosen),median_ms=statistics.median(ms),
                    cv_failed_records=sum(r['summary']['cv_percent']>=3 for r in chosen))
        if mode!='prepare_gpu_only':
            speed=[statistics.median(index[s,r,'control_compute']['summary']['median_ms']/
                                    index[s,r,mode]['summary']['median_ms'] for r in rounds) for s in ids]
            errors=[index[s,0,mode]['mse_vs_o0'] for s in ids]
            record.update(paired_speedup=statistics.median(speed),
                paired_speedup_ci95=list(metrics.bootstrap_median_ci(speed,10000,.95,60)),
                median_mse=statistics.median(errors),mean_mse=statistics.mean(errors))
        out.append(record)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gemm-cubins',type=Path,default=Path('reports/o378_roof_v59'))
    p.add_argument('--use-build',type=Path)
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--samples',type=int,default=4);p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--warmup',type=int,default=50);p.add_argument('--repeats',type=int,default=200)
    p.add_argument('--inner',type=int,default=100)
    args=p.parse_args()
    if (args.output.exists() or not args.output.resolve().is_relative_to(ROOT) or
        not 1<=args.samples<=24 or min(args.repeats,args.rounds,args.inner)<1 or args.warmup<0):
        p.error('fresh repository output and valid counts required')
    import torch
    from adangel import _sm80 as native
    from adangel.trace.storage import load_prepared,sha256_file
    from adangel.trace.prepare import _load_and_validate_raw
    torch.cuda.init();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.get_device_capability()==(8,0)
    args.output.mkdir(parents=True)
    def save(name,obj):(args.output/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    def append(name,obj):
        with (args.output/name).open('a') as f:f.write(json.dumps(obj,allow_nan=False)+'\n')
    if args.use_build:
        d=args.use_build.resolve();b=json.loads((d/'build.json').read_text())
        lib=d/'libgpu_factor_driver.so';prep=d/'factor_prepare.cubin'
        assert sha256_file(lib)==b['driver_sha256'] and sha256_file(prep)==b['preparation_cubin_sha256']
        for path,digest in b['sources'].items():assert sha256_file(ROOT/path)==digest
        cubins=checked_cubins(args.gemm_cubins)
        assert {str(i):sha256_file(path) for i,path in cubins.items()}==b['gemm_cubins']
    else:lib,prep,cubins=build(args.output/'build',args.gemm_cubins)
    save('environment.json',dict(git_commit=command('git','rev-parse','HEAD'),
        extension_sha256=sha256_file(Path(native.__file__)),torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),gemm_cubins={str(i):sha256_file(path) for i,path in cubins.items()},
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        driver_sha256=sha256_file(lib),preparation_cubin_sha256=sha256_file(prep),
        scope='prepared-payload core: conversion/packing absent; NOT full Cold or steady-state',
        timing=dict(control_compute='one control GEMM',cached_compute='guard/prep outside Event, one v59 GEMM',
            prepare_decide_compute='one Event around prepare, DtoH verdict, stream sync, host decision and GEMM',
            prepare_gpu_only='kernel repeated inner times in one Event / inner, verdict outside'),
        policy='unlocked GPU, cyclic order, all raw values retained',production_default_changed=False))
    driver=Pipeline(lib,prep,cubins)
    try:
        save('validation.json',validate(driver))
        if args.validate_only:return
        data=ROOT/'data/prepared/llama2_7b_prefill_o0_o4';rawdata=ROOT/'data/raw/llama2_7b_prefill'
        manifest,mh=inspect_inputs(data)
        raw_manifest,rh=inspect_raw_inputs(rawdata,manifest,ROOT/'configs/trace/llama2_7b_prefill.yaml')
        rawindex={r['sample_id']:r for r in raw_manifest['samples']}
        save('input_provenance.json',dict(prepared_manifest_sha256=mh,raw_manifest_sha256=rh))
        rows=[]
        for si,e in enumerate(manifest['samples'][:args.samples]):
            re=rawindex[e['sample_id']]
            assert sha256_file(data/e['file'])==e['sha256'] and sha256_file(rawdata/re['file'])==re['sha256']
            x=load_prepared(data/e['file'],device='cuda')
            raw=_load_and_validate_raw(rawdata/re['file'],re['layer'],re['projection'])
            verify_raw_prepared(x,(raw['activation_fp16'],raw['weight_fp16']))
            base=native.benchmark('o3','compute_only',x.A_int8,x.A_scale,x.W_mxfp4_g128,x.W_scale_g128,0,1,100,'production')
            best=native._benchmark_roof_candidate('o3',54,base['converted_activation'],x.A_scale,
                base['converted_weight'],x.W_scale_g128,0,1)
            o0=native.benchmark_o0(x.A_int8,x.A_scale,x.W_mxfp4,x.W_scale,'compute_only',0,1,100)['output']
            ws=best['converted_weight_scale'];columns=ws.T.cpu().tolist();anchors,guard=guard_columns(columns)
            assert guard['safe']
            expected=torch.tensor([[1<<(c[g]-h) for c,h in zip(columns,anchors)] for g in range(32)]+
                                  [anchors],device='cuda',dtype=torch.int32)
            for ri in range(args.rounds):
                order=measurement_order(list(MODES),si,0,ri)
                append('gpu_snapshots.jsonl',dict(sample_id=x.sample_id,round=ri,time=time.time(),
                    gpu=command('nvidia-smi','--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu','--format=csv')))
                for mode in order:
                    r=driver.run_mode(mode,best,x.A_scale,ws,args.warmup,args.repeats,args.inner)
                    y=r['output'];equal=None if y is None else bool(torch.equal(y.view(torch.int32),best['output'].view(torch.int32)))
                    exact=None if mode=='control_compute' else bool(torch.equal(r['metadata'],expected))
                    assert equal is not False and exact is not False
                    row=dict(sample_id=x.sample_id,round=ri,mode=mode,order=order,
                        raw_ms=r['raw_ms'],summary=stats(r['raw_ms']),guard_status=r['status'],metadata_exact=exact,
                        output_bitwise_best=equal,mse_vs_o0=None if y is None else mse(y,o0),
                        guard=guard,executed_policy=r['executed_policy'])
                    rows.append(row);append('results.jsonl',row)
            print(x.sample_id,'complete',flush=True)
        save('summary.json',dict(records=summarize(rows),production_default_changed=False,
            scope='not full Cold/steady/conversion; only preparation and prepared-payload core'))
    finally:driver.close()


if __name__=='__main__':main()
