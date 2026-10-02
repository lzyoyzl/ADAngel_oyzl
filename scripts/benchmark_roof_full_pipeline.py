#!/usr/bin/env python3
"""v62: original O3 inputs, identical conversion2, paired complete four-mode timing."""
import argparse
import json
from pathlib import Path
import time
from roof_full_pipeline_probe import ROOT,MODES,Pipeline,build,stage_contract
from benchmark_a100_o1 import stats,command
from benchmark_a100_mixed_trace import inspect_inputs,inspect_raw_inputs,verify_raw_prepared,mse
from benchmark_conversion_pipeline import summarize as base_summary
from benchmark_integer_conversion_probe import order


def summarize(rows):
    keys={(r['sample_id'],r['round'],r['mode'],r['implementation']) for r in rows}
    ids={r['sample_id'] for r in rows};rounds={r['round'] for r in rows}
    if len(keys)!=len(rows) or keys!={(s,i,m,p) for s in ids for i in rounds for m in MODES for p in (0,1)}:
        raise ValueError('incomplete or duplicate paired four-mode records')
    if any(not r['bitwise_equal_current_best'] or not r['payload_bitwise'] for r in rows):
        raise ValueError('trace correctness regression')
    return base_summary(rows)


def timing_check(r,mode,inner,repeats,policy):
    assert r['stage_timing_inner_repeats']==stage_contract(mode,inner)
    assert r['weight_cached']==(mode not in ('conversion_only','cold'))
    assert r['activation_prepared']==(mode=='compute_only')
    assert r['kernel']['weight_conversion_kernels']==1+policy
    assert r['kernel']['activation_conversion_kernels']==1
    assert set(r['timings_ms'])==set(stage_contract(mode,inner))
    assert all(len(v)==repeats and all(t>0 for t in v) for v in r['timings_ms'].values())
    if mode=='conversion_only':
        # C++ sums batched float32 samples, not the medians of different stages.
        for total,w,a in zip(r['timings_ms']['total'],r['timings_ms']['weight_conversion'],r['timings_ms']['activation_conversion']):
            assert abs(total-w-a)<1e-5*max(total,1e-6)


def validate(driver):
    import torch
    from adangel import _sm80 as native
    from adangel.quantization.arbitrary_bits import split_int8_to_packed_int4
    from adangel.quantization.mixed_formats import _pack_nibbles
    from roof_reduction_validation import reference_fp64
    from o3_fullk_probe import guard_columns
    checks=[];rejected=[]
    for m,n in ((64,128),(128,256)):
        for pattern in ('all_codes','zero','alternating','random','unsafe','mixed_tiles'):
            torch.manual_seed(62)
            a=((torch.arange(m*4096,device='cuda')*13)%256-128).to(torch.int8).reshape(m,4096)
            w=(torch.arange(n*2048,device='cuda')%256).byte().reshape(n,2048)
            asc=torch.linspace(0,.01,m,device='cuda')
            ws=((torch.arange(n*32,device='cuda')*7)%5+116).byte().reshape(n,32)
            if pattern=='zero':a.zero_();w.zero_()
            if pattern=='alternating':a[:,::2]=-128;a[:,1::2]=127;w.fill_(0x7f)
            if pattern=='random':
                a=torch.randint(-128,128,a.shape,device='cuda',dtype=torch.int8)
                w=torch.randint(0,256,w.shape,device='cuda',dtype=torch.uint8)
            if pattern in ('unsafe','mixed_tiles'):
                ws.fill_(116);ws[:,-1]=131
                if pattern=='mixed_tiles':ws[:128,-1]=116
            base=native.benchmark('o3','compute_only',a,asc,w,ws,0,1,2,'production',54,2)
            lut=torch.tensor([0,0,1,2,2,3,4,6,0,0,-1,-2,-2,-3,-4,-6],device='cuda',dtype=torch.int8)
            wq=torch.stack((lut[(w&15).long()],lut[(w>>4).long()]),-1).reshape(n,4096)
            pa=split_int8_to_packed_int4(a);pw=_pack_nibbles(wq.byte()&15)
            ref=reference_fp64('o3',(pa,asc,pw,ws))
            expected_a=pa.reshape(2,m,32,64).permute(0,2,1,3).contiguous()
            expected_w=pw.reshape(n,32,64).permute(1,0,2).contiguous()
            safe=guard_columns(ws.cpu().tolist())[1]['safe']
            for mode in MODES:
                for policy in (0,1):
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):r=driver.run_four(policy,mode,a,asc,w,ws,0,2,2)
                    stream.synchronize();timing_check(r,mode,2,2,policy)
                    assert torch.equal(r['packed_activation_g128_major'],expected_a)
                    assert torch.equal(r['packed_weight_g128_major'],expected_w)
                    assert torch.equal(r['converted_weight_scale'],ws.T.contiguous())
                    torch.testing.assert_close(r['output'].double(),ref,rtol=1e-3,atol=1e-3)
                    if not policy:assert torch.equal(r['output'],base['output'])
                    if policy:assert r['status']==(0 if safe else 1)
                    checks.append(dict(shape=[m,n,4096],pattern=pattern,mode=mode,policy=policy,
                        payload_exact=True,scale_exact=True,finite_fp32=True,semantic_tolerance_passed=True,status=r['status']))
    for policy in (0,1):
        for code in (0,255):
            bad=ws.clone();bad[0,0]=code
            try:driver.run_four(policy,'cold',a,asc,w,bad,0,1,2)
            except ValueError:rejected.append(dict(policy=policy,case='scale',code=code))
            else:raise AssertionError('invalid scale accepted')
        for operand in ('a','w'):
            original=a if operand=='a' else w
            odd=torch.empty(original.numel()+1,device='cuda',dtype=original.dtype)[1:].reshape(original.shape)
            odd.copy_(original)
            try:driver.run_four(policy,'cold',odd if operand=='a' else a,asc,odd if operand=='w' else w,ws,0,1,2)
            except RuntimeError:rejected.append(dict(policy=policy,case='alignment',operand=operand))
            else:raise AssertionError('unaligned vector read accepted')
    return dict(passed=True,checks=checks,rejected=rejected,
        scope='small MN/full K4096; nondefault stream; raw-source conversion + guarded GEMM; not 4096-cubed sanitizer')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=4);p.add_argument('--rounds',type=int,default=1)
    p.add_argument('--warmup',type=int,default=50);p.add_argument('--repeats',type=int,default=200)
    p.add_argument('--inner',type=int,default=100);p.add_argument('--validate-only',action='store_true')
    p.add_argument('--gemm-cubins',type=Path,default=Path('reports/o378_roof_v59'))
    p.add_argument('--device-cubin-dir',type=Path,default=Path('reports/o378_roof_v61'))
    p.add_argument('--prep-build',type=Path,default=Path('runs/o378_roof_v60_screen/build'))
    args=p.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT) or not 1<=args.samples<=24 or args.warmup<0 or min(args.repeats,args.inner,args.rounds)<1:
        p.error('fresh repository output and valid counts required')
    import torch
    from adangel import _sm80 as native
    from adangel.trace.storage import load_prepared,sha256_file
    from adangel.trace.prepare import _load_and_validate_raw
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.get_device_capability()==(8,0)
    context_anchor=torch.empty(1,device='cuda');args.output.mkdir(parents=True)
    def save(name,obj):(args.output/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    def append(name,obj):
        with (args.output/name).open('a') as f:f.write(json.dumps(obj,allow_nan=False)+'\n')
    built=build(args.output/'build',args.gemm_cubins,args.device_cubin_dir,args.prep_build)
    save('environment.json',dict(git_commit=command('git','rev-parse','HEAD'),extension_sha256=sha256_file(Path(native.__file__)),
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        policy='unlocked shared GPU; cyclic paired order; all raw samples retained',production_default_changed=False,
        scope='complete four-mode GPU timing from existing INT8/MXFP4-G128 inputs; initial source quantization excluded',
        timing_contract_version=2,conversion_source='unmodified roof_o3_conversion.cu, vectorized=true',
        guard_cost='candidate weight conversion includes GPU factor/status preparation; cached outside compute/steady',
        host_preflight='normal-scale/finite-row checks and post-batch status validation are outside GPU Event timing'))
    driver=Pipeline(*built)
    try:
        save('resources.json',driver.resources);save('validation.json',validate(driver))
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
            x=load_prepared(data/e['file'],device='cuda');raw=_load_and_validate_raw(rawdata/re['file'],re['layer'],re['projection'])
            verify_raw_prepared(x,(raw['activation_fp16'],raw['weight_fp16']))
            base=native.benchmark('o3','compute_only',x.A_int8,x.A_scale,x.W_mxfp4_g128,x.W_scale_g128,0,1,2,'production',54,2)
            o0=native.benchmark_o0(x.A_int8,x.A_scale,x.W_mxfp4,x.W_scale,'compute_only',0,1,2)['output']
            for ri in range(args.rounds):
                append('gpu_snapshots.jsonl',dict(sample_id=x.sample_id,round=ri,time=time.time(),
                    gpu=command('nvidia-smi','--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu','--format=csv')))
                for mi,mode in enumerate(MODES):
                    for policy in order([0,1],si,mi,ri):
                        r=driver.run_four(policy,mode,x.A_int8,x.A_scale,x.W_mxfp4_g128,x.W_scale_g128,args.warmup,args.repeats,args.inner)
                        timing_check(r,mode,args.inner,args.repeats,policy)
                        equal=bool(torch.equal(r['output'].view(torch.int32),base['output'].view(torch.int32)))
                        payload=all(torch.equal(r[key],base[key]) for key in
                            ('packed_activation_g128_major','packed_weight_g128_major','converted_weight_scale'))
                        assert equal and payload and (not policy or r['status']==0)
                        summaries={s:stats(v) for s,v in r['timings_ms'].items()}
                        selected='gemm' if mode=='compute_only' else 'total'
                        row=dict(sample_id=x.sample_id,variant='o3',round=ri,mode=mode,implementation=policy,
                            raw_ms=r['timings_ms'],stage_summaries=summaries,summary=summaries[selected],selected_stage=selected,
                            total_timing=r['total_timing'],stage_timing_inner_repeats=r['stage_timing_inner_repeats'],
                            weight_cached=r['weight_cached'],activation_prepared=r['activation_prepared'],kernel=r['kernel'],
                            guard_status=r['status'],payload_bitwise=payload,bitwise_equal_current_best=equal,
                            mse_vs_current_best=mse(r['output'],base['output']),mse_vs_paired_fp16=mse(r['output'],o0),paired_reference='o0')
                        rows.append(row);append('results.jsonl',row)
            print(x.sample_id,'complete',flush=True)
        save('summary.json',dict(records=summarize(rows),production_default_changed=False))
    finally:driver.close()


if __name__=='__main__':main()
