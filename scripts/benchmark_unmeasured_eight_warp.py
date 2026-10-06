#!/usr/bin/env python3
"""v104: one full24 cached-GEMM runtime completion for exact unlaunched v98.

Not a new geometry, parameter sweep or repetition of an existing GPU test.
Do not change the immutable v98 failed static heuristic. Shared preparation,
input, scales and one host Event loop for both128/256-thread kernels.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from validate_unmeasured_eight_warp import Driver,ROOT,resources_gate,o3_case,summarize


def tensor_pointers_o78(case):
    return tuple(case.state[key] for key in ('a','w','as','ws','af','wf','ab','wb','status','y'))


def numerical_pair(driver,kind,tensors,m,n):
    import torch
    old,_=driver.run(kind,0,tensors,m,n,0,1);old=old.clone()
    new,_=driver.run(kind,1,tensors,m,n,0,1)
    if old.dtype!=torch.float32 or not bool(torch.isfinite(old).all()) or not bool(torch.isfinite(new).all()):
        raise ValueError('nonfinite or non-FP32 candidate')
    if not torch.equal(old.view(torch.int32),new.view(torch.int32)):
        raise ValueError('exact v98 output differs from current best')
    return old


def validate(driver,prep,native):
    import torch
    from types import SimpleNamespace
    from adangel.quantization.mxfp4 import mxfp4_to_q4_packed
    from adangel.quantization import mixed_formats as mf
    from benchmark_o78_fused_prepare import Case
    checks=[]
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for pattern in ('random','zero','saturated','fallback'):
            m,n=(576,384) if pattern=='random' else (64,128)
            torch.manual_seed(20261007)
            a=torch.randint(-128,128,(m,4096),device='cuda',dtype=torch.int8)
            w=torch.randint(0,256,(n,2048),device='cuda',dtype=torch.uint8)
            asc=torch.linspace(.001,.01,m,device='cuda')
            ws=torch.full((n,32),116,device='cuda',dtype=torch.uint8)
            if pattern=='zero':a.zero_();w.zero_()
            if pattern=='saturated':a[:,::2]=-128;a[:,1::2]=127
            if pattern=='fallback':ws[:,0]=90
            x=SimpleNamespace(A_int8=a,A_scale=asc,W_mxfp4_g128=w,W_scale_g128=ws,
                W_q4=mxfp4_to_q4_packed(w))
            tensors,guard,base=o3_case(x,native)
            old=numerical_pair(driver,'o3',tensors,m,n)
            torch.testing.assert_close(old,base['output'],rtol=1e-3,atol=1e-3)
            checks.append(dict(variant='o3',pattern=pattern,shape=[m,n,4096],nondefault_stream=True,
                finite_fp32=True,bitwise_current_best=True,production_semantics_close=True,guard=guard))
        for variant,(wf,af) in mf.VARIANTS.items():
            for pattern in ('random','zero','saturated','fallback'):
                m,n=(128,256) if pattern in ('random','fallback') else (64,128)
                torch.manual_seed(20261007)
                a=(torch.randn(m,4096,device='cuda')*.4).half()
                w=(torch.randn(n,4096,device='cuda')*.1).half()
                if pattern=='zero':a.zero_();w.zero_()
                if pattern=='saturated':
                    a[:,::2]=-8;a[:,1::2]=7;w[:,::2]=-7;w[:,1::2]=6
                ws,asrc=mf.quantize_source(w,wf),mf.quantize_source(a,af)
                if pattern=='fallback':
                    if variant=='o7':asrc['scale'].fill_(127);asrc['scale'][:,-1]=159
                    else:ws['scale'].fill_(1);ws['scale'][:,-1]=192
                base=native._benchmark_mixed(variant,'compute_only',ws,asrc,0,1,2,
                    '64x128x256','group_major',59,5)
                case=Case(variant,ws,asrc,base)
                guard=prep.prepare(case)
                if guard['invalid_ctas']:raise ValueError('invalid validation source')
                old=numerical_pair(driver,'o78',tensor_pointers_o78(case),m,n)
                torch.testing.assert_close(old,base['output'],rtol=1e-3,atol=1e-3)
                checks.append(dict(variant=variant,pattern=pattern,shape=[m,n,4096],nondefault_stream=True,
                    finite_fp32=True,bitwise_current_best=True,production_semantics_close=True,guard=guard))
    stream.synchronize()
    return dict(passed=True,checks=checks,count=len(checks),
        scope='small_MN_fullK_correctness_only_not_performance_screen_or_sanitizer')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('data/prepared/llama2_7b_prefill_o0_o4'))
    p.add_argument('--raw-data',type=Path,default=Path('data/raw/llama2_7b_prefill'))
    p.add_argument('--trace-config',type=Path,default=Path('configs/trace/llama2_7b_prefill.yaml'))
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--warmup',type=int,default=50)
    p.add_argument('--repeats',type=int,default=200)
    args=p.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT) or min(args.rounds,args.repeats)<1 or args.warmup<0:
        p.error('fresh project output and valid counts required')
    import torch
    from adangel import _sm80 as native
    from adangel.trace.storage import load_prepared,sha256_file
    from adangel.trace.prepare import _load_and_validate_raw
    from adangel.quantization import mixed_formats as mf
    from benchmark_a100_mixed_trace import inspect_inputs,inspect_raw_inputs,source_identity,mse
    from benchmark_a100_roof_trace import measurement_order
    from benchmark_a100_o1 import stats,command
    from benchmark_o78_row_fused import Driver as PrepDriver
    from benchmark_o78_fullk_gpu_prepare import checked_gpu_build
    from benchmark_o78_fused_prepare import Case
    torch.cuda.init();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    context_anchor=torch.empty(1,device='cuda')
    if torch.cuda.get_device_capability()!=(8,0):raise ValueError('A100 SM80 required')
    extension_hash=sha256_file(Path(native.__file__))
    driver=Driver(args.output/'build',{'o3':ROOT/'reports/o378_roof_v98_o3_codegen',
                                     'o78':ROOT/'reports/o378_roof_v98_o78_codegen'})
    lib,preparation_receipt=checked_gpu_build(ROOT/'reports/o378_roof_v73_codegen')
    prep=PrepDriver(lib,ROOT/'reports/o378_roof_v67_codegen')
    def save(name,value):(args.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    def append(name,value):
        with (args.output/name).open('a') as out:out.write(json.dumps(value,allow_nan=False)+'\n')
    try:
        if not resources_gate(driver.resources):raise ValueError('actual capacity gain missing')
        save('validation.json',validate(driver,prep,native))
        manifest,prepared_hash=inspect_inputs(args.data)
        raw,raw_hash=inspect_raw_inputs(args.raw_data,manifest,args.trace_config)
        raw_index={r['sample_id']:r for r in raw['samples']}
        old_sources_path=ROOT/'docs/evidence/a100_o378_roof_v99/runs/o378_roof_v99_o78/source_provenance.jsonl'
        old_sources={(r['sample_id'],r['variant']):r for r in map(json.loads,old_sources_path.read_text().splitlines())}
        if len(old_sources)!=48:raise ValueError('full24 source authority missing')
        save('environment.json',dict(git_commit=command('git','rev-parse','HEAD'),
            extension_sha256=extension_hash,prepared_manifest_sha256=prepared_hash,raw_manifest_sha256=raw_hash,
            torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
            resources=driver.receipt['resources'],original_codegen_receipts=driver.receipts,
            preparation_receipt=preparation_receipt,source_provenance_sha256=sha256_file(old_sources_path),
            sources={str(s.relative_to(ROOT)):sha256_file(s) for s in
                (Path(__file__),ROOT/'scripts/validate_unmeasured_eight_warp.py',ROOT/'csrc/sm80/roof_latency_validation_driver.cpp')},
            rationale=driver.receipt['rationale'],modes=['compute_only'],timing_contract_version=2,
            source_quantization='original_FP16_direct_source_quantization_excluded',
            conversion_excluded=True,metadata_prepared_once_shared=True,O3_guard='existing_CPU_conservative_column_guard_cached',
            native_extension_rebuilt=False,production_default_changed=False,no_filtering=True,
            args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}))
        rows=[]
        for si,entry in enumerate(manifest['samples']):
            sid=entry['sample_id'];re=raw_index[sid]
            x=load_prepared(args.data/entry['file'],device='cuda')
            record=_load_and_validate_raw(args.raw_data/re['file'],re['layer'],re['projection'])
            for vi,variant in enumerate(('o3','o7','o8')):
                append('gpu_snapshots.jsonl',dict(sample_id=sid,variant=variant,time=time.time(),
                    snapshot=command('nvidia-smi','--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu','--format=csv')))
                if variant=='o3':
                    tensors,guard,base=o3_case(x,native);kind='o3'
                    reference=native.benchmark('o0','compute_only',x.A_int8,x.A_scale,x.W_mxfp4,x.W_scale,0,1,100,'production')['output']
                    m,n,_=x.shape
                else:
                    wf,af=mf.VARIANTS[variant]
                    ws=mf.quantize_source(record['weight_fp16'].cuda(),wf)
                    asrc=mf.quantize_source(record['activation_fp16'].cuda(),af)
                    identity=dict(sample_id=sid,variant=variant,raw_sha256=re['sha256'],
                        weight=source_identity(ws),activation=source_identity(asrc))
                    if identity!=old_sources[sid,variant]:raise ValueError('full source differs from v99')
                    append('source_provenance.jsonl',identity)
                    base=native._benchmark_mixed(variant,'compute_only',ws,asrc,0,1,2,'64x128x256','group_major',59,5)
                    case=Case(variant,ws,asrc,base);guard=prep.prepare(case)
                    if guard['invalid_ctas']:raise ValueError('invalid real sample')
                    tensors=tensor_pointers_o78(case);kind='o78';m,n=case.m,case.n
                    reference=native._benchmark_mixed(mf.PAIRED_BASELINE[variant],'compute_only',ws,asrc,0,1,2,
                        '64x128x256','row_major')['output']
                expected=numerical_pair(driver,kind,tensors,m,n)
                expected_error=mse(expected,reference)
                for ri in range(args.rounds):
                    order=measurement_order((0,1),si,vi,ri)
                    for policy in order:
                        output,times=driver.run(kind,policy,tensors,m,n,args.warmup,args.repeats)
                        if not bool(torch.isfinite(output).all()) or not torch.equal(
                                output.view(torch.int32),expected.view(torch.int32)):
                            raise ValueError('finite FP32 / current-best bitwise regression')
                        error=mse(output,reference)
                        if error!=expected_error:raise ValueError('MSE changed for exact candidate')
                        row=dict(sample_id=sid,variant=variant,mode='compute_only',round=ri,policy=policy,
                            order=list(order),raw_ms=times,summary=stats(times),guard=guard,
                            finite_fp32=True,bitwise_equal_control=True,MSE_regression_passed=True,
                            mse_vs_reference=error,reference={'o3':'o0','o7':'o5','o8':'o6'}[variant],
                            resource=driver.resources[kind,policy],total_timing='single_execution_cuda_event',
                            source_prepared=True,conversion_included=False,original_v98_static_gate_unchanged=True)
                        rows.append(row);append('results.jsonl',row)
                print(sid,variant,'full24 paired progress',flush=True)
                if variant!='o3':del case,ws,asrc
                del tensors,base,expected,reference,output
            del x,record
        if sha256_file(Path(native.__file__))!=extension_hash:raise AssertionError('production extension changed')
        save('summary.json',summarize(rows))
        print(json.dumps(summarize(rows),indent=2),flush=True)
    finally:prep.close();driver.close()


if __name__=='__main__':main()
