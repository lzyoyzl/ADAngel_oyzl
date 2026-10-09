#!/usr/bin/env python3
"""Release validation and 24-trace paired tests through the formal native API.

No filtering. Conversion is batched, E2E is direct; source quantization and
diagnostic copies are outside timing. Frozen best cubins are correctness-only
oracles, never used by the production implementation.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import statistics
import time

from benchmark_a100_o1 import command, stats
from benchmark_a100_mixed import MODES, validate_fp16_result
from benchmark_a100_mixed_trace import inspect_inputs, inspect_raw_inputs, verify_raw_prepared, mse
from roof_reduction_validation import mse_regression_ok, reference_fp64

ROOT=Path(__file__).resolve().parents[1]
FROZEN={
    'o3':('reports/o378_roof_v89_o3_codegen','o3_grouped_cta.cubin','adangel_roof_o3_grouped_cta_candidate',50688),
    'o78':('reports/o378_roof_v99_o78_codegen','o78_output_streaming.cubin','adangel_roof_o78_output_streaming_candidate',34304),
}


class Frozen:
    """Read-only launch of hash-checked accepted GEMM binaries on new payloads."""
    def __init__(self):
        self.lib=ct.CDLL('libcuda.so.1');self.handles={};self.modules=[]
        self.lib.cuModuleLoad.argtypes=[ct.POINTER(ct.c_void_p),ct.c_char_p]
        self.lib.cuModuleGetFunction.argtypes=[ct.POINTER(ct.c_void_p),ct.c_void_p,ct.c_char_p]
        self.lib.cuFuncSetAttribute.argtypes=[ct.c_void_p,ct.c_int,ct.c_int]
        self.lib.cuLaunchKernel.argtypes=[ct.c_void_p]+[ct.c_uint]*7+[ct.c_void_p,ct.c_void_p,ct.c_void_p]
        self.lib.cuModuleUnload.argtypes=[ct.c_void_p]
        for key,(directory,file,symbol,smem) in FROZEN.items():
            directory=ROOT/directory;receipt=json.loads((directory/'codegen.json').read_text())
            binary=directory/file
            if hashlib.sha256(binary.read_bytes()).hexdigest()!=receipt['cubin_sha256']:
                raise ValueError('frozen best binary drift')
            module=ct.c_void_p();function=ct.c_void_p()
            self.check(self.lib.cuModuleLoad(ct.byref(module),str(binary).encode()))
            self.modules.append(module)
            self.check(self.lib.cuModuleGetFunction(ct.byref(function),module,symbol.encode()))
            self.check(self.lib.cuFuncSetAttribute(function,8,smem))
            self.handles[key]=(function,smem)

    @staticmethod
    def check(code):
        if code:raise RuntimeError('CUDA driver error '+str(code))

    def output(self,variant,result,asc=None):
        import torch
        y=torch.empty_like(result['output']);m,n=y.shape
        if variant=='o3':
            tensors=[result['packed_activation_g128_major'],result['packed_weight_g128_major'],asc,
                result['converted_weight_scale'],result['factor_metadata'],result['guard_status'],y]
        else:
            state=result['prepared_state']
            tensors=[state[k] for k in ('pa','pw','as','ws','af','wf','ab','wb','status')]+[y]
        values=[ct.c_uint64(t.data_ptr()) for t in tensors]+[ct.c_int(x) for x in (m,n,4096)]
        pointers=(ct.c_void_p*len(values))(*(ct.addressof(x) for x in values))
        function,smem=self.handles['o3' if variant=='o3' else 'o78']
        self.check(self.lib.cuLaunchKernel(function,n//128,m//64,1,128,1,1,smem,
            torch.cuda.current_stream().cuda_stream,pointers,None))
        torch.cuda.current_stream().synchronize()
        return y

    def close(self):
        for module in self.modules:self.check(self.lib.cuModuleUnload(module))


def call(native,variant,mode,inputs,policy,warmup,repeats,inner):
    if variant=='o3':
        return native.benchmark(variant,mode,*inputs,warmup,repeats,inner,policy)
    return native._benchmark_mixed(variant,mode,*inputs,warmup,repeats,inner,
                                  '64x128x256','row_major',-1,0,policy)


def contract(result,mode,repeats,inner):
    from roof_full_pipeline_probe import stage_contract
    expected=stage_contract(mode,inner)
    assert dict(result['stage_timing_inner_repeats'])==expected
    assert set(result['timings_ms'])==set(expected)
    assert all(len(x)==repeats and all(t>0 for t in x) for x in result['timings_ms'].values())
    assert result['total_timing']==('sum_of_batched_stage_samples' if mode=='conversion_only' else 'single_execution_cuda_event')
    assert result['weight_cached']==(mode in ('compute_only','steady_state'))
    assert result['activation_prepared']==(mode=='compute_only')
    assert result['kernel']['production_default']


def mixed_payload(result,w,a):
    import numpy as np
    import torch
    from adangel.quantization import mixed_formats as mf
    from adangel.quantization.arbitrary_bits import split_int8_to_packed_int4
    from validate_a100_split_grouped import pack_q4
    from o78_fullk_integer_metadata import prepare_fullk_metadata
    from benchmark_o78_fullk_gpu_prepare import check_metadata_arrays
    aq,asc=mf.to_fixed_reference(a);wq,wsc=mf.to_fixed_reference(w)
    for got,want in zip(result['converted_activation'],(split_int8_to_packed_int4(aq),asc)):
        assert torch.equal(got.contiguous().view(torch.uint8),want.contiguous().view(torch.uint8))
    for got,want in zip(result['converted_weight'],(pack_q4(wq),wsc)):
        assert torch.equal(got.contiguous().view(torch.uint8),want.contiguous().view(torch.uint8))
    squares=lambda q:q.reshape(q.shape[0],32,128).long().square().sum(-1).cpu().numpy()
    an,wn=squares(aq),squares(wq);o7=w['format']=='nvfp4_g128'
    mult=float((w if o7 else a)['tensor_scale'].item())
    oracle=prepare_fullk_metadata(a['scale'].cpu().numpy(),w['scale'].cpu().numpy(),an,wn,
        activation_kind='ue8m0' if o7 else 'e4m3',weight_kind='e4m3' if o7 else 'e6m2',
        activation_base_multiplier=np.float32(4 if o7 else np.float32(mult)*np.float32(.25)),
        weight_base_multiplier=np.float32(mult if o7 else 1))
    st=result['prepared_state'];arrays={}
    for key in ('af','wf','ab','wb','an','wn','am','wm','ast','wst','status'):
        value=st[key].cpu().numpy()
        if key in ('ast','wst','status'):value=value.view(np.uint32)
        if key in ('an','wn'):value=value.view(np.uint64)
        if key=='status':value=value.reshape(-1)
        arrays[key]=value
    guard=check_metadata_arrays(oracle,arrays)
    assert np.array_equal(st['asq'].cpu().numpy(),an) and np.array_equal(st['wsq'].cpu().numpy(),wn)
    return guard


def validate(native,frozen):
    import torch
    from adangel.quantization import mixed_formats as mf
    checks=[];stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for pattern in ('random','zero','alternating','wide_scale'):
            torch.manual_seed(20261009)
            m,n,k=64,128,4096
            a=torch.randint(-128,128,(m,k),device='cuda',dtype=torch.int8)
            w=torch.randint(0,256,(n,k//2),device='cuda',dtype=torch.uint8)
            asc=torch.linspace(.001,.01,m,device='cuda');ws=torch.full((n,32),120,device='cuda',dtype=torch.uint8)
            if pattern=='zero':a.zero_();w.zero_()
            if pattern=='alternating':a[:,::2]=-128;a[:,1::2]=127
            if pattern=='wide_scale':ws[:,-1]=135
            cases={'o3':(a,asc,w,ws)}
            ar=(torch.randn(m,k,device='cuda')*.4).half();wr=(torch.randn(n,k,device='cuda')*.1).half()
            if pattern=='zero':ar.zero_();wr.zero_()
            if pattern=='alternating':ar[:,::2]=-8;ar[:,1::2]=7;wr[:,::2]=-7;wr[:,1::2]=6
            for variant,(wf,af) in mf.VARIANTS.items():
                ww,aa=mf.quantize_source(wr,wf),mf.quantize_source(ar,af)
                if pattern=='wide_scale':
                    if variant=='o7':aa['scale'].fill_(127);aa['scale'][:,-1]=159
                    else:ww['scale'].fill_(1);ww['scale'][:,-1]=192
                cases[variant]=(ww,aa)
            for variant,inputs in cases.items():
                old=call(native,variant,'compute_only',inputs,'legacy',0,1,2)
                if variant=='o3':values=(old['converted_activation'],asc,old['converted_weight'],ws)
                else:values=(*old['converted_activation'],*old['converted_weight'])
                reference=reference_fp64(variant,values)
                for mode in MODES:
                    result=call(native,variant,mode,inputs,'production',0,2,2)
                    contract(result,mode,2,2)
                    torch.testing.assert_close(result['output'].double(),reference,rtol=1e-3,atol=1e-3)
                    expected=frozen.output(variant,result,asc)
                    assert torch.equal(result['output'].view(torch.int32),expected.view(torch.int32))
                    guard=mixed_payload(result,*inputs) if variant!='o3' else dict(fallback_ctas=result['kernel']['fallback_tiles'])
                    checks.append(dict(variant=variant,pattern=pattern,mode=mode,nondefault_stream=True,
                        bitwise_frozen_best=True,semantic_tolerance=True,guard=guard))
        # Invalid codes must never expose an unwritten or stale output.
        rejected=[]
        for variant,inputs in cases.items():
            if variant=='o3':inputs[-1].fill_(255)
            else:inputs[0]['scale'].fill_(255)
            try:call(native,variant,'compute_only',inputs,'production',0,1,2)
            except (ValueError,RuntimeError):rejected.append(variant)
            else:raise AssertionError('invalid source accepted: '+variant)
        stream.synchronize()
    return dict(passed=True,checks=checks,invalid_codes_rejected=rejected,
                scope='small_MN_K4096_correctness_not_performance_screen')


def summary(rows):
    from adangel.benchmark.metrics import bootstrap_median_ci
    result=[]
    ids=sorted({r['sample_id'] for r in rows});rounds=sorted({r['round'] for r in rows})
    index={(r['sample_id'],r['variant'],r['mode'],r['round'],r['policy']):r for r in rows}
    for variant in ('o3','o7','o8'):
        for mode in MODES:
            stage='gemm' if mode=='compute_only' else 'total'
            for policy in ('legacy','production'):
                selected=[index[s,variant,mode,i,policy] for s in ids for i in rounds]
                latency=[statistics.median(index[s,variant,mode,i,policy]['stages'][stage]['median_ms'] for i in rounds) for s in ids]
                speed=[statistics.median(index[s,variant,mode,i,'legacy']['stages'][stage]['median_ms']/index[s,variant,mode,i,policy]['stages'][stage]['median_ms'] for i in rounds) for s in ids]
                errors=[index[s,variant,mode,0,policy]['mse'] for s in ids]
                result.append(dict(variant=variant,mode=mode,policy=policy,samples=len(ids),records=len(selected),
                    median_ms=statistics.median(latency),paired_speedup=statistics.median(speed),
                    speedup_ci95=list(bootstrap_median_ci(speed,10000,.95,20261009)),
                    mse_median=statistics.median(errors),mse_mean=statistics.mean(errors),
                    primary_cv_failed=sum(r['stages'][stage]['cv_percent']>=3 for r in selected),
                    any_stage_cv_failed=sum(any(v['cv_percent']>=3 for v in r['stages'].values()) for r in selected)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--warmup',type=int,default=1000)
    p.add_argument('--repeats',type=int,default=200)
    p.add_argument('--inner',type=int,default=100)
    args=p.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT):p.error('fresh repository output required')
    if min(args.rounds,args.repeats,args.inner)<1 or args.warmup<0:p.error('invalid counts')
    import torch
    from adangel import _sm80 as native
    from adangel.trace.storage import load_prepared,sha256_file
    from adangel.trace.prepare import _load_and_validate_raw
    from adangel.quantization import mixed_formats as mf
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.init()
    assert torch.cuda.get_device_capability()==(8,0)
    args.output.mkdir(parents=True)
    def save(file,value):(args.output/file).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    def append(file,value):
        with (args.output/file).open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')
    frozen=Frozen()
    try:
        save('environment.json',dict(commit=command('git','rev-parse','HEAD'),extension_sha256=sha256_file(Path(native.__file__)),
            gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
            settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            defaults_applied=True,no_filtering=True,clock_policy='unlocked',source_quantization_timed=False))
        save('validation.json',validate(native,frozen));print('FORMAL NATIVE CORRECTNESS PASSED',flush=True)
        if args.validate_only:return
        data=ROOT/'data/prepared/llama2_7b_prefill_o0_o4';rawdata=ROOT/'data/raw/llama2_7b_prefill'
        manifest,mh=inspect_inputs(data);raw,rh=inspect_raw_inputs(rawdata,manifest,ROOT/'configs/trace/llama2_7b_prefill.yaml')
        rawindex={r['sample_id']:r for r in raw['samples']};save('data_provenance.json',dict(prepared_sha256=mh,raw_sha256=rh))
        rows=[]
        for si,entry in enumerate(manifest['samples']):
            x=load_prepared(data/entry['file'],device='cuda');re=rawindex[entry['sample_id']]
            raw=_load_and_validate_raw(rawdata/re['file'],re['layer'],re['projection'])
            verify_raw_prepared(x,(raw['activation_fp16'],raw['weight_fp16']))
            for vi,variant in enumerate(('o3','o7','o8')):
                if variant=='o3':
                    inputs=(x.A_int8,x.A_scale,x.W_mxfp4_g128,x.W_scale_g128)
                    fp=native.benchmark_o0(x.A_int8,x.A_scale,x.W_mxfp4,x.W_scale,'compute_only',0,1,2)
                    reference_name='o0'
                else:
                    wf,af=mf.VARIANTS[variant]
                    inputs=(mf.quantize_source(raw['weight_fp16'].cuda(),wf),mf.quantize_source(raw['activation_fp16'].cuda(),af))
                    reference_name=mf.PAIRED_BASELINE[variant]
                    fp=native._benchmark_mixed(reference_name,'compute_only',*inputs,0,1,2)
                    validate_fp16_result(fp,*inputs)
                old=call(native,variant,'compute_only',inputs,'legacy',0,1,2)
                baseline_error=mse(old['output'],fp['output'])
                prime=call(native,variant,'compute_only',inputs,'production',0,1,2)
                expected=frozen.output(variant,prime,x.A_scale)
                assert torch.equal(prime['output'].view(torch.int32),expected.view(torch.int32))
                if variant!='o3':mixed_payload(prime,*inputs)
                error=mse(prime['output'],fp['output'])
                assert mse_regression_ok(error,baseline_error),(variant,error,baseline_error)
                for ri in range(args.rounds):
                    append('gpu_snapshots.jsonl',dict(sample_id=x.sample_id,variant=variant,round=ri,time=time.time(),
                        gpu=command('nvidia-smi','--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu','--format=csv')))
                    for mi,mode in enumerate(MODES):
                        order=('legacy','production') if (si+vi+ri+mi)%2==0 else ('production','legacy')
                        for policy in order:
                            result=call(native,variant,mode,inputs,policy,args.warmup,args.repeats,args.inner)
                            current=mse(result['output'],fp['output'])
                            if policy=='production':
                                contract(result,mode,args.repeats,args.inner)
                                assert torch.equal(result['output'].view(torch.int32),expected.view(torch.int32))
                                assert current==error
                            else:assert current==baseline_error
                            row=dict(sample_id=x.sample_id,variant=variant,mode=mode,round=ri,policy=policy,
                                execution_order=order,kernel=dict(result['kernel']),mse=current,reference=reference_name,
                                mse_regression_passed=True,bitwise_frozen_best=policy=='production',
                                raw_ms=dict(result['timings_ms']),stages={k:stats(v) for k,v in result['timings_ms'].items()},
                                total_timing=result['total_timing'],stage_timing_inner_repeats=dict(result['stage_timing_inner_repeats']))
                            rows.append(row);append('results.jsonl',row)
                print(x.sample_id,variant,'four modes passed',flush=True)
            save('summary.json',summary(rows))
        assert len(rows)==24*3*4*args.rounds*2
        save('completion.json',dict(passed=True,records=len(rows),samples=24,defaults_applied=True))
        print('PRODUCTION 24-TRACE ACCEPTANCE PASSED',flush=True)
    finally:frozen.close()


if __name__=='__main__':main()
