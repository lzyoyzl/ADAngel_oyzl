#!/usr/bin/env python3
"""v101 full4096 real-sample warp timelines, not an optimization A/B result.

Capture all four warps in every CTA. Bracketed CUDA Event measurements quantify
instrumentation perturbation only. No new default, quantizer, reduction or
cross-CTA partial buffer. No small-shape performance screening.
"""
import argparse
import ctypes as ct
import json
from pathlib import Path
import subprocess
import numpy as np

from analyze_cta_timeline import analyze
from probe_cta_timeline_codegen import ROOT,CONFIG


class Driver:
    def __init__(self,directory,kind):
        from adangel.trace.storage import sha256_file
        self.cfg=CONFIG[kind];self.handles=[]
        receipt=json.loads((directory/'codegen.json').read_text())
        if receipt['kind']!=kind or not receipt['gate']['passed'] or not receipt['control_comparison']['passed']:
            raise ValueError('instrumentation compile/capacity gate failed')
        for name,digest in receipt['sources'].items():
            if sha256_file(ROOT/name)!=digest:raise ValueError('instrumentation source drift: '+name)
        for name,digest in receipt['artifact_sha256'].items():
            if sha256_file(directory/name)!=digest:raise ValueError('instrumentation artifact drift: '+name)
        self.codegen=receipt
        self.lib=ct.CDLL(str((directory/'libcta_timeline.so').resolve()))
        self.lib.roof_probe_error.restype=ct.c_char_p
        self.lib.roof_probe_open.argtypes=[ct.c_char_p,ct.c_char_p,ct.c_uint,ct.POINTER(ct.c_void_p)]
        self.lib.roof_probe_resources.argtypes=[ct.c_void_p,ct.POINTER(ct.c_int)]
        self.lib.roof_probe_close.argtypes=[ct.c_void_p]
        self.lib.roof_timeline_launch.argtypes=[ct.c_void_p,ct.POINTER(ct.c_uint64)]+[ct.c_int]*5+[
            ct.c_void_p,ct.POINTER(ct.c_float)]
        self.resources={}
        try:
            binary=directory/(kind+'_cta_timeline.cubin')
            self.control=self.open(binary,self.cfg['control'],self.cfg['shared'])
            self.instrumented=self.open(binary,self.cfg['symbol'],self.cfg['shared'])
            for name,h in (('control',self.control),('instrumented',self.instrumented)):
                values=(ct.c_int*4)();self.check(self.lib.roof_probe_resources(h,values))
                if values[0]>168 or values[2]!=128 or values[3]<3:
                    raise ValueError('actual function changed CTA capacity')
                self.resources[name]=dict(registers_per_thread=values[0],local_bytes=values[1],
                    threads=values[2],active_blocks_per_SM=values[3],shared_bytes=self.cfg['shared'])
        except Exception:self.close();raise

    def check(self,status):
        if status:raise RuntimeError(self.lib.roof_probe_error().decode())

    def open(self,binary,symbol,shared):
        h=ct.c_void_p()
        self.check(self.lib.roof_probe_open(str(binary.resolve()).encode(),symbol.encode(),shared,ct.byref(h)))
        self.handles.append(h);return h

    def launch(self,handle,parameters,grid,warmup=0,repeats=1):
        import torch
        argv=(ct.c_uint64*len(parameters))(*parameters);times=(ct.c_float*repeats)()
        self.check(self.lib.roof_timeline_launch(handle,argv,len(argv),*grid,warmup,repeats,
            torch.cuda.current_stream().cuda_stream,times))
        return list(times)

    def close(self):
        for h in self.handles:self.check(self.lib.roof_probe_close(h))
        self.handles=[]


def capture(driver,parameters,output_index,reference,paired_fp16,out,variant):
    import torch
    from benchmark_a100_o1 import stats,command
    from benchmark_a100_mixed_trace import mse
    m,n=reference.shape;ctas=(m//64)*(n//128)
    # 32 bytes/warp; tensor allocation is outside every measured interval.
    storage=torch.empty(m*n+ctas*4*8,device=reference.device,dtype=torch.float32)
    y=storage[:m*n].view(m,n)
    scratch=storage[m*n:]
    argv=list(parameters);argv[output_index]=storage.data_ptr()
    grid=(n//128,m//64)
    before=command('nvidia-smi','--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu','--format=csv')
    control_pre=driver.launch(driver.control,argv,grid,50,200)
    assert torch.equal(y.view(torch.int32),reference.view(torch.int32))
    timeline_events=[];analyses=[]
    for iteration in range(3):
        scratch.zero_()
        # Last warmup's data is overwritten by the ONE captured launch.
        timeline_events+=driver.launch(driver.instrumented,argv,grid,50,1)
        assert bool(torch.isfinite(y).all()) and torch.equal(y.view(torch.int32),reference.view(torch.int32))
        raw=scratch.view(torch.uint64).reshape(ctas,4,4).cpu().numpy()
        analysis=analyze(raw,sm_count=torch.cuda.get_device_properties(0).multi_processor_count)
        np.save(out/f'{variant}_timeline_{iteration}.npy',raw,allow_pickle=False)
        analyses.append(analysis)
    control_post=driver.launch(driver.control,argv,grid,50,200)
    assert torch.equal(y.view(torch.int32),reference.view(torch.int32))
    after=command('nvidia-smi','--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu','--format=csv')
    return dict(variant=variant,shape=[m,n,4096],scratch_bytes=scratch.numel()*4,
        captures=analyses,resources=driver.resources,all_instrumented_outputs_bitwise_previous_best=True,
        mse_vs_previous_best=0.0,mse_vs_paired_fp16=mse(y,paired_fp16),
        control_before_raw_ms=control_pre,control_before=stats(control_pre),
        instrumented_capture_Event_ms=timeline_events,control_after_raw_ms=control_post,control_after=stats(control_post),
        GPU_before=before,GPU_after=after,production_default_changed=False,new_best_result=False,
        scope='one_real_full4096_sample_diagnostic; Event_times_measure_instrumentation_perturbation_not_speedup')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=('o3','o7','o8'),required=True)
    p.add_argument('--codegen',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh repository output required')
    import torch
    from adangel import _sm80 as native
    from adangel.trace.storage import sha256_file,load_prepared
    from adangel.trace.prepare import _load_and_validate_raw
    from benchmark_a100_mixed_trace import inspect_inputs,inspect_raw_inputs,verify_raw_prepared
    from benchmark_a100_o1 import command
    torch.cuda.init();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    if torch.cuda.get_device_capability()!=(8,0):p.error('A100 SM80 only')
    context_anchor=torch.empty(1,device='cuda')
    extension=Path(native.__file__);extension_sha=sha256_file(extension)
    data=ROOT/'data/prepared/llama2_7b_prefill_o0_o4';rawdata=ROOT/'data/raw/llama2_7b_prefill'
    manifest,mh=inspect_inputs(data)
    rawmanifest,rh=inspect_raw_inputs(rawdata,manifest,ROOT/'configs/trace/llama2_7b_prefill.yaml')
    entry=manifest['samples'][0];assert entry['sample_id']=='layer_00_q_proj'
    re=next(x for x in rawmanifest['samples'] if x['sample_id']==entry['sample_id'])
    assert sha256_file(data/entry['file'])==entry['sha256'] and sha256_file(rawdata/re['file'])==re['sha256']
    raw=_load_and_validate_raw(rawdata/re['file'],re['layer'],re['projection'])
    prepared=load_prepared(data/entry['file'],device='cuda')
    verify_raw_prepared(prepared,(raw['activation_fp16'],raw['weight_fp16']))
    out.mkdir(parents=True)
    driver=Driver(a.codegen,'o3' if a.variant=='o3' else 'o78')
    try:
        if a.variant=='o3':
            from benchmark_o3_grouped_cta import build,Pipeline
            best=ROOT/CONFIG['o3']['baseline'];built=build(out/'preparation_build',best)
            pipeline=Pipeline(*built)
            try:
                source=(prepared.A_int8,prepared.A_scale,prepared.W_mxfp4_g128,prepared.W_scale_g128)
                r=pipeline.run_four(1,'compute_only',*source,0,1,2)
                assert r['status']==0
            finally:pipeline.close()
            m,n=r['output'].shape
            meta=torch.empty((33,n),device='cuda',dtype=torch.int32)
            status=torch.empty(n//128,device='cuda',dtype=torch.uint32)
            preparation=driver.open(built[1],'adangel_roof_factor_prepare',0)
            driver.launch(preparation,[r['converted_weight_scale'].data_ptr(),meta.data_ptr(),status.data_ptr(),n],(n//128,1))
            # PyTorch2.7 CUDA has no UInt32 any/or reduction. This host guard
            # check is outside all timed/captured launches; do not alter flags.
            assert not np.any(status.cpu().numpy()),'diagnostic selects sample with all integer CTA guards'
            fp=native.benchmark_o0(prepared.A_int8,prepared.A_scale,prepared.W_mxfp4,
                prepared.W_scale,'compute_only',0,1,2)['output']
            args=[r['packed_activation_g128_major'].data_ptr(),r['packed_weight_g128_major'].data_ptr(),
                prepared.A_scale.data_ptr(),r['converted_weight_scale'].data_ptr(),meta.data_ptr(),status.data_ptr(),
                0,m,n,4096]
            result=capture(driver,args,6,r['output'],fp,out,a.variant)
            result['guard']=dict(integer_ctas=2048,fallback_ctas=0,invalid_ctas=0)
        else:
            from adangel.quantization import mixed_formats as mf
            from benchmark_o78_eight_chain_probe import Driver as EightDriver
            from benchmark_o78_fused_prepare import Case
            from benchmark_o78_fullk_gpu_prepare import checked_gpu_build
            wf,af=mf.VARIANTS[a.variant]
            wsrc=mf.quantize_source(raw['weight_fp16'].cuda(),wf)
            asrc=mf.quantize_source(raw['activation_fp16'].cuda(),af)
            library,prep_receipt=checked_gpu_build(ROOT/'reports/o378_roof_v73_codegen')
            pipeline=EightDriver(library,ROOT/'reports/o378_roof_v67_codegen',ROOT/CONFIG['o78']['baseline'])
            try:
                fp=native._benchmark_mixed(mf.PAIRED_BASELINE[a.variant],'compute_only',wsrc,asrc,0,1,2,
                    '64x128x256','row_major')['output']
                old=native._benchmark_mixed(a.variant,'compute_only',wsrc,asrc,0,1,2,'64x128x256','group_major',59,5)
                case=Case(a.variant,wsrc,asrc,old);guard=pipeline.prepare(case);case.check_payload(old)
                assert guard['integer_ctas']==2048 and not guard['fallback_ctas'] and not guard['invalid_ctas']
                reference,_=pipeline.run(case,1,'compute_only',0,1,2);reference=reference.clone()
                parameters=[case.state[name].data_ptr() for name in ('a','w','as','ws','af','wf','ab','wb','status')]
                parameters.extend([0,case.m,case.n,4096])
                result=capture(driver,parameters,9,reference,fp,out,a.variant)
                result.update(guard=guard,preparation='v73_row_fused',preparation_build=prep_receipt)
            finally:pipeline.close()
        assert sha256_file(extension)==extension_sha
        result.update(sample_id=entry['sample_id'],raw_manifest_sha256=rh,prepared_manifest_sha256=mh,
            raw_sample_sha256=re['sha256'],prepared_sample_sha256=entry['sha256'],
            extension_sha256=extension_sha,source_commit=command('git','rev-parse','HEAD'),
            profiler_sha256=sha256_file(Path(__file__)),analyzer_sha256=sha256_file(ROOT/'scripts/analyze_cta_timeline.py'),
            codegen=driver.codegen)
        (out/'receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(variant=a.variant,bitwise_previous_best=True,resources=driver.resources,
            summaries=[{k:v for k,v in r.items() if k!='per_SM'} for r in result['captures']],
            capture_Event_ms=result['instrumented_capture_Event_ms'],
            control_before=result['control_before'],control_after=result['control_after']),indent=2),flush=True)
    finally:driver.close()


if __name__=='__main__':main()
