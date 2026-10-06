#!/usr/bin/env python3
"""Profile one selected v89/control symbol after50 warmups, not an Event result."""
import argparse
import json
from pathlib import Path

from probe_grouped_cta_codegen import ROOT, CONFIG, checked


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=('o3','o7','o8'),required=True)
    p.add_argument('--policy',type=int,choices=(0,1),required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT):p.error('fresh repository output required')
    import torch
    from adangel import _sm80 as native
    from adangel.trace.storage import sha256_file,load_prepared
    from adangel.trace.prepare import _load_and_validate_raw
    from benchmark_a100_o1 import command
    from benchmark_a100_mixed_trace import inspect_inputs,inspect_raw_inputs,verify_raw_prepared,mse
    torch.cuda.init();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.get_device_capability()==(8,0)
    context_anchor=torch.empty(1,device='cuda')
    kind='o3' if a.variant=='o3' else 'o78';cfg=CONFIG[kind]
    directory=ROOT/f'reports/o378_roof_v89_{kind}_codegen';codegen=checked(directory,kind)
    data=ROOT/'data/prepared/llama2_7b_prefill_o0_o4';rawdata=ROOT/'data/raw/llama2_7b_prefill'
    manifest,mh=inspect_inputs(data)
    rawmanifest,rh=inspect_raw_inputs(rawdata,manifest,ROOT/'configs/trace/llama2_7b_prefill.yaml')
    entry=manifest['samples'][0];assert entry['sample_id']=='layer_00_q_proj'
    raw_entry=next(x for x in rawmanifest['samples'] if x['sample_id']==entry['sample_id'])
    assert sha256_file(data/entry['file'])==entry['sha256'] and sha256_file(rawdata/raw_entry['file'])==raw_entry['sha256']
    raw=_load_and_validate_raw(rawdata/raw_entry['file'],raw_entry['layer'],raw_entry['projection'])
    prepared=load_prepared(data/entry['file'],device='cuda')
    verify_raw_prepared(prepared,(raw['activation_fp16'],raw['weight_fp16']))
    a.output.mkdir(parents=True)
    symbol=cfg['control'] if a.policy==0 else cfg['symbol']
    binary=ROOT/cfg['baseline']/cfg['cubin'] if a.policy==0 else directory/(cfg['stem']+'.cubin')
    receipt=dict(scope='one_real_sample_NCU_diagnostic_not_Event_performance',variant=a.variant,
        policy=a.policy,expected_kernel=symbol,sample_id=entry['sample_id'],shape=[4096]*3,
        # The single reference launch also matches the filter for policy0.
        filtered_launch_skip=51 if a.policy==0 else 50,filtered_launch_count=1,
        raw_manifest_sha256=rh,prepared_manifest_sha256=mh,raw_sample_sha256=raw_entry['sha256'],
        prepared_sample_sha256=entry['sha256'],git_commit=command('git','rev-parse','HEAD'),
        extension_sha256=sha256_file(Path(native.__file__)),profiler_script_sha256=sha256_file(Path(__file__)),
        gemm_binary_sha256=sha256_file(binary),gemm_codegen=codegen,
        production_default_changed=False,new_performance_result=False)
    if a.variant=='o3':
        from benchmark_o3_grouped_cta import build,Pipeline
        driver=Pipeline(*build(a.output/'build',directory))
        try:
            source=(prepared.A_int8,prepared.A_scale,prepared.W_mxfp4_g128,prepared.W_scale_g128)
            reference=driver.run_four(0,'compute_only',*source,0,1,2)['output']
            fp=native.benchmark_o0(prepared.A_int8,prepared.A_scale,prepared.W_mxfp4,
                prepared.W_scale,'compute_only',0,1,2)['output']
            result=driver.run_four(a.policy,'compute_only',*source,50,1,2)
            assert result['status']==0
            out=result['output']
            receipt.update(resources=driver.resources['v79' if a.policy==0 else 'v89'],
                paired_fp16='o0',preparation='conversion2_gpu_guard',
                guard=dict(integer_ctas=2048,fallback_ctas=0,invalid_ctas=0))
        finally:driver.close()
    else:
        from adangel.quantization import mixed_formats as mf
        from benchmark_o78_grouped_cta import Driver
        from benchmark_o78_fused_prepare import Case
        from benchmark_o78_fullk_gpu_prepare import checked_gpu_build
        wf,af=mf.VARIANTS[a.variant]
        ws,acs=mf.quantize_source(raw['weight_fp16'].cuda(),wf),mf.quantize_source(raw['activation_fp16'].cuda(),af)
        library,prep=checked_gpu_build(ROOT/'reports/o378_roof_v73_codegen')
        driver=Driver(library,ROOT/'reports/o378_roof_v67_codegen',directory)
        try:
            fp=native._benchmark_mixed(mf.PAIRED_BASELINE[a.variant],'compute_only',ws,acs,0,1,2,
                '64x128x256','row_major')['output']
            old=native._benchmark_mixed(a.variant,'compute_only',ws,acs,0,1,2,'64x128x256','group_major',59,5)
            case=Case(a.variant,ws,acs,old);guard=driver.prepare(case);case.check_payload(old)
            assert guard['integer_ctas']==2048 and not guard['fallback_ctas'] and not guard['invalid_ctas']
            reference,_=driver.run(case,0,'compute_only',0,1,2);reference=reference.clone()
            out,_=driver.run(case,a.policy,'compute_only',50,1,2)
            receipt.update(resources=driver.resources[a.policy],paired_fp16=mf.PAIRED_BASELINE[a.variant],
                guard=guard,preparation='v73_row_fused_gpu_guard',preparation_build=prep)
        finally:driver.close()
    assert out.dtype==torch.float32 and bool(torch.isfinite(out).all())
    assert torch.equal(out.view(torch.int32),reference.view(torch.int32))
    receipt.update(numerical_checks_passed=True,bitwise_previous_best=True,
        mse_vs_paired_fp16=mse(out,fp),mse_vs_previous_best=mse(out,reference))
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(a.variant,a.policy,'identity and numerical profile checks passed',flush=True)


if __name__=='__main__':main()
