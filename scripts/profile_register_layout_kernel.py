#!/usr/bin/env python3
"""Existing v78/v85 O7 kernels, one real input, no benchmark or new candidate."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SYMBOLS=('adangel_roof_o78_eight_chain_candidate','adangel_roof_o78_register_layout_candidate')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=int,choices=(0,1),required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT):p.error('fresh repository output required')
    import torch
    from adangel import _sm80 as native
    from adangel.quantization import mixed_formats as mf
    from adangel.trace.storage import sha256_file,load_prepared
    from adangel.trace.prepare import _load_and_validate_raw
    from benchmark_a100_o1 import command
    from benchmark_a100_mixed_trace import inspect_inputs,inspect_raw_inputs,verify_raw_prepared,mse,source_identity
    from benchmark_a100_mixed import validate_fp16_result
    from benchmark_o78_register_layout import Driver
    from benchmark_o78_fused_prepare import Case
    from benchmark_o78_fullk_gpu_prepare import checked_gpu_build
    from roof_reduction_validation import mse_regression_ok
    torch.cuda.init();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.get_device_capability()==(8,0)
    context_anchor=torch.empty(1,device='cuda')
    data=ROOT/'data/prepared/llama2_7b_prefill_o0_o4'
    rawdata=ROOT/'data/raw/llama2_7b_prefill'
    manifest,mh=inspect_inputs(data)
    rawmanifest,rh=inspect_raw_inputs(rawdata,manifest,ROOT/'configs/trace/llama2_7b_prefill.yaml')
    e=manifest['samples'][0];assert e['sample_id']=='layer_00_q_proj'
    re=next(r for r in rawmanifest['samples'] if r['sample_id']==e['sample_id'])
    assert sha256_file(data/e['file'])==e['sha256'] and sha256_file(rawdata/re['file'])==re['sha256']
    raw=_load_and_validate_raw(rawdata/re['file'],re['layer'],re['projection'])
    prepared=load_prepared(data/e['file'],device='cpu')
    verify_raw_prepared(prepared,(raw['activation_fp16'],raw['weight_fp16']));del prepared
    directory=ROOT/'reports/o378_roof_v85_codegen_checked'
    library,build=checked_gpu_build(directory)
    driver=Driver(library,ROOT/'reports/o378_roof_v67_codegen',directory)
    try:
        wf,af=mf.VARIANTS['o7']
        ws=mf.quantize_source(raw['weight_fp16'].cuda(),wf)
        acs=mf.quantize_source(raw['activation_fp16'].cuda(),af)
        fp=native._benchmark_mixed('o5','compute_only',ws,acs,0,1,2,'64x128x256','row_major')
        validate_fp16_result(fp,ws,acs)
        old=native._benchmark_mixed('o7','compute_only',ws,acs,0,1,2,'64x128x256','group_major',59,5)
        case=Case('o7',ws,acs,old);guard=driver.prepare(case);case.check_payload(old)
        assert guard['integer_ctas']==guard['ctas']==2048
        assert guard['fallback_ctas']==guard['invalid_ctas']==0
        ref,_=driver.run(case,2,'compute_only',0,1,2);ref=ref.clone()
        out,_=driver.run(case,a.policy,'compute_only',50,1,2)
        assert out.dtype==torch.float32 and bool(torch.isfinite(out).all())
        assert torch.equal(out.view(torch.int32),ref.view(torch.int32))
        assert mse_regression_ok(mse(out,fp['output']),mse(old['output'],fp['output']))
        receipt=dict(scope='one_sample_existing_layout_NCU_not_Event_speedup',variant='o7',policy=a.policy,
            expected_kernel=SYMBOLS[a.policy],sample_id=e['sample_id'],shape=[4096]*3,
            filtered_launch_skip=50,filtered_launch_count=1,raw_manifest_sha256=rh,
            prepared_manifest_sha256=mh,raw_sample_sha256=re['sha256'],prepared_sample_sha256=e['sha256'],
            git_commit=command('git','rev-parse','HEAD'),extension_sha256=sha256_file(Path(native.__file__)),
            profiler_script_sha256=sha256_file(Path(__file__)),torch=torch.__version__,cuda=torch.version.cuda,
            production_default_changed=False,new_performance_result=False,resources=driver.resources[a.policy],
            guard=guard,preparation_build=build,paired_fp16='o5',
            source_provenance=dict(weight=source_identity(ws),activation=source_identity(acs)),
            gemm_codegen=json.loads((directory/'codegen.json').read_text()),
            numerical_checks_passed=True,bitwise_previous_fullk=True,mse_vs_paired_fp16=mse(out,fp['output']),
            mse_vs_previous_fullk=mse(out,ref),packed_candidate_verified=bool(a.policy==0 or case.register_layout_verified))
        a.output.mkdir(parents=True)
        (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
        print('REGISTER LAYOUT PROFILE NUMERICAL CHECKS PASSED',flush=True)
    finally:driver.close()


if __name__=='__main__':main()
