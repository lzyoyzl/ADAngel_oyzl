#!/usr/bin/env python3
"""v80: profile the already validated v78/v79 best kernels, not a new tune.

One real 4096^3 sample per variant, original quantization and GPU guards.
Exactly 50 launches of the filtered symbol precede one profiled invocation.
Profiling duration is never treated as an Event benchmark or a new speedup.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = {'o3': 'adangel_roof_o3_eight_chain_candidate',
           'o7': 'adangel_roof_o78_eight_chain_candidate',
           'o8': 'adangel_roof_o78_eight_chain_candidate'}


def kernel_symbol(variant, metadata_copy_candidate=False):
    if variant not in SYMBOLS or (metadata_copy_candidate and variant=='o3'):
        raise ValueError('unsupported profile variant/candidate')
    return 'adangel_roof_o78_warp_metadata_candidate' if metadata_copy_candidate else SYMBOLS[variant]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=tuple(SYMBOLS), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--metadata-copy-candidate', action='store_true')
    a = parser.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT) or (a.metadata_copy_candidate and a.variant=='o3'):
        parser.error('fresh repository output required')
    import torch
    from adangel import _sm80 as native
    from adangel.trace.storage import sha256_file, load_prepared
    from adangel.trace.prepare import _load_and_validate_raw
    from benchmark_a100_o1 import command
    from benchmark_a100_mixed_trace import inspect_inputs, inspect_raw_inputs, verify_raw_prepared, mse
    from roof_reduction_validation import mse_regression_ok
    torch.cuda.init(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.get_device_capability() == (8, 0)
    context_anchor = torch.empty(1, device='cuda')
    data = ROOT / 'data/prepared/llama2_7b_prefill_o0_o4'
    rawdata = ROOT / 'data/raw/llama2_7b_prefill'
    manifest, mh = inspect_inputs(data)
    rawmanifest, rh = inspect_raw_inputs(rawdata, manifest, ROOT / 'configs/trace/llama2_7b_prefill.yaml')
    e = manifest['samples'][0]
    assert e['sample_id'] == 'layer_00_q_proj'
    re = next(r for r in rawmanifest['samples'] if r['sample_id'] == e['sample_id'])
    assert sha256_file(data/e['file']) == e['sha256'] and sha256_file(rawdata/re['file']) == re['sha256']
    raw = _load_and_validate_raw(rawdata/re['file'], re['layer'], re['projection'])
    x = load_prepared(data/e['file'], device='cuda')
    verify_raw_prepared(x, (raw['activation_fp16'], raw['weight_fp16']))
    a.output.mkdir(parents=True)
    receipt = dict(scope='one_sample_best_kernel_NCU_not_Event_performance_or_new_candidate',
        variant=a.variant, expected_kernel=kernel_symbol(a.variant,a.metadata_copy_candidate),
        metadata_copy_candidate=a.metadata_copy_candidate, sample_id=e['sample_id'], shape=[4096]*3,
        filtered_launch_skip=50, filtered_launch_count=1, raw_manifest_sha256=rh,
        prepared_manifest_sha256=mh, raw_sample_sha256=re['sha256'], prepared_sample_sha256=e['sha256'],
        git_commit=command('git','rev-parse','HEAD'), extension_sha256=sha256_file(Path(native.__file__)),
        profiler_script_sha256=sha256_file(Path(__file__)), torch=torch.__version__, cuda=torch.version.cuda,
        production_default_changed=False, new_performance_result=False)
    if a.variant == 'o3':
        from benchmark_o3_eight_chain_probe import build, Pipeline
        directory = ROOT/'reports/o378_roof_v79_codegen'
        driver = Pipeline(*build(a.output/'build', directory))
        try:
            source = (x.A_int8, x.A_scale, x.W_mxfp4_g128, x.W_scale_g128)
            control = driver.run_four(0, 'compute_only', *source, 0, 1, 2)
            fp = native.benchmark_o0(x.A_int8,x.A_scale,x.W_mxfp4,x.W_scale,'compute_only',0,1,2)['output']
            result = driver.run_four(1, 'compute_only', *source, 50, 1, 2)
            assert result['status'] == control['status'] == 0
            assert all(torch.equal(result[k], control[k]) for k in
                ('packed_activation_g128_major','packed_weight_g128_major','converted_weight_scale'))
            out, reference = result['output'], control['output']
            receipt.update(resources=driver.resources['eight_chain'], paired_fp16='o0',
                guard=dict(integer_ctas=2048,fallback_ctas=0,invalid_ctas=0), preparation='conversion2_gpu_guard')
        finally:
            driver.close()
    else:
        from adangel.quantization import mixed_formats as mf
        from benchmark_a100_mixed import validate_fp16_result
        from benchmark_a100_mixed_trace import source_identity
        from benchmark_o78_eight_chain_probe import Driver
        from benchmark_o78_fused_prepare import Case
        from benchmark_o78_fullk_gpu_prepare import checked_gpu_build
        directory = ROOT/'reports/o378_roof_v78_codegen'
        if a.metadata_copy_candidate:
            from benchmark_o78_warp_metadata import Driver
            directory = ROOT/'reports/o378_roof_v81_codegen'
        library, preparation = checked_gpu_build(ROOT/'reports/o378_roof_v73_codegen')
        driver = Driver(library, ROOT/'reports/o378_roof_v67_codegen', directory)
        try:
            wf, af = mf.VARIANTS[a.variant]
            ws = mf.quantize_source(raw['weight_fp16'].cuda(),wf)
            acs = mf.quantize_source(raw['activation_fp16'].cuda(),af)
            fp_result = native._benchmark_mixed(mf.PAIRED_BASELINE[a.variant], 'compute_only',ws,acs,0,1,2,'64x128x256','row_major')
            validate_fp16_result(fp_result,ws,acs); fp = fp_result['output']
            old = native._benchmark_mixed(a.variant,'compute_only',ws,acs,0,1,2,'64x128x256','group_major',59,5)
            case = Case(a.variant,ws,acs,old)
            guard = driver.prepare(case); case.check_payload(old)
            assert guard['integer_ctas'] == guard['ctas'] == 2048
            assert guard['fallback_ctas'] == guard['invalid_ctas'] == 0
            reference, _ = driver.run(case,0,'compute_only',0,1,2)
            reference = reference.clone()
            out, _ = driver.run(case,1,'compute_only',50,1,2)
            assert mse_regression_ok(mse(out,fp),mse(old['output'],fp))
            receipt.update(resources=driver.resources[1], guard=guard, preparation_build=preparation,
                paired_fp16=mf.PAIRED_BASELINE[a.variant], preparation='v73_row_fused_gpu_guard',
                source_provenance=dict(weight=source_identity(ws),activation=source_identity(acs)))
        finally:
            driver.close()
    assert out.dtype == torch.float32 and bool(torch.isfinite(out).all())
    assert torch.equal(out.view(torch.int32), reference.view(torch.int32))
    receipt.update(gemm_codegen=json.loads((directory/'codegen.json').read_text()),
        numerical_checks_passed=True, bitwise_previous_fullk=True,
        mse_vs_paired_fp16=mse(out,fp), mse_vs_previous_fullk=mse(out,reference))
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(a.variant,'best-kernel profile numerical checks passed',flush=True)


if __name__ == '__main__': main()
