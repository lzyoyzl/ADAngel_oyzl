#!/usr/bin/env python3
"""v70 single real-sample diagnostic; not a new kernel or Event performance run.

Use NCU's exact requested symbol and --launch-skip 50 --launch-count 1.
All preparation and reference kernels have other symbols. The one captured
launch uses the same audited GEMM and GPU-generated metadata as v69.
"""
import argparse
import json
from pathlib import Path

from benchmark_o78_fused_prepare import Case, Driver
from benchmark_o78_fullk_gpu_prepare import checked_gpu_build

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('o7', 'o8'), required=True)
    parser.add_argument('--policy', type=int, choices=(0, 1), required=True)
    parser.add_argument('--gpu-build', type=Path, required=True)
    parser.add_argument('--gemm-cubins', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT):
        parser.error('fresh repository output required')
    import torch
    from adangel import _sm80 as native
    from adangel.quantization import mixed_formats as mf
    from adangel.trace.storage import sha256_file, load_prepared
    from adangel.trace.prepare import _load_and_validate_raw
    from benchmark_a100_o1 import command
    from benchmark_a100_mixed import validate_fp16_result
    from benchmark_a100_mixed_trace import inspect_inputs, inspect_raw_inputs, verify_raw_prepared, source_identity, mse
    from roof_reduction_validation import mse_regression_ok
    torch.cuda.init()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.get_device_capability() == (8, 0)
    data = ROOT / 'data/prepared/llama2_7b_prefill_o0_o4'
    raw_data = ROOT / 'data/raw/llama2_7b_prefill'
    manifest, mh = inspect_inputs(data)
    raw, rh = inspect_raw_inputs(raw_data, manifest, ROOT / 'configs/trace/llama2_7b_prefill.yaml')
    entry = manifest['samples'][0]
    assert entry['sample_id'] == 'layer_00_q_proj'
    re = next(r for r in raw['samples'] if r['sample_id'] == entry['sample_id'])
    path = raw_data / re['file']
    assert sha256_file(path) == re['sha256']
    record = _load_and_validate_raw(path, re['layer'], re['projection'])
    prepared_path = data / entry['file']
    assert sha256_file(prepared_path) == entry['sha256']
    verify_raw_prepared(load_prepared(prepared_path, device='cpu'),
                       (record['activation_fp16'], record['weight_fp16']))
    wf, af = mf.VARIANTS[args.variant]
    ws = mf.quantize_source(record['weight_fp16'].cuda(), wf)
    acs = mf.quantize_source(record['activation_fp16'].cuda(), af)
    fp16 = native._benchmark_mixed(mf.PAIRED_BASELINE[args.variant], 'compute_only', ws, acs,
                                  0, 1, 2, '64x128x256', 'row_major')
    validate_fp16_result(fp16, ws, acs)
    old = native._benchmark_mixed(args.variant, 'compute_only', ws, acs, 0, 1, 2,
                                 '64x128x256', 'group_major', 59, 5)
    case = Case(args.variant, ws, acs, old)
    library, build = checked_gpu_build(args.gpu_build)
    driver = Driver(library, args.gemm_cubins)
    args.output.mkdir(parents=True)
    try:
        guard = driver.prepare(case)
        case.check_payload(old)
        assert guard['integer_ctas'] == guard['ctas'] == 2048
        assert guard['fallback_ctas'] == guard['invalid_ctas'] == 0
        # One policy only: exactly 50 matching warmups then one captured launch.
        output, _ = driver.run(case, args.policy, 'compute_only', 50, 1, 2)
        assert output.dtype == torch.float32 and bool(torch.isfinite(output).all())
        torch.testing.assert_close(output, old['output'], rtol=1e-3, atol=1e-3)
        if args.policy == 0:
            assert torch.equal(output.view(torch.int32), old['output'].view(torch.int32))
        error = mse(output, fp16['output'])
        assert mse_regression_ok(error, mse(old['output'], fp16['output']))
        result = dict(scope='one_real_sample_NCU_diagnostic_not_Event_performance',
            git_commit=command('git', 'rev-parse', 'HEAD'), variant=args.variant, policy=args.policy,
            sample_id=entry['sample_id'], shape=[4096,4096,4096], filtered_launch_skip=50,
            filtered_launch_count=1, expected_kernel='adangel_roof_o78_fullk_' + ('candidate' if args.policy else 'control'),
            raw_manifest_sha256=rh, prepared_manifest_sha256=mh, raw_sample_sha256=re['sha256'],
            source_provenance=dict(weight=source_identity(ws), activation=source_identity(acs)),
            extension_sha256=sha256_file(Path(native.__file__)), gemm_codegen=driver.codegen,
            preparation_build=build, resources=driver.resources[args.policy], guard=guard,
            numerical_checks_passed=True, paired_fp16=mf.PAIRED_BASELINE[args.variant],
            mse_vs_paired_fp16=error, mse_vs_best=mse(output, old['output']),
            max_abs_vs_best=(output-old['output']).abs().max().item(),
            production_default_changed=False)
        (args.output/'receipt.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(args.variant, args.policy, 'profile target numerical checks passed', flush=True)
    finally:
        driver.close()


if __name__ == '__main__':
    main()
