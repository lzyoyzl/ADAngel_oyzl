#!/usr/bin/env python3
"""v114: one current-v78 O8 real-sample diagnostic after50 warmups.

Application replay launches a fresh process per pass. Keep a unique receipt
for every process; never overwrite a previous pass or save new trace tensors.
This is not an Event benchmark, a small performance screen, or a new kernel.
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = 'layer_12_o_proj'
SYMBOL = 'adangel_roof_o78_eight_chain_candidate'


def select_entry(manifest):
    found = [r for r in manifest['samples'] if r['sample_id'] == SAMPLE]
    if len(found) != 1:
        raise ValueError('one exact representative sample required')
    return found[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    if not out.is_relative_to(ROOT) or not a.validation.resolve().is_relative_to(ROOT):
        p.error('repository-only paths required')
    import torch
    from adangel import _sm80 as native
    from adangel.quantization import mixed_formats as mf
    from adangel.trace.storage import sha256_file, load_prepared
    from adangel.trace.prepare import _load_and_validate_raw
    from benchmark_a100_mixed_trace import verify_raw_prepared, source_identity, mse
    from benchmark_a100_mixed import validate_fp16_result
    from benchmark_o78_eight_chain_probe import Driver, checked
    from benchmark_o78_fused_prepare import Case
    from benchmark_o78_fullk_gpu_prepare import checked_gpu_build
    from roof_reduction_validation import mse_regression_ok
    import subprocess

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.cuda.get_device_capability() != (8, 0):
        raise ValueError('A100 required')
    context_anchor = torch.empty(1, device='cuda')  # Keep the primary context alive.
    v = json.loads(a.validation.read_text())
    data = ROOT / 'data/prepared/llama2_7b_prefill_o0_o4'
    rawdata = ROOT / 'data/raw/llama2_7b_prefill'
    if not v['full24_hashes_checked'] or v['sample_id'] != SAMPLE:
        raise ValueError('full24 preflight receipt required')
    if (sha256_file(data / 'manifest.json') != v['prepared_manifest_sha256'] or
            sha256_file(rawdata / 'trace_manifest.json') != v['raw_manifest_sha256']):
        raise ValueError('manifest changed during application replay')
    entry = select_entry(json.loads((data / 'manifest.json').read_text()))
    re = select_entry(json.loads((rawdata / 'trace_manifest.json').read_text()))
    if sha256_file(data / entry['file']) != entry['sha256'] or sha256_file(rawdata / re['file']) != re['sha256']:
        raise ValueError('selected sample changed')
    record = _load_and_validate_raw(rawdata / re['file'], re['layer'], re['projection'])
    verify_raw_prepared(load_prepared(data / entry['file'], device='cpu'),
                       (record['activation_fp16'], record['weight_fp16']))
    wf, af = mf.VARIANTS['o8']
    ws = mf.quantize_source(record['weight_fp16'].cuda(), wf)
    acs = mf.quantize_source(record['activation_fp16'].cuda(), af)
    identity = dict(sample_id=SAMPLE, variant='o8', raw_sha256=re['sha256'],
                    weight=source_identity(ws), activation=source_identity(acs))
    if identity != v['v99_source']:
        raise ValueError('full source identity differs from measured v99')
    baseline = ROOT / 'reports/o378_roof_v67_codegen'
    candidate = ROOT / 'reports/o378_roof_v78_codegen'
    codegen = checked(candidate)
    library, preparation = checked_gpu_build(ROOT / 'reports/o378_roof_v73_codegen')
    extension_before = sha256_file(Path(native.__file__))
    fp = native._benchmark_mixed('o6', 'compute_only', ws, acs, 0, 1, 2,
                                 '64x128x256', 'row_major')
    validate_fp16_result(fp, ws, acs)
    old = native._benchmark_mixed('o8', 'compute_only', ws, acs, 0, 1, 2,
                                  '64x128x256', 'group_major', 59, 5)
    case = Case('o8', ws, acs, old)
    driver = Driver(library, baseline, candidate)
    try:
        guard = driver.prepare(case)
        case.check_payload(old)
        if guard['integer_ctas'] != 2048 or guard['fallback_ctas'] or guard['invalid_ctas']:
            raise ValueError('representative full-integer capture required')
        reference, _ = driver.run(case, 2, 'compute_only', 0, 1, 2)
        reference = reference.clone()
        # Only this symbol matches NCU: exactly50 warmups and1 target launch.
        output, _ = driver.run(case, 1, 'compute_only', 50, 1, 2)
        if output.dtype != torch.float32 or not torch.isfinite(output).all():
            raise ValueError('nonfinite profile output')
        if not torch.equal(output.view(torch.int32), reference.view(torch.int32)):
            raise ValueError('current best differs from original exact full-K reference')
        error = mse(output, fp['output'])
        if not mse_regression_ok(error, mse(reference, fp['output'])):
            raise ValueError('profile MSE regression')
        resources = driver.resources[1]
    finally:
        driver.close()
    if sha256_file(Path(native.__file__)) != extension_before:
        raise ValueError('production extension changed')
    receipt = dict(scope='one_real_sample_application_replay_not_Event_performance',
        pid=os.getpid(), sample_id=SAMPLE, variant='o8', shape=[4096]*3,
        expected_kernel=SYMBOL, filtered_launch_skip=50, filtered_launch_count=1,
        replay_mode='application', cache_control='none', clock_control='none',
        numerical_checks_passed=True, bitwise_previous_fullK=True, finite_fp32=True,
        mse_vs_paired_fp16=error, paired_fp16='o6', mse_vs_previous_fullK=mse(output, reference),
        source_identity=identity, guard=guard, resources=resources,
        gemm_binary_sha256=sha256_file(candidate / 'o78_eight_chain.cubin'),
        codegen=codegen, preparation_build=preparation,
        raw_manifest_sha256=v['raw_manifest_sha256'], prepared_manifest_sha256=v['prepared_manifest_sha256'],
        extension_sha256_before=extension_before, extension_sha256_after=extension_before,
        source_sha256=sha256_file(Path(__file__)),
        validation_sha256=sha256_file(a.validation),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        production_default_changed=False, new_GEMM_measured=False, new_candidate_implemented=False)
    directory = out / f'pass_{os.getpid()}'
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print('warm O8 profile pass identity/output checks passed', os.getpid(), flush=True)


if __name__ == '__main__':
    main()
