#!/usr/bin/env python3
"""Current-best O3 only: one cache-primed application-replay NCU launch.

No new GEMM, no performance screen, no production extension rebuild. The
runner builds the existing host/conversion adapter once before replay.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = 'layer_12_o_proj'
SYMBOL = 'adangel_roof_o3_grouped_cta_candidate'
WARMUP = 1000


def select_entry(manifest):
    rows = [r for r in manifest['samples'] if r['sample_id'] == SAMPLE]
    if len(rows) != 1:
        raise ValueError('exact representative sample required')
    return rows[0]


def encode_driver_args(args):
    library, prep, cubins, device, conversion, symbols, control, best = args
    return dict(library=str(library), prep=str(prep),
        cubins={str(k): str(v) for k, v in cubins.items()}, device=str(device),
        conversion=str(conversion), symbols=symbols, control=str(control), best=str(best))


def driver_files(args):
    return [Path(args[k]) for k in ('library', 'prep', 'device', 'conversion', 'control', 'best')] + [
        Path(v) for v in args['cubins'].values()]


def decode_driver_args(args, hashes):
    for path in driver_files(args):
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError('driver artifact outside repository')
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path)]:
            raise ValueError('driver artifact changed during application replay')
    return (Path(args['library']), Path(args['prep']),
        {int(k): Path(v) for k, v in args['cubins'].items()}, Path(args['device']),
        Path(args['conversion']), args['symbols'], Path(args['control']), Path(args['best']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not all(path.resolve().is_relative_to(ROOT) for path in (a.validation, a.output)):
        p.error('repository-only paths required')
    import torch
    from adangel import _sm80 as native
    from adangel.trace.storage import sha256_file, load_prepared
    from adangel.trace.prepare import _load_and_validate_raw
    from benchmark_a100_mixed_trace import verify_raw_prepared, mse
    from benchmark_o3_grouped_cta import Pipeline
    from probe_grouped_cta_codegen import checked

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.cuda.get_device_capability() != (8, 0):
        raise ValueError('A100 SM80 required')
    context_anchor = torch.empty(1, device='cuda')
    v = json.loads(a.validation.read_text())
    if not v['full24_hashes_checked'] or v['sample_id'] != SAMPLE:
        raise ValueError('full24 preflight receipt required')
    data = ROOT / 'data/prepared/llama2_7b_prefill_o0_o4'
    rawdata = ROOT / 'data/raw/llama2_7b_prefill'
    if (sha256_file(data / 'manifest.json') != v['prepared_manifest_sha256'] or
            sha256_file(rawdata / 'trace_manifest.json') != v['raw_manifest_sha256']):
        raise ValueError('manifest changed during replay')
    entry = select_entry(json.loads((data / 'manifest.json').read_text()))
    raw_entry = select_entry(json.loads((rawdata / 'trace_manifest.json').read_text()))
    for directory, row in ((data, entry), (rawdata, raw_entry)):
        if sha256_file(directory / row['file']) != row['sha256']:
            raise ValueError('selected sample changed')
    raw = _load_and_validate_raw(rawdata / raw_entry['file'], raw_entry['layer'], raw_entry['projection'])
    prepared = load_prepared(data / entry['file'], device='cuda')
    verify_raw_prepared(prepared, (raw['activation_fp16'], raw['weight_fp16']))
    directory = ROOT / 'reports/o378_roof_v89_o3_codegen'
    codegen = checked(directory, 'o3')
    extension_before = sha256_file(Path(native.__file__))
    if extension_before != v['extension_sha256']:
        raise ValueError('production extension changed')
    source = (prepared.A_int8, prepared.A_scale, prepared.W_mxfp4_g128, prepared.W_scale_g128)
    driver = Pipeline(*decode_driver_args(v['driver_args'], v['driver_sha256']))
    try:
        # v79 does not match the profiler's exact-v89 symbol filter.
        reference = driver.run_four(0, 'compute_only', *source, 0, 1, 2)['output']
        fp = native.benchmark_o0(prepared.A_int8, prepared.A_scale, prepared.W_mxfp4,
            prepared.W_scale, 'compute_only', 0, 1, 2)['output']
        result = driver.run_four(1, 'compute_only', *source, WARMUP, 1, 2)
        output = result['output']
        if result['status'] != 0:
            raise ValueError('representative capture must use full integer guard path')
        if output.dtype != torch.float32 or not bool(torch.isfinite(output).all()):
            raise ValueError('nonfinite profile output')
        if not torch.equal(output.view(torch.int32), reference.view(torch.int32)):
            raise ValueError('current best differs from prior exact full-K output')
        resources = driver.resources['v89']
        error = mse(output, fp)
    finally:
        driver.close()
    if sha256_file(Path(native.__file__)) != extension_before:
        raise ValueError('production extension changed during replay')
    receipt = dict(scope='current_best_O3_application_replay_not_Event_performance',
        pid=os.getpid(), sample_id=SAMPLE, variant='o3', shape=[4096]*3, expected_kernel=SYMBOL,
        policy=1, filtered_launch_skip=WARMUP, filtered_launch_count=1,
        replay_mode='application', cache_control='none', clock_control='none',
        numerical_checks_passed=True, bitwise_previous_best=True, finite_fp32=True,
        mse_vs_paired_fp16=error, paired_fp16='o0', mse_vs_previous_best=mse(output, reference),
        guard=dict(integer_ctas=2048, fallback_ctas=0, invalid_ctas=0), resources=resources,
        gemm_binary_sha256=sha256_file(directory / 'o3_grouped_cta.cubin'), gemm_codegen=codegen,
        raw_manifest_sha256=v['raw_manifest_sha256'], prepared_manifest_sha256=v['prepared_manifest_sha256'],
        raw_sample_sha256=raw_entry['sha256'], prepared_sample_sha256=entry['sha256'],
        extension_sha256_before=extension_before, extension_sha256_after=extension_before,
        source_sha256=sha256_file(Path(__file__)), validation_sha256=sha256_file(a.validation),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        production_default_changed=False, new_performance_result=False, new_candidate_implemented=False)
    out = a.output / f'pass_{os.getpid()}'
    out.mkdir(parents=True, exist_ok=False)
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print('current-best O3 warm profile pass identity/output passed', os.getpid(), flush=True)


if __name__ == '__main__':
    main()
