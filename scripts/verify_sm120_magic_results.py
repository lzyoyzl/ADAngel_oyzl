#!/usr/bin/env python3
"""Verify complete post-sync formal runs; never filter or rewrite measurements."""
import argparse
import itertools
import json
import math
from pathlib import Path


def verify(paired, four_modes, safety_run, audit_path):
    load = lambda p: json.loads(p.read_text())
    safety = load(safety_run / 'environment.json')
    audit = load(audit_path)
    assert audit['passed'] and audit['binary_sha256'] == safety['binary_sha256']
    implementations = ['o0', 'o1', 'o1_magic', 'o3', 'o3_magic']
    identities = {f'layer_{i:02d}_{p}' for i in [0, 6, 12, 18, 24, 31]
                  for p in ['q_proj', 'k_proj', 'v_proj', 'o_proj']}
    results = []
    common_manifest = None
    common_mse = {}
    for directory, modes, rounds in [
        (paired, ['compute_only'], 3),
        (four_modes, ['conversion_only', 'compute_only', 'cold', 'steady_state'], 1),
    ]:
        if directory is None:
            continue
        env = load(directory / 'environment.json')
        for field in ['binary_sha256', 'cuda_sources_sha256', 'torch', 'cuda', 'gpu']:
            assert env[field] == safety[field], (directory, field)
        args = env['args']
        assert (args['warmup'], args['repeats'], args['rounds'], args['inner'], args['samples']) == (50, 200, rounds, 100, 0)
        assert args['variants'] == ['o1', 'o3']
        manifest = load(directory / 'data_manifest.json')
        assert manifest['matrix_shape'] == [4096, 4096, 4096]
        assert len(manifest['samples']) == 24
        assert {s['sample_id'] for s in manifest['samples']} == identities
        if common_manifest is None:
            common_manifest = manifest
        assert manifest == common_manifest
        rows = [json.loads(line) for line in (directory / 'results.jsonl').read_text().splitlines() if line]
        expected = set(itertools.product(identities, implementations, modes, range(rounds)))
        found = set()
        cv_failures = []
        for r in rows:
            key = (r['sample_id'], r['implementation'], r['mode'], r['round'])
            assert key in expected and key not in found, ('duplicate/unexpected', key)
            found.add(key)
            _, impl, mode, _ = key
            variant = impl.split('_')[0]
            stages = {'total'}
            if mode != 'conversion_only': stages.add('gemm')
            if mode in ['conversion_only', 'cold']: stages.add('weight_conversion')
            if mode != 'compute_only' and variant in ['o0', 'o3']: stages.add('activation_conversion')
            assert set(r['timings_ms']) == set(r['summary']) == stages, key
            for stage, values in r['timings_ms'].items():
                assert len(values) == 200 and all(math.isfinite(v) and v > 0 for v in values), (key, stage)
                if r['summary'][stage]['cv_percent'] >= 3:
                    cv_failures.append([*key, stage, r['summary'][stage]['cv_percent']])
            timing = r['timing_method']
            assert timing['strategy'] == 'conversion_amortized_end_to_end_direct'
            assert timing['conversion_inner_repeats'] == 100
            assert timing['end_to_end_total_timing'] == 'direct_single_path'
            assert timing['mode_total_inner_repeats'] == (100 if mode == 'conversion_only' else 1)
            mse = r['mse_vs_o0']
            assert math.isfinite(mse) and mse >= 0
            mse_key = (r['sample_id'], variant)
            common_mse.setdefault(mse_key, mse)
            assert mse == common_mse[mse_key], ('MSE regression', key)
            if variant == 'o0':
                assert mse == 0
                continue
            assert r['bitwise_equal_baseline'] is True and mse == r['baseline_mse_vs_o0']
            kernel = r['kernel']
            assert kernel['scale_publication'] == 'per_writer_acquire_release_v3'
            assert kernel['full_barrier_arrivals_per_stage'] == 33
            assert kernel['integer_conversion'] == ('exact_magic_bias' if impl.endswith('_magic') else 'i2f')
            assert kernel['cta_tile'] == ([128, 64, 64] if variant == 'o1' else [64, 32, 128])
            assert kernel['output_dtype'] == 'fp32' and kernel['data_movement'] == 'TMA'
        assert found == expected, ('missing records', directory, len(found), len(expected))
        results.append(dict(run=str(directory), records=len(rows), samples=24,
                            cv_failed_stage_measurements=cv_failures))
    return dict(passed=True, scope='paired_only' if four_modes is None else 'paired_and_four_modes',
                binary_sha256=safety['binary_sha256'], runs=results,
                policy='All outliers retained; CV findings are disclosed, not filtered or relabeled')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paired', type=Path, required=True)
    scope = p.add_mutually_exclusive_group(required=True)
    scope.add_argument('--four-modes', type=Path)
    scope.add_argument('--paired-only', action='store_true',
                       help='Verify only completed primary pairing; does not certify four-mode coverage')
    p.add_argument('--safety-run', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): p.error('Use a new verification output file')
    result = verify(a.paired, a.four_modes, a.safety_run, a.audit)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
