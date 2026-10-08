#!/usr/bin/env python3
"""Recover v138 after SSH loss; preserve the original run, rerun incomplete samples."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MODES = ('conversion_only', 'compute_only', 'cold', 'steady_state')
COUNTS = dict(samples=24, rounds=3, warmup=1000, repeats=200, inner=100)


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return list(map(json.loads, path.read_text().splitlines()))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completed_prefix(rows, sample_ids):
    """Selection depends only on completeness, never timing/CV/performance."""
    if len(sample_ids) != 24 or len(set(sample_ids)) != 24:
        raise ValueError('24 unique ordered source identities required')
    expected = {(m, r, p) for m in MODES for r in range(3) for p in (0, 1)}
    by_id = {s: set() for s in sample_ids}
    for row in rows:
        s = row['sample_id']
        key = (row['mode'], row['round'], row['candidate'])
        if row['variant'] != 'o8' or s not in by_id or key not in expected or key in by_id[s]:
            raise ValueError('unknown/duplicate interrupted record')
        by_id[s].add(key)
    start = next((i for i, s in enumerate(sample_ids) if by_id[s] != expected), 24)
    if start == 24 or any(by_id[s] for s in sample_ids[start + 1:]):
        raise ValueError('expected a complete prefix, one interrupted sample and missing suffix')
    keep = set(sample_ids[:start])
    return start, [r for r in rows if r['sample_id'] in keep], [r for r in rows if r['sample_id'] not in keep]


def save(directory, name, obj, jsonl=False):
    with (directory / name).open('x') as out:
        if jsonl:
            for row in obj:
                out.write(json.dumps(row, allow_nan=False) + '\n')
        else:
            out.write(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def merge_runs(interrupted, resumed, combined, sample_ids):
    old_rows = lines(interrupted / 'results.jsonl')
    start, keep, incomplete = completed_prefix(old_rows, sample_ids)
    new_rows = lines(resumed / 'results.jsonl')
    if {r['sample_id'] for r in new_rows} != set(sample_ids[start:]):
        raise ValueError('resumed sample identities differ')
    old_env, new_env = read(interrupted / 'environment.json'), read(resumed / 'environment.json')
    for key in ('extension_sha256', 'codegen', 'gpu_preparation_build', 'resources', 'variants',
                'torch', 'cuda', 'gpu', 'prepared_manifest_sha256', 'raw_manifest_sha256',
                'production_default_changed', 'control', 'candidate', 'source_quantization',
                'timing_scope', 'no_filtering'):
        if old_env[key] != new_env[key]:
            raise ValueError('acquisition environment/implementation changed: ' + key)
    for env in (old_env, new_env):
        if any(env['args'][k] != v for k, v in COUNTS.items()) or not env['args']['full_modes']:
            raise ValueError('measurement protocol changed')
    if new_env['sample_start'] != start or old_env.get('sample_start', 0) != 0:
        raise ValueError('acquisition offset changed')
    validation = read(interrupted / 'validation.json')
    if read(resumed / 'validation.json') != validation:
        raise ValueError('preflight numerical/guard proof changed')
    prefix = set(sample_ids[:start])
    source = [r for r in lines(interrupted / 'source_provenance.jsonl') if r['sample_id'] in prefix]
    source += lines(resumed / 'source_provenance.jsonl')
    reference_path = ROOT / 'docs/evidence/a100_o378_roof_v99/runs/o378_roof_v99_o78/source_provenance.jsonl'
    reference = {r['sample_id']: r for r in lines(reference_path) if r['variant'] == 'o8'}
    if len(source) != 24 or {r['sample_id'] for r in source} != set(reference) or any(r != reference[r['sample_id']] for r in source):
        raise ValueError('source identity changed')
    snapshots = [r for r in lines(interrupted / 'gpu_snapshots.jsonl') if r['sample_id'] in prefix]
    snapshots += lines(resumed / 'gpu_snapshots.jsonl')
    if combined.exists() or not combined.resolve().is_relative_to(ROOT):
        raise ValueError('fresh combined repository output required')
    combined.mkdir(parents=True)
    receipt = dict(reason='SSH_connection_reset_and_process_confirmed_terminal',
        selection='retain_complete_sample_prefix_rerun_entire_incomplete_sample_and_suffix',
        not_CV_or_performance_selection=True, completed_samples=start, rerun_samples=sample_ids[start:],
        accepted_initial_records=len(keep), accepted_resumed_records=len(new_rows),
        interrupted_incomplete_records_retained_in_source=len(incomplete),
        initial_run=str(interrupted.relative_to(ROOT)), resumed_run=str(resumed.relative_to(ROOT)),
        initial_git_commit=old_env['git_commit'], resumed_git_commit=new_env['git_commit'],
        initial_results_sha256=sha(interrupted / 'results.jsonl'),
        resumed_results_sha256=sha(resumed / 'results.jsonl'),
        initial_environment_sha256=sha(interrupted / 'environment.json'),
        resumed_environment_sha256=sha(resumed / 'environment.json'),
        recovery_script_sha256=sha(Path(__file__)),
        common_protocol_sha256=sha(ROOT / 'scripts/benchmark_o78_coefficient_probe.py'),
        kernel_binaries_and_runtime_driver_unchanged=True)
    save(combined, 'recovery.json', receipt)
    save(combined, 'environment.json', dict(old_env, acquisition_recovery=receipt))
    save(combined, 'validation.json', validation)
    save(combined, 'results.jsonl', keep + new_rows, True)
    save(combined, 'source_provenance.jsonl', source, True)
    save(combined, 'gpu_snapshots.jsonl', snapshots, True)
    save(combined, 'source_identity_checked.json', dict(passed=True, samples=24, variant='o8',
        full_v99_source_identity_equal=True, old_provenance_sha256=sha(reference_path)))
    from analyze_hif4_swar import analyze
    analysis = analyze(combined)
    save(combined, 'summary.json', analysis)
    print(json.dumps({k: v for k, v in analysis.items() if k != 'conversion_compile_audit'}, indent=2))
    print('HIF4 RECOVERED FULL24 PASSED', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--interrupted', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--combined-output', type=Path, required=True)
    a = p.parse_args()
    interrupted, output, combined = (x.resolve() for x in (a.interrupted, a.output, a.combined_output))
    if any(not x.is_relative_to(ROOT) for x in (interrupted, output, combined)) or output.exists() or combined.exists():
        p.error('existing repository source and two fresh repository outputs required')
    env = read(interrupted / 'environment.json')
    if any(env['args'][k] != v for k, v in COUNTS.items()):
        p.error('original measurement protocol differs')
    # The paired driver revalidates all 24 prepared and raw input identities.
    manifest = read(ROOT / env['args']['data'] / 'manifest.json')
    sample_ids = [e['sample_id'] for e in manifest['samples']]
    start, _, _ = completed_prefix(lines(interrupted / 'results.jsonl'), sample_ids)
    from benchmark_o8_hif4_swar import Driver, timing_contract, validate
    from benchmark_o78_coefficient_probe import main as paired_main
    sys.argv = [sys.argv[0], '--output', str(output), '--full-modes']
    for key in ('gpu_build', 'baseline', 'cubins', 'data', 'raw_data', 'trace_config', *COUNTS):
        sys.argv += ['--' + key.replace('_', '-'), str(env['args'][key])]
    paired_main(driver_cls=Driver, labels=(env['control'], env['candidate']),
        experiment='HiF4_packed_conversion_not_GEMM', banner='HIF4 PACKED SWAR RECOVERY',
        contract=timing_contract, variants=('o8',), validation_fn=validate, sample_start=start)
    merge_runs(interrupted, output, combined, sample_ids)


if __name__ == '__main__':
    main()
