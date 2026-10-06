#!/usr/bin/env python3
"""Recompute v89 full-24 paired evidence; never replace/filter the first run."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from adangel.benchmark.metrics import bootstrap_median_ci

SAMPLES = {f'layer_{layer:02d}_{projection}_proj'
           for layer in (0, 6, 12, 18, 24, 31) for projection in ('q', 'k', 'v', 'o')}


def policy(row):
    return row.get('candidate', row.get('implementation'))


def summarize(rows, rounds, repeats):
    index = {(r['sample_id'], r['variant'], r['mode'], r['round'], policy(r)): r for r in rows}
    variants = sorted({r['variant'] for r in rows})
    modes = sorted({r['mode'] for r in rows})
    expected = {(sid, variant, mode, round_id, impl) for sid in SAMPLES for variant in variants
                for mode in modes for round_id in range(rounds) for impl in (0, 1)}
    if not rows or len(index) != len(rows) or set(index) != expected:
        raise ValueError('complete, non-duplicated all24 paired evidence required')
    for r in rows:
        equal = r.get('bitwise_equal_current_best', r.get('bitwise_equal_v67', False))
        if not equal or not r.get('payload_bitwise', r.get('metadata_exact', False)):
            raise ValueError('output or preparation equivalence failed')
        if r.get('mse_vs_current_best', r.get('mse_vs_v67', -1)) != 0:
            raise ValueError('changed output versus previous best')
        if not math.isfinite(r['mse_vs_paired_fp16']):
            raise ValueError('nonfinite MSE')
        for stage, values in r['raw_ms'].items():
            if len(values) != repeats or any(not math.isfinite(x) or x <= 0 for x in values):
                raise ValueError('incomplete or invalid raw timings')
            mean = statistics.fmean(values)
            calculated = dict(median_ms=statistics.median(values), mean_ms=mean,
                              cv_percent=statistics.pstdev(values) / mean * 100)
            for name, value in calculated.items():
                if not math.isclose(value, r['stage_summaries'][stage][name], rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError('stored statistic disagrees with raw timings')
        selected = 'gemm' if r['mode'] == 'compute_only' else 'total'
        if r['summary'] != r['stage_summaries'][selected]:
            raise ValueError('incorrect selected timing stage')
    result = []
    for variant in variants:
        for mode in modes:
            for impl in (0, 1):
                selected = [index[s, variant, mode, ri, impl] for s in sorted(SAMPLES) for ri in range(rounds)]
                latencies, ratios, errors = [], [], []
                for sid in sorted(SAMPLES):
                    group = [index[sid, variant, mode, ri, impl] for ri in range(rounds)]
                    latencies.append(statistics.median(r['summary']['median_ms'] for r in group))
                    ratios.append(statistics.median(index[sid, variant, mode, ri, 0]['summary']['median_ms'] /
                                                    index[sid, variant, mode, ri, impl]['summary']['median_ms']
                                                    for ri in range(rounds)))
                    mse_values = {index[sid, variant, md, ri, p]['mse_vs_paired_fp16']
                                  for md in modes for ri in range(rounds) for p in (0, 1)}
                    if len(mse_values) != 1:
                        raise ValueError('per-sample MSE changed between policies/rounds/modes')
                    errors.append(next(iter(mse_values)))
                result.append(dict(variant=variant, mode=mode, policy=impl, samples=24, records=len(selected),
                    median_ms=statistics.median(latencies), paired_speedup=statistics.median(ratios),
                    paired_speedup_ci95=list(bootstrap_median_ci(ratios, 10000, .95, 20261006)),
                    median_mse=statistics.median(errors), mean_mse=statistics.fmean(errors),
                    selected_cv_failed_records=sum(r['summary']['cv_percent'] >= 3 for r in selected),
                    any_stage_cv_failed_records=sum(any(x['cv_percent'] >= 3 for x in r['stage_summaries'].values())
                                                    for r in selected),
                    median_cv_percent=statistics.median(r['summary']['cv_percent'] for r in selected),
                    max_cv_percent=max(r['summary']['cv_percent'] for r in selected),
                    samples_faster_than_control=sum(x > 1 for x in ratios) if impl else 0,
                    bitwise_previous_best=True, no_filtering=True))
    return result


def read_run(directory):
    env = json.loads((directory / 'environment.json').read_text())
    args = env['args']
    if args['samples'] != 24 or env['production_default_changed']:
        raise ValueError('nonformal all24 candidate run required')
    rows = [json.loads(line) for line in (directory / 'results.jsonl').read_text().splitlines()]
    summary = summarize(rows, args['rounds'], args['repeats'])
    key = 'input_provenance.json' if (directory / 'input_provenance.json').exists() else 'environment.json'
    provenance = json.loads((directory / key).read_text())
    fingerprint = dict(extension=env['extension_sha256'], raw=provenance['raw_manifest_sha256'],
                       prepared=provenance['prepared_manifest_sha256'])
    if key == 'input_provenance.json':
        receipt = json.loads((directory / 'build/grouped_build.json').read_text())
        fingerprint['codegen'] = receipt['codegen']
        fingerprint['runtime_sources'] = receipt['runtime_sources']
    else:
        fingerprint['codegen'] = env['codegen']
    return dict(args=args, summary=summary, fingerprint=fingerprint,
                input_sha256={name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                              for name in ('results.jsonl', 'environment.json', 'validation.json')})


def compare(first, retry):
    if first['fingerprint'] != retry['fingerprint']:
        raise ValueError('kernel, runtime source, extension or input identity changed')
    for name in ('samples', 'rounds', 'repeats', 'inner'):
        if first['args'][name] != retry['args'][name]:
            raise ValueError('counts other than untimed warmup changed')
    old = {(r['variant'], r['mode'], r['policy']): r for r in first['summary']}
    new = {(r['variant'], r['mode'], r['policy']): r for r in retry['summary']}
    if set(old) != set(new):
        raise ValueError('variant/mode coverage changed')
    rows = []
    for key in sorted(old):
        a, b = old[key], new[key]
        if (a['median_mse'], a['mean_mse']) != (b['median_mse'], b['mean_mse']):
            raise ValueError('MSE changed across runs')
        rows.append(dict(variant=key[0], mode=key[1], policy=key[2], first=a, retry=b,
            latency_shift_percent=(b['median_ms'] / a['median_ms'] - 1) * 100,
            paired_gain_percent=(b['paired_speedup'] - 1) * 100,
            strict_cv_passed=b['any_stage_cv_failed_records'] == 0))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first', type=Path, required=True)
    p.add_argument('--retry', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT):
        p.error('fresh repository output file required')
    first, retry = read_run(args.first), read_run(args.retry)
    result = dict(first_run=str(args.first), retry_run=str(args.retry),
                  first_warmup=first['args']['warmup'], retry_warmup=retry['args']['warmup'],
                  no_raw_filtering=True, no_default_change=True,
                  conclusion_scope='independent full24 repeat, not best-of-many CV selection',
                  records=compare(first, retry), first_input_sha256=first['input_sha256'],
                  retry_input_sha256=retry['input_sha256'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for r in result['records']:
        print(r['variant'], r['policy'], 'paired_gain_percent=', round(r['paired_gain_percent'], 3),
              'CV_failed=', r['retry']['any_stage_cv_failed_records'], '/', r['retry']['records'])


if __name__ == '__main__':
    main()
