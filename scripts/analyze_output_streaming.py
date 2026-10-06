#!/usr/bin/env python3
"""Recompute complete v99 paired raw timings; do not discard CV/outlier records."""
import argparse
import hashlib
import json
from pathlib import Path

from analyze_grouped_cta_retest import summarize
from probe_output_streaming_codegen import ROOT, CONFIG


def analyze_run(directory, kind, four_modes=False):
    environment = json.loads((directory/'environment.json').read_text())
    args = environment['args']
    rounds = 1 if four_modes else 3
    if (args['samples'], args['rounds'], args['warmup'], args['repeats'], args['inner']) != (24,rounds,1000,200,100):
        raise ValueError('predeclared direct24 /1000 /200 /100 run required')
    if four_modes and (kind != 'o78' or not args.get('full_modes')):
        raise ValueError('independent O7/O8 four-mode confirmation required')
    if environment['production_default_changed']:
        raise ValueError('independent candidate only')
    rows = [json.loads(line) for line in (directory/'results.jsonl').read_text().splitlines()]
    if {r['variant'] for r in rows} != ({'o3'} if kind == 'o3' else {'o7','o8'}):
        raise ValueError('variant coverage changed')
    expected_modes = {'conversion_only', 'compute_only', 'cold', 'steady_state'} if four_modes else {'compute_only'}
    if {r['mode'] for r in rows} != expected_modes:
        raise ValueError('mode coverage changed')
    cfg = CONFIG[kind]
    for row in rows:
        policy = row.get('candidate', row.get('implementation'))
        if policy not in (0, 1):
            raise ValueError('policy0/1 required')
        # The established O3 schema stores bitwise equality/finite MSE, not
        # the explicit flags emitted by the O7/O8 protocol.
        if kind == 'o78' and (not row.get('finite_fp32') or not row.get('MSE_regression_passed')):
            raise ValueError('finite output and MSE regression required')
        expected = cfg['control'] if policy == 0 else cfg['symbol']
        metadata = row.get('kernel', row.get('resources', {}))
        if metadata.get('kernel_symbol') != expected:
            raise ValueError('wrong kernel identity')
        if metadata.get('cta_tile') != [64,128,128]:
            raise ValueError('tile drift')
        if metadata.get('registers_per_thread') != 168 or metadata.get('active_blocks_per_sm') != 3:
            raise ValueError('actual resource capacity changed')
        if metadata.get('integer_output_policy') != ('default_wb' if policy == 0 else 'streaming_cs'):
            raise ValueError('output-policy identity changed')
    summaries = summarize(rows, args['rounds'], args['repeats'])
    return dict(run=str(directory.relative_to(ROOT)), records=len(rows),
        rounds=rounds, timing_scope='four_modes_confirmation' if four_modes else 'cached_compute_only',
        raw_sequence_filtering=False,
        raw_sha256={name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                    for name in ('environment.json','validation.json','results.jsonl','summary.json')},
        summaries=summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--o3', type=Path, required=True)
    parser.add_argument('--o78', type=Path, required=True)
    parser.add_argument('--four', type=Path, help='separate complete O7/O8 four-mode confirmation')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):
        parser.error('fresh repository output required')
    result = dict(scope='v99_output_streaming_full24_paired_not_cached_MMA_microbenchmark',
        production_default_changed=False, no_small_performance_screen=True,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runs={kind: analyze_run(directory.resolve(), kind) for kind, directory in (('o3',args.o3),('o78',args.o78))})
    if args.four:
        result['runs']['o78_four'] = analyze_run(args.four.resolve(), 'o78', four_modes=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for run in result['runs'].values():
        for row in run['summaries']:
            print(row['variant'], row['mode'], row['policy'],
                  'median_ms=',round(row['median_ms'],6),
                  'paired_gain_percent=',round((row['paired_speedup']-1)*100,3),
                  'CI=',row['paired_speedup_ci95'],
                  'CV_failed=',row['any_stage_cv_failed_records'],'/',row['records'])


if __name__ == '__main__':
    main()
