#!/usr/bin/env python3
"""Diagnose O3 full-K instruction substitution, not a new performance candidate.

Profile both existing cubins with the existing paired harness, one sample,
one round, warmup50/repeats1. Kernel-filtered launch skips are 50 and101.
Do not use profiler-affected Event records as ordinary performance results.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from analyze_roof_scale_ncu import analyze


def analyze_capture(raw, source, policy):
    if policy not in (0, 1):
        raise ValueError('control0 or approved full-K1 only')
    result = analyze(raw, source, 54, 'o3', True, True,
                     expected_symbol='adangel_roof_fullk_integer_o3',
                     fullk_integer=bool(policy))
    metrics = list(csv.DictReader(io.StringIO(raw)))[1]
    if int(metrics['launch__block_size'].replace(',', '')) != 128:
        raise ValueError('unexpected block size')
    if int(metrics['launch__grid_size'].replace(',', '')) != 2048:
        raise ValueError('expected full4096^3 grid')
    result['policy'] = policy
    result['reference_math_tune'] = result.pop('tune')
    result['expected_filtered_launch_skip'] = 50 if policy == 0 else 101
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    args = p.parse_args()
    rows, sources = [], []
    for policy, label in enumerate(('control', 'fullk')):
        texts = []
        for suffix in ('raw', 'source_sass'):
            path = args.directory / f'{label}_{suffix}.csv'
            value = path.read_bytes()
            sources.append(dict(file=str(path), sha256=hashlib.sha256(value).hexdigest()))
            texts.append(value.decode('utf-8-sig'))
        rows.append(analyze_capture(*texts, policy))
    print(json.dumps(dict(scope='single real sample NCU diagnostic; not paired Event speedup or a new kernel',
                          rows=rows, sources=sources), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
