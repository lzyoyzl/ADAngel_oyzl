#!/usr/bin/env python3
"""Reconcile one profiled conversion entry; NCU time is not Event acceptance."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re

FORMATS = ('nvfp4_g128', 'mxfp8_e4m3_g128', 'hif4_g128', 'nvstyle_fp6_e2m3_g128')


def analyze(raw_text, source_text, fmt, policy):
    kind = FORMATS.index(fmt)
    if policy not in (0, 2):
        raise ValueError('only current scalar control and vector16 are profiled')
    rows = list(csv.DictReader(io.StringIO(raw_text)))
    if len(rows) != 2:
        raise ValueError('expected units and exactly one kernel')
    units, raw = rows
    source = io.StringIO(source_text)
    identity = next(csv.reader(source))
    symbol = ('adangel_sm80_vector_fixed_conversion' if policy == 2 else
              'adangel_sm80_mixed_fused_g128_conversion' if kind == 1 else
              'adangel_sm80_integer_fixed_conversion')
    expected = f'{symbol}<{kind}'
    normalize = lambda s: re.sub(r'\([^()]*\)|\s+', '', s)
    # Match both exports and template arguments, not just a generic symbol.
    for name in (raw['Kernel Name'], identity[1]):
        name = normalize(name)
        token = expected + (',16>' if policy == 2 else '>' if kind == 1 else ',1>')
        if token not in name:
            raise ValueError(f'wrong format/policy kernel: {name}')
    if identity[0] != 'Kernel Name':
        raise ValueError('missing source identity')
    counts = {}
    for row in csv.DictReader(source):
        match = re.match(r'\s*(?:@!?U?P(?:T|\d+)\s+)?([A-Z][A-Z0-9_]*)', row['Source'])
        if not match:
            raise ValueError('unknown SASS opcode')
        count = int(row['Instructions Executed'].replace(',', ''))
        if count < 0:
            raise ValueError('negative instruction count')
        counts[match[1]] = counts.get(match[1], 0) + count

    def metric(key, unit=None):
        if unit is not None and units[key] != unit:
            raise ValueError(f'wrong unit: {key}')
        value = float(raw[key].replace(',', ''))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'invalid metric: {key}')
        return value

    if sum(counts.values()) != metric('smsp__inst_executed.sum', 'inst'):
        raise ValueError('SASS/raw dynamic instruction mismatch')
    if any(counts.get(op, 0) for op in ('LDL', 'STL', 'IMMA', 'HMMA')):
        raise ValueError('unexpected local access or GEMM in conversion target')
    duration = metric('gpu__time_duration.sum') * {
        'ns': 1e-6, 'us': 1e-3, 'ms': 1, 's': 1000}[units['gpu__time_duration.sum']]
    return dict(format=fmt, policy=policy, kernel_name=raw['Kernel Name'],
        ncu_ms=duration, dynamic_warp_instructions=sum(counts.values()), opcode_counts=counts,
        registers=metric('launch__registers_per_thread', 'register/thread'),
        sm_throughput_pct=metric('sm__throughput.avg.pct_of_peak_sustained_elapsed', '%'),
        dram_throughput_pct=metric('gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed', '%'),
        l1tex_throughput_pct=metric('l1tex__throughput.avg.pct_of_peak_sustained_elapsed', '%'),
        l2_throughput_pct=metric('lts__throughput.avg.pct_of_peak_sustained_elapsed', '%'),
        issue_active_pct=metric('smsp__issue_active.avg.pct_of_peak_sustained_active', '%'),
        eligible_warps=metric('smsp__warps_eligible.avg.per_cycle_active', 'warp'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    rows, files = [], []
    for fmt in FORMATS:
        for policy in (0, 2):
            texts = []
            for suffix in ('raw', 'source_sass'):
                path = args.directory / f'{fmt}_p{policy}_{suffix}.csv'
                payload = path.read_bytes()
                files.append(dict(file=path.name, sha256=hashlib.sha256(payload).hexdigest()))
                texts.append(payload.decode('utf-8-sig'))
            rows.append(analyze(*texts, fmt, policy))
    print(json.dumps(dict(scope='one real input; NCU full set; cold cache; unlocked; not Event timing',
                          files=files, rows=rows), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
