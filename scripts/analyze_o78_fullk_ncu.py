#!/usr/bin/env python3
"""v70 full-K/control NCU analysis, bound to the audited v67 code fingerprint."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re

from analyze_roof_scale_ncu import analyze


def analyze_capture(raw, source, receipt):
    variant, policy = receipt['variant'], receipt['policy']
    symbol = 'adangel_roof_o78_fullk_' + ('candidate' if policy else 'control')
    if variant not in ('o7','o8') or policy not in (0,1) or receipt['expected_kernel'] != symbol:
        raise ValueError('unexpected full-K profile identity')
    if receipt['guard']['integer_ctas'] != 2048 or receipt['guard']['fallback_ctas'] or receipt['guard']['invalid_ctas']:
        raise ValueError('full-K accounting requires all 2048 CTAs accepted')
    row = analyze(raw, source, 59, variant, True, True, expected_symbol=symbol, fullk_integer=bool(policy))
    metrics = list(csv.DictReader(io.StringIO(raw)))[1]
    if int(metrics['launch__block_size'].replace(',','')) != 128 or int(metrics['launch__grid_size'].replace(',','')) != 2048:
        raise ValueError('wrong launch shape')
    if row['registers_per_thread'] != receipt['resources']['registers_per_thread']:
        raise ValueError('NCU resources differ from loaded cubin')
    source_rows = list(csv.DictReader(io.StringIO(source.split('\n',1)[1])))
    actual = Counter()
    for item in source_rows:
        match = re.match(r'\s*(?:@!?U?P(?:T|\d+)\s+)?([A-Z][A-Z0-9_]*)', item['Source'])
        if not match:
            raise ValueError('unknown source opcode')
        actual[match[1]] += 1
    expected = receipt['gemm_codegen']['entries'][symbol]
    if actual != Counter({op.rstrip(';'):n for op,n in expected['opcode_counts'].items()}) or sum(actual.values()) != expected['instructions']:
        raise ValueError('NCU source fingerprint differs from audited cubin')
    row.update(policy=policy, fullk_integer=bool(policy), static_fingerprint_verified=True)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    args = p.parse_args()
    rows, sources = [], []
    for variant in ('o7','o8'):
        for policy in (0,1):
            stem = f'{variant}_p{policy}'
            payloads = []
            for suffix in ('raw','source_sass'):
                path = args.directory / f'{stem}_{suffix}.csv'
                data = path.read_bytes()
                sources.append(dict(file=str(path), sha256=hashlib.sha256(data).hexdigest()))
                payloads.append(data.decode('utf-8-sig'))
            receipt = json.loads((args.directory / stem / 'receipt.json').read_text())
            rows.append(analyze_capture(*payloads, receipt))
    print(json.dumps(dict(scope='one_sample_four_NCU_captures_not_Event_speedup_or_new_kernel',
                          rows=rows, sources=sources), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
