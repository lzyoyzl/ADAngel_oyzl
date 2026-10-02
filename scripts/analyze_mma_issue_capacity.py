#!/usr/bin/env python3
"""Verify v82 instruction-capacity evidence; never label it trace GEMM/MSE."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re

from probe_a100_mma_issue_capacity import audit, summarize, SYMBOLS


def analyze(directory):
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    build = json.loads((directory / 'codegen.json').read_text())
    entries = json.loads(json.dumps(audit((directory / 'kernel.sass').read_text())))
    if entries != build['entries']:
        raise ValueError('compiled instruction audit mismatch')
    rows = [json.loads(s) for s in (directory / 'results.jsonl').read_text().splitlines()]
    saved = json.loads((directory / 'summary.json').read_text())
    if summarize(rows) != saved['records'] or saved['original_experiment_MSE_measured']:
        raise ValueError('Event summary or scope mismatch')
    raw = list(csv.DictReader(io.StringIO((directory / 'merged_raw.csv').read_text())))
    if len(raw) != 2:
        raise ValueError('one NCU launch plus units row required')
    units, metrics = raw
    def num(key):
        return float(metrics[key].replace(',', ''))
    if (metrics['Kernel Name'] != SYMBOLS[2] or num('launch__grid_size') != 2048
            or num('launch__block_size') != 128 or num('launch__registers_per_thread') != 40
            or num('launch__occupancy_limit_shared_mem') != 3
            or num('sm__ops_path_tensor_src_int4_sparsity_off.sum') != 2199023255552
            or num('sm__ops_path_tensor_src_int8_sparsity_off.sum') != 0):
        raise ValueError('NCU identity/resource/physical-work mismatch')
    source = (directory / 'merged_source_sass.csv').read_text()
    if next(csv.reader([source.splitlines()[0]]))[1] != SYMBOLS[2]:
        raise ValueError('source report symbol mismatch')
    static, dynamic = Counter(), Counter()
    for row in csv.DictReader(io.StringIO(source.split('\n', 1)[1])):
        opcode = re.match(r'\s*(?:@!?U?P(?:T|\d+)\s+)?([A-Z][A-Z0-9_.]*)', row['Source'])
        if not opcode:
            raise ValueError('unknown source instruction')
        static[opcode[1]] += 1
        dynamic[opcode[1].split('.')[0]] += int(row['Instructions Executed'].replace(',', ''))
    expected = Counter()
    for op, count in entries[SYMBOLS[2]]['opcode_counts'].items():
        expected[op.rstrip(';')] += count
    if static != expected or dynamic['IMMA'] != 134217728 or sum(dynamic.values()) != num('smsp__inst_executed.sum'):
        raise ValueError('source SASS fingerprint/dynamic count mismatch')
    if units['gpu__time_duration.sum'] != 'ms' or units['sm__cycles_elapsed.avg.per_second'] != 'Ghz':
        raise ValueError('unexpected NCU units')
    paths = ['codegen.json', 'kernel.sass', 'resources.txt', 'build.log', 'results.jsonl',
             'summary.json', 'merged_raw.csv', 'merged_source_sass.csv', 'ncu.log']
    return dict(scope='native_MMA_capacity_not_trace_GEMM_or_attainable_peak',
        event_records=saved['records'], static_and_dynamic_work_verified=True,
        ncu=dict(duration_ms=num('gpu__time_duration.sum'), sm_clock_ghz=num('sm__cycles_elapsed.avg.per_second'),
                 tensor_active_percent=num('sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed'),
                 dynamic_warp_instructions=dict(sorted(dynamic.items())),
                 physical_int4_operations=num('sm__ops_path_tensor_src_int4_sparsity_off.sum'),
                 issue_active_percent=num('smsp__issue_active.avg.pct_of_peak_sustained_active'),
                 eligible_warps=num('smsp__warps_eligible.avg.per_cycle_active')),
        no_new_trace_MSE_or_conversion_or_end_to_end=True, production_default_changed=False,
        input_sha256={p: digest(directory / p) for p in paths})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(analyze(args.input), indent=2, allow_nan=False))
