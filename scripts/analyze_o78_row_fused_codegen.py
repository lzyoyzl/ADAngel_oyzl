#!/usr/bin/env python3
"""Audit v73 preparation resources and unchanged v69 control instructions.

Static counts only; this is not an NCU or latency attribution model.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def entries(sass):
    result = {}
    for block in re.split(r'(?=Function\s*:\s*)', sass):
        if not block.startswith('Function'):
            continue
        symbol = block.splitlines()[0].split(':', 1)[1].strip()
        instructions = []
        for line in block.splitlines()[1:]:
            match = re.match(r'\s*/\*[0-9a-f]+\*/\s*(.*?)\s*;', line)
            if match:
                instructions.append(re.sub(r'^@!?U?P\w+\s+', '', match[1]))
        if symbol in result:
            raise ValueError('duplicate SASS entry: ' + symbol)
        result[symbol] = dict(
            encoded_words=re.findall(r'/\* (0x[0-9a-f]+) \*/', block),
            opcode_counts=dict(Counter(i.split()[0] for i in instructions)),
            instructions=len(instructions),
        )
    if not result or any(not e['encoded_words'] for e in result.values()):
        raise ValueError('missing encoded SASS instructions')
    return result


def analyze(build, control):
    new = entries((build / 'prepare.sass').read_text())
    old = entries((control / 'prepare.sass').read_text())
    unchanged = {s: s in new and old[s]['encoded_words'] == new[s]['encoded_words'] for s in old}
    resources = {}
    for symbol, registers, stack, shared, local in re.findall(
            r' Function (\S+):\s+REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)',
            (build / 'resources.txt').read_text()):
        resources[symbol] = dict(registers=int(registers), stack=int(stack), shared=int(shared), local=int(local))
    added = sorted(set(new) - set(old))
    rows = []
    for symbol in added:
        ops = new[symbol]['opcode_counts']
        row = dict(symbol=symbol, **resources[symbol], instructions=new[symbol]['instructions'], opcode_counts=ops)
        row['local_instructions'] = sum(n for op, n in ops.items() if op.startswith(('LDL', 'STL')))
        row['cta_barriers'] = sum(n for op, n in ops.items() if op.startswith('BAR.SYNC'))
        rows.append(row)
    if len(rows) != 4 or not all('adangel_sm80_row_conversion_metadata' in r['symbol'] for r in rows):
        raise ValueError('expected exactly four row-fused conversion entries')
    if not all(unchanged.values()):
        raise ValueError('v69 control encoded instructions changed')
    if not all(r['stack'] == r['local'] == r['local_instructions'] == 0 and
               r['shared'] == 256 and r['cta_barriers'] == 1 for r in rows):
        raise ValueError('unexpected candidate local memory or CTA synchronization')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    return dict(scope='static_preparation_audit_not_GEMM_or_dynamic_profile',
                sass_sha256=digest(build / 'prepare.sass'), control_sass_sha256=digest(control / 'prepare.sass'),
                original_control_encoded_identical=unchanged, rows=rows,
                gemm_examined_by_this_static_audit=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = analyze(a.build, a.control)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(unchanged_entries=len(result['original_control_encoded_identical']),
                         new_resources=[{k: r[k] for k in ('registers', 'shared', 'local', 'cta_barriers')} for r in result['rows']]), indent=2))


if __name__ == '__main__':
    main()
