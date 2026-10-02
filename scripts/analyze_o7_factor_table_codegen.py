#!/usr/bin/env python3
"""v76 loop work audit; static counts are NOT measured time or dynamic traffic."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from probe_o7_factor_table_codegen import CONTROL, CANDIDATE


def analyze(sass):
    result = []
    for block in re.split(r'(?=Function\s*:\s*)', sass):
        if not block.startswith('Function'):
            continue
        symbol = block.splitlines()[0].split(':', 1)[1].strip()
        if symbol not in (CONTROL, CANDIDATE):
            continue
        ins = []
        for line in block.splitlines()[1:]:
            match = re.match(r'\s*/\*([0-9a-f]+)\*/\s*(.*?)\s*;', line)
            if match:
                text = re.sub(r'^@!?U?P\w+\s+', '', match[2])
                ins.append((int(match[1], 16), text.split()[0], text))
        loops = []
        for pc, op, text in ins:
            branch = re.fullmatch(r'BRA\s+0x([0-9a-f]+)', text)
            if not branch or int(branch[1], 16) >= pc:
                continue
            body = [x for x in ins if int(branch[1], 16) <= x[0] <= pc]
            counts = Counter(i[1] for i in body)
            if sum(v for k, v in counts.items() if k.startswith('IMMA.')) != 64:
                continue
            if counts.get('I2F', 0):
                kind = 'fp32_fallback'
            elif any(k.startswith('STS') for k in counts):
                kind = 'factor_table'
            else:
                kind = 'original_integer'
            loops.append(dict(kind=kind, begin_pc=hex(body[0][0]), backedge_pc=hex(body[-1][0]),
                instructions=len(body), opcodes=dict(sorted(counts.items())),
                imma=sum(v for k, v in counts.items() if k.startswith('IMMA.')),
                imad_family=sum(v for k, v in counts.items() if k.split('.')[0] == 'IMAD'),
                scalar_shared_loads=sum(v for k, v in counts.items() if k.split('.')[0] == 'LDS'),
                scalar_shared_stores=sum(v for k, v in counts.items() if k.split('.')[0] == 'STS'),
                local_loads=[dict(pc=hex(p), instruction=t) for p, o, t in body if o.startswith('LDL')],
                local_stores=[dict(pc=hex(p), instruction=t) for p, o, t in body if o.startswith('STL')]))
        expected = {'original_integer', 'fp32_fallback'} | ({'factor_table'} if symbol == CANDIDATE else set())
        if len(loops) != len(expected) or {r['kind'] for r in loops} != expected:
            raise ValueError('ambiguous or missing mainloops for exact entry ' + symbol)
        result.append(dict(symbol=symbol, loops=loops))
    if {r['symbol'] for r in result} != {CONTROL, CANDIDATE}:
        raise ValueError('exact control/candidate entries required')
    return dict(scope='static_same_entry_mainloop_cost_not_performance_or_dynamic_traffic', entries=result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sass', type=Path, required=True)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    data = json.dumps(analyze(a.sass.read_text()), indent=2) + '\n'
    if a.output:
        if a.output.exists():
            p.error('fresh output required')
        a.output.write_text(data)
    else:
        print(data, end='')
