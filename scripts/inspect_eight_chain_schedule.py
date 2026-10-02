#!/usr/bin/env python3
"""Read exact eight-chain SASS dataflow, not an execution/latency simulation.

Track the four accumulator registers through each native high/high/low/low
MMA chain and intervening integer instructions. This answers how ptxas orders
the logical chains; it does not measure how many instructions are in flight.
Fail closed if a C fragment cannot be attributed to one chain or any chain
does not have the expected two signed then two unsigned MMA operations.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from inspect_o78_register_liveness import analyze

ROOT = Path(__file__).resolve().parents[1]


def instructions(sass, symbol, loop):
    block = next((b for b in re.split(r'(?=\s*Function\s*:\s*)', sass)
                  if re.match(r'\s*Function\s*:\s*' + re.escape(symbol) + r'\s', b)), None)
    if block is None:
        raise ValueError('exact SASS entry missing')
    low, high = int(loop['begin_pc'], 16), int(loop['end_pc'], 16)
    result = []
    for line in block.splitlines():
        m = re.search(r'/\*([0-9a-fA-F]+)\*/\s*(.*?)\s*;', line)
        if m and low <= int(m[1], 16) <= high:
            result.append((int(m[1], 16), re.sub(r'^@!?\w+\s+', '', m[2])))
    return result


def trace(sass, symbol, liveness):
    loop = next(x for x in liveness['loops'] if x['kind'] == 'integer')
    ops = instructions(sass, symbol, loop)
    tags, chains, schedule, loads = {}, [], [], []
    active, peak, mma_count = set(), 0, 0
    for pc, instruction in ops:
        op = instruction.split()[0]
        mma = re.match(r'IMMA\.16864\.(S4|U4)\.S4\s+R(\d+),.*?,.*?,\s*(RZ|R\d+)$', instruction)
        if mma:
            kind, dest, c = mma[1], int(mma[2]), mma[3]
            if c == 'RZ':
                if kind != 'S4':
                    raise ValueError('expected merged chain to start with signed-high MMA')
                cid = len(chains)
                chains.append(dict(id=cid, start_pc=hex(pc), mma=[]))
                active.add(cid)
            else:
                sources = [tags.get(int(c[1:]) + i, frozenset()) for i in range(4)]
                if any(len(s) != 1 for s in sources) or len(set(sources)) != 1:
                    raise ValueError(f'unresolved C fragment at {pc:x}: {sources}')
                cid = next(iter(sources[0]))
                if cid not in active:
                    raise ValueError('completed chain reused as new MMA accumulator')
            chain = chains[cid]
            expected = ('S4', 'S4', 'U4', 'U4')
            if len(chain['mma']) >= 4 or kind != expected[len(chain['mma'])]:
                raise ValueError('unexpected merged MMA stage order')
            chain['mma'].append(dict(pc=hex(pc), kind=kind, destination=dest, c=c))
            for i in range(4):
                tags[dest + i] = frozenset({cid})
            peak = max(peak, len(active))
            mma_count += 1
            if len(chain['mma']) == 4:
                chain['end_pc'] = hex(pc)
                active.remove(cid)
            schedule.append(dict(pc=hex(pc), chain=cid, stage=len(chain['mma']),
                                 active_after=len(active), ordinal=mma_count))
            continue
        # Stores, branches, barriers and predicate/uniform operations do not
        # define a GPR operand in the first position.
        dest = re.match(r'\S+\s+R(\d+)(?:\.[a-zA-Z0-9]+)?\s*,(.*)', instruction)
        if not dest:
            continue
        d, rest = int(dest[1]), dest[2]
        width = 1
        if op.startswith('LDSM.'):
            width = int(op.rsplit('.', 1)[1])
            loads.append(dict(pc=hex(pc), destination=d, width=width, mma_before=mma_count))
        elif op.startswith(('LDG', 'LDS', 'LDL')):
            width = 4 if '.128' in op else 2 if '.64' in op else 1
        elif op == 'CS2R' or op.startswith('IMAD.WIDE'):
            width = 2
        clear = op.startswith(('LDG', 'LDS', 'LDL', 'LDSM', 'S2R', 'CS2R'))
        if not clear and op.split('.')[0] not in {
                'MOV', 'IMAD', 'IADD3', 'SHF', 'LOP3', 'LEA', 'PRMT', 'SEL',
                'I2F', 'FMUL', 'FADD', 'FFMA', 'FSEL', 'FLO'}:
            raise ValueError('unhandled GPR writer: ' + instruction)
        source_tags = frozenset() if clear else frozenset().union(
            *(tags.get(int(r), frozenset()) for r in re.findall(r'\bR(\d+)\b', rest)))
        for i in range(width):
            tags[d + i] = source_tags
    if len(chains) != 16 or mma_count != 64 or active or any(len(c['mma']) != 4 for c in chains):
        raise ValueError('incomplete native MMA chains')
    return dict(symbol=symbol, loop=loop, chains=chains, schedule=schedule, loads=loads,
        total_mma=mma_count, chains_per_group=len(chains), peak_started_not_finished_chains=peak,
        active_count_histogram=dict(sorted(Counter(s['active_after'] for s in schedule).items())),
        scope='static programmed order, not concurrent hardware execution or a latency/throughput model')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--codegen', type=Path, required=True)
    p.add_argument('--stem', required=True)
    p.add_argument('--symbol', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT):
        p.error('fresh repository output required')
    receipt = json.loads((a.codegen / 'codegen.json').read_text())
    live = analyze((a.codegen / 'liveness.txt').read_text(), a.symbol)
    if live != receipt['liveness'][a.symbol]:
        raise ValueError('liveness receipt mismatch')
    result = trace((a.codegen / (a.stem + '.sass')).read_text(), a.symbol, live)
    result.update(input_sha256={name: hashlib.sha256((a.codegen / name).read_bytes()).hexdigest()
        for name in ('codegen.json', 'liveness.txt', a.stem + '.sass')},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        production_default_changed=False, new_performance_result=False)
    a.output.mkdir(parents=True)
    (a.output / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('symbol', 'total_mma', 'chains_per_group',
        'peak_started_not_finished_chains', 'active_count_histogram', 'scope')}, indent=2))


if __name__ == '__main__':
    main()
