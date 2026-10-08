#!/usr/bin/env python3
"""Audit scaled-partial handoff in archived SASS, without predicting latency.

The existing eight-chain audit ends a chain at its fourth MMA. This companion
tracks its four values through scaling until the final integer accumulator
updates. A new MMA ordinal alone cannot tell whether scaling already overlaps
the next output slice. No CUDA code, numerical policy or dispatch is changed.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from inspect_eight_chain_schedule import instructions, trace


def reg(value):
    match = re.fullmatch(r"R(\d+)(?:\.reuse)?", value.strip())
    return int(match[1]) if match else None


def reduction_updates(ops, chains):
    """Track final MMA lanes into in-place IMAD updates; fail on lost values.

    This is deliberately narrow: current eight-chain INT32 fast paths only.
    Intermediate row multiplication has C=RZ and is not an accumulator update.
    Addresses, predicates, uniform registers and scale loads are not partials.
    """
    seeds = {int(c['end_pc'], 16): (c['id'], c['mma'][-1]['destination'])
             for c in chains}
    tags, updates = {}, []
    for ordinal, (pc, instruction) in enumerate(ops):
        if pc in seeds:
            cid, destination = seeds[pc]
            for vi in range(4):
                tags[destination + vi] = (cid, vi)
            continue
        fields = instruction.split(None, 1)
        op = fields[0]
        if len(fields) != 2:
            continue
        operands = [x.strip() for x in fields[1].split(',')]
        destination = reg(operands[0])
        if destination is None:
            continue
        # A non-final MMA overwrites four registers. It cannot carry a previous
        # output's finished partial: the original MMA-chain audit checks this.
        if op.startswith('IMMA.'):
            for i in range(4):
                tags.pop(destination + i, None)
            continue
        width = (int(op.rsplit('.', 1)[1]) if op.startswith('LDSM.') else
                 4 if '.128' in op else 2 if '.64' in op or op == 'CS2R'
                 or op.startswith('IMAD.WIDE') else 1)
        if op.startswith(('LD', 'S2R', 'CS2R')):
            for i in range(width):
                tags.pop(destination + i, None)
            continue
        if op == 'IMAD' and len(operands) == 4 and reg(operands[3]) == destination:
            products = {tags[r] for x in operands[1:3]
                        if (r := reg(x)) in tags}
            if products:
                if len(products) != 1:
                    raise ValueError('two partials unexpectedly multiplied')
                cid, vi = products.pop()
                updates.append(dict(pc=hex(pc), instruction_index=ordinal,
                                    chain=cid, value=vi))
                # Accumulator state is not a live temporary partial.
                tags.pop(destination, None)
                continue
        if op.split('.')[0] not in {
                'MOV', 'IMAD', 'IADD3', 'SHF', 'LOP3', 'LEA', 'PRMT', 'SEL',
                'I2F', 'FMUL', 'FADD', 'FFMA', 'FSEL', 'FLO'}:
            raise ValueError('unhandled GPR writer: ' + instruction)
        sources = {tags[r] for x in operands[1:] if (r := reg(x)) in tags}
        if len(sources) > 1:
            raise ValueError('unexpected merger of distinct finished partials')
        if sources:
            if width != 1:
                raise ValueError('wide arithmetic on a finished partial is unsupported')
            tags[destination] = sources.pop()
        else:
            for i in range(width):
                tags.pop(destination + i, None)
    expected = Counter((c['id'], vi) for c in chains for vi in range(4))
    observed = Counter((u['chain'], u['value']) for u in updates)
    if observed != expected:
        raise ValueError(f'incomplete/duplicate scaled updates: {observed - expected}; '
                         f'missing: {expected - observed}')
    return updates


def inspect(sass, symbol, liveness):
    schedule = trace(sass, symbol, liveness)
    ops = instructions(sass, symbol, schedule['loop'])
    updates = reduction_updates(ops, schedule['chains'])
    start_next = int(schedule['chains'][8]['start_pc'], 16)
    old_updates = [u for u in updates if u['chain'] < 8]
    old_after = [u for u in old_updates if int(u['pc'], 16) > start_next]
    last_old = max(int(u['pc'], 16) for u in old_updates)
    next_mma_before_drain = [x for x in schedule['schedule']
                            if x['chain'] >= 8 and int(x['pc'], 16) < last_old]
    return dict(symbol=symbol, scaled_accumulator_updates=len(updates),
                old_slice_updates_before_next_mma=len(old_updates)-len(old_after),
                old_slice_updates_after_next_mma=len(old_after),
                next_slice_mma_before_old_scale_drains=len(next_mma_before_drain),
                first_next_mma_pc=hex(start_next), last_old_scale_pc=hex(last_old),
                integer_loop_instructions=len(ops), updates=updates,
                scope='static interleaving only, not hardware overlap or measured speedup')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codegen', type=Path, required=True)
    parser.add_argument('--sass', type=Path, required=True)
    parser.add_argument('--symbols', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.exists() or not args.output.resolve().is_relative_to(root):
        parser.error('fresh repository output required')
    receipt = json.loads(args.codegen.read_text())
    result = dict(analyses=[inspect(args.sass.read_text(), symbol,
                                   receipt['liveness'][symbol]) for symbol in args.symbols],
                  input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (args.codegen, args.sass)},
                  new_performance_result=False, production_default_changed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    for analysis in result['analyses']:
        print(json.dumps({k:v for k,v in analysis.items() if k != 'updates'}, indent=2))


if __name__ == '__main__':
    main()
