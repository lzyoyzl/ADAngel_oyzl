#!/usr/bin/env python3
"""v72 static integer-loop dependencies; not a cycle/latency prediction.

Track scalar shared loads and MMA results through register definitions. This
does not infer lane coordinates or measured stall contributions. Full source,
PTX and SASS must accompany the report.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

SYMBOLS = ('adangel_roof_o78_coefficient_control',
           'adangel_roof_o78_coefficient_candidate')


def integer_loops(sass):
    output = {}
    for block in re.split(r'(?=Function\s*:\s*)', sass):
        if not block.startswith('Function'):
            continue
        symbol = block.splitlines()[0].split(':', 1)[1].strip()
        if symbol not in SYMBOLS:
            continue
        instructions = []
        for line in block.splitlines()[1:]:
            match = re.match(r'\s*/\*([0-9a-f]+)\*/\s*(.*?)\s*;', line)
            if match:
                text = re.sub(r'^@!?U?P\w+\s+', '', match[2])
                instructions.append((int(match[1], 16), text.split()[0], text))
        matches = []
        for pc, _, text in instructions:
            jump = re.fullmatch(r'BRA\s+0x([0-9a-f]+)', text)
            if not jump or int(jump[1], 16) >= pc:
                continue
            loop = [x for x in instructions if int(jump[1], 16) <= x[0] <= pc]
            if sum(x[1].startswith('IMMA.') for x in loop) == 64 and not any(
                    x[1].split('.')[0] in ('FFMA', 'FMUL', 'I2F') for x in loop):
                matches.append(loop)
        if len(matches) != 1:
            raise ValueError('expected one unique integer loop: ' + symbol)
        output[symbol] = matches[0]
    if set(output) != set(SYMBOLS):
        raise ValueError('missing exact entries')
    return output


def dependencies(loop):
    registers = {}
    coefficients, partial_first, updates = [], [], []
    for pc, opcode, text in loop:
        args = text.split(None, 1)[1] if ' ' in text else ''
        operands = args.split(',')
        dest = re.fullmatch(r'\s*R(\d+)\s*', operands[0])
        if not dest:
            continue
        dst = 'R' + dest[1]
        src = re.findall(r'\bR(?:\d+|Z)\b', ','.join(operands[1:]))
        origins = [registers.get(r, frozenset()) for r in src]
        tag = frozenset().union(*origins)
        width = 1
        if opcode.startswith('IMMA.'):
            tag, width = frozenset({'partial'}), 4
        elif opcode.startswith('LDSM.'):
            tag, width = frozenset({'payload'}), 4
        elif opcode == 'LDS' or opcode.startswith('LDS.'):
            tag = frozenset({'scalar_shared'})
            width = 2 if '.64' in opcode else 1
        elif opcode.startswith(('LDL', 'LDG')):
            tag = frozenset({'other_load'})
        elif opcode == 'IMAD' and len(operands) == 4:
            def operand_tag(value):
                regs = re.findall(r'\bR(?:\d+|Z)\b', value)
                return frozenset().union(*(registers.get(r, frozenset()) for r in regs))
            a, b = operand_tag(operands[1]), operand_tag(operands[2])
            zero = operands[3].strip() == 'RZ'
            if zero and a == b == {'scalar_shared'}:
                tag = frozenset({'coefficient'})
                coefficients.append(dict(pc=hex(pc), instruction=text))
            elif zero and ((a == {'partial'} and b == {'scalar_shared'}) or
                           (b == {'partial'} and a == {'scalar_shared'})):
                tag = frozenset({'partial_times_one_scale'})
                partial_first.append(dict(pc=hex(pc), instruction=text))
            elif ((a == {'partial'} and b == {'coefficient'}) or
                  (b == {'partial'} and a == {'coefficient'})):
                tag = frozenset({'scaled_sum'})
                updates.append(dict(pc=hex(pc), instruction=text))
        for offset in range(width):
            registers['R' + str(int(dest[1]) + offset)] = tag
    return dict(independent_coefficient_multiplies=coefficients,
                partial_first_multiplies=partial_first,
                coefficient_based_accumulator_updates=updates)


def analyze(sass):
    rows = []
    for symbol, loop in integer_loops(sass).items():
        dep = dependencies(loop)
        rows.append(dict(symbol=symbol, start_pc=hex(loop[0][0]), backedge_pc=hex(loop[-1][0]),
            instructions=len(loop), opcode_counts=dict(Counter(x[1] for x in loop)),
            mainloop_local_loads=[dict(pc=hex(pc), instruction=t) for pc, op, t in loop if op.startswith('LDL')],
            mainloop_local_stores=[dict(pc=hex(pc), instruction=t) for pc, op, t in loop if op.startswith('STL')],
            dependency_counts={key: len(value) for key, value in dep.items()}, **dep))
    return dict(scope='static_def_use_and_loop_work_not_dynamic_NCU_or_latency', rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sass', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.dumps(analyze(args.sass.read_text()), indent=2) + '\n'
    if args.output:
        if args.output.exists():
            parser.error('fresh output required')
        args.output.write_text(result)
    else:
        print(result, end='')


if __name__ == '__main__':
    main()
