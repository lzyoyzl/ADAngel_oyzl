#!/usr/bin/env python3
"""Locate each actual full-K loop, excluding the cold FP32 fallback.

Static work is not dynamic NCU execution or measured latency. The fixed loop
is found by its backward branch and 64 native INT4 instructions, never by a
hard-coded line interval. This is an evidence-based early rejection gate.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


def fullk_loops(sass):
    rows = []
    for block in re.split(r'(?=Function\s*:\s*)',sass):
        if not block.startswith('Function'):
            continue
        symbol = block.splitlines()[0].split(':',1)[1].strip()
        if symbol not in ('adangel_roof_o7_pow2_control','adangel_roof_o7_pow2_candidate'):
            continue
        instructions = []
        for line in block.splitlines()[1:]:
            match = re.match(r'\s*/\*([0-9a-f]+)\*/\s*(.*?)\s*;',line)
            if match:
                pc, text = int(match[1],16),match[2]
                instruction = re.sub(r'^@!?U?P\w+\s+','',text)
                instructions.append((pc,instruction.split()[0],text))
        matches = []
        for pc,op,text in instructions:
            jump = re.fullmatch(r'BRA\s+0x([0-9a-f]+)',text)
            if not jump or int(jump[1],16)>=pc:
                continue
            start = int(jump[1],16)
            loop = [x for x in instructions if start<=x[0]<=pc]
            if sum(x[1].startswith('IMMA.') for x in loop) != 64:
                continue
            if any(x[1].split('.')[0] in ('FFMA','FMUL','I2F') for x in loop):
                continue
            matches.append((start,pc,loop))
        if len(matches)!=1:
            raise ValueError('expected one integer full-K loop: '+symbol)
        start,end,loop = matches[0]
        counts = dict(Counter(op for _,op,_ in loop))
        rows.append(dict(symbol=symbol,start_pc=hex(start),backedge_pc=hex(end),
            instructions=len(loop),opcode_counts=counts,
            mainloop_local_loads=[dict(pc=hex(pc),instruction=text) for pc,op,text in loop if op.startswith('LDL')],
            mainloop_local_stores=[dict(pc=hex(pc),instruction=text) for pc,op,text in loop if op.startswith('STL')],
            scope='static_loop_body_not_dynamic_NCU_or_latency'))
    if {r['symbol'] for r in rows} != {'adangel_roof_o7_pow2_control','adangel_roof_o7_pow2_candidate'}:
        raise ValueError('missing unique entries')
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sass',type=Path,required=True)
    args = p.parse_args()
    print(json.dumps(dict(rows=fullk_loops(args.sass.read_text())),indent=2))


if __name__ == '__main__':
    main()
