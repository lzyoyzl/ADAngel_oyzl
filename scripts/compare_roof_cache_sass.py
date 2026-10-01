#!/usr/bin/env python3
"""Compare cache-policy counterparts inside one SM80 binary, not performance."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def compare(payload):
    functions={}
    for block in re.split(r'(?=Function\s*:\s*)',payload):
        if not block.startswith('Function'): continue
        symbol=block.splitlines()[0].split(':',1)[1].strip()
        match=re.search(r'adangel_sm80_roof_candidateILb([01])ELb0ELi(54|59|61|62)EE',symbol)
        if not match: continue
        tune=int(match[2])
        if tune in functions: raise ValueError('duplicate target entry')
        # Exclude address/control-word comments, retain operands/predicates/order.
        instructions=[]
        for line in block.splitlines():
            if re.match(r'\s*/\*[0-9a-f]+\*/',line):
                text=re.sub(r'/\*.*?\*/','',line).strip()
                if text: instructions.append(text)
        if not instructions: raise ValueError('missing target SASS')
        functions[tune]=(symbol,instructions)
    if set(functions)!={54,59,61,62}: raise ValueError('missing cache pair')
    pairs=[]
    for old,new in ((54,61),(59,62)):
        a,b=functions[old][1],functions[new][1]
        def normalize(xs): return [s.replace('.BYPASS','') for s in xs]
        copies=lambda xs:[s for s in xs if 'LDGSTS' in s]
        pairs.append(dict(reference=old,candidate=new,
            reference_symbol=functions[old][0],candidate_symbol=functions[new][0],
            reference_instructions=len(a),candidate_instructions=len(b),
            identical_instruction_text_except_bypass=normalize(a)==normalize(b),
            reference_copy_instructions=copies(a),candidate_copy_instructions=copies(b),
            reference_bypasses_l1=bool(copies(a)) and all('.BYPASS' in s for s in copies(a)),
            candidate_uses_l1=bool(copies(b)) and all('.BYPASS' not in s for s in copies(b))))
    return pairs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sass',type=Path,required=True)
    a=p.parse_args(); data=a.sass.read_bytes()
    rows=compare(data.decode())
    print(json.dumps(dict(source=str(a.sass),sha256=hashlib.sha256(data).hexdigest(),
        scope='static_instruction_text_not_control_words_dynamic_counts_or_performance',pairs=rows),indent=2))
    if not all(r['reference_bypasses_l1'] and r['candidate_uses_l1'] for r in rows):
        raise SystemExit(1)


if __name__=='__main__': main()
