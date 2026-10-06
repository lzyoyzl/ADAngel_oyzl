#!/usr/bin/env python3
"""Read-only exact-entry comparison for v96; not GPU/performance acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from compare_a100_codegen import compare, instructions
from probe_interleaved_tail_codegen import ROOT, CONFIG, checked


def exact_entry(text,symbol):
    matches=[b for b in re.split(r'(?=Function\s*:\s*)',text)
        if b.startswith('Function') and b.splitlines()[0].split(':',1)[1].strip()==symbol]
    if len(matches)!=1:raise ValueError('one exact entry required: '+symbol)
    return matches[0]


def equivalence(sass,control,candidate):
    old=exact_entry(sass,control);new=exact_entry(sass,candidate)
    # Only normalize the symbol's text, never opcodes/registers/control bits.
    normalized=new.replace(candidate,control)
    result=compare(old,normalized,'^'+re.escape(control)+'$')
    a=instructions(old,'^'+re.escape(control)+'$')[control]
    b=instructions(normalized,'^'+re.escape(control)+'$')[control]
    result.update(control=control,candidate=candidate,
        control_instructions=len(a)//2,candidate_instructions=len(b)//2,
        control_encoded_words_sha256=hashlib.sha256('\n'.join(a).encode()).hexdigest(),
        candidate_encoded_words_sha256=hashlib.sha256('\n'.join(b).encode()).hexdigest(),
        scope='exact encoded words after symbol-text normalization; not a launch, output/MSE test or speedup')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=CONFIG,required=True)
    p.add_argument('--codegen',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg=CONFIG[a.kind]
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT):p.error('fresh repository output required')
    receipt=checked(a.codegen,a.kind)
    sass=a.codegen/(cfg['stem']+'.sass')
    r=equivalence(sass.read_text(),cfg['control'],cfg['symbol'])
    r.update(codegen_sha256=hashlib.sha256((a.codegen/'codegen.json').read_bytes()).hexdigest(),
        sass_sha256=hashlib.sha256(sass.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        compiler_gate=receipt['worth_runtime_validation'],new_runtime_measurement=False,
        production_default_changed=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))


if __name__=='__main__':main()
