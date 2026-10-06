#!/usr/bin/env python3
"""v97 exact-entry encoding inspection; not a GPU/performance acceptance."""
import argparse
import hashlib
import json
from pathlib import Path

from inspect_interleaved_tail_equivalence import equivalence
from probe_tail_lookahead_codegen import ROOT, CONFIG, checked


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=CONFIG,required=True)
    p.add_argument('--codegen',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg=CONFIG[a.kind]
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT):
        p.error('fresh repository output required')
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
