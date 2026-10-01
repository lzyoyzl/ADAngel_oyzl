#!/usr/bin/env python3
"""Conversion codegen/resource check; not a Tensor Core audit."""
import argparse
import json
from pathlib import Path
import re
from compare_a100_codegen import instructions


def analyze(sass, resources, reference):
    pattern = r'adangel_sm80_(?:integer_fixed_conversion|mixed_fused_g128_conversion)'
    old, new = instructions(reference, pattern), instructions(sass, pattern)
    # The library builds both original TUs unmodified. Compare all 12 entries,
    # including the selected four controls, not only source-level names.
    matched = {symbol: old[symbol] == words for symbol, words in new.items() if symbol in old}
    rows = []
    for block in re.split(r'(?=Function\s*:\s*)', sass):
        if not block.startswith('Function'):
            continue
        symbol = block.splitlines()[0].split(':', 1)[1].strip()
        if 'adangel_sm80_vector_fixed_conversion' not in symbol:
            continue
        resource = re.search(r'Function\s+(?:\:\s*)?' + re.escape(symbol) + r'\s*:\s*([^\n]+)', resources)
        fields = {k: int(v) for k, v in re.findall(r'(REG|LOCAL|STACK):(\d+)', resource[0] if resource else '')}
        rows.append(dict(symbol=symbol, resources=fields,
                         local_sass=bool(re.search(r'\b(?:LDL|STL)(?:\.|\s)', block)),
                         vector_load=bool(re.search(r'\bLDG[^;\n]*\.(?:64|128)\b', block)),
                         static_instructions=len(re.findall(r'/\*\s*[0-9a-f]+\s*\*/', block))))
    return dict(passed=len(matched)==12 and all(matched.values()) and len(rows)==8 and
                all(set(r['resources'])=={'REG','LOCAL','STACK'} and r['resources']['STACK']==0
                    and r['resources']['LOCAL']==0 and not r['local_sass'] for r in rows),
                controls_exact_sass=matched, functions=rows,
                scope='isolated_conversion_scalar_best_controls_and_vector_resources_not_gemm')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--reference-sass',type=Path,required=True)
    a=p.parse_args()
    result=analyze((a.directory/'conversion.sass').read_text(),
                   (a.directory/'conversion.resources.txt').read_text(),a.reference_sass.read_text())
    (a.directory/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['passed']: raise SystemExit(1)


if __name__=='__main__':main()
