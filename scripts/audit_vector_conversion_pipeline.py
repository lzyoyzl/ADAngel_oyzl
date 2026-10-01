#!/usr/bin/env python3
"""v54 exact GEMM regression, vector16 codegen parity and resource audit."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from compare_a100_codegen import compare, instructions


def analyze(before, after, resources, probe):
    preserved = compare(before, after, r'adangel_sm80_(?:o3_swizzled_bound2|split_grouped_major|roof_candidate)')
    full = compare(before, after, r'adangel_')
    old = instructions(probe, r'adangel_sm80_vector_fixed_conversion.*ELi16EE')
    new = instructions(after, r'adangel_sm80_vector_fixed_conversion')
    exact = {s: new.get(s) == words for s, words in old.items()}
    entries=[];gemms=[]
    for block in re.split(r'(?=Function\s*:\s*)',after):
        if not block.startswith('Function'): continue
        symbol=block.splitlines()[0].split(':',1)[1].strip()
        if 'adangel_sm80_roof_candidateILb1ELb0ELi59EE' in symbol:
            gemms.append(dict(symbol=symbol,u4_s4=bool(re.search(r'IMMA\.[^;\n]*U4\.S4',block)),
                s4_s4=bool(re.search(r'IMMA\.[^;\n]*S4\.S4',block)),cp_async='LDGSTS' in block,
                no_int8=not bool(re.search(r'IMMA\.[^;\n]*[SU]8\.',block))))
        if 'adangel_sm80_vector_fixed_conversion' not in symbol: continue
        resource=re.search(r'Function\s+(?:\:\s*)?'+re.escape(symbol)+r'\s*:\s*([^\n]+)',resources)
        fields={k:int(v) for k,v in re.findall(r'(REG|STACK|LOCAL):(\d+)',resource[0] if resource else '')}
        entries.append(dict(symbol=symbol,resources=fields,
            no_local=not bool(re.search(r'\b(?:LDL|STL)(?:\.|\s)',block))))
    return dict(passed=preserved['passed'] and preserved['old_symbols']>=160
                and len(old)==len(new)==len(entries)==4 and all(exact.values())
                and len(gemms)==1 and all(gemms[0][key] for key in ('u4_s4','s4_s4','cp_async','no_int8'))
                and all(set(r['resources'])=={'REG','STACK','LOCAL'} and r['resources']['STACK']==0
                        and r['resources']['LOCAL']==0 and r['no_local'] for r in entries),
                gemm_regression=preserved,all_old_entries=full,vector_exact_v53=exact,
                vector_entries=entries,gemm59=gemms,
                scope='old primary/candidate GEMMs must match; other codegen changes disclosed and require regression')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('before','after','resources','probe','output'):p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args()
    result=analyze(a.before.read_text(),a.after.read_text(),a.resources.read_text(),a.probe.read_text())
    result['sources']=[dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                       for path in (a.before,a.after,a.resources,a.probe)]
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],old_gemms=result['gemm_regression']['old_symbols'],
                         other_changed=result['all_old_entries']['changed'],vector_entries=len(result['vector_entries']))))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
