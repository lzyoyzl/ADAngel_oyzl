#!/usr/bin/env python3
"""Audit cubin L2 hints against exact best-kernel controls, not probe hitcounts."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from compare_a100_codegen import instructions
from probe_roof_l2_codegen import static_entries, PATTERN


def text_entries(payload, pattern=PATTERN):
    result={}
    for block in re.split(r'(?=Function\s*:\s*)',payload):
        if not block.startswith('Function'): continue
        symbol=block.splitlines()[0].split(':',1)[1].strip()
        if not re.search(pattern,symbol): continue
        if symbol in result: raise ValueError('duplicate matching function')
        result[symbol]=[re.sub(r'/\*.*?\*/','',line).strip() for line in block.splitlines()
                        if re.match(r'\s*/\*[0-9a-f]+\*/',line)]
    return result


def audit(directory,best_sass):
    files=[directory/f'l2_{size}.sass' for size in (0,128,256)]+[best_sass]
    payloads=[p.read_text() for p in files]
    old=instructions(payloads[-1],r'adangel_sm80_roof_candidateILb[01]ELb0ELi(?:54|59)EE')
    control=instructions(payloads[0],PATTERN)
    matches={('adangel_roof_l2_o3' if 'ELi54EE' in symbol else 'adangel_roof_l2_o78'):words
             for symbol,words in old.items()}
    if set(matches)!=set(control): raise ValueError('missing best/control pair')
    control_matches={key:matches[key]==control[key] for key in control}
    text0=text_entries(payloads[0]);rows=[]
    for size,source in zip((0,128,256),payloads):
        text=text_entries(source);entries=static_entries(source)
        for symbol,details in entries.items():
            copies=details['copies']
            marker=f'.LTC{size}B'
            row=dict(prefetch_bytes=size,symbol=symbol,**details,
                expected_hint=(all(marker in s for s in copies) if size else all('.LTC' not in s for s in copies)),
                only_ltc_text_difference=[re.sub(r'\.LTC(?:128|256)B','',s) for s in text[symbol]]==text0[symbol])
            rows.append(row)
    passed=all(control_matches.values()) and all(r['expected_hint'] and r['only_ltc_text_difference'] and
        r['all_copies_bypass_l1'] and r['native_u4_s4'] and r['native_s4_s4'] and not r['int8_mma'] for r in rows)
    return dict(passed=passed,control_encoded_sass_matches_best=control_matches,entries=rows,
        sources=[dict(file=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files],
        scope='same_entry_native_INT4_and_only_LTC_instruction_text_change; control_exact_encoded_SASS; not_runtime_safety')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--best-sass',type=Path,required=True)
    a=p.parse_args();result=audit(a.directory,a.best_sass)
    print(json.dumps(result,indent=2))
    if not result['passed']: raise SystemExit(1)


if __name__=='__main__': main()
