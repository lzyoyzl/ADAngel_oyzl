#!/usr/bin/env python3
"""Same-entry native INT4 audit and exact control for guarded/shared-parameter full-K integer alignment."""
import argparse
import hashlib
import json
from pathlib import Path

from compare_a100_codegen import instructions
from probe_roof_fullk_integer_codegen import PATTERN, static_entries


def audit(directory,best_sass):
    sources=[directory/f'fullk_integer_{i}.sass' for i in (0,1)]
    payloads=[p.read_text() for p in sources]
    old=instructions(best_sass.read_text(),r'adangel_sm80_roof_candidateILb[01]ELb0ELi(?:54|59)EE')
    reference={('adangel_roof_fullk_integer_o3' if 'ELi54EE' in symbol else 'adangel_roof_fullk_integer_o78'):words
               for symbol,words in old.items()}
    control=instructions(payloads[0],PATTERN)
    matched={key:control[key] for key in reference if key in control}
    if set(matched)!=set(reference) or len(old)!=2:
        raise ValueError('missing or ambiguous best/control pair')
    exact={key:matched[key]==reference[key] for key in matched}
    rows=[]
    for policy,payload in enumerate(payloads):
        words=instructions(payload,PATTERN)
        for symbol,details in static_entries(payload).items():
            rows.append(dict(fullk_integer=policy,symbol=symbol,**details,
                encoded_identical_to_control=words[symbol]==control[symbol]))
    return dict(passed=all(exact.values()) and
        all(r['encoded_identical_to_control'] for r in rows if r['symbol'].endswith(('_o78',))) and all(r['all_copies_bypass_l1'] and
        r['native_u4_s4'] and r['native_s4_s4'] and not r['int8_mma'] for r in rows),
        control_encoded_sass_matches_best=exact,entries=rows,
        scope='same-entry native INT4; exact control; candidate correctness still needs GPU checks',
        sources=[dict(file=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources+[best_sass]])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--best-sass',type=Path,required=True)
    a=p.parse_args();result=audit(a.directory,a.best_sass)
    print(json.dumps(result,indent=2))
    if not result['passed']: raise SystemExit(1)


if __name__=='__main__': main()
