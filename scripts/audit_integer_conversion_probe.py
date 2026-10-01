#!/usr/bin/env python3
"""Audit only v34 conversion kernels; this is NOT a Tensor Core audit."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def analyze(sass,resources):
    rows=[]
    for block in re.split(r'(?=Function\s*:\s*)',sass):
        if not block.startswith('Function'): continue
        symbol=block.splitlines()[0].split(':',1)[1].strip()
        if 'adangel_sm80_integer_fixed_conversion' not in symbol: continue
        match=re.search(r'Function\s+(?:\:\s*)?'+re.escape(symbol)+r'\s*:\s*([^\n]+)',resources)
        resource=match[0] if match else ''
        fields={k:int(v) for k,v in re.findall(r'(REG|LOCAL|STACK):(\d+)',resource)}
        checks=dict(resource_present=set(fields)=={'REG','LOCAL','STACK'},
            no_local_sass=not bool(re.search(r'\b(?:LDL|STL)(?:\.|\s)',block)),
            no_float_to_integer=not bool(re.search(r'\bF2I(?:\.|\s)',block)),
            no_stack=fields.get('STACK')==0,no_local_resource=fields.get('LOCAL')==0)
        rows.append(dict(symbol=symbol,checks=checks,passed=all(checks.values()),resource=resource))
    return dict(passed=len(rows)==8 and all(r['passed'] for r in rows),functions=rows,
        scope='conversion_only_integer_payload_and_resource_audit_not_mma')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): p.error('fresh output required')
    import torch
    from adangel import _sm80 as native
    args.output.mkdir(parents=True)
    binary=Path(native.__file__)
    outputs={}
    for flag,name in (('--dump-sass','extension.sass'),('--dump-resource-usage','resources.txt')):
        outputs[name]=subprocess.check_output(['cuobjdump','-arch','sm_80',flag,str(binary)],text=True)
        (args.output/name).write_text(outputs[name])
    report=analyze(outputs['extension.sass'],outputs['resources.txt'])
    report['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__': main()
