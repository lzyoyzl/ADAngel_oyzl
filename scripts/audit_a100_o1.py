#!/usr/bin/env python3
"""Audit every instantiated optimized O1 function, never a probe or old kernel."""
import argparse
import json
from pathlib import Path
import re
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variant',choices=['o1','o3'],default='o1')
    args=p.parse_args()
    if args.output.exists(): raise SystemExit('Use a fresh output directory')
    import torch  # load libc10 before the extension
    from adangel import _sm80 as native
    args.output.mkdir(parents=True)
    binary=native.__file__
    outputs={}
    for flag,name in [('--dump-sass','extension.sass'),('--dump-ptx','extension.ptx'),
                      ('--dump-resource-usage','resources.txt')]:
        outputs[name]=subprocess.check_output(['cuobjdump',flag,binary],text=True)
        (args.output/name).write_text(outputs[name])
    functions=[]
    for block in re.split(r'(?=Function\s*:\s*)',outputs['extension.sass']):
        if not block.startswith('Function'): continue
        symbol=block.splitlines()[0].split(':',1)[1].strip()
        if f'adangel_sm80_{args.variant}_swizzled' not in symbol: continue
        rm=re.search(r'Function\s+(?:\:\s*)?'+re.escape(symbol)+r'\s*:\s*([^\n]+)',outputs['resources.txt'])
        resource=rm[0] if rm else ''
        local=re.search(r'LOCAL:(\d+)',resource)
        stack=re.search(r'STACK:(\d+)',resource)
        entry=re.search(r'\.entry\s+'+re.escape(symbol)+r'\s*\(',outputs['extension.ptx'])
        ptx=''
        if entry:
            start=outputs['extension.ptx'].find('{',entry.end())
            depth=1; end=start+1
            while depth and end<len(outputs['extension.ptx']):
                depth+=(outputs['extension.ptx'][end]=='{')-(outputs['extension.ptx'][end]=='}')
                end+=1
            ptx=outputs['extension.ptx'][start:end]
        checks=dict(sass_int8=bool(re.search(r'IMMA\.\w+\.S8\.S8',block)),
            sass_async='LDGSTS' in block, sass_no_local=not bool(re.search(r'\b(?:LDL|STL)\b',block)),
            resource_no_local=bool(local and int(local[1])==0),
            resource_no_stack=bool(stack and int(stack[1])==0),
            ptx_async='cp.async' in ptx,
            ptx_int8=bool(re.search(r'mma\.sync[^;]*\.s32\.s8\.s8\.s32',ptx)))
        if args.variant=='o3':
            del checks['sass_int8'];del checks['ptx_int8']
            checks.update(sass_u4s4=bool(re.search(r'IMMA\.\w+\.U4\.S4',block)),
                sass_s4s4=bool(re.search(r'IMMA\.\w+\.S4\.S4',block)),
                sass_no_int8=not bool(re.search(r'IMMA[^;]*\.[SU]8',block)),
                ptx_u4s4=bool(re.search(r'mma\.sync[^;]*\.s32\.u4\.s4\.s32',ptx)),
                ptx_s4s4=bool(re.search(r'mma\.sync[^;]*\.s32\.s4\.s4\.s32',ptx)))
        # Itanium template arguments: ExponentScale=true, PairMma=false,
        # MagicCast=true. Match both streaming and non-streaming instances.
        magic='Lb1ELb0ELb1E' in symbol
        if args.variant=='o3': magic='Lb1ELb' in symbol
        instruction_counts={op:len(re.findall(r'\b'+op+r'(?:\.|\s)',block))
                            for op in ('I2F','FADD','FFMA','IMMA','LDSM','LDGSTS','LDL','STL')}
        if magic:
            checks['magic_no_i2f']=instruction_counts['I2F']==0
            checks['magic_has_fadd']=instruction_counts['FADD']>0
        functions.append(dict(symbol=symbol,checks=checks,passed=all(checks.values()),resource=resource,
                              instruction_counts=instruction_counts))
    passed=bool(functions) and all(f['passed'] for f in functions)
    (args.output/'audit.json').write_text(json.dumps(dict(binary=binary,passed=passed,functions=functions),indent=2)+'\n')
    print(json.dumps(dict(passed=passed,functions=functions),indent=2))
    if not passed: raise SystemExit(1)


if __name__=='__main__': main()
