#!/usr/bin/env python3
"""Audit instantiated SM80 O1/O3 functions; retain explicit spill diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def audit_policy(checks, allow_spills=False):
    """Keep raw checks intact; only known spill checks can become warnings."""
    spill_checks={'sass_no_local','resource_no_local','resource_no_stack'}
    failed=[name for name,value in checks.items() if not value]
    warnings=[name for name in failed if allow_spills and name in spill_checks]
    errors=[name for name in failed if name not in warnings]
    return dict(passed=not errors,strict_passed=not failed,
                errors=errors,warnings=warnings)


def roof_scale_checks(symbol, instruction_counts):
    """Recognize only guarded dual-scale specializations, not probes."""
    if not re.search(r'adangel_sm80_roof_candidateILb1ELb1ELi1[12]EE', symbol):
        return {}
    return dict(power2_scale_no_fmul=instruction_counts.get('FMUL', 0)==0,
                power2_scale_keeps_i2f=instruction_counts.get('I2F', 0)>0,
                power2_scale_keeps_ffma=instruction_counts.get('FFMA', 0)>0)


def roof_pipeline_checks(symbol, ptx, sass):
    """Require overlapping and draining waits in the same three-stage entry."""
    if not re.search(r'adangel_sm80_roof_candidateILb[01]ELb[01]ELi(?:1[6-9]|23|31|33|42|46|48|50|52|54|56|58)EE',symbol):
        return {}
    return dict(pipeline_ptx_wait_one=bool(re.search(r'cp\.async\.wait_group\s+1\s*;',ptx)),
                pipeline_ptx_drain=bool(re.search(r'cp\.async\.wait_group\s+0\s*;',ptx)),
                pipeline_sass_wait_one=bool(re.search(r'DEPBAR\.LE\s+SB\d+,\s*0x1\b',sass)),
                pipeline_sass_drain=bool(re.search(r'DEPBAR\.LE\s+SB\d+,\s*0x0\b',sass)))


def roof_async_payload_checks(symbol, ptx, sass):
    if not re.search(r'adangel_sm80_roof_candidateILb1ELb0ELi5[5-8]EE',symbol):
        return {}
    # These kernels have no other scalar global inputs: payload and both scale
    # panels must all arrive through LDGSTS. Do not confuse LDG with LDGSTS.
    return dict(async_scales_no_scalar_ldg=not bool(re.search(r'\bLDG(?:\.|\s)',sass)),
                async_scales_no_scalar_sts=not bool(re.search(r'\bSTS(?:\.|\s)',sass)),
                async_scales_ptx_commit='cp.async.commit_group' in ptx,
                async_scales_ptx_wait='cp.async.wait_group' in ptx,
                async_scales_cta_barrier=bool(re.search(r'\bBAR\.SYNC',sass)))


def roof_reduction_checks(symbol, counts):
    if re.search(r'adangel_sm80_roof_candidateILb[01]ELb[01]ELi(?:3[4-9]|40)EE',symbol):
        return dict(products_keep_i2f=counts.get('I2F',0)>0,
                    products_keep_fmul=counts.get('FMUL',0)>0,
                    products_keep_fadd=counts.get('FADD',0)>0,
                    products_not_contracted=counts.get('FFMA',0)==0)
    if not re.search(r'adangel_sm80_roof_candidateILb[01]ELb[01]ELi2[4-7]EE',symbol):
        return {}
    # Does not prove the dependency graph; numeric tests cover the mapping.
    return dict(reduction_keeps_i2f=counts.get('I2F',0)>0,
                reduction_keeps_ffma=counts.get('FFMA',0)>0,
                reduction_has_final_fadd=counts.get('FADD',0)>0)


def roof_paired_pipeline_checks(symbol, ptx, sass):
    """Paired ring drains the copy pair before consumers; unlike23 no wait1."""
    if not re.search(r'adangel_sm80_roof_candidateILb[01]ELb[01]ELi(?:29|39|40)EE',symbol):
        return {}
    return dict(paired_ptx_drain=bool(re.search(r'cp\.async\.wait_group\s+0\s*;',ptx)),
                paired_ptx_commit='cp.async.commit_group' in ptx,
                paired_ptx_barrier=bool(re.search(r'bar\.sync',ptx)),
                paired_sass_drain=bool(re.search(r'DEPBAR\.LE\s+SB\d+,\s*0x0\b',sass)),
                paired_sass_barrier=bool(re.search(r'\bBAR\.SYNC',sass)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variant',choices=['o1','o3','split_grouped','mixed_binary','roof_candidate'],default='o1')
    p.add_argument('--allow-spills',action='store_true',
                   help='INT4 paths: report spill checks as warnings, never waive ISA checks or missing resource data')
    args=p.parse_args()
    if args.allow_spills and args.variant=='o1': p.error('--allow-spills is only supported for INT4 paths')
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
        prefix=('adangel_sm80_'+args.variant) if args.variant in ('split_grouped','mixed_binary','roof_candidate') else f'adangel_sm80_{args.variant}_swizzled'
        if prefix not in symbol: continue
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
            resource_metadata_present=bool(local and stack),
            sass_async='LDGSTS' in block, sass_no_local=not bool(re.search(r'\b(?:LDL|STL)\b',block)),
            resource_no_local=bool(local and int(local[1])==0),
            resource_no_stack=bool(stack and int(stack[1])==0),
            ptx_async='cp.async' in ptx,
            ptx_int8=bool(re.search(r'mma\.sync[^;]*\.s32\.s8\.s8\.s32',ptx)))
        if args.variant in ('o3','split_grouped','roof_candidate'):
            del checks['sass_int8'];del checks['ptx_int8']
            checks.update(sass_u4s4=bool(re.search(r'IMMA\.\w+\.U4\.S4',block)),
                sass_s4s4=bool(re.search(r'IMMA\.\w+\.S4\.S4',block)),
                sass_no_int8=not bool(re.search(r'IMMA[^;]*\.[SU]8',block)),
                ptx_u4s4=bool(re.search(r'mma\.sync[^;]*\.s32\.u4\.s4\.s32',ptx)),
                ptx_s4s4=bool(re.search(r'mma\.sync[^;]*\.s32\.s4\.s4\.s32',ptx)))
        if args.variant=='mixed_binary':
            del checks['sass_int8'];del checks['ptx_int8']
            checks.update(sass_binary=bool(re.search(r'\bBMMA\.[^;]*\.AND',block)),
                sass_no_integer_mma=not bool(re.search(r'\bIMMA\.',block)),
                ptx_binary=bool(re.search(r'mma\.sync[^;]*\.m16n8k128[^;]*\.s32\.b1\.b1\.s32\.and\.popc',ptx)))
        # Itanium template arguments: ExponentScale=true, PairMma=false,
        # MagicCast=true. Match both streaming and non-streaming instances.
        magic='Lb1ELb0ELb1E' in symbol
        if args.variant=='roof_candidate': magic=False
        # Decode explicit O3 template arguments; do not infer Magic from Fast
        # or accidentally match the independent integer-merge boolean.
        if args.variant=='o3':
            args_match=re.search(r'swizzled(?:_bound2)?ILi\d+ELi\d+ELi\d+ELb([01])ELb([01])ELb([01])ELi[24]ELb([01])E',symbol)
            if not args_match: raise RuntimeError(f'Unknown O3 template signature: {symbol}')
            magic=args_match[3]=='1'
        instruction_counts={op:len(re.findall(r'\b'+op+r'(?:\.|\s)',block))
                            for op in ('I2F','FADD','FMUL','FFMA','IMAD','IADD3','IMMA','BMMA','LDSM','LDGSTS','LDL','STL')}
        instruction_counts['STG64']=len(re.findall(r'\bSTG(?:\.[A-Z]+)*\.64\b',block))
        instruction_counts['LDG_U16']=len(re.findall(r'\bLDG(?:\.[A-Z]+)*\.U16\b',block))
        if magic:
            checks['magic_no_i2f']=instruction_counts['I2F']==0
            checks['magic_has_fadd']=instruction_counts['FADD']>0
        if args.variant=='roof_candidate':
            checks.update(roof_scale_checks(symbol,instruction_counts))
            checks.update(roof_pipeline_checks(symbol,ptx,block))
            checks.update(roof_async_payload_checks(symbol,ptx,block))
            checks.update(roof_reduction_checks(symbol,instruction_counts))
            checks.update(roof_paired_pipeline_checks(symbol,ptx,block))
        functions.append(dict(symbol=symbol,checks=checks,**audit_policy(checks,args.allow_spills),resource=resource,
                              instruction_counts=instruction_counts))
    passed=bool(functions) and all(f['passed'] for f in functions)
    strict_passed=bool(functions) and all(f['strict_passed'] for f in functions)
    policy='spill_warnings_isa_required' if args.allow_spills else 'zero_spill_required'
    (args.output/'audit.json').write_text(json.dumps(dict(binary=binary,
        binary_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
        passed=passed,strict_passed=strict_passed,policy=policy,functions=functions),indent=2)+'\n')
    print(json.dumps(dict(passed=passed,strict_passed=strict_passed,policy=policy,functions=functions),indent=2))
    if not passed: raise SystemExit(1)


if __name__=='__main__': main()
