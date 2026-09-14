#!/usr/bin/env python3
"""Audit exact SM120 old/magic O1/O3 functions, not probes or unrelated kernels."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def classify(symbol):
    o1 = 'adangel_o1_register_partial_128x64_k64_scale_shared_row_dedup'
    if o1+'_magic' in symbol: return 'o1_magic'
    if o1 in symbol and 'sparse_scale' not in symbol: return 'o1'
    if 'adangel_o3_split_tma_ws' in symbol:
        if 'O3M64N32K128AlignedFactor16WMagicConfig' in symbol: return 'o3_magic'
        if 'O3M64N32K128AlignedFactor16WConfig' in symbol: return 'o3'
    return None


def ptx_entry(ptx, symbol):
    match = re.search(r'\.entry\s+'+re.escape(symbol)+r'\s*\(', ptx)
    if not match: return ''
    start = ptx.find('{', match.end())
    if start < 0: return ''
    end, depth = start+1, 1
    while depth and end < len(ptx):
        depth += (ptx[end] == '{')-(ptx[end] == '}')
        end += 1
    return ptx[start:end] if depth == 0 else ''


def instruction_counts(block):
    result = {op: len(re.findall(r'\b'+op+r'(?:\.|\s)', block))
              for op in ['I2F', 'I2FP', 'IADD3', 'IADD', 'FADD', 'FFMA', 'IMMA', 'UTMALDG', 'LDL', 'STL']}
    # For these four instantiated kernels the producer code precedes the
    # consumer MMA/postprocessing region. Preserve global counts as well;
    # producer scale decoding deliberately still contains integer conversions.
    first = re.search(r'\bIMMA\.', block)
    result['post_mma_i2f'] = (len(re.findall(r'\bI2F(?:P)?\.', block[first.start():]))
                              if first else -1)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('Use a fresh output directory')
    import torch
    from adangel import _sm120 as native
    args.output.mkdir(parents=True)
    outputs = {}
    for flag, name in [('--dump-sass', 'extension.sass'), ('--dump-ptx', 'extension.ptx'),
                       ('--dump-resource-usage', 'resources.txt')]:
        outputs[name] = subprocess.check_output(['cuobjdump', flag, native.__file__], text=True)
        (args.output/name).write_text(outputs[name])
    functions = []
    for block in re.split(r'(?=Function\s*:\s*)', outputs['extension.sass']):
        if not block.startswith('Function'): continue
        symbol = block.splitlines()[0].split(':', 1)[1].strip()
        variant = classify(symbol)
        if variant is None: continue
        ptx = ptx_entry(outputs['extension.ptx'], symbol)
        match = re.search(r'Function\s+(?:\:\s*)?'+re.escape(symbol)+r'\s*:\s*([^\n]+)', outputs['resources.txt'])
        resource = match[0] if match else ''
        local, stack = re.search(r'LOCAL:(\d+)', resource), re.search(r'STACK:(\d+)', resource)
        counts = instruction_counts(block)
        checks = dict(ptx_present=bool(ptx), ptx_tma='cp.async.bulk.tensor' in ptx,
                      sass_tma=counts['UTMALDG'] > 0, sass_imma=counts['IMMA'] > 0,
                      resource_present=bool(local and stack),
                      no_local_stack=bool(local and stack and int(local[1]) == int(stack[1]) == 0),
                      no_spill_instructions=counts['LDL'] == counts['STL'] == 0)
        if variant.startswith('o1'):
            checks['ptx_int8'] = bool(re.search(r'mma\.sync[^;]*\.s32\.s8\.s8\.s32', ptx))
            checks['sass_int8'] = bool(re.search(r'IMMA[^;]*\.S8\.S8', block))
        else:
            checks['ptx_u4s4'] = bool(re.search(r'mma\.sync[^;]*\.s32\.u4\.s4\.s32', ptx))
            checks['ptx_s4s4'] = bool(re.search(r'mma\.sync[^;]*\.s32\.s4\.s4\.s32', ptx))
            # Preserve and explicitly document CUDA 12.8 SM120 legacy lowering.
            checks['sass_int8_lowering'] = bool(re.search(r'IMMA[^;]*\.[SU]8\.[SU]8', block))
        if variant.endswith('_magic'):
            checks['magic_no_post_mma_i2f'] = counts['post_mma_i2f'] == 0
            checks['magic_fadd'] = counts['FADD'] > 0
            checks['magic_iadd'] = counts['IADD3']+counts['IADD'] > 0
        else:
            checks['baseline_post_mma_i2f'] = counts['post_mma_i2f'] > 0
        functions.append(dict(variant=variant, symbol=symbol, resource=resource,
                              instruction_counts=counts, checks=checks,
                              passed=all(checks.values())))
    coverage = len(functions) == 4 and {f['variant'] for f in functions} == {'o1', 'o1_magic', 'o3', 'o3_magic'}
    pairs = {}
    if coverage:
        by_name = {f['variant']: f for f in functions}
        for variant in ['o1', 'o3']:
            old = by_name[variant]['instruction_counts']
            new = by_name[variant+'_magic']['instruction_counts']
            removed = old['I2FP']-new['I2FP']
            pairs[variant] = dict(
                partial_i2fp_to_fadd=removed > 0 and removed == new['FADD']-old['FADD'] == old['post_mma_i2f'],
                producer_i2f_preserved=old['I2F'] == new['I2F'] and old['I2FP']-old['post_mma_i2f'] == new['I2FP'],
                mma_tma_preserved=old['IMMA'] == new['IMMA'] and old['UTMALDG'] == new['UTMALDG'])
    result = dict(binary=native.__file__, binary_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
                  complete_coverage=coverage, functions=functions, paired_checks=pairs,
                  passed=coverage and all(f['passed'] for f in functions) and all(all(x.values()) for x in pairs.values()),
                  note='O3 remains legacy U4/S4 PTX lowered to INT8 IMMA; not native INT4 SASS')
    (args.output/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if not result['passed']: raise SystemExit(1)


if __name__ == '__main__': main()
