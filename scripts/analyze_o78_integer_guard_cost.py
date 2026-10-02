#!/usr/bin/env python3
"""Read-only v65 follow-up: guard simplification and work budget, NOT a kernel.

No timing prediction, authorization for arithmetic changes, or GPU launch.
Use already-hashed norm evidence and existing NCU instruction counts only.
"""
import argparse
import hashlib
import json
from math import isqrt
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
LIMIT=2**31-1


def ceil_sqrt(n):
    if type(n) is not int or n<0: raise ValueError('nonnegative integer required')
    return isqrt(n-1)+1 if n else 0


def rooted_ctas(a,w,tile_m=64,tile_n=128):
    """Conservative integer-root guard; never accepts a failed exact norm gate."""
    ar=[ceil_sqrt(n) for n in a['norm2']];wr=[ceil_sqrt(n) for n in w['norm2']]
    failed=[]
    for m in range(0,len(ar),tile_m):
        for n in range(0,len(wr),tile_n):
            root_ok=max(ar[m:m+tile_m])*max(wr[n:n+tile_n])<=LIMIT
            coeff_ok=max(a['factor_max'][m:m+tile_m])*max(w['factor_max'][n:n+tile_n])<=LIMIT
            if not root_ok or not coeff_ok: failed.append([m//tile_m,n//tile_n])
    return dict(unsafe_cta_coordinates=failed,max_root=max(ar+wr),
                max_norm_bits=max(a['norm2']+w['norm2']).bit_length())


def work_budget(row):
    """Illustrative 3->2 scalar work replacement, NOT guaranteed SASS/speedup."""
    count=4096*4096//32*32
    for op in ('I2F','FMUL','FFMA'):
        if row['opcodes'][op]!=count: raise ValueError('unexpected reference instruction scope')
    # Existing partial=lo+16*hi and all MMA/copy work remain.
    # Remove I2F, scale FMUL, FFMA; add factor multiply and integer MAD.
    saved=count
    return dict(reference_dynamic_warp_instructions=row['dynamic_instructions'],
        old_group_postprocess_warp_instructions=3*count,
        hypothetical_integer_group_warp_instructions=2*count,
        hypothetical_reduction_before_epilogue_and_guard=saved,
        hypothetical_reduction_percent=100*saved/row['dynamic_instructions'],
        final_I2F_warp_instructions=4096*4096//32,
        unchanged_IMMA_warp_instructions=row['opcodes']['IMMA'],
        caveat='instruction-work illustration only; excludes epilogue scale/address/guard/spill changes; not latency or a strict compiler bound')


def analyze():
    root=ROOT/'docs/evidence/a100_o378_roof_v65/reports/o378_roof_v65_trace24'
    manifest=json.loads((root/'summary.json').read_text())
    cases=[]
    for entry in manifest['samples']:
        raw=(root/entry['file']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=entry['sha256']: raise ValueError('evidence hash mismatch')
        case=json.loads(raw);stats=rooted_ctas(case['a'],case['w'])
        exact={tuple(c) for c in entry['guard_unsafe_cta_coordinates']}
        rounded={tuple(c) for c in stats['unsafe_cta_coordinates']}
        if not exact.issubset(rounded): raise AssertionError('non-conservative guard')
        cases.append(dict(sample_id=entry['sample_id'],variant=entry['variant'],
            additional_rejected_ctas=len(rounded-exact),**stats))
    sources=[];budgets=[]
    for variant in ('o7','o8'):
        path=ROOT/f'docs/evidence/a100_o378_roof_v33/reports/o378_roof_v33/ncu_{variant}_analysis.json'
        raw=path.read_bytes();source=json.loads(raw)
        row=next(r for r in source['rows'] if r['tune']==59)
        budgets.append(dict(variant=variant,**work_budget(row)))
        sources.append(dict(file=str(path.relative_to(ROOT)),sha256=hashlib.sha256(raw).hexdigest()))
    return dict(scope='CPU evidence analysis; no CUDA candidate, benchmark, MSE or default change',
        norm_manifest_sha256=hashlib.sha256((root/'summary.json').read_bytes()).hexdigest(),
        ncu_sources=sources,cases=cases,work_budget=budgets,
        max_norm_bits=max(c['max_norm_bits'] for c in cases),
        max_root=max(c['max_root'] for c in cases),
        additional_rejected_ctas=sum(c['additional_rejected_ctas'] for c in cases),
        proposed_layout_bytes=dict(activation_payload_extra_read_if_separate_pass=4096*4096,
            source_scale_codes_per_operand=4096*32,
            integer_factors_per_operand=4096*32*4,
            row_metadata_per_operand=4096*(8+4+4),
            cta_flags=(4096//64)*(4096//128)*4,
            group_square_sums_if_conversion_fused=4096*32*8),
        caveats=['layout byte counts are logical buffer/read work, not measured DRAM traffic',
            'fusing group square sums into conversion avoids an extra payload pass but still needs row reduction and CTA decision',
            'UINT64 norms fit observed samples only; a general implementation must detect overflow and reject/fallback',
            'GPU sqrt implementation must guarantee an upward integer bound; ordinary float sqrt+cast is not a proof',
            'A-dependent preparation is paid in conversion, Cold and steady; static W work can be version-cached',
            'same two INT4 MMA paths: reducing I2F does not reduce the necessary MMA capacity bound'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT):
        parser.error('fresh repository output required')
    result=analyze();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('max_norm_bits','max_root','additional_rejected_ctas','work_budget')},indent=2))


if __name__=='__main__': main()
