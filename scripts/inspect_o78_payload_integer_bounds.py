#!/usr/bin/env python3
"""v65 read-only gate: payload-aware separable full-K INT32 bounds.

No new CUDA kernel, source quantizer, payload format, default or timing result.
The common tensor scale is factored in REAL arithmetic; existing FP32 rounding
points are not preserved. Passing this gate is NOT MSE/performance acceptance.
"""
import argparse
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
import subprocess
import time

from inspect_o78_integer_alignment_feasibility import ROOT, INT32_MAX, dyadic


def weighted_norms(scale_codes, group_sum_squares, kind):
    """Exact Python integers; no FP sqrt, capped shift or int64 overflow.

    For each row: scale[g] = factor[g] * 2**anchor, factor[g] >= 0.
    norm2 = sum_g sum_k(q[g,k]**2) * factor[g]**2.
    Anchor includes all nonzero source scales, as in the prior separable gate;
    it is not silently changed to omit payload zeros.
    """
    if not scale_codes or len(scale_codes)!=len(group_sum_squares):
        raise ValueError('nonempty matching row sequences required')
    norms=[];anchors=[];factors_max=[]
    for codes,squares in zip(scale_codes,group_sum_squares):
        if not codes or len(codes)!=len(squares):
            raise ValueError('matching nonempty group sequences required')
        if any(type(v) is not int or v<0 for v in squares):
            raise ValueError('sum of squares must be nonnegative integers')
        values=[dyadic(c,kind) for c in codes]
        anchor=min((e for m,e in values if m),default=0)
        factors=[m << (e-anchor) if m else 0 for m,e in values]
        norms.append(sum(v*f*f for v,f in zip(squares,factors)))
        anchors.append(anchor);factors_max.append(max(factors))
    return dict(norm2=norms,anchors=anchors,factor_max=factors_max)


def inspect_norms(a,w,tile_m=64,tile_n=128):
    """Cauchy controls the absolute sum, hence every group/prefix, not only Y.

    sum_g |P_g| * af_g * wf_g <= sqrt(A_norm2[m] * W_norm2[n]).
    A separate conservative factor gate protects af*wf before the INT32 IMAD.
    Counts describe a proposed guard, not observed arithmetic overflow.
    """
    if tile_m<1 or tile_n<1:
        raise ValueError('positive tiles required')
    an,wn=a['norm2'],w['norm2']
    if not an or not wn or any(v<0 for v in an+wn):
        raise ValueError('nonnegative nonempty norms required')
    if len(an)!=len(a['factor_max']) or len(wn)!=len(w['factor_max']):
        raise ValueError('factor shape mismatch')
    sorted_w=sorted(wn)
    count=lambda limit: sum(len(wn) if v==0 else bisect_right(sorted_w,limit*limit//v) for v in an)
    safe=count(INT32_MAX)
    safe24=count(2**24)
    max_coefficient=max(a['factor_max'])*max(w['factor_max'])
    safe_tiles=0;tiles=0;covered=0;unsafe_coordinates=[]
    for m in range(0,len(an),tile_m):
        for n in range(0,len(wn),tile_n):
            tiles+=1
            norm_ok=max(an[m:m+tile_m])*max(wn[n:n+tile_n])<=INT32_MAX**2
            factor_ok=max(a['factor_max'][m:m+tile_m])*max(w['factor_max'][n:n+tile_n])<=INT32_MAX
            if norm_ok and factor_ok:
                safe_tiles+=1;covered+=min(tile_m,len(an)-m)*min(tile_n,len(wn)-n)
            else:
                unsafe_coordinates.append([m//tile_m,n//tile_n])
    total=len(an)*len(wn)
    return dict(outputs=total,int32_norm_safe_outputs=safe,int32_norm_safe_fraction=safe/total,
        bound_at_most_2pow24_outputs=safe24,maximum_norm_product=max(an)*max(wn),
        maximum_coefficient_bound=max_coefficient,coefficient_int32_safe=max_coefficient<=INT32_MAX,
        all_outputs_guaranteed_safe=safe==total and max_coefficient<=INT32_MAX,
        cta_tile_mn=[tile_m,tile_n],ctas=tiles,guard_safe_ctas=safe_tiles,
        guard_safe_cta_fraction=safe_tiles/tiles,guard_safe_cta_outputs=covered,
        guard_unsafe_cta_coordinates=unsafe_coordinates)


def export_norm_evidence(source,output):
    """Publish all row/column norms without duplicating 32-group statistics.

    Full group-level files remain on A100 and in the transfer archive. Retain
    the one rejected sample in full so its weighted norms can be independently
    rebuilt, not merely rechecked against the already-produced norms.
    """
    source=source.resolve();output=output.resolve()
    if not source.is_relative_to(ROOT) or not output.is_relative_to(ROOT) or output.exists():
        raise ValueError('repository input and fresh repository output required')
    raw=(source/'summary.json').read_bytes();result=json.loads(raw)
    output.mkdir(parents=True)
    for entry in result['samples']:
        if Path(entry['file']).name!=entry['file']: raise ValueError('unsafe source filename')
        content=(source/entry['file']).read_bytes()
        if hashlib.sha256(content).hexdigest()!=entry['sha256']: raise ValueError('source evidence hash mismatch')
        case=json.loads(content)
        keep_groups=not entry['all_outputs_guaranteed_safe']
        if not keep_groups:
            for side in ('a','w'):
                case[side].pop('scale_codes');case[side].pop('group_sum_squares')
        case['group_statistics_retained']=keep_groups
        encoded=(json.dumps(case,separators=(',',':'),allow_nan=False)+'\n').encode()
        (output/entry['file']).write_bytes(encoded)
        entry['full_source_sha256']=entry['sha256']
        entry['sha256']=hashlib.sha256(encoded).hexdigest()
    result['evidence_export']=dict(original_directory=source.name,
        original_summary_sha256=hashlib.sha256(raw).hexdigest(),
        retained='all per-row norms/anchors/factor bounds; full groups for rejected samples',
        exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/prepared/llama2_7b_prefill_o0_o4'))
    p.add_argument('--raw-data',type=Path,default=Path('data/raw/llama2_7b_prefill'))
    p.add_argument('--trace-config',type=Path,default=Path('configs/trace/llama2_7b_prefill.yaml'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,choices=(4,24),default=4)
    p.add_argument('--export-from',type=Path,help='export existing evidence; no GPU or new inspection')
    args=p.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT):
        p.error('fresh repository output required')
    if args.export_from:
        export_norm_evidence(args.export_from,args.output)
        return
    import torch
    from adangel.quantization import mixed_formats as mf
    from adangel.trace.prepare import _load_and_validate_raw
    from adangel.trace.storage import sha256_file
    from benchmark_a100_mixed_trace import inspect_inputs,inspect_raw_inputs,tensor_identity
    torch.cuda.init();torch.set_num_threads(4)
    if torch.cuda.get_device_capability()!=(8,0):
        p.error('A100 SM80 evidence required')
    manifest,ph=inspect_inputs(args.data)
    raw,rh=inspect_raw_inputs(args.raw_data,manifest,args.trace_config)
    args.output.mkdir(parents=True)
    results=[];start=time.monotonic()
    for entry in raw['samples'][:args.samples]:
        path=args.raw_data/entry['file']
        if sha256_file(path)!=entry['sha256']: raise ValueError('raw trace changed')
        record=_load_and_validate_raw(path,entry['layer'],entry['projection'])
        activation=record['activation_fp16'].cuda();weight=record['weight_fp16'].cuda()
        for variant,(wf,af) in mf.VARIANTS.items():
            asrc=mf.quantize_source(activation,af);wsrc=mf.quantize_source(weight,wf)
            # These are the existing scalar-semantics fixed-point references.
            # No new quantization rule or precision is introduced by the gate.
            aq,_=mf.to_fixed_reference(asrc);wq,_=mf.to_fixed_reference(wsrc)
            def side(source,q,kind):
                assert q.dtype==torch.int8 and tuple(q.shape)==(4096,4096)
                sq=q.reshape(4096,32,128).long().square().sum(-1)
                codes=source['scale'].cpu().tolist();squares=sq.cpu().tolist()
                v=weighted_norms(codes,squares,kind)
                v.update(scale_codes=codes,group_sum_squares=squares,
                    payload_identity=tensor_identity(q),scale_identity=tensor_identity(source['scale']),
                    kind=kind,abs_payload_max=int(q.int().abs().max().item()))
                return v
            a=side(asrc,aq,'ue8m0' if variant=='o7' else 'e4m3')
            w=side(wsrc,wq,'e4m3' if variant=='o7' else 'e6m2')
            stats=inspect_norms(a,w)
            sample=dict(sample_id=entry['sample_id'],variant=variant,raw_sha256=entry['sha256'],
                activation_format=af,weight_format=wf,a=a,w=w,stats=stats,
                common_real_factor='4*T_W' if variant=='o7' else 'T_A/4',
                tensor_scale=float((wsrc if variant=='o7' else asrc)['tensor_scale'].item()))
            name=f"{entry['sample_id']}_{variant}.json"
            (args.output/name).write_text(json.dumps(sample,indent=2,allow_nan=False)+'\n')
            results.append(dict(sample_id=entry['sample_id'],variant=variant,file=name,
                sha256=sha256_file(args.output/name),**stats))
            print(entry['sample_id'],variant,'outputs_safe=',stats['int32_norm_safe_fraction'],
                'ctas_safe=',stats['guard_safe_cta_fraction'],'coefficient_safe=',stats['coefficient_int32_safe'],flush=True)
            del asrc,wsrc,aq,wq,a,w
        del record,activation,weight
    summary=[]
    for variant in mf.VARIANTS:
        selected=[r for r in results if r['variant']==variant]
        total=sum(r['outputs'] for r in selected);tiles=sum(r['ctas'] for r in selected)
        summary.append(dict(variant=variant,samples=len(selected),
            int32_norm_safe_fraction=sum(r['int32_norm_safe_outputs'] for r in selected)/total,
            all_outputs_safe_samples=sum(r['all_outputs_guaranteed_safe'] for r in selected),
            guard_safe_cta_fraction=sum(r['guard_safe_ctas'] for r in selected)/tiles,
            coefficient_safe_samples=sum(r['coefficient_int32_safe'] for r in selected),
            maximum_coefficient_bound=max(r['maximum_coefficient_bound'] for r in selected)))
    if sha256_file(args.data/'manifest.json')!=ph or sha256_file(args.raw_data/'trace_manifest.json')!=rh:
        raise ValueError('manifest changed during inspection')
    result=dict(scope='read_only_payload_range_feasibility_not_CUDA_MSE_or_performance_acceptance',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        script_sha256=sha256_file(Path(__file__)),prepared_manifest_sha256=ph,raw_manifest_sha256=rh,
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),wall_seconds=time.monotonic()-start,
        summary=summary,samples=results,
        caveats=['source scales factored in real arithmetic, not existing FP32 rounding identity',
                 'failure of sufficient bound is NOT observed arithmetic overflow',
                 'no new kernel, performance measurement, MSE result, default or quantization change',
                 'activation-dependent guard must be paid online; no timing claim for this Python oracle'])
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
