#!/usr/bin/env python3
"""v113: full24 lossless sparse-H feasibility; no candidate GEMM or pruning.

Unlike v103 (Q4 W), inspect A's high radix16 digit. Reuse v100's exact signed
representation for O7/O8 only; do not retest its dense kernel. SM80 INT4 sparse
A requires paired4of8, not arbitrary2of4. Keep nonzeros outside the main term
as exact residuals. Fixed K/G128 order, quantizer/scale/default remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import subprocess

import numpy as np

from analyze_sparse_q4_feasibility import (
    weight_statistics, scalar_residual_model, REFERENCE_PROVENANCE, PTX_SOURCE,
)

ROOT=Path(__file__).resolve().parents[1]
POLICIES=(('o3','ordinary'),('o7','ordinary'),('o7','balanced'),
          ('o8','ordinary'),('o8','balanced'))
# Investment budget, NOT a theorem that scalar costs add to MMA time.
DENSE_MMA_FLOOR_MS=.220347
CORRECTION_BUDGET_MS=DENSE_MMA_FLOOR_MS/4


def digits(q,variant,policy):
    q=np.asarray(q)
    if (q.dtype!=np.int8 or q.ndim!=2 or not q.shape[0] or not q.shape[1]
            or q.shape[1]%128 or (variant,policy) not in POLICIES):
        raise ValueError('valid G128 int8 activation and fixed policy required')
    bound={'o3':128,'o7':112,'o8':30}[variant]
    if np.any(q.astype(np.int16)<-bound) or np.any(q.astype(np.int16)>bound):
        raise ValueError('source fixed integer exceeds variant codebook bound')
    wide=q.astype(np.int16)
    high=(wide+(8 if policy=='balanced' else 0))//16
    low=wide-high*16
    if np.any(high<-8) or np.any(high>7):raise ValueError('high does not fit S4')
    if policy=='ordinary':
        if np.any(low<0) or np.any(low>15):raise AssertionError('low U4 failed')
    elif np.any(low<-8) or np.any(low>7):raise AssertionError('low S4 failed')
    if not np.array_equal(low+16*high,wide):raise AssertionError('radix16 reconstruction failed')
    return low.astype(np.int8),high.astype(np.int8)


def high_statistics(q,variant,policy):
    low,high=digits(q,variant,policy)
    s=weight_statistics(high)
    fields=('minimum_residual_nonzeros','minimum_residual_fraction',
            'active_pair_histogram','minimum_residual_nonzeros_per8_histogram',
            'chunks_requiring_residual_fraction','g128_without_residual_fraction',
            'decomposition_exact','ordered_metadata_valid')
    return dict({k:s[k] for k in fields},shape=list(q.shape),elements=q.size,
        high_min=int(high.min()),high_max=int(high.max()),
        high_zero_fraction=s['q4_zero_fraction'],
        source_fixed_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
        low_sha256=hashlib.sha256(low.tobytes()).hexdigest(),
        high_sha256=hashlib.sha256(high.tobytes()).hexdigest(),
        representation_exact=True,low_MMA_type='U4' if policy=='ordinary' else 'S4',
        high_MMA_type='S4',selection='minimum_nonzero_residual_for_fixed_paired4of8',
        no_pruning=True,no_K_reordering=True)


def model(count,residual,n):
    r=scalar_residual_model(count,residual,m=n)
    r.update(output_columns=n,dense_both_routes_MMA_floor_ms=DENSE_MMA_FLOOR_MS,
        sparse_high_dense_low_MMA_floor_ms=DENSE_MMA_FLOOR_MS*.75,
        max_ideal_MMA_capacity_saving_ms=CORRECTION_BUDGET_MS,
        scalar_correction_investment_budget_ms=CORRECTION_BUDGET_MS,
        data_gate_passed=r['scalar_correction_only_optimistic_ms']<=CORRECTION_BUDGET_MS,
        gate_is_investment_filter_not_impossibility_proof=True,
        gather_decode_scale_metadata_conversion_overheads_not_charged=True,
        capacity_model_not_measured_kernel_time=True)
    return r


def summarize(rows):
    if len(rows)!=120 or len({(r['sample_id'],r['variant'],r['policy']) for r in rows})!=120:
        raise ValueError('exactly24x5 unique observations required')
    ids={r['sample_id'] for r in rows}
    if len(ids)!=24 or {(s,v,p) for s,(v,p) in itertools.product(ids,POLICIES)}!={
            (r['sample_id'],r['variant'],r['policy']) for r in rows}:
        raise ValueError('incomplete full24 policy coverage')
    out=[]
    for variant,policy in POLICIES:
        selected=[r for r in rows if (r['variant'],r['policy'])==(variant,policy)]
        costs=[r['model']['scalar_correction_only_optimistic_ms'] for r in selected]
        pass_count=sum(r['model']['data_gate_passed'] for r in selected)
        out.append(dict(variant=variant,policy=policy,samples=24,
            median_high_zero_fraction=statistics.median(r['statistics']['high_zero_fraction'] for r in selected),
            median_residual_fraction=statistics.median(r['statistics']['minimum_residual_fraction'] for r in selected),
            maximum_residual_fraction=max(r['statistics']['minimum_residual_fraction'] for r in selected),
            median_scalar_correction_model_ms=statistics.median(costs),
            maximum_scalar_correction_model_ms=max(costs),data_gate_samples=pass_count,
            investment_gate_passed=pass_count==24))
    return dict(scope='full24_exact_activation_H_data_gate_not_GPU_performance',records=120,
        variants=out,budget_ms=CORRECTION_BUDGET_MS,
        fixed_model_inputs=dict(sm_count=108,clock_mhz=1410,generous_lane_corrections_per_cycle=128),
        interpretation='scalar_correction_budget_not_additive_lower_bound_or_measured_speedup',
        candidate_GEMM_launched=False,new_GEMM_measured=False,new_MSE_measured=False,
        production_default_changed=False,quantization_semantics_changed=False,
        sparse_kernel_implemented=False,ptx_rule=PTX_SOURCE)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/prepared/llama2_7b_prefill_o0_o4'))
    p.add_argument('--raw-data',type=Path,default=Path('data/raw/llama2_7b_prefill'))
    p.add_argument('--trace-config',type=Path,default=Path('configs/trace/llama2_7b_prefill.yaml'))
    p.add_argument('--source-provenance',type=Path,default=REFERENCE_PROVENANCE)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh repository output required')
    import torch
    from adangel.trace.storage import load_prepared,sha256_file
    from adangel.trace.prepare import _load_and_validate_raw
    from adangel.quantization import mixed_formats as mf
    from benchmark_a100_mixed_trace import inspect_inputs,inspect_raw_inputs,source_identity
    torch.set_num_threads(4)
    manifest,prepared_hash=inspect_inputs(args.data)
    raw,raw_hash=inspect_raw_inputs(args.raw_data,manifest,args.trace_config)
    refs=[json.loads(l) for l in args.source_provenance.read_text().splitlines() if l.strip()]
    expected={(r['sample_id'],r['variant']):r for r in refs}
    if len(refs)!=48 or len(expected)!=48:raise ValueError('exact v99 O7/O8 full24 provenance required')
    raw_index={r['sample_id']:r for r in raw['samples']}
    extensions=list((ROOT/'python/adangel').glob('_sm80*.so'))
    if len(extensions)!=1:raise ValueError('one unchanged production extension required')
    extension_hash=sha256_file(extensions[0])
    out.mkdir(parents=True)
    names=('scripts/inspect_activation_high_sparsity.py','scripts/analyze_sparse_q4_feasibility.py',
           'python/adangel/quantization/mixed_formats.py','scripts/benchmark_a100_mixed_trace.py')
    environment=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_hashes={name:sha256_file(ROOT/name) for name in names},torch=torch.__version__,cuda=torch.version.cuda,
        source_device=args.device,prepared_manifest_sha256=prepared_hash,raw_manifest_sha256=raw_hash,
        v99_provenance_sha256=sha256_file(args.source_provenance),
        native_extension_sha256_before=extension_hash,no_candidate_GEMM_or_timing=True,
        no_input_data_saved_or_changed=True,source_quantizer_reused=True,ptx_rule=PTX_SOURCE,
        budget_predeclared_ms=CORRECTION_BUDGET_MS,
        no_repeat=['v103_Q4_weight_sparsity','v100_dense_balanced_radix16','v112_direct_fragment_feed'])
    (out/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    rows=[]
    for entry in manifest['samples']:
        sid=entry['sample_id'];re=raw_index[sid]
        prepared=load_prepared(args.data/entry['file'],device='cpu')
        q3=prepared.A_int8.numpy()
        record=_load_and_validate_raw(args.raw_data/re['file'],re['layer'],re['projection'])
        variants=[('o3',q3,dict(prepared_sha256=entry['sha256'],original_A_int8=True))]
        for variant in ('o7','o8'):
            source=mf.quantize_source(record['activation_fp16'].to(args.device),mf.VARIANTS[variant][1])
            identity=source_identity(source);ref=expected[sid,variant]
            if ref['raw_sha256']!=re['sha256'] or identity!=ref['activation']:
                raise ValueError('activation source differs from measured v99: '+sid+'/'+variant)
            q,scale=mf.to_fixed_reference(source)
            variants.append((variant,q.cpu().numpy(),dict(v99_source_exact=True,activation_source=identity)))
            del source,q,scale
        for variant,q,provenance in variants:
            for _,policy in (pair for pair in POLICIES if pair[0]==variant):
                measured=high_statistics(q,variant,policy)
                row=dict(sample_id=sid,variant=variant,policy=policy,raw_sha256=re['sha256'],
                    provenance=provenance,statistics=measured,
                    model=model(measured['elements'],measured['minimum_residual_nonzeros'],prepared.shape[1]))
                rows.append(row)
                with (out/'results.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(sid,'exact sources and five high-digit observations passed',flush=True)
        del record,prepared,variants,q3,q
    environment['native_extension_sha256_after']=sha256_file(extensions[0])
    if environment['native_extension_sha256_after']!=extension_hash:raise AssertionError('extension changed')
    (out/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    result=summarize(rows)
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
