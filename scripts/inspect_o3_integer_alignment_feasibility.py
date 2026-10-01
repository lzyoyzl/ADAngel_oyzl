#!/usr/bin/env python3
"""Read-only O3 scale/bound inspection; no alternative GEMM or MSE experiment.

This does NOT authorize/implement cross-G128 integer accumulation. All bounds
assume the original two-route INT8 x Q4 G128 partial, with independent scales.
It is a data/overflow feasibility report, not a performance prediction.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
PARTIAL_BOUND=128*128*8  # conservative across the full signed INT8 / INT4 domains
INT32_MAX=2**31-1
EXACT_FP32_INTEGER_BOUND=2**24
WINDOWS=(2,4,8,32)


def aligned_integer_bound(codes):
    """UE8M0 value 2**(code-127), including code0; code255 is not a scale.

    q_g*2**e_g = (q_g*2**(e_g-min(e)))*2**min(e).
    This Python integer bound cannot overflow and never truncates a scale.
    The bound includes all intermediate prefix sums by triangle inequality.
    """
    if not codes or any(type(c) is not int or not 0<=c<=254 for c in codes):
        raise ValueError('nonempty integer UE8M0 codes in [0,254] required')
    anchor=min(codes)
    return PARTIAL_BOUND*sum(1<<(c-anchor) for c in codes)


def inspect_scale_rows(rows,window):
    if window not in WINDOWS or not rows:
        raise ValueError('supported window and nonempty rows required')
    groups=len(rows[0])
    if not groups or groups%window or any(len(row)!=groups for row in rows):
        raise ValueError('uniform group count divisible by window required')
    deltas=Counter();safe=exact=windows=zero=0;largest=0;columns_safe=0
    for row in rows:
        column_safe=True
        for first in range(0,groups,window):
            codes=row[first:first+window]
            bound=aligned_integer_bound(codes)
            deltas[max(codes)-min(codes)]+=1
            okay=bound<=INT32_MAX
            safe+=okay;column_safe &= okay
            exact+=bound<=EXACT_FP32_INTEGER_BOUND
            zero+=0 in codes
            windows+=1;largest=max(largest,bound)
        columns_safe+=column_safe
    return dict(window_groups=window,columns=len(rows),groups=groups,windows=windows,
        int32_safe_windows=safe,int32_unsafe_windows=windows-safe,
        int32_safe_fraction=safe/windows,all_windows_safe_columns=columns_safe,
        exact_fp32_integer_bound_windows=exact,zero_code_windows=zero,
        maximum_absolute_integer_bound=largest,
        exponent_spread_histogram={str(k):deltas[k] for k in sorted(deltas)},
        note='integer bound only; not FP32-output equality, MSE, speed, or kernel safety')


def aggregate(rows,window):
    entries=[r for sample in rows for r in sample['windows'] if r['window_groups']==window]
    if len(entries)!=len(rows) or not entries: raise ValueError('incomplete sample coverage')
    result={key:sum(e[key] for e in entries) for key in
        ('columns','windows','int32_safe_windows','int32_unsafe_windows',
         'all_windows_safe_columns','exact_fp32_integer_bound_windows','zero_code_windows')}
    result.update(window_groups=window,samples=len(rows),
        maximum_absolute_integer_bound=max(e['maximum_absolute_integer_bound'] for e in entries))
    result['int32_safe_fraction']=result['int32_safe_windows']/result['windows']
    histogram=Counter()
    for entry in entries:
        histogram.update({int(k):v for k,v in entry['exponent_spread_histogram'].items()})
    result['exponent_spread_histogram']={str(k):histogram[k] for k in sorted(histogram)}
    if sum(histogram.values())!=result['windows']: raise ValueError('histogram coverage mismatch')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/prepared/llama2_7b_prefill_o0_o4'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT):
        p.error('fresh output directory inside repository required')
    import torch
    from adangel.trace.storage import load_prepared,sha256_file
    from benchmark_a100_mixed_trace import inspect_inputs
    torch.set_num_threads(4)
    manifest,digest=inspect_inputs(a.data)
    samples=[]
    for entry in manifest['samples']:
        path=a.data/entry['file']
        if sha256_file(path)!=entry['sha256']: raise ValueError('sample changed after validation')
        x=load_prepared(path,device='cpu')
        if x.sample_id!=entry['sample_id'] or list(x.shape)!=[4096,4096,4096]:
            raise ValueError('sample identity or shape mismatch')
        scale=x.W_scale_g128
        if scale is None or scale.dtype!=torch.uint8 or list(scale.shape)!=[4096,32]:
            raise ValueError('natural UE8M0 G128 scale contract mismatch')
        values=scale.tolist()
        windows=[inspect_scale_rows(values,w) for w in WINDOWS]
        samples.append(dict(sample_id=x.sample_id,file=entry['file'],sha256=entry['sha256'],
            scale_code_min=int(scale.min()),scale_code_max=int(scale.max()),windows=windows))
        print(x.sample_id,'inspected on CPU',flush=True)
    result=dict(scope='read_only_O3_integer_bound_feasibility_no_new_kernel_or_MSE_or_timing',
        input_policy='existing_O3_G128_Q4_prepared_trace_unmodified',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        prepared_manifest_sha256=digest,source_trace=manifest.get('source_trace'),
        partial_bound=PARTIAL_BOUND,int32_max=INT32_MAX,fp32_exact_integer_bound=EXACT_FP32_INTEGER_BOUND,
        assumptions=['UE8M0 decoded as 2**(code-127); reject255',
            'original independent G128 scales; anchor is only an integer coordinate',
            'integer shifts are mathematical nonnegative multiply-by-power2; C++ signed left-shift is not prescribed',
            'int32-safe bound includes any prefix order by triangle inequality',
            'FP32 reassociation, register pressure, new instructions, and fallback still require approval and GPU validation',
            'coverage is per weight column/window, not all possible future inputs',
            'O7/O8 non-power2 multi-level scales are outside this analysis'],
        samples=samples,summary=[aggregate(samples,w) for w in WINDOWS])
    if sha256_file(a.data/'manifest.json')!=digest: raise ValueError('manifest changed while reading')
    a.output.mkdir(parents=True)
    (a.output/'feasibility.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__': main()
