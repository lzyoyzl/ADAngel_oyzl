#!/usr/bin/env python3
"""24 original-trace crops, actual 512/1024 cubed, accepted native plans only.

Crop FP16 before independently applying the existing public quantizers. No
4096 padding, secondary quantization, synthetic timing corpus, or legacy shape
fallback. Source quantization and correctness work are outside CUDA events.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from benchmark_a100_o1 import command, stats
from benchmark_a100_mixed import MODES, aligned_timings, validate_fp16_result
from roof_reduction_validation import reference_fp64

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('o1', 'o3', 'o5', 'o6', 'o7', 'o8')
REFERENCES = dict(o1='o0', o3='o0', o5='o5', o6='o6', o7='o5', o8='o6')


def call(native, variant, mode, inputs, warmup=0, repeats=1, inner=2):
    if variant in ('o1', 'o3'):
        return native.benchmark(variant, mode, *inputs, warmup, repeats, inner, 'production')
    return native._benchmark_mixed(variant, mode, *inputs, warmup, repeats, inner,
                                  '64x128x256', 'row_major', -1, 0, 'production')


def check_path(result, variant, size):
    import torch
    meta = dict(result['kernel'])
    assert result['output'].shape == (size, size)
    assert result['output'].dtype == torch.float32 and torch.isfinite(result['output']).all()
    if variant == 'o1':
        assert meta['implementation'] == 'swizzle_128x64_k128_magic', meta
        assert meta['requested_implementation'] == 'production'
    elif variant in ('o3', 'o7', 'o8'):
        prefix = 'adangel_sm80_o3_fullk_grouped' if variant == 'o3' else 'adangel_sm80_o78_fullk_streaming'
        assert meta['kernel_symbol'] == f'{prefix}_k{size}', meta
        assert meta['physical_k'] == size and meta['group_count'] == size // 128
        assert meta['production_default'] and not meta['legacy_shape_fallback']
        assert meta['guarded_fullk_int32'] and meta['cta_tile'] == [64, 128, 128]
    else:
        assert meta['tensor_core'] and meta['compute_type'] == 'CUBLAS_COMPUTE_32F'
        assert meta['split_k'] <= 1 and meta['output_dtype'] == 'fp32'


def mixed_payload(result, w, a):
    """Exact packed values, effective scales and independent CPU range guard."""
    import numpy as np
    import torch
    from adangel.quantization import mixed_formats as mf
    from adangel.quantization.arbitrary_bits import split_int8_to_packed_int4
    from validate_a100_split_grouped import pack_q4
    from o78_fullk_integer_metadata import prepare_fullk_metadata
    from benchmark_o78_fullk_gpu_prepare import check_metadata_arrays
    aq, asc = mf.to_fixed_reference(a)
    wq, wsc = mf.to_fixed_reference(w)
    for got, want in zip(result['converted_activation'], (split_int8_to_packed_int4(aq), asc)):
        assert torch.equal(got.contiguous().view(torch.uint8), want.contiguous().view(torch.uint8))
    for got, want in zip(result['converted_weight'], (pack_q4(wq), wsc)):
        assert torch.equal(got.contiguous().view(torch.uint8), want.contiguous().view(torch.uint8))
    groups = aq.shape[1] // 128
    square = lambda q: q.reshape(q.shape[0], groups, 128).long().square().sum(-1).cpu().numpy()
    an, wn = square(aq), square(wq)
    o7 = w['format'] == 'nvfp4_g128'
    mult = float((w if o7 else a)['tensor_scale'].item())
    oracle = prepare_fullk_metadata(a['scale'].cpu().numpy(), w['scale'].cpu().numpy(), an, wn,
        activation_kind='ue8m0' if o7 else 'e4m3', weight_kind='e4m3' if o7 else 'e6m2',
        activation_base_multiplier=np.float32(4 if o7 else np.float32(mult) * np.float32(.25)),
        weight_base_multiplier=np.float32(mult if o7 else 1))
    st, arrays = result['prepared_state'], {}
    for key in ('af', 'wf', 'ab', 'wb', 'an', 'wn', 'am', 'wm', 'ast', 'wst', 'status'):
        val = st[key].cpu().numpy()
        if key in ('ast', 'wst', 'status'): val = val.view(np.uint32)
        if key in ('an', 'wn'): val = val.view(np.uint64)
        if key == 'status': val = val.reshape(-1)
        arrays[key] = val
    checked = check_metadata_arrays(oracle, arrays)
    assert np.array_equal(st['asq'].cpu().numpy(), an)
    assert np.array_equal(st['wsq'].cpu().numpy(), wn)
    return checked


def correctness(result, variant, inputs):
    import torch
    if variant in ('o5', 'o6'):
        return validate_fp16_result(result, *inputs)
    if variant == 'o1':
        from adangel.quantization.mxfp4 import unpack_e2m1_tensor, decode_ue8m0_tensor
        a, asc, w, ws = inputs
        lut = torch.tensor([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], device=a.device)
        wi = lut[unpack_e2m1_tensor(w).long()]
        ref = torch.zeros_like(result['output'])
        sc = decode_ue8m0_tensor(ws)
        for g in range(a.shape[1] // 32):
            sl = slice(g * 32, (g + 1) * 32)
            p = a[:, sl].double() @ wi[:, sl].double().T
            contribution = (p.float() * asc[:, None]) * (sc[None, :, g] * .5)
            ref = (ref + contribution).float()
    elif variant == 'o3':
        from adangel.quantization.mxfp4 import mxfp4_to_q4_packed
        from adangel.quantization.arbitrary_bits import split_int8_to_packed_int4
        a, asc, w, ws = inputs
        assert torch.equal(result['converted_activation'], split_int8_to_packed_int4(a))
        assert torch.equal(result['converted_weight'], mxfp4_to_q4_packed(w))
        assert torch.equal(result['converted_weight_scale'].T, ws)
        ref = reference_fp64(variant, (result['converted_activation'], asc, result['converted_weight'], ws))
    else:
        guard = mixed_payload(result, *inputs)
        ref = reference_fp64(variant, (*result['converted_activation'], *result['converted_weight']))
    torch.testing.assert_close(result['output'].double(), ref.double(), rtol=1e-3, atol=1e-3)
    return dict(semantic_reference_passed=True,
                max_abs_error=float((result['output'].double()-ref.double()).abs().max()),
                **({'guard': guard} if variant in ('o7', 'o8') else {}))


def prepare(a, w):
    from adangel.quantization.int8 import quantize_int8_per_row
    from adangel.quantization.mxfp4 import quantize_mxfp4
    from adangel.quantization import mixed_formats as mf
    ai, asc = quantize_int8_per_row(a)
    w32, ws32 = quantize_mxfp4(w, 32)
    w128, ws128 = quantize_mxfp4(w, 128)
    pairs = {v: (mf.quantize_source(w, wf), mf.quantize_source(a, af)) for v, (wf, af) in mf.VARIANTS.items()}
    return dict(o1=(ai, asc, w32, ws32), o3=(ai, asc, w128, ws128),
                o5=pairs['o7'], o7=pairs['o7'], o6=pairs['o8'], o8=pairs['o8'])


def mse(a, b):
    return float((a.double() - b.double()).square().mean())


def summarize(rows):
    out = []
    for size in sorted({r['size'] for r in rows}):
        for variant in VARIANTS:
            for mode in MODES:
                selected = [r for r in rows if (r['size'], r['variant'], r['mode']) == (size, variant, mode)]
                samples = sorted({r['sample_id'] for r in selected})
                stages = {}
                for stage in selected[0]['stages']:
                    values = [statistics.median(r['stages'][stage]['median_ms'] for r in selected if r['sample_id'] == s) for s in samples]
                    stages[stage] = dict(median_ms=statistics.median(values), mean_ms=statistics.mean(values))
                errors = [next(r['mse'] for r in selected if r['sample_id'] == s) for s in samples]
                out.append(dict(size=size, variant=variant, mode=mode, samples=len(samples), records=len(selected),
                    stages=stages, mse_median=statistics.median(errors), mse_mean=statistics.mean(errors),
                    reference=REFERENCES[variant],
                    cv_failed_records=sum(any(v['cv_percent'] >= 3 for v in r['stages'].values()) for r in selected),
                    primary_cv_failed=sum(r['stages']['gemm' if mode == 'compute_only' else 'total']['cv_percent'] >= 3 for r in selected),
                    fallback_tiles_max=max(r['kernel'].get('fallback_tiles', 0) for r in selected)))
    return out


def edge_validation(native):
    import torch
    from adangel.quantization import mixed_formats as mf
    checks = []
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for size in (512, 1024):
            for pattern in ('random', 'zero', 'alternating', 'wide_scale'):
                torch.manual_seed(174 + size)
                a = (torch.randn(size, size, device='cuda') * .5).half()
                w = (torch.randn(size, size, device='cuda') * .1).half()
                if pattern == 'zero': a.zero_(); w.zero_()
                if pattern == 'alternating': a[:, ::2] = -8; a[:, 1::2] = 7; w[:, ::2] = -7; w[:, 1::2] = 6
                inputs = prepare(a, w)
                if pattern == 'wide_scale':
                    inputs['o3'][-1][:, -1] = 150
                    inputs['o7'][1]['scale'].fill_(127); inputs['o7'][1]['scale'][:, -1] = 159
                    inputs['o8'][0]['scale'].fill_(1); inputs['o8'][0]['scale'][:, -1] = 192
                for v in VARIANTS:
                    result = call(native, v, 'compute_only', inputs[v])
                    check_path(result, v, size)
                    # Mutated extreme sources are only for integer fallback, not HMMA overflow tests.
                    if pattern == 'wide_scale' and v in ('o5', 'o6'): continue
                    numerical = correctness(result, v, inputs[v])
                    if pattern == 'wide_scale' and v in ('o3', 'o7', 'o8'):
                        assert result['kernel']['fallback_tiles'] > 0
                    expected = result['output'].clone()
                    for mode in MODES:
                        got = call(native, v, mode, inputs[v], repeats=2)
                        assert torch.equal(got['output'].view(torch.int32), expected.view(torch.int32))
                    checks.append(dict(size=size, variant=v, pattern=pattern, nondefault_stream=True,
                                       modes_bitwise_equal=True, **numerical))
                for v in ('o3', 'o7', 'o8'):
                    bad = inputs[v][-1] if v == 'o3' else inputs[v][0]['scale']
                    bad.fill_(255)
                    try: call(native, v, 'compute_only', inputs[v])
                    except (ValueError, RuntimeError): pass
                    else: raise AssertionError('invalid source not rejected: ' + v)
        stream.synchronize()
    return dict(passed=True, checks=checks, invalid_codes_rejected=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--sizes', type=int, nargs='+', default=[512, 1024], choices=[512, 1024])
    p.add_argument('--samples', type=int, default=24)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--warmup', type=int, default=1000)
    p.add_argument('--repeats', type=int, default=200)
    p.add_argument('--inner', type=int, default=100)
    p.add_argument('--validate-only', action='store_true')
    args = p.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT): p.error('fresh repository output required')
    if min(args.rounds, args.repeats, args.inner, args.samples) < 1 or args.samples > 24 or args.warmup < 0: p.error('invalid counts')
    import torch
    from adangel import _sm80 as native
    from adangel.trace.raw import validate_raw_trace
    from adangel.trace.storage import sha256_file
    from adangel.trace.prepare import _load_and_validate_raw
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.get_device_capability() == (8, 0)
    args.output.mkdir(parents=True)
    def save(file, obj): (args.output / file).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    def append(file, obj):
        with (args.output / file).open('a') as f: f.write(json.dumps(obj, allow_nan=False) + '\n')
    save('environment.json', dict(commit=command('git', 'rev-parse', 'HEAD'), extension_sha256=sha256_file(Path(native.__file__)),
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
        settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        no_filtering=True, clock_policy='unlocked', no_4096_padding=True, source_quantization_timed=False))
    if args.validate_only:
        save('validation.json', edge_validation(native)); print('SIZE EDGE VALIDATION PASSED', flush=True); return
    rawdir = ROOT / 'data/raw/llama2_7b_prefill'
    manifest = validate_raw_trace(rawdir, ROOT / 'configs/trace/llama2_7b_prefill.yaml', deep=False)
    save('data_provenance.json', dict(raw_manifest_sha256=sha256_file(rawdir / 'trace_manifest.json'),
        input_policy='top_left_original_fp16_crop_then_independent_source_quantization',
        slices='A[:S,:S], W[:S,:S], Y=A@W.T', sample_hashes=manifest['samples']))
    rows = []
    for si, entry in enumerate(manifest['samples'][:args.samples]):
        record = _load_and_validate_raw(rawdir / entry['file'], entry['layer'], entry['projection'])
        for size in args.sizes:
            a = record['activation_fp16'][:size, :size].contiguous().cuda()
            w = record['weight_fp16'][:size, :size].contiguous().cuda()
            inputs = prepare(a, w)
            refs = dict(o0=native.benchmark_o0(*inputs['o1'], 'compute_only', 0, 1, 2)['output'].clone())
            for v in ('o5', 'o6'): refs[v] = call(native, v, 'compute_only', inputs[v])['output'].clone()
            expected, errors = {}, {}
            for v in VARIANTS:
                result = call(native, v, 'compute_only', inputs[v])
                check_path(result, v, size)
                check = correctness(result, v, inputs[v])
                expected[v] = result['output'].clone()
                errors[v] = mse(expected[v], refs[REFERENCES[v]])
                append('correctness.jsonl', dict(sample_id=entry['sample_id'], size=size, variant=v,
                    mse=errors[v], reference=REFERENCES[v], kernel=dict(result['kernel']), **check))
            for ri in range(args.rounds):
                append('gpu_snapshots.jsonl', dict(sample_id=entry['sample_id'], size=size, round=ri, time=time.time(),
                    state=command('nvidia-smi', '--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu', '--format=csv')))
                # Cyclic, reversed ordering reduces systematic thermal/order bias.
                shift = (si + ri) % len(VARIANTS)
                order = VARIANTS[shift:] + VARIANTS[:shift]
                if (si + ri) % 2: order = tuple(reversed(order))
                for mode in MODES:
                    for v in order:
                        result = call(native, v, mode, inputs[v], args.warmup, args.repeats, args.inner)
                        check_path(result, v, size)
                        assert torch.equal(result['output'].view(torch.int32), expected[v].view(torch.int32))
                        raw, native_total, method = aligned_timings(result, mode)
                        assert all(len(values) == args.repeats and min(values) > 0 for values in raw.values())
                        row = dict(sample_id=entry['sample_id'], size=size, variant=v, mode=mode, round=ri,
                            execution_order=order, kernel=dict(result['kernel']), mse=errors[v], reference=REFERENCES[v],
                            raw_ms=raw, native_total_ms=native_total, stages={k: stats(vals) for k, vals in raw.items()},
                            total_timing=method, stage_timing_inner_repeats=dict(result['stage_timing_inner_repeats']))
                        assert all(count == (args.inner if 'conversion' in stage or mode == 'conversion_only' else 1)
                                   for stage, count in row['stage_timing_inner_repeats'].items())
                        rows.append(row); append('results.jsonl', row)
            save('summary.json', summarize(rows))
            print(entry['sample_id'], size, 'all six accepted paths / four modes passed', flush=True)
    expected_count = args.samples * len(args.sizes) * len(VARIANTS) * len(MODES) * args.rounds
    assert len(rows) == expected_count
    save('completion.json', dict(passed=True, records=len(rows), samples=args.samples, sizes=args.sizes, no_filtering=True))
    print('SIZE EXPERIMENT COMPLETED', expected_count, flush=True)


if __name__ == '__main__': main()
