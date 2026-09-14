#!/usr/bin/env python3
"""Isolated exact-partial-cast A/B tests on RTX 5090; production is unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from benchmark_a100_o1 import command, stats

BASE = {
    'o1': 'register_128x64_k64_scale_shared_row_dedup',
    'o3': 'm64_n32_k128_aligned_factor_16w',
}
MODES = ['conversion_only', 'compute_only', 'cold', 'steady_state']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path('data/prepared/llama2_7b_prefill_o0_o4'))
    p.add_argument('--variants', nargs='+', choices=['o1', 'o3'], default=['o1', 'o3'])
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--repeats', type=int, default=200)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--inner', type=int, default=100)
    p.add_argument('--samples', type=int, default=0)
    p.add_argument('--all-modes', action='store_true')
    p.add_argument('--validate', action='store_true')
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--profile', choices=['o1', 'o1_magic', 'o3', 'o3_magic'])
    p.add_argument('--snapshot-only', action='store_true',
                   help='Save output hashes/MSE for before/after synchronization regression')
    p.add_argument('--compare-snapshot', type=Path,
                   help='Require identical hashes/MSE to a previous snapshot.json')
    args = p.parse_args()
    if args.output.exists(): p.error('Use a fresh output directory')
    if min(args.repeats, args.rounds) < 1 or args.warmup < 0 or args.inner < 2:
        p.error('Invalid timing parameters')
    import torch
    from adangel import _sm120 as native
    from adangel.trace.storage import load_prepared, sha256_file, validate_manifest
    assert torch.cuda.get_device_capability() == (12, 0)
    args.output.mkdir(parents=True)

    def save(name, obj):
        (args.output / name).write_text(json.dumps(obj, indent=2) + '\n')

    def append(name, obj):
        with (args.output / name).open('a') as f: f.write(json.dumps(obj) + '\n')

    root = Path(__file__).resolve().parents[1]
    previous_snapshot = (json.loads(args.compare_snapshot.read_text())
                         if args.compare_snapshot else None)
    snapshots = {}
    save('environment.json', dict(
        commit=command('git', 'rev-parse', 'HEAD'), torch=torch.__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        binary_sha256=sha256_file(Path(native.__file__)),
        cuda_sources_sha256={s: sha256_file(root/s) for s in [
            'csrc/sm120/o1_gemm.cu', 'csrc/sm120/o3_gemm.cu',
            'include/adangel/exact_partial_cast.cuh']},
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        policy='No filtering or clock changes; cyclic/reversed same-process paired order; exact old-output comparison'))

    def call(variant, magic, mode, a, asc, w, ws, warmup, repeats, inner):
        impl = BASE[variant] + ('_magic' if magic else '')
        result = getattr(native, f'_benchmark_{variant}_impl')(
            impl, mode, a, asc, w, ws, warmup, repeats, inner)
        meta = dict(result['kernel'])
        assert meta['implementation_key'] == impl
        assert meta['integer_conversion'] == ('exact_magic_bias' if magic else 'i2f')
        assert list(meta['cta_tile']) == ([128, 64, 64] if variant == 'o1' else [64, 32, 128])
        return result

    def equal_output(actual, expected):
        assert actual.dtype == torch.float32 and torch.isfinite(actual).all()
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))

    if args.validate or args.validate_only:
        checks = []
        for variant in args.variants:
            shapes = ([(64, 32, 32), (64, 64, 64), (128, 96, 96),
                       (128, 128, 256), (256, 256, 4096)] if variant == 'o1'
                      else [(64, 32, 128), (128, 64, 256), (128, 96, 384), (256, 256, 4096)])
            group = 32 if variant == 'o1' else 128
            for m, n, k in shapes:
                for pattern in ['zero', 'random', 'saturation', 'alternating', 'all_codes', 'zero_scale']:
                    torch.manual_seed(5090 + k)
                    a = torch.randint(-128, 128, (m, k), dtype=torch.int8, device='cuda')
                    w = torch.randint(0, 256, (n, k//2), dtype=torch.uint8, device='cuda')
                    asc = torch.linspace(.0001, .1, m, device='cuda')
                    ws = (torch.arange(n*(k//group), device='cuda').reshape(n, k//group)%21+106).to(torch.uint8)
                    if pattern == 'zero': a.zero_()
                    if pattern == 'zero_scale': asc.zero_()
                    if pattern == 'saturation': a[:] = (torch.arange(k, device='cuda')%256-128).to(torch.int8)
                    if pattern == 'alternating': a[:] = torch.where(torch.arange(k, device='cuda')%2 == 0, -128, 127).to(torch.int8)
                    if pattern == 'all_codes':
                        # Every packed E2M1 pair; zero UE8M0 code also exercised.
                        w[:] = (torch.arange(n*k//2, device='cuda').reshape(n, k//2)%256).to(torch.uint8)
                        ws[:, 0] = 0
                    base = call(variant, False, 'compute_only', a, asc, w, ws, 0, 1, 2)
                    for mode in MODES:
                        result = call(variant, True, mode, a, asc, w, ws, 0, 1, 2)
                        equal_output(result['output'], base['output'])
                        for key in ['converted_weight', 'converted_activation']:
                            x, y = result.get(key), base.get(key)
                            if isinstance(x, torch.Tensor): assert torch.equal(x, y)
                        checks.append(dict(variant=variant, shape=[m, n, k], pattern=pattern,
                                           mode=mode, bitwise_equal=True))
        save('validation.json', dict(passed=True, checks=checks))
        print('Exact GPU checks:', len(checks), flush=True)
        if args.validate_only: return

    manifest = json.loads((args.data/'manifest.json').read_text())
    validate_manifest(manifest, formal=True, require_arbitrary_bits=True)
    save('data_manifest.json', manifest)
    modes = MODES if args.all_modes else ['compute_only']
    for i, entry in enumerate(manifest['samples'][:args.samples or None]):
        path = args.data/entry['file']
        assert sha256_file(path) == entry['sha256']
        x = load_prepared(path, device='cuda')

        def run(impl, mode):
            if impl == 'o0':
                return native.benchmark('o0', mode, x.A_int8, x.A_scale, x.W_mxfp4, x.W_scale,
                                        args.warmup, args.repeats, args.inner)
            variant = impl.split('_')[0]
            w, ws = ((x.W_mxfp4, x.W_scale) if variant == 'o1'
                     else (x.W_mxfp4_g128, x.W_scale_g128))
            return call(variant, impl.endswith('_magic'), mode, x.A_int8, x.A_scale,
                        w, ws, args.warmup, args.repeats, args.inner)

        if args.profile:
            run(args.profile, 'compute_only')
            return
        if args.snapshot_only:
            ref = run('o0', 'compute_only')['output']
            for impl in ['o0'] + [name for v in args.variants for name in [v, v+'_magic']]:
                result = run(impl, 'compute_only')
                y = result['output']
                assert y.dtype == torch.float32 and torch.isfinite(y).all()
                key = x.sample_id + '/' + impl
                value = dict(output_sha256=hashlib.sha256(
                    y.contiguous().cpu().numpy().tobytes()).hexdigest(),
                    mse_vs_o0=float((y.double()-ref.double()).square().mean()))
                if previous_snapshot is not None:
                    assert value == previous_snapshot[key], (key, value, previous_snapshot[key])
                snapshots[key] = value
            print(x.sample_id, 'snapshot passed', flush=True)
            continue
        ref = run('o0', 'compute_only')['output']
        bases = {v: run(v, 'compute_only')['output'] for v in args.variants}
        base_mse = {v: float((y.double()-ref.double()).square().mean()) for v, y in bases.items()}
        append('gpu_snapshots.jsonl', dict(sample_id=x.sample_id, time=time.time(),
            gpu=command('nvidia-smi', '--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu', '--format=csv')))
        for mode in modes:
            for round_id in range(args.rounds):
                order = ['o0'] + [name for v in args.variants for name in [v, v+'_magic']]
                shift = (i+round_id)%len(order)
                order = order[shift:]+order[:shift]
                if round_id%2: order.reverse()
                for impl in order:
                    result = run(impl, mode)
                    y = result['output']
                    assert torch.isfinite(y).all()
                    mse = float((y.double()-ref.double()).square().mean())
                    variant = impl.split('_')[0]
                    if impl != 'o0':
                        equal_output(y, bases[variant])
                        assert mse == base_mse[variant]
                    append('results.jsonl', dict(sample_id=x.sample_id, implementation=impl,
                        mode=mode, round=round_id, timings_ms=dict(result['timings_ms']),
                        summary={s: stats(v) for s, v in result['timings_ms'].items()},
                        timing_method=dict(result['timing_method']), kernel=dict(result['kernel']),
                        mse_vs_o0=mse, baseline_mse_vs_o0=base_mse.get(variant, 0),
                        bitwise_equal_baseline=impl != 'o0'))
            print(x.sample_id, mode, 'done', flush=True)
    if args.snapshot_only:
        if previous_snapshot is not None:
            assert snapshots.keys() == previous_snapshot.keys(), 'Snapshot coverage differs'
        save('snapshot.json', snapshots)
        print('Snapshot entries:', len(snapshots), flush=True)


if __name__ == '__main__': main()
