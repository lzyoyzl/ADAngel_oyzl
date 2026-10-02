#!/usr/bin/env python3
"""v79: isolated O3 v61/eight-chain pair; same conversion2 and GPU guard.

Both policies execute the full-K guarded ABI. No formal dispatch is changed.
Start with compute-only; all four timing modes are available for confirmation.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import time

from benchmark_a100_o1 import command, stats
from benchmark_a100_mixed_trace import inspect_inputs, inspect_raw_inputs, verify_raw_prepared, mse
from benchmark_conversion_pipeline import summarize as base_summary
from benchmark_integer_conversion_probe import order
from benchmark_roof_full_pipeline import timing_check
from probe_o3_eight_chain_codegen import CONTROL, CANDIDATE
from roof_full_pipeline_probe import ROOT, MODES, Pipeline as FullPipeline, build as full_build


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output, codegen):
    receipt = json.loads((codegen / 'codegen.json').read_text())
    cubin = codegen / 'o3_eight_chain.cubin'
    if digest(cubin) != receipt['cubin_sha256']:
        raise ValueError('candidate cubin drift')
    for name, sha in receipt['sources'].items():
        if digest(ROOT / name) != sha:
            raise ValueError('audited source drift: ' + name)
    if not receipt['control_comparison']['passed'] or not receipt['sentinel_comparison']['passed']:
        raise ValueError('control/sentinel codegen mismatch')
    built = full_build(output, ROOT / 'reports/o378_roof_v59',
                       ROOT / 'reports/o378_roof_v61', ROOT / 'runs/o378_roof_v60_screen/build')
    if digest(built[3]) != receipt['baseline_cubin_sha256']:
        raise ValueError('O3 control changed')
    sources = ('scripts/benchmark_o3_eight_chain_probe.py', 'scripts/roof_full_pipeline_probe.py',
               'scripts/roof_device_factor_probe.py', 'scripts/roof_gpu_factor_probe.py',
               'scripts/o3_fullk_probe.py', 'scripts/benchmark_roof_full_pipeline.py',
               'scripts/benchmark_conversion_pipeline.py')
    (output / 'eight_chain_build.json').write_text(json.dumps(dict(
        candidate_cubin_sha256=digest(cubin), control_cubin_sha256=digest(built[3]),
        codegen_receipt_sha256=digest(codegen / 'codegen.json'),
        sources={name: digest(ROOT / name) for name in sources},
        policies={0: CONTROL, 1: CANDIDATE}, both_policies_use_gpu_guard=True,
        conversion_candidate=2, production_default_changed=False), indent=2) + '\n')
    return (*built, cubin)


class Pipeline(FullPipeline):
    def __init__(self, *built):
        self.eight_chain = None
        super().__init__(*built[:-1])
        try:
            self.eight_chain = ct.c_void_p()
            self.check(self.lib.roof_probe_open(str(built[-1]).encode(), CANDIDATE.encode(),
                                               50688, ct.byref(self.eight_chain)))
            values = (ct.c_int * 4)()
            self.check(self.lib.roof_probe_resources(self.eight_chain, values))
            self.resources['eight_chain'] = dict(registers_per_thread=values[0],
                local_size_bytes=values[1], threads=values[2], active_blocks_per_sm=values[3],
                shared_memory_bytes=50688)
            assert values[2] == 128
        except Exception:
            self.close()
            raise

    def close(self):
        if getattr(self, 'eight_chain', None):
            self.check(self.lib.roof_probe_close(self.eight_chain))
            self.eight_chain = None
        super().close()

    def run_four(self, policy, *args, **kwargs):
        if policy not in (0, 1):
            raise ValueError('O3 pair policy must be 0 or 1')
        original = self.device
        try:
            if policy == 1:
                self.device = self.eight_chain
            # Both cases use v61's guarded ABI and identical preparation costs.
            result = super().run_four(1, *args, **kwargs)
        finally:
            self.device = original
        result['kernel'].update(gemm_tune='v61_device_fullk' if policy == 0 else 'v79_eight_chain',
                                kernel_symbol=CONTROL if policy == 0 else CANDIDATE,
                                cta_tile=[64, 128, 128], pipeline_stages=3)
        return result


def summarize(rows, sample_ids, rounds, modes):
    keys = {(r['sample_id'], r['round'], r['mode'], r['implementation']) for r in rows}
    expected = {(s, i, m, p) for s in sample_ids for i in range(rounds) for m in modes for p in (0, 1)}
    if not rows or len(keys) != len(rows) or keys != expected:
        raise ValueError('incomplete or duplicate O3 paired records')
    if any(not r['bitwise_equal_current_best'] or not r['payload_bitwise'] for r in rows):
        raise ValueError('O3 exact algebra regression')
    return base_summary(rows)


def validate(driver):
    import torch
    from adangel.quantization.arbitrary_bits import split_int8_to_packed_int4
    from adangel.quantization.mixed_formats import _pack_nibbles
    from roof_reduction_validation import reference_fp64
    from o3_fullk_probe import guard_columns
    checks, rejected = [], []
    for m, n in ((64, 128), (128, 256)):
        for pattern in ('all_codes', 'zero', 'alternating', 'random', 'unsafe', 'mixed_tiles'):
            torch.manual_seed(79)
            a = ((torch.arange(m * 4096, device='cuda') * 13) % 256 - 128).to(torch.int8).reshape(m, 4096)
            w = (torch.arange(n * 2048, device='cuda') % 256).byte().reshape(n, 2048)
            asc = torch.linspace(0, .01, m, device='cuda')
            ws = ((torch.arange(n * 32, device='cuda') * 7) % 5 + 116).byte().reshape(n, 32)
            if pattern == 'zero':
                a.zero_(); w.zero_()
            if pattern == 'alternating':
                a[:, ::2] = -128; a[:, 1::2] = 127; w.fill_(0x7f)
            if pattern == 'random':
                a = torch.randint(-128, 128, a.shape, device='cuda', dtype=torch.int8)
                w = torch.randint(0, 256, w.shape, device='cuda', dtype=torch.uint8)
            if pattern in ('unsafe', 'mixed_tiles'):
                ws.fill_(116); ws[:, -1] = 131
                if pattern == 'mixed_tiles':
                    ws[:128, -1] = 116
            lut = torch.tensor([0, 0, 1, 2, 2, 3, 4, 6, 0, 0, -1, -2, -2, -3, -4, -6],
                               device='cuda', dtype=torch.int8)
            wq = torch.stack((lut[(w & 15).long()], lut[(w >> 4).long()]), -1).reshape(n, 4096)
            pa = split_int8_to_packed_int4(a); pw = _pack_nibbles(wq.byte() & 15)
            ref = reference_fp64('o3', (pa, asc, pw, ws))
            expected_a = pa.reshape(2, m, 32, 64).permute(0, 2, 1, 3).contiguous()
            expected_w = pw.reshape(n, 32, 64).permute(1, 0, 2).contiguous()
            safe = guard_columns(ws.cpu().tolist())[1]['safe']
            control = driver.run_four(0, 'compute_only', a, asc, w, ws, 0, 1, 2)['output']
            for mode in MODES:
                for policy in (0, 1):
                    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        r = driver.run_four(policy, mode, a, asc, w, ws, 0, 2, 2)
                    stream.synchronize()
                    timing_check(r, mode, 2, 2, 1)
                    assert torch.equal(r['packed_activation_g128_major'], expected_a)
                    assert torch.equal(r['packed_weight_g128_major'], expected_w)
                    assert torch.equal(r['converted_weight_scale'], ws.T.contiguous())
                    torch.testing.assert_close(r['output'].double(), ref, rtol=1e-3, atol=1e-3)
                    assert torch.equal(r['output'].view(torch.int32), control.view(torch.int32))
                    assert r['status'] == (0 if safe else 1)
                    checks.append(dict(shape=[m, n, 4096], pattern=pattern, mode=mode, policy=policy,
                        bitwise_v61=True, payload_exact=True, scale_exact=True, finite_fp32=True,
                        semantic_tolerance_passed=True, status=r['status']))
    for policy in (0, 1):
        for code in (0, 255):
            bad = ws.clone(); bad[0, 0] = code
            try:
                driver.run_four(policy, 'cold', a, asc, w, bad, 0, 1, 2)
            except ValueError:
                rejected.append(dict(policy=policy, case='scale', code=code))
            else:
                raise AssertionError('invalid scale accepted')
        for operand in ('a', 'w'):
            original = a if operand == 'a' else w
            odd = torch.empty(original.numel() + 1, device='cuda', dtype=original.dtype)[1:].reshape(original.shape)
            odd.copy_(original)
            try:
                driver.run_four(policy, 'cold', odd if operand == 'a' else a, asc,
                                odd if operand == 'w' else w, ws, 0, 1, 2)
            except RuntimeError:
                rejected.append(dict(policy=policy, case='alignment', operand=operand))
            else:
                raise AssertionError('unaligned vector read accepted')
    return dict(passed=True, checks=checks, rejected=rejected,
                scope='small MN/full K4096, nondefault stream, safe/unsafe/mixed CTA; not 4096-cubed sanitizer')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--codegen', type=Path, default=Path('reports/o378_roof_v79_codegen'))
    p.add_argument('--samples', type=int, default=4); p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--warmup', type=int, default=50); p.add_argument('--repeats', type=int, default=200)
    p.add_argument('--inner', type=int, default=100)
    p.add_argument('--modes', nargs='+', choices=MODES, default=['compute_only'])
    p.add_argument('--validate-only', action='store_true')
    args = p.parse_args()
    if (args.output.exists() or not args.output.resolve().is_relative_to(ROOT)
            or not 1 <= args.samples <= 24 or args.warmup < 0 or min(args.repeats, args.inner, args.rounds) < 1
            or len(set(args.modes)) != len(args.modes)):
        p.error('fresh repository output and valid counts/modes required')
    import torch
    from adangel import _sm80 as native
    from adangel.trace.storage import load_prepared, sha256_file
    from adangel.trace.prepare import _load_and_validate_raw
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    if torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError('A100 required')
    context_anchor = torch.empty(1, device='cuda')
    args.output.mkdir(parents=True)
    def save(name, obj):
        (args.output / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    def append(name, obj):
        with (args.output / name).open('a') as f:
            f.write(json.dumps(obj, allow_nan=False) + '\n')
    built = build(args.output / 'build', args.codegen)
    save('environment.json', dict(git_commit=command('git', 'rev-parse', 'HEAD'),
        extension_sha256=sha256_file(Path(native.__file__)), torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), production_default_changed=False,
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        policy='unlocked shared GPU, cyclic pair order, all raw samples retained', timing_contract_version=2,
        scope='O3 v61 vs v79, identical conversion2 and online GPU guard; initial source quantization excluded',
        guard_cost='both: weight conversion includes factors/status; cached outside compute/steady',
        host_preflight='same normal-scale/finite-row preflight and post-batch status outside GPU Event timing'))
    driver = Pipeline(*built)
    try:
        save('resources.json', driver.resources); save('validation.json', validate(driver))
        if args.validate_only:
            print('O3 EIGHT-CHAIN VALIDATION PASSED', flush=True)
            return
        data = ROOT / 'data/prepared/llama2_7b_prefill_o0_o4'
        rawdata = ROOT / 'data/raw/llama2_7b_prefill'
        manifest, mh = inspect_inputs(data)
        raw_manifest, rh = inspect_raw_inputs(rawdata, manifest, ROOT / 'configs/trace/llama2_7b_prefill.yaml')
        rawindex = {r['sample_id']: r for r in raw_manifest['samples']}
        entries = manifest['samples'][:args.samples]
        save('input_provenance.json', dict(prepared_manifest_sha256=mh, raw_manifest_sha256=rh,
                                          samples=entries))
        rows = []
        for si, e in enumerate(entries):
            re = rawindex[e['sample_id']]
            assert sha256_file(data / e['file']) == e['sha256'] and sha256_file(rawdata / re['file']) == re['sha256']
            x = load_prepared(data / e['file'], device='cuda')
            raw = _load_and_validate_raw(rawdata / re['file'], re['layer'], re['projection'])
            verify_raw_prepared(x, (raw['activation_fp16'], raw['weight_fp16']))
            source = (x.A_int8, x.A_scale, x.W_mxfp4_g128, x.W_scale_g128)
            base = driver.run_four(0, 'compute_only', *source, 0, 1, 2)
            o0 = native.benchmark_o0(x.A_int8, x.A_scale, x.W_mxfp4, x.W_scale,
                                       'compute_only', 0, 1, 2)['output']
            for ri in range(args.rounds):
                append('gpu_snapshots.jsonl', dict(sample_id=x.sample_id, round=ri, time=time.time(),
                    gpu=command('nvidia-smi', '--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu', '--format=csv')))
                for mi, mode in enumerate(args.modes):
                    for policy in order([0, 1], si, mi, ri):
                        r = driver.run_four(policy, mode, *source, args.warmup, args.repeats, args.inner)
                        timing_check(r, mode, args.inner, args.repeats, 1)
                        equal = bool(torch.equal(r['output'].view(torch.int32), base['output'].view(torch.int32)))
                        payload = all(torch.equal(r[key], base[key]) for key in
                            ('packed_activation_g128_major', 'packed_weight_g128_major', 'converted_weight_scale'))
                        assert equal and payload and r['status'] == base['status']
                        summaries = {s: stats(v) for s, v in r['timings_ms'].items()}
                        selected = 'gemm' if mode == 'compute_only' else 'total'
                        row = dict(sample_id=x.sample_id, variant='o3', round=ri, mode=mode, implementation=policy,
                            raw_ms=r['timings_ms'], stage_summaries=summaries, summary=summaries[selected],
                            selected_stage=selected, total_timing=r['total_timing'],
                            stage_timing_inner_repeats=r['stage_timing_inner_repeats'], weight_cached=r['weight_cached'],
                            activation_prepared=r['activation_prepared'], kernel=r['kernel'], guard_status=r['status'],
                            payload_bitwise=payload, bitwise_equal_current_best=equal,
                            mse_vs_current_best=mse(r['output'], base['output']),
                            mse_vs_paired_fp16=mse(r['output'], o0), paired_reference='o0')
                        rows.append(row); append('results.jsonl', row)
                        print(x.sample_id, mode, policy, summaries[selected]['median_ms'], flush=True)
        save('summary.json', dict(records=summarize(rows, [e['sample_id'] for e in entries], args.rounds, args.modes),
                                  production_default_changed=False))
    finally:
        driver.close()


if __name__ == '__main__':
    main()
