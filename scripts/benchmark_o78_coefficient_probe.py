#!/usr/bin/env python3
"""v72: coefficient-first versus v67 full-K, with identical v69 GPU preparation.

The default is a cached GEMM screen. --full-modes additionally measures all
online costs, without switching production dispatch. Retain all timings.
"""
import argparse
import ctypes as ct
import json
from pathlib import Path
import statistics
import time

import numpy as np

import benchmark_o78_fused_prepare as fused
from analyze_o78_coefficient_codegen import analyze, SYMBOLS

base = fused.base
ROOT = base.ROOT


def checked(directory):
    from adangel.trace.storage import sha256_file
    receipt = json.loads((directory / 'codegen.json').read_text())
    for path, digest in receipt['sources'].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError('coefficient source drift: ' + path)
    if sha256_file(directory / 'o78_coefficient.cubin') != receipt['cubin_sha256']:
        raise ValueError('coefficient cubin drift')
    if not receipt['control_opcode_counts_match_v67'] or not receipt['control_instruction_count_match_v67']:
        raise ValueError('control no longer matches v67 opcode work')
    if set(receipt['entries']) != set(SYMBOLS) or not all(
            x['native_u4_s4'] and x['native_s4_s4'] and not x['int8_mma'] and x['all_copies_bypass_l1']
            for x in receipt['entries'].values()):
        raise ValueError('native INT4/copy audit failed')
    deps = analyze((directory / 'o78_coefficient.sass').read_text())
    by_symbol = {r['symbol']: r for r in deps['rows']}
    if by_symbol[SYMBOLS[1]]['dependency_counts']['coefficient_based_accumulator_updates'] != 64:
        raise ValueError('intended coefficient-first dependency did not survive ptxas')
    return dict(receipt=receipt, loop_analysis=deps)


class Driver(fused.Driver):
    def __init__(self, library, baseline, candidate):
        candidate_receipt = checked(candidate)
        super().__init__(library, baseline)
        # Retain an original v67 full-K handle for independent bitwise checks.
        self.check(self.lib.roof_probe_close(self.handles.pop(0)))
        self.handles[2] = self.handles.pop(1)
        original_resource = self.resources[1]
        self.resources = {2: original_resource}
        try:
            for policy, symbol in enumerate(SYMBOLS):
                handle = ct.c_void_p()
                self.check(self.lib.roof_probe_open(str((candidate / 'o78_coefficient.cubin').resolve()).encode(),
                                                   symbol.encode(), 34304, ct.byref(handle)))
                self.handles[policy] = handle
                values = (ct.c_int * 4)()
                self.check(self.lib.roof_probe_resources(handle, values))
                self.resources[policy] = dict(registers_per_thread=values[0], local_size_bytes=values[1],
                    threads=values[2], active_blocks_per_sm=values[3], shared_memory_bytes=34304,
                    cta_tile=[64, 128, 128], pipeline_stages=2, kernel_symbol=symbol)
            self.codegen = dict(v67=self.codegen, coefficient=candidate_receipt)
        except Exception:
            self.close()
            raise

    def run(self, case, policy, mode, warmup, repeats, inner):
        import torch
        if policy not in self.handles or mode not in base.MODES:
            raise ValueError('invalid coefficient test policy/mode')
        if np.any(case.oracle['status_flat'] == 2):
            raise ValueError('invalid source must not expose an unwritten output')
        values = (ct.c_float * (4 * repeats))()
        # IMPORTANT: candidate-preparation=1 for BOTH controls and candidate.
        # Kernel selection must not silently change factor/guard preparation.
        self.check(self.lib.roof_o78_gpu_benchmark(self.handles[policy], int(case.variant[1:]), 1, base.MODES.index(mode),
            case.a_source, case.w_source, case.state_pointers, case.m, case.n, case.a_multiplier,
            case.w_multiplier, warmup, repeats, inner, torch.cuda.current_stream().cuda_stream, values))
        return case.state['y'], base.normalize_timings(mode, np.ctypeslib.as_array(values).reshape(4, repeats), repeats)


def validate(driver):
    import torch
    from adangel import _sm80 as native
    from adangel.quantization import mixed_formats as mf
    from roof_reduction_validation import reference_fp64
    checks = []
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for pattern in ('random', 'zero', 'alternating', 'wide_scale'):
            m, n, k = (128, 256, 4096) if pattern == 'wide_scale' else (64, 128, 4096)
            torch.manual_seed(20261003)
            a = (torch.randn(m, k, device='cuda') * .4).half()
            w = (torch.randn(n, k, device='cuda') * .1).half()
            if pattern == 'zero':
                a.zero_(); w.zero_()
            if pattern == 'alternating':
                a[:, ::2] = -8; a[:, 1::2] = 7
                w[:, ::2] = -7; w[:, 1::2] = 6
            for variant, (wf, af) in mf.VARIANTS.items():
                ws, acs = mf.quantize_source(w, wf), mf.quantize_source(a, af)
                if pattern == 'wide_scale':
                    if variant == 'o7':
                        acs['scale'].fill_(127); acs['scale'][:, -1] = 159
                    else:
                        ws['scale'].fill_(1); ws['scale'][:, -1] = 192
                old = native._benchmark_mixed(variant, 'compute_only', ws, acs, 0, 1, 2,
                                              '64x128x256', 'group_major', 59, 5)
                case = fused.Case(variant, ws, acs, old)
                guard = driver.prepare(case)
                expected, _ = driver.run(case, 2, 'compute_only', 0, 1, 2)
                expected = expected.clone()
                semantic = reference_fp64(variant, (*old['converted_activation'], *old['converted_weight']))
                for policy in (0, 1):
                    for mode in base.MODES:
                        output, _ = driver.run(case, policy, mode, 0, 2, 2)
                        if mode == 'conversion_only':
                            output, _ = driver.run(case, policy, 'compute_only', 0, 1, 2)
                        assert output.dtype == torch.float32 and bool(torch.isfinite(output).all())
                        assert torch.equal(output.view(torch.int32), expected.view(torch.int32)), (variant, pattern, policy, mode)
                        torch.testing.assert_close(output.double(), semantic, rtol=1e-3, atol=1e-3)
                        if guard['integer_ctas'] == 0:
                            assert torch.equal(output.view(torch.int32), old['output'].view(torch.int32))
                        checks.append(dict(variant=variant, pattern=pattern, policy=policy, mode=mode,
                            bitwise_equal_v67=True, finite_fp32=True, semantic_tolerance_passed=True,
                            shape=[m, n, k], nondefault_stream=True, **guard))
        stream.synchronize()
    # Reuse exact invalid-code/saturation guard tests, with the fused Case ABI.
    original_case = base.Case
    try:
        base.Case = fused.Case
        edges = base.validate_edges(driver)
    finally:
        base.Case = original_case
    return dict(passed=True, scope='small_MN_full_K4096_not_4096cubed_sanitizer',
                count=len(checks), checks=checks, edge_count=len(edges), edge_checks=edges)


def summarize(rows, modes):
    from adangel.benchmark.metrics import bootstrap_median_ci
    index = {(r['sample_id'], r['variant'], r['mode'], r['round'], r['candidate']): r for r in rows}
    ids = sorted({r['sample_id'] for r in rows})
    rounds = sorted({r['round'] for r in rows})
    expected = {(s, v, m, r, p) for s in ids for v in ('o7', 'o8') for m in modes for r in rounds for p in (0, 1)}
    if len(index) != len(rows) or set(index) != expected:
        raise ValueError('incomplete paired evidence')
    result = []
    for variant in ('o7', 'o8'):
        for mode in modes:
            for policy in (0, 1):
                selected = [index[s, variant, mode, r, policy] for s in ids for r in rounds]
                if not all(r['bitwise_equal_v67'] and r['MSE_regression_passed'] and r['metadata_exact'] for r in selected):
                    raise ValueError('numerical or metadata acceptance failed')
                lat = [statistics.median(index[s, variant, mode, r, policy]['summary']['median_ms'] for r in rounds) for s in ids]
                speed = [statistics.median(index[s, variant, mode, r, 0]['summary']['median_ms'] /
                                          index[s, variant, mode, r, policy]['summary']['median_ms'] for r in rounds) for s in ids]
                errors = [index[s, variant, mode, 0, policy]['mse_vs_paired_fp16'] for s in ids]
                result.append(dict(variant=variant, mode=mode, candidate=policy, samples=len(ids), records=len(selected),
                    median_ms=statistics.median(lat), paired_speedup=statistics.median(speed),
                    paired_speedup_ci95=list(bootstrap_median_ci(speed, 10000, .95, 20261003)),
                    median_mse=statistics.median(errors), mean_mse=statistics.mean(errors),
                    selected_cv_failed_records=sum(r['summary']['cv_percent'] >= 3 for r in selected),
                    any_stage_cv_failed_records=sum(any(s['cv_percent'] >= 3 for s in r['stage_summaries'].values()) for r in selected),
                    max_abs_vs_v67=max(r['max_abs_vs_v67'] for r in selected)))
    return result


def main(*, driver_cls=Driver, default_gpu_build=Path('reports/o378_roof_v69_codegen'),
         labels=('v67_fullK_same_v69_preparation', 'v72_coefficient_first_same_v69_preparation'),
         experiment='coefficient_first', banner='COEFFICIENT',
         contract=fused.timing_contract, description=__doc__):
    """Shared paired protocol; defaults preserve the original v72 experiment."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--gpu-build', type=Path, default=default_gpu_build)
    p.add_argument('--baseline', type=Path, default=Path('reports/o378_roof_v67_codegen'))
    p.add_argument('--cubins', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, choices=(4, 24), default=4)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--repeats', type=int, default=200)
    p.add_argument('--inner', type=int, default=100)
    p.add_argument('--full-modes', action='store_true')
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--data', type=Path, default=Path('data/prepared/llama2_7b_prefill_o0_o4'))
    p.add_argument('--raw-data', type=Path, default=Path('data/raw/llama2_7b_prefill'))
    p.add_argument('--trace-config', type=Path, default=Path('configs/trace/llama2_7b_prefill.yaml'))
    args = p.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT) or min(args.rounds, args.repeats) < 1 or args.inner < 2 or args.warmup < 0:
        p.error('fresh repository output and valid counts required')
    import torch
    from adangel import _sm80 as native
    from adangel.quantization import mixed_formats as mf
    from adangel.trace.storage import sha256_file, load_prepared
    from adangel.trace.prepare import _load_and_validate_raw
    from benchmark_a100_o1 import command, stats
    from benchmark_a100_roof_trace import measurement_order
    from benchmark_a100_mixed import validate_fp16_result
    from benchmark_a100_mixed_trace import inspect_inputs, inspect_raw_inputs, verify_raw_prepared, source_identity, mse
    from roof_reduction_validation import mse_regression_ok
    torch.cuda.init(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.get_device_capability() == (8, 0)
    context_anchor = torch.empty(1, device='cuda')
    library, build = base.checked_gpu_build(args.gpu_build)
    driver = driver_cls(library, args.baseline, args.cubins)
    args.output.mkdir(parents=True)
    modes = base.MODES if args.full_modes else ('compute_only',)
    def save(name, obj):
        (args.output / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    def append(name, obj):
        with (args.output / name).open('a') as out:
            out.write(json.dumps(obj, allow_nan=False) + '\n')
    try:
        save('validation.json', validate(driver))
        if args.validate_only:
            print(banner + ' VALIDATION PASSED', flush=True)
            return
        manifest, mh = inspect_inputs(args.data)
        raw, rh = inspect_raw_inputs(args.raw_data, manifest, args.trace_config)
        raw_index = {r['sample_id']: r for r in raw['samples']}
        save('environment.json', dict(git_commit=command('git', 'rev-parse', 'HEAD'),
            extension_sha256=sha256_file(Path(native.__file__)), codegen=driver.codegen, gpu_preparation_build=build,
            resources=driver.resources, torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
            prepared_manifest_sha256=mh, raw_manifest_sha256=rh, production_default_changed=False,
            control=labels[0], candidate=labels[1],
            source_quantization='original_FP16_direct_source_quantization_excluded',
            timing_scope='all_four_modes' if args.full_modes else 'cached_compute_only_not_E2E', no_filtering=True,
            args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}))
        rows = []
        for si, entry in enumerate(manifest['samples'][:args.samples]):
            re = raw_index[entry['sample_id']]
            path, rp = args.data / entry['file'], args.raw_data / re['file']
            assert sha256_file(path) == entry['sha256'] and sha256_file(rp) == re['sha256']
            record = _load_and_validate_raw(rp, re['layer'], re['projection'])
            prepared = load_prepared(path, device='cpu')
            verify_raw_prepared(prepared, (record['activation_fp16'], record['weight_fp16'])); del prepared
            for vi, (variant, (wf, af)) in enumerate(mf.VARIANTS.items()):
                ws = mf.quantize_source(record['weight_fp16'].cuda(), wf)
                acs = mf.quantize_source(record['activation_fp16'].cuda(), af)
                append('source_provenance.jsonl', dict(sample_id=entry['sample_id'], variant=variant,
                    raw_sha256=re['sha256'], weight=source_identity(ws), activation=source_identity(acs)))
                fp = native._benchmark_mixed(mf.PAIRED_BASELINE[variant], 'compute_only', ws, acs, 0, 1, 2, '64x128x256', 'row_major')
                validate_fp16_result(fp, ws, acs)
                old = native._benchmark_mixed(variant, 'compute_only', ws, acs, 0, 1, 2, '64x128x256', 'group_major', 59, 5)
                case = fused.Case(variant, ws, acs, old)
                guard = driver.prepare(case)
                expected, _ = driver.run(case, 2, 'compute_only', 0, 1, 2)
                expected = expected.clone()
                expected_mse = mse(expected, fp['output'])
                assert mse_regression_ok(expected_mse, mse(old['output'], fp['output']))
                for ri in range(args.rounds):
                    append('gpu_snapshots.jsonl', dict(sample_id=entry['sample_id'], variant=variant, round=ri, time=time.time(),
                        gpu=command('nvidia-smi', '--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu', '--format=csv')))
                    for mi, mode in enumerate(modes):
                        order = measurement_order((0, 1), si, vi + mi, ri)
                        for policy in order:
                            output, timings = driver.run(case, policy, mode, args.warmup, args.repeats, args.inner)
                            if mode == 'conversion_only':
                                output, _ = driver.run(case, policy, 'compute_only', 0, 1, 2)
                            assert output.dtype == torch.float32 and bool(torch.isfinite(output).all())
                            assert torch.equal(output.view(torch.int32), expected.view(torch.int32))
                            error = mse(output, fp['output'])
                            assert error == expected_mse
                            arrays = {key: case.state[key].cpu().numpy() for key in ('af', 'wf', 'ab', 'wb', 'an', 'wn', 'am', 'wm', 'ast', 'wst', 'status')}
                            current_guard = base.check_metadata_arrays(case.oracle, arrays)
                            stage_summaries = {s: stats(t) for s, t in timings.items()}
                            selected = 'gemm' if mode == 'compute_only' else 'total'
                            row = dict(sample_id=entry['sample_id'], variant=variant, mode=mode, round=ri, candidate=policy,
                                execution_order=list(order), raw_ms=timings, stage_summaries=stage_summaries,
                                summary=stage_summaries[selected], metadata_exact=True, guard=current_guard,
                                finite_fp32=True, bitwise_equal_v67=True, MSE_regression_passed=True,
                                mse_vs_paired_fp16=error, paired_fp16=mf.PAIRED_BASELINE[variant],
                                mse_vs_v67=mse(output, expected), max_abs_vs_v67=(output - expected).abs().max().item(),
                                resources=driver.resources[policy], **contract(mode, args.inner))
                            rows.append(row); append('results.jsonl', row)
                print(entry['sample_id'], variant, 'paired complete', flush=True)
            save('summary.json', dict(scope=experiment + '_paired_' + ('four_modes' if args.full_modes else 'cached_GEMM'),
                production_default_changed=False, no_filtering=True, records=summarize(rows, modes)))
        assert sha256_file(args.data / 'manifest.json') == mh and sha256_file(args.raw_data / 'trace_manifest.json') == rh
        print(banner + ' PAIRED TEST PASSED', flush=True)
    finally:
        driver.close()


if __name__ == '__main__':
    main()
