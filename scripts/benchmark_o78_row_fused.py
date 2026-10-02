#!/usr/bin/env python3
"""v73 row-fused preparation versus v69, SAME v67 GEMM, no default change.

Use --full-modes to measure online preparation and its E2E effect. Compute-only
is a control (unchanged GEMM), never claimed as a new instruction optimization.
"""
import argparse
import ctypes as ct
import json
from pathlib import Path
import time

import numpy as np
import benchmark_o78_fused_prepare as fused
from benchmark_o78_coefficient_probe import validate, summarize
base, ROOT = fused.base, fused.base.ROOT


def timing_contract(mode, inner, policy):
    if policy not in (0, 1, 2):
        raise ValueError('invalid row-fusion policy')
    result = fused.timing_contract(mode, inner)
    candidate = policy == 1
    result.update(
        preparation_implementation=('row_fused_conversion_factor_metadata' if candidate
                                    else 'fused_conversion_group_squares_then_metadata'),
        group_squares_read_for_metadata=not candidate,
        group_squares_global_write_retained=True,
        weight_preparation_launches=1 if candidate else 2,
        activation_preparation_launches=2 if candidate else 3,
        gemm_cufunction_identical_between_policies=True,
    )
    return result


class Driver(fused.Driver):
    def __init__(self, library, baseline):
        super().__init__(library, baseline)
        self.check(self.lib.roof_probe_close(self.handles.pop(0)))
        handle = self.handles[1]
        resource = self.resources[1]
        # All policies use the exact SAME CUfunction, not a relinked GEMM.
        self.handles = {0: handle, 1: handle, 2: handle}
        self.resources = {p: dict(resource, kernel_symbol='adangel_roof_o78_fullk_candidate') for p in self.handles}
        for old, new in (('roof_o78_gpu_prepare', 'roof_o78_row_fused_prepare'),
                         ('roof_o78_gpu_benchmark', 'roof_o78_row_fused_benchmark')):
            function = getattr(self.lib, new)
            function.argtypes = getattr(self.lib, old).argtypes
            function.restype = ct.c_int
            setattr(self.lib, old, function)

    def close(self):
        for handle in {h.value: h for h in self.handles.values()}.values():
            self.check(self.lib.roof_probe_close(handle))
        self.handles.clear()

    def run(self, case, policy, mode, warmup, repeats, inner):
        import torch
        if policy not in self.handles or mode not in base.MODES:
            raise ValueError('invalid row-fusion policy/mode')
        if np.any(case.oracle['status_flat'] == 2):
            raise ValueError('invalid source must not expose an unwritten output')
        values = (ct.c_float * (4 * repeats))()
        # policy=2 is an explicit v69/original-fullK numerical reference.
        preparation = int(policy == 1)
        self.check(self.lib.roof_o78_gpu_benchmark(self.handles[policy], int(case.variant[1:]), preparation,
            base.MODES.index(mode), case.a_source, case.w_source, case.state_pointers, case.m, case.n,
            case.a_multiplier, case.w_multiplier, warmup, repeats, inner,
            torch.cuda.current_stream().cuda_stream, values))
        return case.state['y'], base.normalize_timings(mode, np.ctypeslib.as_array(values).reshape(4, repeats), repeats)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu-build', type=Path, default=Path('reports/o378_roof_v73_codegen'))
    p.add_argument('--baseline', type=Path, default=Path('reports/o378_roof_v67_codegen'))
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
    driver = Driver(library, args.baseline)
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
            print('ROW FUSION VALIDATION PASSED', flush=True)
            return
        manifest, mh = inspect_inputs(args.data)
        raw, rh = inspect_raw_inputs(args.raw_data, manifest, args.trace_config)
        raw_index = {r['sample_id']: r for r in raw['samples']}
        save('environment.json', dict(git_commit=command('git', 'rev-parse', 'HEAD'),
            extension_sha256=sha256_file(Path(native.__file__)), codegen=driver.codegen, gpu_preparation_build=build,
            resources=driver.resources, torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
            prepared_manifest_sha256=mh, raw_manifest_sha256=rh, production_default_changed=False,
            control='v69_separate_conversion_metadata_same_v67_GEMM', candidate='v73_row_fused_metadata_same_v67_GEMM',
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
                                resources=driver.resources[policy], **timing_contract(mode, args.inner, policy))
                            rows.append(row); append('results.jsonl', row)
                print(entry['sample_id'], variant, 'paired complete', flush=True)
            save('summary.json', dict(scope='row_conversion_metadata_fusion_paired_' + ('four_modes' if args.full_modes else 'cached_GEMM'),
                production_default_changed=False, no_filtering=True, records=summarize(rows, modes)))
        assert sha256_file(args.data / 'manifest.json') == mh and sha256_file(args.raw_data / 'trace_manifest.json') == rh
        print('ROW FUSION PAIRED TEST PASSED', flush=True)
    finally:
        driver.close()


if __name__ == '__main__':
    main()
