#!/usr/bin/env python3
"""v76 O7 shared coefficient-table screen, identical v73 preparation.

The guard and table generation are inside GEMM timing, not free preprocessing.
No production/default changes; all paired measurements including CV failures
are retained. Only O7 is performance-tested; synthetic O8 checks exercise the
non-power-of-two fallback through the shared internal test ABI.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import benchmark_o78_row_fused as row_fused
from benchmark_o78_coefficient_probe import validate
from analyze_o7_factor_table_codegen import analyze, CONTROL, CANDIDATE
from probe_o7_factor_table_codegen import table_header, SHARED_BYTES

base, fused, ROOT = row_fused.base, row_fused.fused, row_fused.ROOT


def checked(directory):
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = json.loads((directory / 'codegen.json').read_text())
    for name, sha in receipt['sources'].items():
        if digest(ROOT / name) != sha:
            raise ValueError('table source drift: ' + name)
    if digest(directory / 'o7_factor_table.cubin') != receipt['cubin_sha256']:
        raise ValueError('table binary drift')
    if digest(directory / 'o7_factor_table_generated.cuh') != receipt['generated_header_sha256']:
        raise ValueError('generated header drift')
    if (directory / 'o7_factor_table_generated.cuh').read_text() != table_header((ROOT / 'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()):
        raise ValueError('table generator drift')
    if not receipt['control_opcode_counts_match_v67'] or not receipt['control_instruction_count_match_v67']:
        raise ValueError('recompiled control work changed')
    if not all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma'] and e['all_copies_bypass_l1'] for e in receipt['entries'].values()):
        raise ValueError('INT4/async instruction audit failed')
    return dict(build=receipt, loop_analysis=analyze((directory / 'o7_factor_table.sass').read_text()))


def table_coverage(activation_factors, status):
    af, flags = np.asarray(activation_factors), np.asarray(status)
    if af.ndim != 2 or af.shape[0] != 32 or af.shape[1] % 64 or flags.ndim != 2 or flags.shape[0] != af.shape[1] // 64:
        raise ValueError('expected group-major factors and CTA mask')
    valid = (af > 0) & (af < 1024) & ((af & (af-1)) == 0)
    row_tiles = valid.reshape(32, flags.shape[0], 64).all(axis=(0, 2))
    table = (flags == 0) & row_tiles[:, None]
    return dict(table_ctas=int(table.sum()), original_integer_fallback_ctas=int(((flags == 0) & ~table).sum()),
                fp32_fallback_ctas=int((flags == 1).sum()), invalid_ctas=int((flags > 1).sum()),
                scope='host_prediction_from_checked_metadata_not_device_counter')


def timing_contract(mode, inner):
    result = row_fused.timing_contract(mode, inner, 1)
    result.update(gemm_cufunction_identical_between_policies=False,
                  guard_and_table_generation_in_gemm=True, preparation_identical_between_policies=True)
    return result


class Driver(row_fused.Driver):
    def __init__(self, library, baseline, candidate):
        receipt = checked(candidate)
        super().__init__(library, baseline)
        try:
            handle = ct.c_void_p()
            self.check(self.lib.roof_probe_open(str((candidate / 'o7_factor_table.cubin').resolve()).encode(),
                                               CANDIDATE.encode(), SHARED_BYTES, ct.byref(handle)))
            self.handles[1] = handle
            values = (ct.c_int * 4)()
            self.check(self.lib.roof_probe_resources(handle, values))
            self.resources[1] = dict(registers_per_thread=values[0], local_size_bytes=values[1], threads=values[2],
                active_blocks_per_sm=values[3], shared_memory_bytes=SHARED_BYTES,
                cta_tile=[64, 128, 128], pipeline_stages=2, kernel_symbol=CANDIDATE)
            self.codegen = dict(v67=self.codegen, v76=receipt)
        except Exception:
            self.close()
            raise

    def run(self, case, policy, mode, warmup, repeats, inner):
        import torch
        if policy not in self.handles or mode not in base.MODES or np.any(case.oracle['status_flat'] == 2):
            raise ValueError('invalid input/policy/mode')
        values = (ct.c_float * (4 * repeats))()
        # SAME row-fused preparation for all policies; 0 and2 share the v67 CUfunction.
        self.check(self.lib.roof_o78_gpu_benchmark(self.handles[policy], int(case.variant[1:]), 1,
            base.MODES.index(mode), case.a_source, case.w_source, case.state_pointers, case.m, case.n,
            case.a_multiplier, case.w_multiplier, warmup, repeats, inner,
            torch.cuda.current_stream().cuda_stream, values))
        return case.state['y'], base.normalize_timings(mode, np.ctypeslib.as_array(values).reshape(4, repeats), repeats)


def summarize(rows, modes):
    from adangel.benchmark.metrics import bootstrap_median_ci
    ids = sorted({r['sample_id'] for r in rows})
    rounds = sorted({r['round'] for r in rows})
    index = {(r['sample_id'], r['mode'], r['round'], r['candidate']): r for r in rows}
    expected = {(s, m, r, p) for s in ids for m in modes for r in rounds for p in (0, 1)}
    if len(index) != len(rows) or set(index) != expected or {r['variant'] for r in rows} != {'o7'}:
        raise ValueError('incomplete or wrong-variant paired records')
    output = []
    for mode in modes:
        for policy in (0, 1):
            selected = [index[s, mode, r, policy] for s in ids for r in rounds]
            if not all(r['bitwise_equal_v67'] and r['MSE_regression_passed'] and r['metadata_exact'] for r in selected):
                raise ValueError('numerical checks failed')
            latencies = [statistics.median(index[s, mode, r, policy]['summary']['median_ms'] for r in rounds) for s in ids]
            speed = [statistics.median(index[s, mode, r, 0]['summary']['median_ms'] / index[s, mode, r, policy]['summary']['median_ms'] for r in rounds) for s in ids]
            errors = [index[s, mode, 0, policy]['mse_vs_paired_fp16'] for s in ids]
            output.append(dict(variant='o7', mode=mode, candidate=policy, samples=len(ids), records=len(selected),
                median_ms=statistics.median(latencies), paired_speedup=statistics.median(speed),
                paired_speedup_ci95=list(bootstrap_median_ci(speed, 10000, .95, 20261003)),
                median_mse=statistics.median(errors), mean_mse=statistics.mean(errors),
                selected_cv_failed_records=sum(r['summary']['cv_percent'] >= 3 for r in selected),
                any_stage_cv_failed_records=sum(any(s['cv_percent'] >= 3 for s in r['stage_summaries'].values()) for r in selected)))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu-build', type=Path, default=Path('reports/o378_roof_v73_codegen'))
    p.add_argument('--baseline', type=Path, default=Path('reports/o378_roof_v67_codegen'))
    p.add_argument('--cubins', type=Path, default=Path('reports/o378_roof_v76_codegen'))
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
    a = p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT) or min(a.rounds, a.repeats) < 1 or a.inner < 2 or a.warmup < 0:
        p.error('fresh repository output and valid timing parameters required')
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
    library, build = base.checked_gpu_build(a.gpu_build)
    driver = Driver(library, a.baseline, a.cubins)
    a.output.mkdir(parents=True)
    modes = base.MODES if a.full_modes else ('compute_only',)
    def save(name, data):
        (a.output / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    def append(name, data):
        with (a.output / name).open('a') as f:
            f.write(json.dumps(data, allow_nan=False) + '\n')
    try:
        save('validation.json', validate(driver))
        if a.validate_only:
            print('FACTOR TABLE VALIDATION PASSED', flush=True)
            return
        manifest, mh = inspect_inputs(a.data)
        raw, rh = inspect_raw_inputs(a.raw_data, manifest, a.trace_config)
        raw_index = {r['sample_id']: r for r in raw['samples']}
        save('environment.json', dict(git_commit=command('git', 'rev-parse', 'HEAD'),
            extension_sha256=sha256_file(Path(native.__file__)), codegen=driver.codegen,
            gpu_preparation_build=build, resources=driver.resources, torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(), prepared_manifest_sha256=mh, raw_manifest_sha256=rh,
            production_default_changed=False, control='v67_fullK_v73_preparation', candidate='v76_CTA_factor_table_v73_preparation',
            source_quantization='original_FP16_direct_source_quantization_excluded',
            timing_scope='all_four_modes' if a.full_modes else 'cached_compute_only_not_E2E', no_filtering=True,
            args={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}))
        rows = []
        for si, entry in enumerate(manifest['samples'][:a.samples]):
            original = raw_index[entry['sample_id']]
            path, rp = a.data / entry['file'], a.raw_data / original['file']
            assert sha256_file(path) == entry['sha256'] and sha256_file(rp) == original['sha256']
            record = _load_and_validate_raw(rp, original['layer'], original['projection'])
            prepared = load_prepared(path, device='cpu')
            verify_raw_prepared(prepared, (record['activation_fp16'], record['weight_fp16'])); del prepared
            wf, af = mf.VARIANTS['o7']
            ws = mf.quantize_source(record['weight_fp16'].cuda(), wf)
            acs = mf.quantize_source(record['activation_fp16'].cuda(), af)
            append('source_provenance.jsonl', dict(sample_id=entry['sample_id'], variant='o7', raw_sha256=original['sha256'],
                                                  weight=source_identity(ws), activation=source_identity(acs)))
            fp = native._benchmark_mixed('o5', 'compute_only', ws, acs, 0, 1, 2, '64x128x256', 'row_major')
            validate_fp16_result(fp, ws, acs)
            old = native._benchmark_mixed('o7', 'compute_only', ws, acs, 0, 1, 2, '64x128x256', 'group_major', 59, 5)
            case = fused.Case('o7', ws, acs, old)
            driver.prepare(case)
            expected, _ = driver.run(case, 2, 'compute_only', 0, 1, 2)
            expected = expected.clone()
            expected_mse = mse(expected, fp['output'])
            assert mse_regression_ok(expected_mse, mse(old['output'], fp['output']))
            coverage = table_coverage(case.oracle['activation_factors'], case.oracle['status'])
            for ri in range(a.rounds):
                append('gpu_snapshots.jsonl', dict(sample_id=entry['sample_id'], round=ri, time=time.time(),
                    gpu=command('nvidia-smi', '--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu', '--format=csv')))
                for mi, mode in enumerate(modes):
                    order = measurement_order((0, 1), si, mi, ri)
                    for policy in order:
                        output, timings = driver.run(case, policy, mode, a.warmup, a.repeats, a.inner)
                        if mode == 'conversion_only':
                            output, _ = driver.run(case, policy, 'compute_only', 0, 1, 2)
                        assert output.dtype == torch.float32 and bool(torch.isfinite(output).all())
                        assert torch.equal(output.view(torch.int32), expected.view(torch.int32))
                        error = mse(output, fp['output'])
                        assert error == expected_mse
                        arrays = {k: case.state[k].cpu().numpy() for k in ('af','wf','ab','wb','an','wn','am','wm','ast','wst','status')}
                        guard = base.check_metadata_arrays(case.oracle, arrays)
                        summaries = {s: stats(t) for s, t in timings.items()}
                        result = dict(sample_id=entry['sample_id'], variant='o7', mode=mode, round=ri, candidate=policy,
                            execution_order=list(order), raw_ms=timings, stage_summaries=summaries,
                            summary=summaries['gemm' if mode == 'compute_only' else 'total'],
                            metadata_exact=True, guard=guard, table_coverage=coverage, finite_fp32=True,
                            bitwise_equal_v67=True, MSE_regression_passed=True, mse_vs_paired_fp16=error, paired_fp16='o5',
                            mse_vs_v67=mse(output, expected), max_abs_vs_v67=(output-expected).abs().max().item(),
                            resources=driver.resources[policy], **timing_contract(mode, a.inner))
                        rows.append(result); append('results.jsonl', result)
            save('summary.json', dict(scope='O7_factor_table_' + ('four_modes' if a.full_modes else 'cached_GEMM'),
                production_default_changed=False, no_filtering=True, records=summarize(rows, modes)))
            print(entry['sample_id'], 'O7 paired complete', flush=True)
        assert sha256_file(a.data / 'manifest.json') == mh and sha256_file(a.raw_data / 'trace_manifest.json') == rh
        print('FACTOR TABLE PAIRED TEST PASSED', flush=True)
    finally:
        driver.close()


if __name__ == '__main__':
    main()
