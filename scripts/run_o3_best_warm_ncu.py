#!/usr/bin/env python3
"""Fill current-best O3 cache-primed NCU evidence, not an optimization retest."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

from profile_o3_best_warm import ROOT, SAMPLE, SYMBOL, WARMUP, select_entry, encode_driver_args, driver_files
from run_grouped_cta_ncu import analyze_capture as grouped_analysis


def capture_command(ncu, python, prefix, validation, receipts):
    return [str(ncu), '--set', 'full', '--replay-mode', 'application',
        '--cache-control', 'none', '--clock-control', 'none',
        '--kernel-name-base', 'function', '--kernel-name', 'regex:^'+SYMBOL+'$',
        '--launch-skip', str(WARMUP), '--launch-count', '1', '-o', str(prefix),
        str(python), 'scripts/profile_o3_best_warm.py', '--validation', str(validation),
        '--output', str(receipts)]


def verify_passes(receipts):
    if not receipts:
        raise ValueError('no completed application-replay receipt')
    first = receipts[0]
    keys = ('sample_id', 'variant', 'shape', 'expected_kernel', 'gemm_binary_sha256',
        'extension_sha256_before', 'extension_sha256_after', 'resources', 'guard',
        'validation_sha256', 'git_commit', 'source_sha256', 'raw_manifest_sha256',
        'prepared_manifest_sha256', 'raw_sample_sha256', 'prepared_sample_sha256')
    for r in receipts:
        if any(r.get(k) != first.get(k) for k in keys):
            raise ValueError('application pass identity drift')
        if (r['sample_id'] != SAMPLE or r['variant'] != 'o3' or r['expected_kernel'] != SYMBOL or
                r['shape'] != [4096]*3 or r['policy'] != 1 or
                r['guard'] != dict(integer_ctas=2048, fallback_ctas=0, invalid_ctas=0) or
                not r['numerical_checks_passed'] or not r['bitwise_previous_best'] or
                not r['finite_fp32'] or r['mse_vs_previous_best'] != 0 or
                r['replay_mode'] != 'application' or r['cache_control'] != 'none' or
                r['clock_control'] != 'none' or r['filtered_launch_skip'] != WARMUP or
                r['filtered_launch_count'] != 1 or r['production_default_changed'] or
                r['new_performance_result'] or r['new_candidate_implemented'] or
                r['extension_sha256_before'] != r['extension_sha256_after']):
            raise ValueError('application pass contract failed')
    return first


def analyze_capture(raw, source, receipts):
    first = verify_passes(receipts)
    metric_rows = list(csv.DictReader(io.StringIO(raw)))
    if len(metric_rows) != 2:
        raise ValueError('exactly one launch plus units required')
    result = grouped_analysis(raw, source, first)
    if result['registers_per_thread'] != first['resources']['registers_per_thread']:
        raise ValueError('profile resources differ from audited binary')
    result.update(scope='current_best_O3_cache_primed_NCU_not_Event_speedup',
        sample_id=SAMPLE, completed_application_pass_receipts=len(receipts),
        replay_mode='application', cache_control='none', clock_control='none', warmup=WARMUP,
        numerical_checks_passed_all_completed_passes=True, paired_fp16='o0',
        new_performance_result=False, production_default_changed=False,
        historical_cache_all_profiles_not_paired_speedups=True,
        raw_selected_metrics={k: val for k, val in metric_rows[1].items() if k and k.startswith(
            ('smsp__average_warps_issue_stalled_', 'sm__inst_executed_pipe_',
             'smsp__sass_', 'lts__', 'dram__', 'l1tex__', 'gpc__cycles_elapsed'))})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--analyze-only', action='store_true')
    args = p.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT) or (out.exists() and not args.analyze_only):
        p.error('fresh repository output directory required')
    commands = []
    def run(cmd, name):
        commands.append(dict(command=cmd, output=name))
        (out / 'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        with (out / name).open('w') as f:
            subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
    ncu = Path('/usr/local/cuda-12.8/bin/ncu')
    if not args.analyze_only:
        out.mkdir(parents=True)
        import torch  # Load libtorch before importing the existing extension.
        from adangel import _sm80 as native
        from adangel.trace.storage import sha256_file
        from benchmark_a100_mixed_trace import inspect_inputs, inspect_raw_inputs
        from benchmark_o3_grouped_cta import build
        manifest, mh = inspect_inputs(ROOT / 'data/prepared/llama2_7b_prefill_o0_o4')
        raw, rh = inspect_raw_inputs(ROOT / 'data/raw/llama2_7b_prefill', manifest,
                                   ROOT / 'configs/trace/llama2_7b_prefill.yaml')
        select_entry(manifest); select_entry(raw)
        # Immutable original GEMM cubin; only the existing host/conversion adapter is rebuilt.
        encoded = encode_driver_args(build(out / 'build', ROOT / 'reports/o378_roof_v89_o3_codegen'))
        validation = dict(full24_hashes_checked=True, sample_id=SAMPLE,
            prepared_manifest_sha256=mh, raw_manifest_sha256=rh,
            extension_sha256=sha256_file(Path(native.__file__)), driver_args=encoded,
            driver_sha256={str(path): sha256_file(path) for path in driver_files(encoded)})
        (out / 'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
        run([str(ncu), '--version'], 'ncu_version.txt')
        run(capture_command(ncu, sys.executable, out / 'o3_warm', out / 'validation.json',
                            out / 'pass_receipts'), 'capture.log')
        for suffix, page in (('raw', 'raw'), ('source_sass', 'source')):
            cmd = [str(ncu), '--import', str(out / 'o3_warm.ncu-rep'), '--csv', '--page', page]
            if page == 'source': cmd += ['--print-source', 'sass']
            run(cmd, 'o3_warm_'+suffix+'.csv')
    receipts = [json.loads(path.read_text()) for path in sorted((out / 'pass_receipts').glob('pass_*/receipt.json'))]
    result = analyze_capture((out / 'o3_warm_raw.csv').read_text(encoding='utf-8-sig'),
                             (out / 'o3_warm_source_sass.csv').read_text(encoding='utf-8-sig'), receipts)
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('CURRENT BEST O3 WARM NCU IDENTITY/WORK CHECKS PASSED', flush=True)


if __name__ == '__main__':
    main()
