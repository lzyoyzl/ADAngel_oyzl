#!/usr/bin/env python3
"""v114: fill missing current-best O8/cache-primed NCU evidence, no optimization scan."""
from collections import Counter
import argparse
import csv
import io
import json
from pathlib import Path
import re
import subprocess
import sys

from profile_o8_best_warm import ROOT, SAMPLE, SYMBOL, select_entry
from analyze_roof_scale_ncu import analyze
from run_eight_chain_ncu import normalized_counts


def capture_command(ncu, python, prefix, validation, receipts):
    return [str(ncu), '--set', 'full', '--replay-mode', 'application',
        '--cache-control', 'none', '--clock-control', 'none',
        '--kernel-name-base', 'function', '--kernel-name', 'regex:^'+SYMBOL+'$',
        '--launch-skip', '50', '--launch-count', '1', '-o', str(prefix),
        str(python), 'scripts/profile_o8_best_warm.py',
        '--validation', str(validation), '--output', str(receipts)]


def verify_passes(receipts):
    if not receipts:
        raise ValueError('application replay produced no completed pass receipt')
    first = receipts[0]
    identities = ('sample_id', 'variant', 'shape', 'expected_kernel', 'source_identity',
                  'gemm_binary_sha256', 'extension_sha256_before', 'extension_sha256_after',
                  'resources', 'guard', 'validation_sha256', 'git_commit', 'source_sha256')
    for r in receipts:
        if any(r.get(key) != first.get(key) for key in identities):
            raise ValueError('application pass identity drift')
        if (r['sample_id'] != SAMPLE or r['variant'] != 'o8' or r['expected_kernel'] != SYMBOL or
                r['shape'] != [4096]*3 or r['guard']['integer_ctas'] != 2048 or
                r['guard']['fallback_ctas'] or r['guard']['invalid_ctas'] or
                not r['numerical_checks_passed'] or not r['bitwise_previous_fullK'] or
                r['mse_vs_previous_fullK'] != 0 or not r['finite_fp32'] or
                r['replay_mode'] != 'application' or r['cache_control'] != 'none' or
                r['clock_control'] != 'none' or r['filtered_launch_skip'] != 50 or
                r['filtered_launch_count'] != 1 or r['production_default_changed'] or
                r['new_GEMM_measured'] or r['new_candidate_implemented'] or
                r['extension_sha256_before'] != r['extension_sha256_after']):
            raise ValueError('application pass contract failed')
    return first


def analyze_capture(raw, source, receipts):
    receipt = verify_passes(receipts)
    expected = receipt['codegen']['build']['entries'][SYMBOL]
    if receipt['gemm_binary_sha256'] != receipt['codegen']['build']['cubin_sha256']:
        raise ValueError('loaded binary differs from audited best')
    rows = list(csv.DictReader(io.StringIO(raw)))
    if len(rows) != 2:
        raise ValueError('one profiled launch plus units required')
    metric = rows[1]
    if int(metric['launch__block_size'].replace(',', '')) != 128 or int(metric['launch__grid_size'].replace(',', '')) != 2048:
        raise ValueError('unexpected launch geometry')
    counts = Counter()
    for row in csv.DictReader(io.StringIO(source.split('\n', 1)[1])):
        match = re.match(r'\s*(?:@!?U?P(?:T|\d+)\s+)?([A-Z][A-Z0-9_]*)', row['Source'])
        if not match:
            raise ValueError('unrecognized SASS opcode')
        counts[match[1]] += 1
    if counts != normalized_counts(expected['opcode_counts']) or sum(counts.values()) != expected['instructions']:
        raise ValueError('profile source fingerprint differs from exact audited best')
    result = analyze(raw, source, 59, 'o8', True, True, expected_symbol=SYMBOL, fullk_integer=True)
    if result['registers_per_thread'] != receipt['resources']['registers_per_thread']:
        raise ValueError('NCU and loaded resources disagree')
    result.update(static_fingerprint_verified=True, completed_application_pass_receipts=len(receipts),
        sample_id=SAMPLE, replay_mode='application', cache_control='none', clock_control='none',
        mse_vs_paired_fp16=receipt['mse_vs_paired_fp16'], paired_fp16='o6',
        numerical_checks_passed_all_completed_passes=True,
        raw_selected_metrics={k: val for k, val in metric.items() if k and k.startswith(
            ('smsp__average_warps_issue_stalled_', 'sm__inst_executed_pipe_',
             'smsp__sass_', 'lts__', 'dram__', 'l1tex__', 'gpc__cycles_elapsed'))})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--analyze-only', action='store_true')
    a = p.parse_args()
    out = a.output.resolve()
    if not out.is_relative_to(ROOT) or (out.exists() and not a.analyze_only):
        p.error('fresh repository directory for capture required')
    commands = []
    def run(cmd, name):
        commands.append(dict(command=cmd, output=name))
        (out / 'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        with (out / name).open('w') as f:
            subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
    ncu = Path('/usr/local/cuda-12.8/bin/ncu')
    if not a.analyze_only:
        out.mkdir(parents=True)
        from benchmark_a100_mixed_trace import inspect_inputs, inspect_raw_inputs
        from adangel.trace.storage import sha256_file
        manifest, mh = inspect_inputs(ROOT / 'data/prepared/llama2_7b_prefill_o0_o4')
        raw, rh = inspect_raw_inputs(ROOT / 'data/raw/llama2_7b_prefill', manifest,
                                    ROOT / 'configs/trace/llama2_7b_prefill.yaml')
        select_entry(manifest); re_entry = select_entry(raw)
        provenance = ROOT / 'docs/evidence/a100_o378_roof_v99/runs/o378_roof_v99_o78/source_provenance.jsonl'
        source = [r for r in map(json.loads, provenance.read_text().splitlines())
                  if r['sample_id'] == SAMPLE and r['variant'] == 'o8']
        if len(source) != 1 or source[0]['raw_sha256'] != re_entry['sha256']:
            raise ValueError('exact measured source required')
        validation = dict(full24_hashes_checked=True, sample_id=SAMPLE, raw_manifest_sha256=rh,
            prepared_manifest_sha256=mh, v99_source=source[0], v99_provenance_sha256=sha256_file(provenance))
        (out / 'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
        run([str(ncu), '--version'], 'ncu_version.txt')
        run(capture_command(ncu, sys.executable, out / 'o8_warm', out / 'validation.json',
                            out / 'pass_receipts'), 'capture.log')
        for suffix, page in (('raw', 'raw'), ('source_sass', 'source')):
            cmd = [str(ncu), '--import', str(out / 'o8_warm.ncu-rep'), '--csv', '--page', page]
            if page == 'source':
                cmd += ['--print-source', 'sass']
            run(cmd, 'o8_warm_'+suffix+'.csv')
    paths = sorted((out / 'pass_receipts').glob('pass_*/receipt.json'))
    receipts = [json.loads(path.read_text()) for path in paths]
    result = analyze_capture((out / 'o8_warm_raw.csv').read_text(encoding='utf-8-sig'),
                             (out / 'o8_warm_source_sass.csv').read_text(encoding='utf-8-sig'), receipts)
    result.update(scope='current_best_O8_cache_primed_NCU_diagnostic_not_Event_speedup',
        new_performance_result=False, production_default_changed=False,
        historical_cache_all_profiles_not_paired_speedups=True,
        no_repeat='missing_current_O8_and_cache_primed_profiling_not_an_optimization_retest')
    (out / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('CURRENT BEST O8 WARM NCU IDENTITY/WORK CHECKS PASSED', flush=True)


if __name__ == '__main__':
    main()
