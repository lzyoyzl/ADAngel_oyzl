"""Recompute integrated vector16 timing, numerics, and full codegen evidence."""
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import struct
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
E = ROOT / 'docs/evidence/a100_o378_roof_v54'
P = E / 'reports/o378_roof_v54'
BINARY = '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class V54EvidenceTests(unittest.TestCase):
    def test_four_modes_paired_raw_timing_and_mse(self):
        self.check_run('trace24', [4, 5])

    def test_independent_reversed_order_confirmation(self):
        self.check_run('confirm24', [5, 4])

    def check_run(self, name, order):
        from benchmark_a100_o1 import stats
        from benchmark_vector_conversion_pipeline import summarize
        run = E / f'runs/o378_roof_v54_{name}'
        records = rows(run / 'results.jsonl')
        self.assertEqual(len(records), 384)
        self.assertEqual(len({r['sample_id'] for r in records}), 24)
        self.assertEqual({r['round'] for r in records}, {0})
        self.assertEqual({r['implementation'] for r in records}, {4, 5})
        expected_stages = {
            'conversion_only': {'weight_conversion', 'activation_conversion', 'total'},
            'compute_only': {'gemm', 'total'},
            'cold': {'weight_conversion', 'activation_conversion', 'gemm', 'total'},
            'steady_state': {'activation_conversion', 'gemm', 'total'},
        }
        for r in records:
            self.assertEqual(set(r['raw_ms']), expected_stages[r['mode']])
            self.assertEqual(set(r['stage_summaries']), set(r['raw_ms']))
            for stage, values in r['raw_ms'].items():
                self.assertEqual(len(values), 200)
                self.assertTrue(all(x > 0 for x in values))
                self.assertEqual(r['stage_timing_inner_repeats'][stage],
                    100 if 'conversion' in stage or r['mode'] == 'conversion_only' else 1)
                for key, value in stats(values).items():
                    self.assertAlmostEqual(value, r['stage_summaries'][stage][key],
                        delta=max(1e-14, abs(value) * 1e-12))
                self.assertEqual(statistics.median(values), r['stage_summaries'][stage]['median_ms'])
            selected = 'gemm' if r['mode'] == 'compute_only' else 'total'
            self.assertEqual(r['summary'], r['stage_summaries'][selected])
            self.assertEqual(r['selected_stage'], selected)
            if r['mode'] == 'conversion_only':
                self.assertEqual(r['total_timing'], 'sum_of_batched_stage_samples')
                self.assertEqual(r['raw_ms']['total'], [struct.unpack('f', struct.pack('f', a+w))[0]
                    for a, w in zip(r['raw_ms']['activation_conversion'], r['raw_ms']['weight_conversion'])])
            else:
                self.assertEqual(r['total_timing'], 'single_execution_cuda_event')
            self.assertEqual(r['weight_cached'], r['mode'] not in ('cold', 'conversion_only'))
            self.assertEqual(r['activation_prepared'], r['mode'] == 'compute_only')
            self.assertEqual(r['kernel']['roof_tune'], 59)
            self.assertFalse(r['kernel']['gemm_math_changed'])
            self.assertEqual(r['kernel']['conversion_vector_elements'], 16 if r['implementation'] == 5 else 0)
            if r['implementation'] == 5:
                self.assertEqual(r['kernel']['weight_conversion_impl'], 5)
                self.assertEqual(r['kernel']['activation_conversion_impl'], 5)
                self.assertEqual(r['kernel']['conversion_kernels_per_operand'], 1)
            self.assertTrue(r['bitwise_equal_current_best'])
            self.assertEqual(r['mse_vs_current_best'], 0.)
        self.assertEqual(summarize(records), read(run / 'summary.json'))
        for variant, median, mean, ref in (
                ('o7', .0055361724266658075, .005053635833761954, 'o5'),
                ('o8', .004411084948644645, .0043813792153016215, 'o6')):
            for impl in (4, 5):
                for mode in expected_stages:
                    group = [r for r in records if (r['variant'], r['implementation'], r['mode']) == (variant, impl, mode)]
                    self.assertEqual(len(group), 24)
                    self.assertEqual(statistics.median(r['mse_vs_paired_fp16'] for r in group), median)
                    self.assertAlmostEqual(statistics.fmean(r['mse_vs_paired_fp16'] for r in group), mean, places=16)
                    self.assertTrue(all(r['paired_reference'] == ref for r in group))
        env = read(run / 'environment.json')
        self.assertEqual(env['binary_sha256'], BINARY)
        self.assertFalse(env['gemm_math_changed'])
        self.assertEqual(env['args']['warmup'], 50)
        self.assertEqual(env['args']['repeats'], 200)
        self.assertEqual(env['args']['inner'], 100)
        self.assertEqual(env['args']['implementations'], order)
        self.assertEqual(len(rows(run / 'source_provenance.jsonl')), 48)
        self.assertEqual(read(run / 'validation.json'), dict(
            passed=True, records=384, samples=24, output_bitwise_current_best=True))

    def test_full_codegen_comparison_and_exact_vector_parity(self):
        from audit_vector_conversion_pipeline import analyze
        before = gzip.decompress((P / 'before.sass.gz').read_bytes())
        after = gzip.decompress((P / 'after.sass.gz').read_bytes())
        resources = (P / 'resources.txt').read_bytes()
        probe = (ROOT / 'docs/evidence/a100_o378_roof_v53/reports/o378_roof_v53_build2/conversion.sass').read_bytes()
        saved = read(P / 'codegen_audit.json')
        result = analyze(before.decode(), after.decode(), resources.decode(), probe.decode())
        self.assertEqual(result, {k: v for k, v in saved.items() if k != 'sources'})
        for data, source in zip((before, after, resources, probe), saved['sources']):
            self.assertEqual(hashlib.sha256(data).hexdigest(), source['sha256'])
        self.assertTrue(result['passed'])
        self.assertEqual(result['gemm_regression']['old_symbols'], 162)
        self.assertEqual(len(result['vector_entries']), 4)
        self.assertEqual(len(result['all_old_entries']['changed']), 5)
        self.assertTrue(all('mixed_binary' in s for s in result['all_old_entries']['changed']))
        self.assertTrue(all(result['vector_exact_v53'].values()))

    def test_correctness_and_limited_safety_coverage(self):
        for name, count in (('preflight', 144), ('memcheck', 48), ('synccheck', 48), ('racecheck', 48)):
            result = read(E / f'runs/o378_roof_v54_{name}/validation.json')
            self.assertTrue(result['passed'])
            self.assertEqual(result['binary_sha256'], BINARY)
            self.assertEqual(len(result['checks']), count)
            self.assertEqual(len(result['rejected']), 6)
            self.assertEqual(result['quick'], name != 'preflight')
            self.assertTrue(all(r['bitwise'] and r['nondefault_stream'] and r['semantic_tolerance_passed']
                                for r in result['checks']))
        for name in ('memcheck', 'synccheck'):
            self.assertIn('ERROR SUMMARY: 0 errors', (P / f'{name}.log').read_text())
        self.assertIn('0 hazards displayed (0 errors, 0 warnings)', (P / 'racecheck.log').read_text())
        regression = read(E / 'runs/o378_roof_v54_mixed_regression/validation.json')
        self.assertTrue(regression['passed'])
        self.assertEqual(regression['binary_sha256'], BINARY)
        self.assertEqual(len(regression['binary_gemm_checks']), 480)
        audit = read(P / 'mma_audit/audit.json')
        self.assertTrue(audit['passed'])
        self.assertEqual(audit['binary_sha256'], BINARY)
        self.assertEqual(len(audit['functions']), 150)


if __name__ == '__main__':
    unittest.main()
