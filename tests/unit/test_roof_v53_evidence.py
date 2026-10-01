"""Recompute vector-conversion results from raw Event, SASS and NCU evidence."""
import hashlib
import json
from pathlib import Path
import statistics
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
E = ROOT / 'docs/evidence/a100_o378_roof_v53'
P = E / 'reports/o378_roof_v53_build2'
LIB_SHA = 'b7c47d2bc596cebf6e16bd75912f6ff142ddb6e4723bd066abf0e0e939cc563d'
EXT_SHA = 'fe9c2b18d89787231bf988b704a29d2041edcfdad558c98acee3f5b2195b366f'


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


class V53EvidenceTests(unittest.TestCase):
    def test_all_pairs_raw_events_and_cv(self):
        from benchmark_vector_conversion_probe import summarize
        from benchmark_a100_o1 import stats
        run = E / 'runs/o378_roof_v53_trace24'
        records = lines(run / 'results.jsonl')
        self.assertEqual(len(records), 864)
        self.assertEqual(len({r['sample_id'] for r in records}), 24)
        self.assertEqual({r['round'] for r in records}, {0, 1, 2})
        self.assertEqual({r['implementation'] for r in records}, {0, 1, 2})
        for r in records:
            self.assertEqual(len(r['timings_ms']), 200)
            self.assertTrue(all(x > 0 for x in r['timings_ms']))
            self.assertEqual(r['inner'], 100)
            self.assertTrue(r['bitwise_payload_and_scale'])
            # Python 3.10/3.11 statistics differ in the last CV bit; all raw
            # samples/medians remain exact, do not turn that into a GPU failure.
            recomputed_stats = stats(r['timings_ms'])
            self.assertEqual(recomputed_stats.keys(), r['summary'].keys())
            for key, value in recomputed_stats.items():
                self.assertAlmostEqual(value, r['summary'][key], delta=max(1e-14, abs(value)*1e-12))
            self.assertEqual(recomputed_stats['median_ms'], r['summary']['median_ms'])
        recomputed = summarize(records)
        self.assertEqual(recomputed, read(run / 'summary.json'))
        self.assertEqual(sum(r['cv_failed_records'] for r in recomputed), 1)
        selected = [r for r in recomputed if r['implementation'] == 2]
        self.assertEqual(len(selected), 4)
        self.assertTrue(all(r['cv_failed_records'] == 0 for r in selected))
        self.assertTrue(all(r['paired_speedup_ci95'][0] > 1.6 for r in selected))
        env = read(run / 'environment.json')
        self.assertEqual(env['library_sha256'], LIB_SHA)
        self.assertEqual(env['binary_sha256'], EXT_SHA)
        self.assertFalse(env['gemm_performance_measured'])
        self.assertFalse(env['formal_dispatch_changed'])
        self.assertEqual(env['args']['warmup'], 50)
        self.assertEqual(len(lines(run / 'source_provenance.jsonl')), 96)

    def test_all_output_mse_checks(self):
        run = E / 'runs/o378_roof_v53_trace24'
        checks = lines(run / 'output_mse.jsonl')
        self.assertEqual(len(checks), 144)
        index = {(r['sample_id'], r['variant'], r['implementation']): r for r in checks}
        self.assertEqual(len(index), 144)
        expected = {'o7': (0.0055361724266658075, 0.005053635833761954, 'o5'),
                    'o8': (0.004411084948644645, 0.0043813792153016215, 'o6')}
        for variant, (median, mean, reference) in expected.items():
            for policy in (0, 1, 2):
                rows = [r for r in checks if (r['variant'], r['implementation']) == (variant, policy)]
                self.assertEqual(len(rows), 24)
                self.assertEqual(statistics.median(r['mse_vs_paired_fp16'] for r in rows), median)
                self.assertAlmostEqual(statistics.fmean(r['mse_vs_paired_fp16'] for r in rows), mean, places=16)
                for r in rows:
                    self.assertTrue(r['output_bitwise_current_best'])
                    self.assertEqual(r['mse_vs_current_best'], 0)
                    self.assertEqual(r['paired_reference'], reference)
                    self.assertEqual(r['mse_vs_paired_fp16'],
                                     index[(r['sample_id'], variant, 0)]['mse_vs_paired_fp16'])
        self.assertEqual(read(run / 'validation.json')['output_checks'], 144)

    def test_gpu_safety_and_reference_coverage(self):
        for mode in ('preflight', 'memcheck', 'synccheck', 'racecheck'):
            result = read(E / f'runs/o378_roof_v53_{mode}/validation.json')
            self.assertTrue(result['passed'])
            self.assertFalse(result['quick'])
            self.assertEqual(result['library_sha256'], LIB_SHA)
            self.assertEqual(result['binary_sha256'], EXT_SHA)
            self.assertEqual(len(result['checks']), 393)
            self.assertEqual(len(result['rejected']), 18)
            self.assertTrue(all(r['bitwise'] and r['nondefault_stream'] for r in result['checks']))
            self.assertEqual({r['shape'][1] for r in result['checks']}, {128, 256, 384, 640, 4096})
        for mode in ('memcheck', 'synccheck'):
            self.assertIn('ERROR SUMMARY: 0 errors', (P / f'{mode}.log').read_text())
        self.assertIn('0 hazards displayed (0 errors, 0 warnings)', (P / 'racecheck.log').read_text())

    def test_scalar_control_and_resource_audit(self):
        from audit_vector_conversion_probe import analyze
        result = analyze((P / 'conversion.sass').read_text(),
                         (P / 'conversion.resources.txt').read_text(),
                         (P / 'scalar_reference.sass').read_text())
        self.assertTrue(result['passed'])
        self.assertEqual(result, read(P / 'audit.json'))
        self.assertEqual(len(result['controls_exact_sass']), 12)
        self.assertEqual(len(result['functions']), 8)
        self.assertTrue(all(r['resources']['REG'] <= 32 for r in result['functions']))
        self.assertEqual(hashlib.sha256((P / 'scalar_reference.sass').read_bytes()).hexdigest(),
                         '17806897913e0b7c3f4a632b0c1b655990b4514d6f39fb71754a6c67c9772355')

    def test_same_entry_ncu_and_work_reduction(self):
        from analyze_vector_conversion_ncu import analyze, FORMATS
        saved = read(P / 'ncu_analysis.json')['rows']
        rows = []
        for fmt in FORMATS:
            for policy in (0, 2):
                base = P / 'ncu' / f'{fmt}_p{policy}'
                raw = base.with_name(base.name + '_raw.csv').read_text()
                source = base.with_name(base.name + '_source_sass.csv').read_text()
                rows.append(analyze(raw, source, fmt, policy))
                receipt = read(E / f'runs/o378_roof_v53_profile_{fmt}_p{policy}/profile_receipt.json')
                self.assertTrue(receipt['passed'])
                self.assertEqual(receipt['library_sha256'], LIB_SHA)
                self.assertEqual(receipt['target_launch_index_zero_based'], 50)
                with self.assertRaises(ValueError):
                    analyze(raw, source, fmt, 2 - policy)
            control, candidate = rows[-2:]
            self.assertLess(candidate['dynamic_warp_instructions'], control['dynamic_warp_instructions'])
            self.assertLess(candidate['ncu_ms'], control['ncu_ms'])
        self.assertEqual(rows, saved)


if __name__ == '__main__':
    unittest.main()
