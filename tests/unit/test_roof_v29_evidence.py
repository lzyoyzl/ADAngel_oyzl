"""Reconcile archived v29 measurements; this is not another GPU experiment."""
import hashlib
import json
from pathlib import Path
import runpy
import statistics
import unittest

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'docs/evidence/a100_o378_roof_v29'
REPORTS = ROOT / 'reports/o378_roof_v29'
BINARY = 'cb59c3a0e3c6856eeda503bcce00bfbc06c1a539836d5d698206c2ea80711774'


@unittest.skipUnless((REPORTS / 'audit/audit.json').exists(), 'completed archive not present')
class RowScaleEvidenceTests(unittest.TestCase):
    def load(self, path):
        return json.loads((ROOT / path).read_text())

    def records(self, run):
        return [json.loads(line) for line in
                (ROOT / 'runs' / run / 'results.jsonl').read_text().splitlines()]

    def test_native_int4_old_target_preservation_and_spill_disclosure(self):
        x = self.load('reports/o378_roof_v29/audit/audit.json')
        self.assertEqual(x['binary_sha256'], BINARY)
        self.assertTrue(x['passed'])
        self.assertFalse(x['strict_passed'])
        self.assertEqual(len(x['functions']), 140)
        new = [f for f in x['functions'] if any(f'Li{t}E' in f['symbol'] for t in (51, 52))]
        self.assertEqual(len(new), 2)
        for f in new:
            for key in ('sass_u4s4', 'sass_s4s4', 'sass_no_int8', 'sass_async'):
                self.assertTrue(f['checks'][key])
            self.assertFalse(f['strict_passed'])
            self.assertIn('REG:168', f['resource'])
        for name, count in (('production_codegen.json', 12), ('candidate_codegen.json', 138)):
            x = self.load('reports/o378_roof_v29/' + name)
            self.assertTrue(x['passed'])
            self.assertEqual(len(x['unchanged']), count)
        x = self.load('reports/o378_roof_v29/all_sm80_codegen.json')
        self.assertFalse(x['passed'])
        self.assertEqual(x['old_symbols'], 270)
        self.assertEqual(len(x['changed']), 5)
        self.assertTrue(all('mixed_binary' in s for s in x['changed']))

    def test_synthetic_safety_guard_and_non_target_regression(self):
        for kind, count in (('preflight', 192), ('memcheck', 84), ('synccheck', 84)):
            x = self.load(f'runs/o378_roof_v29_{kind}/validation.json')
            self.assertTrue(x['passed'])
            self.assertEqual(len(x['checks']), count)
            self.assertTrue(all(r['semantic_tolerance_passed'] for r in x['checks']))
            self.assertEqual(self.load(f'runs/o378_roof_v29_{kind}/environment.json')['binary_sha256'], BINARY)
            if kind != 'preflight':
                self.assertIn('ERROR SUMMARY: 0 errors', (REPORTS / f'{kind}.log').read_text())
        self.assertIn('0 hazards displayed (0 errors, 0 warnings)', (REPORTS / 'racecheck.log').read_text())
        x = self.load('reports/o378_roof_v29/guard_checks.json')
        self.assertTrue(x['passed'])
        self.assertEqual(x['binary_sha256'], BINARY)
        self.assertEqual(len(x['checks']), 20)
        self.assertTrue(all(r['rejected'] for r in x['checks']))
        x = self.load('runs/o378_roof_v29_mixed_regression/validation.json')
        self.assertTrue(x['passed'])
        self.assertEqual(len(x['binary_gemm_checks']), 480)
        self.assertEqual(x['binary_sha256'], BINARY)

    def test_real_trace_reassociation_mse_and_complete_coverage(self):
        rows = self.records('o378_roof_v29_trace24')
        self.assertEqual(len(rows), 600)
        self.assertEqual({r['variant'] for r in rows}, {'o3'})
        for tune in (-1, 41, 42, 51, 52):
            group = [r for r in rows if r['tune'] == tune]
            self.assertEqual(len(group), 120)
            self.assertEqual(len({r['sample_id'] for r in group}), 24)
            self.assertEqual({r['round'] for r in group}, set(range(5)))
            for r in group:
                self.assertEqual(len(r['raw_ms']), 200)
                self.assertTrue(r['semantic_tolerance_passed'])
                self.assertTrue(r['mse_regression_passed'])
                if tune not in (51, 52):
                    self.assertTrue(r['bitwise_equal_production'])
                    continue
                self.assertFalse(r['bitwise_equal_production'])
                self.assertGreater(r['mse_vs_production'], 0)
                self.assertLess(r['mse_vs_production'], 3e-14)
                self.assertLessEqual(r['max_abs_vs_production'], 3.1e-5)
                self.assertLess(r['max_abs_vs_semantic_fp64'], 7e-6)
                old = r['baseline_mse']['mse_vs_o0']
                self.assertLess(abs(r['mse_vs_o0'] - old), 1e-12 + 1e-5 * old)
                k = r['kernel']
                self.assertEqual(k['cta_tile'], [64, 128, 128])
                self.assertEqual(k['cross_group_accumulator_dtype'], 'fp32')
                self.assertTrue(k['row_scale_in_epilogue'])
                self.assertTrue(k['unscaled_fp32_bound_checked'])
                self.assertEqual(k['pipeline_stages'], 2 if tune == 51 else 3)

    def test_paired_compute_improvement_retains_failed_cv(self):
        rows = self.records('o378_roof_v29_trace24')
        for candidate in (51, 52):
            x = self.load(f'reports/o378_roof_v29/trace24_{candidate}_vs41.json')
            self.assertEqual(x['source_binary_sha256'], BINARY)
            x = x['rows'][0]
            ratios = []
            for sid in sorted({r['sample_id'] for r in rows}):
                lookup = {(r['round'], r['tune']): r for r in rows if r['sample_id'] == sid}
                ratios.append(statistics.median(lookup[i, 41]['summary']['median_ms'] /
                              lookup[i, candidate]['summary']['median_ms'] for i in range(5)))
            self.assertEqual(statistics.median(ratios), x['paired_speedup_median'])
            self.assertGreater(x['paired_speedup_ci95'][0], 1)
            for tune in (41, candidate):
                failed = sum(r['tune'] == tune and r['summary']['cv_percent'] >= 3 for r in rows)
                self.assertEqual(x['cv'][str(tune)]['selected_stage_failed'], failed)
                self.assertGreater(failed, 0)

    def test_four_modes_charge_layout_conversion_and_use_direct_totals(self):
        run = 'o378_roof_v29_four24'
        rows = self.records(run)
        self.assertEqual(len(rows), 288)
        env = self.load(f'runs/{run}/environment.json')
        self.assertEqual(env['binary_sha256'], BINARY)
        self.assertEqual((env['args']['warmup'], env['args']['repeats'], env['args']['inner']), (50, 200, 100))
        modes = ('conversion_only', 'compute_only', 'cold', 'steady_state')
        for t in (-1, 41, 52):
            for mode in modes:
                group = [r for r in rows if (r['tune'], r['mode']) == (t, mode)]
                self.assertEqual(len(group), 24)
                for r in group:
                    self.assertTrue(r['mse_regression_passed'])
                    self.assertEqual(len(r['raw_ms']), 200)
                    self.assertEqual(r['total_timing'], 'sum_of_batched_stage_samples' if mode == 'conversion_only'
                                     else 'single_execution_cuda_event')
                    if t == 52:
                        self.assertTrue(r['kernel']['row_scale_in_epilogue'])
                        self.assertEqual(r['kernel']['activation_payload_reorder_traffic_bytes'], 33554432)
                        self.assertEqual(r['kernel']['weight_payload_reorder_traffic_bytes'], 16777216)
        compare = runpy.run_path(str(REPO / 'scripts/compare_roof_trace_candidates.py'))['compare']
        x = self.load('reports/o378_roof_v29/four24_52_vs41.json')
        self.assertEqual(compare(rows, 41, 52, 24, 1, ['o3'], modes, True), x['rows'])

    def test_ncu_reanalysis_less_support_work_not_fewer_mma_or_i2f(self):
        x = self.load('reports/o378_roof_v29/ncu_analysis.json')
        for source in x['sources']:
            self.assertEqual(hashlib.sha256((ROOT / source['file']).read_bytes()).hexdigest(), source['sha256'])
        analyze = runpy.run_path(str(REPO / 'scripts/analyze_roof_scale_ncu.py'))['analyze']
        rows = {r['tune']: r for r in x['rows']}
        self.assertEqual(set(rows), {41, 42, 51, 52})
        for t, r in rows.items():
            raw = (REPORTS / f'ncu_o3_t{t}_raw.csv').read_text(encoding='utf-8-sig')
            sass = (REPORTS / f'ncu_o3_t{t}_source_sass.csv').read_text(encoding='utf-8-sig')
            self.assertEqual(analyze(raw, sass, t, 'o3', True, True), r)
            for op in ('IMMA', 'I2F', 'FFMA'):
                self.assertEqual(r['opcodes'][op], 16777216)
            self.assertEqual(r['opcodes'].get('FMUL', 0), 524288 if t in (51, 52) else 0)
            self.assertEqual(r['source_memory_work']['L1 Wavefronts Shared'], 29622272)
            self.assertEqual(r['registers_per_thread'], 168)
            self.assertEqual(r['max_ctas_per_sm_from_launch_limits'], 3)
            self.assertEqual(r['binding_modeled_resources'], ['mma', 'i2f'])
        for old, new in ((41, 51), (42, 52)):
            self.assertLess(rows[new]['dynamic_instructions'], rows[old]['dynamic_instructions'])
            self.assertLess(rows[new]['source_memory_work']['L2 Theoretical Sectors Local'],
                            rows[old]['source_memory_work']['L2 Theoretical Sectors Local'])
            # Faster here does not mean every utilization counter must rise.
            self.assertLess(rows[new]['eligible_warps'], rows[old]['eligible_warps'])


if __name__ == '__main__':
    unittest.main()
