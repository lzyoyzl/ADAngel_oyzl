"""Recompute v51 shared-parameter alignment evidence; no new GPU work."""
import hashlib
import json
from pathlib import Path
import runpy
import statistics
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / 'docs/evidence/a100_o378_roof_v51'
P = E / 'reports/o378_roof_v51'
sys.path.insert(0, str(ROOT / 'scripts'))


def read(path):
    return json.loads(path.read_text())


class V51EvidenceTests(unittest.TestCase):
    def test_complete_pairs_raw_timing_and_mse(self):
        run = E / 'runs/o378_roof_v51_trace24'
        rows = [json.loads(s) for s in (run / 'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 144)
        self.assertEqual(len({r['sample_id'] for r in rows}), 24)
        for row in rows:
            self.assertEqual(len(row['raw_ms']), 200)
            self.assertTrue(all(v > 0 for v in row['raw_ms']))
            self.assertEqual(statistics.median(row['raw_ms']), row['summary']['median_ms'])
            self.assertTrue(row['bitwise_equal_current_best'])
            self.assertEqual(row['mse_vs_current_best'], 0)
            self.assertEqual(row['mse_vs_o0'], row['current_best_mse_vs_o0'])
            self.assertEqual(row['executed_policy'], row['pair_scale_shared'])
            self.assertTrue(row['guard']['safe_all_pairs'])
            self.assertLessEqual(row['guard']['max_exponent_difference'], 5)
            self.assertGreater(row['guard']['range_check_wall_ms'], 0)
            self.assertIn('not Event', row['guard']['scope'])
        fn = runpy.run_path(str(ROOT / 'scripts/benchmark_roof_pair_scale_shared_probe.py'))['summary']
        result = fn(rows)
        self.assertEqual(result, read(run / 'summary.json')['records'])
        self.assertEqual([r['cv_failed_records'] for r in result], [36, 21])
        self.assertLess(result[1]['paired_speedup_ci95'][1], .82)
        env = read(run / 'environment.json')
        self.assertEqual(env['args']['warmup'], 50)
        self.assertEqual(env['args']['repeats'], 200)
        self.assertEqual(env['args']['rounds'], 3)

    def test_native_isa_control_and_unchanged_sentinel(self):
        fn = runpy.run_path(str(ROOT / 'scripts/audit_roof_pair_scale_shared_probe.py'))['audit']
        result = fn(P, P / 'best_controls.sass')
        self.assertTrue(result['passed'])
        self.assertTrue(all(result['control_encoded_sass_matches_best'].values()))
        self.assertEqual(len(result['entries']), 6)
        self.assertEqual(result['entries'], read(P / 'audit.json')['entries'])
        for r in result['entries']:
            self.assertTrue(r['native_u4_s4'] and r['native_s4_s4'] and r['all_copies_bypass_l1'])
            self.assertFalse(r['int8_mma'])
            if r['symbol'].endswith(('_o78', '_fallback')):
                self.assertTrue(r['encoded_identical_to_control'])

    def test_safety_resources_and_fallback_dispatch(self):
        for mode in ('preflight', 'memcheck', 'synccheck', 'racecheck'):
            d = read(E / f'runs/o378_roof_v51_{mode}/validation.json')
            self.assertTrue(d['passed'])
            self.assertEqual(len(d['checks']), 75)
            self.assertEqual({r['shape'][2] for r in d['checks']}, {128, 256, 384, 640, 4096})
            self.assertEqual(d['extension_sha256'], 'fe9c2b18d89787231bf988b704a29d2041edcfdad558c98acee3f5b2195b366f')
            self.assertEqual({r['pattern'] for r in d['checks']},
                             {'random', 'zero', 'extrema', 'zero_scale', 'guard13', 'fallback14', 'fallback15', 'subnormal'})
            subnormal = [r for r in d['checks'] if r['pattern'] == 'subnormal']
            self.assertEqual(len(subnormal), 5)
            self.assertTrue(all(r['policy'] == 1 and r['best_reference'] == 'fp64_semantic_float' for r in subnormal))
            for r in d['checks']:
                self.assertTrue(r['finite_fp32'])
                p = r['executed_policy']
                resources = r['probe_resources']
                self.assertEqual(resources['registers_per_thread'], [168, 255, 245][p])
                self.assertEqual(resources['active_blocks_per_sm'], [3, 2, 2][p])
                self.assertEqual(resources['shared_memory_bytes'], [50688, 68608, 67584][p])
                self.assertEqual(resources['local_size_bytes'], [16, 64, 0][p])
                if r['policy'] == 1 and r['pattern'].startswith('fallback') and r['shape'][2] > 128:
                    self.assertEqual(p, 2)
            if mode != 'preflight':
                marker = 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if mode == 'racecheck' else 'ERROR SUMMARY: 0 errors'
                self.assertIn(marker, (P / f'{mode}.log').read_text())

    def test_cached_guard_is_invalidated_after_inplace_mutation(self):
        for mode in ('preflight', 'memcheck', 'synccheck', 'racecheck'):
            checks = read(E / f'runs/o378_roof_v51_{mode}/validation.json')['guard_cache_checks']
            self.assertEqual(len(checks), 4)
            for row in checks:
                for key in ('safe_then_unsafe_fallback', 'invalid_after_cached_guard_rejected', 'output_matches_fp64'):
                    self.assertTrue(row[key])

    def test_ncu_identity_math_and_resource_model_recomputed(self):
        fn = runpy.run_path(str(ROOT / 'scripts/profile_roof_pair_scale_shared_probe.py'))['analyze_profile']
        saved = read(P / 'ncu_full/ncu_o3_analysis.json')
        cg = read(P / 'codegen.json')
        for source in saved['sources']:
            self.assertEqual(hashlib.sha256((E / source['file']).read_bytes()).hexdigest(), source['sha256'])
        computed = []
        for policy in (0, 1):
            prefix = P / f'ncu_full/ncu_o3_p{policy}'
            raw = Path(str(prefix) + '_raw.csv').read_text(encoding='utf-8-sig')
            sass = Path(str(prefix) + '_source_sass.csv').read_text(encoding='utf-8-sig')
            rows = [json.loads(s) for s in (E / f'runs/o378_roof_v51_ncu_o3_p{policy}/results.jsonl').read_text().splitlines()]
            resources = next(r['probe_resources'] for r in rows if r['pair_scale_shared'] == policy)
            r = fn(raw, sass, 'o3', policy, resources, cg['variants'][str(policy)]['entries']['adangel_roof_pair_scale_shared_o3'])
            self.assertEqual(r['opcodes']['IMMA'], 16777216)
            self.assertEqual(r['opcodes']['I2F'], [16777216, 8388608][policy])
            self.assertEqual(r['opcodes']['FFMA'], [16777216, 8388608][policy])
            self.assertEqual(r['dynamic_instructions'], [99319808, 114352128][policy])
            self.assertEqual(r['source_memory_work']['L2 Theoretical Sectors Local'], [3178496, 16711680][policy])
            computed.append(r)
        self.assertEqual(computed, saved['rows'])


if __name__ == '__main__':
    unittest.main()
