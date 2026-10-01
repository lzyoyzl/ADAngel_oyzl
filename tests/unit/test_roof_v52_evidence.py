"""Recompute v52 fragment-reuse measurements and same-entry NCU evidence."""
import hashlib
import json
from pathlib import Path
import runpy
import statistics
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / 'docs/evidence/a100_o378_roof_v52'
P = E / 'reports/o378_roof_v52'
sys.path.insert(0, str(ROOT / 'scripts'))


def read(path):
    return json.loads(path.read_text())


class V52EvidenceTests(unittest.TestCase):
    def test_complete_timing_pairs_and_mse(self):
        run = E / 'runs/o378_roof_v52_trace24'
        rows = [json.loads(s) for s in (run / 'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 216)
        self.assertEqual(len({r['sample_id'] for r in rows}), 24)
        for r in rows:
            self.assertEqual(len(r['raw_ms']), 200)
            self.assertTrue(all(t > 0 for t in r['raw_ms']))
            self.assertEqual(statistics.median(r['raw_ms']), r['summary']['median_ms'])
            self.assertTrue(r['bitwise_equal_current_best'])
            self.assertEqual(r['mse_vs_current_best'], 0)
            self.assertEqual(r['mse_vs_o0'], r['current_best_mse_vs_o0'])
            self.assertEqual(r['executed_policy'], r['pair_fragment_reuse'])
            self.assertTrue(r['guard']['safe_all_pairs'])
            self.assertLessEqual(r['guard']['max_exponent_difference'], 5)
            self.assertGreater(r['guard']['range_check_wall_ms'], 0)
            self.assertIn('not Event', r['guard']['scope'])
        fn = runpy.run_path(str(ROOT / 'scripts/benchmark_roof_pair_fragment_reuse_probe.py'))['summary']
        computed = fn(rows)
        self.assertEqual(computed, read(run / 'summary.json')['records'])
        self.assertEqual([r['cv_failed_records'] for r in computed], [28, 18, 11])
        self.assertTrue(all(r['paired_speedup_ci95'][1] < .84 for r in computed[1:]))
        env = read(run / 'environment.json')
        self.assertEqual((env['args']['warmup'], env['args']['repeats'], env['args']['rounds']), (50, 200, 3))

    def test_same_entry_native_int4_and_exact_control(self):
        fn = runpy.run_path(str(ROOT / 'scripts/audit_roof_pair_fragment_reuse_probe.py'))['audit']
        result = fn(P, P / 'best_controls.sass')
        self.assertTrue(result['passed'])
        self.assertTrue(all(result['control_encoded_sass_matches_best'].values()))
        self.assertEqual(len(result['entries']), 9)
        self.assertEqual(result['entries'], read(P / 'audit.json')['entries'])
        for r in result['entries']:
            self.assertTrue(r['native_u4_s4'] and r['native_s4_s4'] and r['all_copies_bypass_l1'])
            self.assertFalse(r['int8_mma'])
            if r['symbol'].endswith(('_o78', '_fallback')):
                self.assertTrue(r['encoded_identical_to_control'])

    def test_safety_resources_and_fallback(self):
        for mode in ('preflight', 'memcheck', 'synccheck', 'racecheck'):
            d = read(E / f'runs/o378_roof_v52_{mode}/validation.json')
            self.assertTrue(d['passed'])
            self.assertEqual(len(d['checks']), 115)
            self.assertEqual({r['shape'][2] for r in d['checks']}, {128, 256, 384, 640, 4096})
            self.assertEqual(d['extension_sha256'], 'fe9c2b18d89787231bf988b704a29d2041edcfdad558c98acee3f5b2195b366f')
            self.assertEqual({r['pattern'] for r in d['checks']},
                             {'random', 'zero', 'extrema', 'zero_scale', 'guard13', 'fallback14', 'fallback15', 'subnormal'})
            subs = [r for r in d['checks'] if r['pattern'] == 'subnormal']
            self.assertEqual(len(subs), 10)
            self.assertTrue(all(r['policy'] in (1, 2) and r['best_reference'] == 'fp64_semantic_float' for r in subs))
            for r in d['checks']:
                self.assertTrue(r['finite_fp32'])
                p, resources = r['executed_policy'], r['probe_resources']
                self.assertEqual(resources['registers_per_thread'], [168, 255, 255, 245][p])
                self.assertEqual(resources['active_blocks_per_sm'], [3, 2, 2, 2][p])
                self.assertEqual(resources['shared_memory_bytes'], [50688, 68608, 68608, 67584][p])
                self.assertEqual(resources['local_size_bytes'], [16, 40, 16, 0][p])
                if r['policy'] in (1, 2) and r['pattern'].startswith('fallback') and r['shape'][2] > 128:
                    self.assertEqual(p, 3)
            if mode != 'preflight':
                marker = 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if mode == 'racecheck' else 'ERROR SUMMARY: 0 errors'
                self.assertIn(marker, (P / f'{mode}.log').read_text())

    def test_guard_cache_mutations(self):
        for mode in ('preflight', 'memcheck', 'synccheck', 'racecheck'):
            checks = read(E / f'runs/o378_roof_v52_{mode}/validation.json')['guard_cache_checks']
            self.assertEqual(len(checks), 8)
            for row in checks:
                for key in ('safe_then_unsafe_fallback', 'invalid_after_cached_guard_rejected', 'output_matches_fp64'):
                    self.assertTrue(row[key])

    def test_ncu_identity_and_resource_model(self):
        fn = runpy.run_path(str(ROOT / 'scripts/profile_roof_pair_fragment_reuse_probe.py'))['analyze_profile']
        cg = read(P / 'codegen.json')
        for tag, candidate in (('b', 1), ('ab', 2)):
            saved = read(P / f'ncu_{tag}/ncu_o3_analysis.json')
            for source in saved['sources']:
                self.assertEqual(hashlib.sha256((E / source['file']).read_bytes()).hexdigest(), source['sha256'])
            computed = []
            for p in (0, candidate):
                prefix = P / f'ncu_{tag}/ncu_o3_p{p}'
                raw = Path(str(prefix) + '_raw.csv').read_text(encoding='utf-8-sig')
                sass = Path(str(prefix) + '_source_sass.csv').read_text(encoding='utf-8-sig')
                rows = [json.loads(s) for s in (E / f'runs/o378_roof_v52_ncu_{tag}_o3_p{p}/results.jsonl').read_text().splitlines()]
                resources = next(r['probe_resources'] for r in rows if r['pair_fragment_reuse'] == p)
                r = fn(raw, sass, 'o3', p, resources, cg['variants'][str(p)]['entries']['adangel_roof_pair_fragment_reuse_o3'])
                self.assertEqual(r['opcodes']['IMMA'], 16777216)
                self.assertEqual(r['opcodes']['I2F'], 16777216 if p == 0 else 8388608)
                self.assertEqual(r['opcodes']['FFMA'], 16777216 if p == 0 else 8388608)
                self.assertEqual(r['opcodes']['LDSM'], [4194304, 4194304, 6291456][p])
                self.assertEqual(r['dynamic_instructions'], [99319808, 118743040, 119668736][p])
                self.assertEqual(r['source_memory_work']['L2 Theoretical Sectors Local'], [3178496, 10027008, 3342336][p])
                computed.append(r)
            self.assertEqual(computed, saved['rows'])


if __name__ == '__main__':
    unittest.main()
