"""Recompute v42 prefetch-position evidence without filtering timing outliers."""
import hashlib
import json
from pathlib import Path
import runpy
import statistics
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / 'docs/evidence/a100_o378_roof_v42'
P = E / 'reports/o378_roof_v42'
sys.path.insert(0, str(ROOT / 'scripts'))


def read(path):
    return json.loads(path.read_text())


class V42EvidenceTests(unittest.TestCase):
    def test_trace_pairing_and_unchanged_mse(self):
        run = E / 'runs/o378_roof_v42_trace24'
        rows = [json.loads(s) for s in (run / 'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 648)
        self.assertEqual(len({r['sample_id'] for r in rows}), 24)
        for r in rows:
            self.assertEqual(len(r['raw_ms']), 200)
            self.assertTrue(all(x > 0 for x in r['raw_ms']))
            self.assertEqual(statistics.median(r['raw_ms']), r['summary']['median_ms'])
            self.assertTrue(r['bitwise_equal_current_best'])
            self.assertEqual(r['mse_vs_current_best'], 0)
        aggregate = runpy.run_path(str(ROOT / 'scripts/benchmark_roof_prefetch_probe.py'))['summary']
        result = aggregate(rows)
        self.assertEqual(result, read(run / 'summary.json')['records'])
        for r in result:
            self.assertEqual(r['records'], 72)
            self.assertGreater(r['cv_failed_records'], 0)
            if r['prefetch_position']:
                self.assertLess(r['paired_speedup_ci95'][1], 1)

    def test_same_entry_native_int4_and_exact_control(self):
        audit = runpy.run_path(str(ROOT / 'scripts/audit_roof_prefetch_probe.py'))['audit']
        result = audit(P, P / 'best_controls.sass')
        self.assertTrue(result['passed'])
        self.assertTrue(all(result['control_encoded_sass_matches_best'].values()))
        self.assertEqual(len(result['entries']), 6)
        self.assertEqual(result['entries'], read(P / 'audit.json')['entries'])
        for r in result['entries']:
            self.assertTrue(r['native_u4_s4'] and r['native_s4_s4'])
            self.assertFalse(r['int8_mma'])
            self.assertTrue(r['all_copies_bypass_l1'])

    def test_preflight_safety_and_fixed_geometry(self):
        for mode in ('preflight', 'racecheck', 'memcheck', 'synccheck'):
            validation = read(E / f'runs/o378_roof_v42_{mode}/validation.json')
            self.assertTrue(validation['passed'])
            self.assertEqual(len(validation['checks']), 180)
            self.assertEqual({r['shape'][2] for r in validation['checks']}, {128, 256, 384, 640, 4096})
            self.assertEqual(validation['extension_sha256'],
                'fe9c2b18d89787231bf988b704a29d2041edcfdad558c98acee3f5b2195b366f')
            for r in validation['checks']:
                self.assertTrue(r['finite_fp32'] and r['bitwise_equal_best'])
                resource = r['probe_resources']
                self.assertEqual(resource['cta_tile'], [64, 128, 128])
                self.assertEqual(resource['consumer_warp_layout'], [2, 2])
                self.assertEqual(resource['accumulators_per_thread'], 64)
                self.assertEqual(resource['threads'], 128)
                self.assertEqual(resource['active_blocks_per_sm'], 3)
                self.assertEqual(resource['active_consumer_warps_per_sm'], 12)
                self.assertEqual(resource['registers_per_thread'], 168)
                self.assertEqual(resource['prefetch_position'], r['policy'])
            if mode != 'preflight':
                marker = ('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'
                          if mode == 'racecheck' else 'ERROR SUMMARY: 0 errors')
                self.assertIn(marker, (P / f'{mode}.log').read_text())

    def test_ncu_static_identity_and_recomputed_math(self):
        analyze = runpy.run_path(str(ROOT / 'scripts/profile_roof_prefetch_probe.py'))['analyze_profile']
        codegen = read(P / 'codegen.json')
        for candidate in (1, 2):
            for variant in ('o3', 'o7'):
                saved = read(P / f'ncu_pos{candidate}/ncu_{variant}_analysis.json')
                for source in saved['sources']:
                    self.assertEqual(hashlib.sha256((E / source['file']).read_bytes()).hexdigest(), source['sha256'])
                computed = []
                for policy in (0, candidate):
                    prefix = P / f'ncu_pos{candidate}/ncu_{variant}_p{policy}'
                    raw = Path(str(prefix) + '_raw.csv').read_text(encoding='utf-8-sig')
                    source = Path(str(prefix) + '_source_sass.csv').read_text(encoding='utf-8-sig')
                    run = E / f'runs/o378_roof_v42_ncu_pos{candidate}_{variant}_p{policy}'
                    rows = [json.loads(s) for s in (run / 'results.jsonl').read_text().splitlines()]
                    resources = next(r['probe_resources'] for r in rows if r['prefetch_position'] == policy)
                    symbol = 'adangel_roof_prefetch_o3' if variant == 'o3' else 'adangel_roof_prefetch_o78'
                    row = analyze(raw, source, variant, policy, resources,
                                  codegen['variants'][str(policy)]['entries'][symbol])
                    computed.append(row)
                    for op in ('IMMA', 'I2F', 'FFMA'):
                        self.assertEqual(row['opcodes'][op], 16777216)
                    self.assertEqual(row['opcodes']['LDSM'], 4194304)
                    self.assertEqual(row['opcodes']['BAR'], 262144)
                self.assertEqual(computed, saved['rows'])


if __name__ == '__main__':
    unittest.main()
