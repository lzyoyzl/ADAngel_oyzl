"""Recompute v47 warp-aspect evidence without filtering timing outliers."""
import hashlib
import json
from pathlib import Path
import runpy
import statistics
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / 'docs/evidence/a100_o378_roof_v47'
P = E / 'reports/o378_roof_v47'
sys.path.insert(0, str(ROOT / 'scripts'))


def read(path):
    return json.loads(path.read_text())


class V47EvidenceTests(unittest.TestCase):
    def check_run(self, suffix, count, policies):
        run = E / f'runs/o378_roof_v47_{suffix}'
        rows = [json.loads(s) for s in (run / 'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), count)
        self.assertEqual(len({r['sample_id'] for r in rows}), 24)
        for row in rows:
            self.assertEqual(len(row['raw_ms']), 200)
            self.assertTrue(all(x > 0 for x in row['raw_ms']))
            self.assertEqual(statistics.median(row['raw_ms']), row['summary']['median_ms'])
            self.assertTrue(row['bitwise_equal_current_best'])
            self.assertEqual(row['mse_vs_current_best'], 0)
        aggregate = runpy.run_path(str(ROOT / 'scripts/benchmark_roof_warp_aspect_probe.py'))['summary']
        computed = aggregate(rows, policies)
        self.assertEqual(computed, read(run / 'summary.json')['records'])
        for row in computed:
            self.assertGreater(row['cv_failed_records'], 0)
        return computed

    def test_initial_screen_pairs_and_mse(self):
        result = self.check_run('trace24', 648, (0, 1, 2))
        for row in result:
            self.assertEqual(row['records'], 72)
            if row['warp_aspect']:
                self.assertLess(row['paired_speedup_ci95'][1],1)

    def test_same_entry_native_int4_and_control(self):
        audit = runpy.run_path(str(ROOT / 'scripts/audit_roof_warp_aspect_probe.py'))['audit']
        result = audit(P, P / 'best_controls.sass')
        self.assertTrue(result['passed'])
        self.assertTrue(all(result['control_encoded_sass_matches_best'].values()))
        self.assertEqual(len(result['entries']), 6)
        self.assertEqual(result['entries'], read(P / 'audit.json')['entries'])
        for row in result['entries']:
            self.assertTrue(row['native_u4_s4'] and row['native_s4_s4'])
            self.assertFalse(row['int8_mma'])
            self.assertTrue(row['all_copies_bypass_l1'])

    def test_safety_and_resources(self):
        for mode in ('preflight', 'racecheck', 'memcheck', 'synccheck'):
            validation = read(E / f'runs/o378_roof_v47_{mode}/validation.json')
            self.assertTrue(validation['passed'])
            self.assertEqual(len(validation['checks']), 180)
            self.assertEqual({r['shape'][2] for r in validation['checks']}, {128, 256, 384, 640, 4096})
            self.assertEqual(validation['extension_sha256'],
                'fe9c2b18d89787231bf988b704a29d2041edcfdad558c98acee3f5b2195b366f')
            for row in validation['checks']:
                self.assertTrue(row['finite_fp32'] and row['bitwise_equal_best'])
                resource = row['probe_resources']
                self.assertEqual(resource['cta_tile'], [64, 128, 128])
                self.assertEqual(resource['consumer_warp_layout'], [[2,2],[1,4],[4,1]][row['policy']])
                self.assertEqual(resource['accumulators_per_thread'], 64)
                self.assertEqual(resource['threads'], 128)
                self.assertEqual(resource['active_blocks_per_sm'], 3)
                self.assertEqual(resource['active_consumer_warps_per_sm'], 12)
                self.assertEqual(resource['registers_per_thread'], 168)
                self.assertEqual(resource['warp_aspect'], row['policy'])
                expected_local=(16,24,8) if row['variant']=='o3' else (8,0,16)
                self.assertEqual(resource['local_size_bytes'],expected_local[row['policy']])
            if mode != 'preflight':
                marker = ('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'
                          if mode == 'racecheck' else 'ERROR SUMMARY: 0 errors')
                self.assertIn(marker, (P / f'{mode}.log').read_text())

    def test_ncu_identity_and_recomputed_work(self):
        analyze = runpy.run_path(str(ROOT / 'scripts/profile_roof_warp_aspect_probe.py'))['analyze_profile']
        codegen = read(P / 'codegen.json')
        for directory, variant, candidate in (('ncu_1x4', 'o3', 1),
                                                ('ncu_1x4', 'o7', 1),
                                                ('ncu_4x1', 'o3', 2),
                                                ('ncu_4x1', 'o7', 2)):
            saved = read(P / f'{directory}/ncu_{variant}_analysis.json')
            for source in saved['sources']:
                self.assertEqual(hashlib.sha256((E / source['file']).read_bytes()).hexdigest(), source['sha256'])
            computed = []
            for policy in (0, candidate):
                prefix = P / f'{directory}/ncu_{variant}_p{policy}'
                raw = Path(str(prefix) + '_raw.csv').read_text(encoding='utf-8-sig')
                source = Path(str(prefix) + '_source_sass.csv').read_text(encoding='utf-8-sig')
                run = E / f'runs/o378_roof_v47_{directory}_{variant}_p{policy}'
                rows = [json.loads(s) for s in (run / 'results.jsonl').read_text().splitlines()]
                resources = next(r['probe_resources'] for r in rows if r['warp_aspect'] == policy)
                symbol = 'adangel_roof_warp_aspect_o3' if variant == 'o3' else 'adangel_roof_warp_aspect_o78'
                row = analyze(raw, source, variant, policy, resources,
                              codegen['variants'][str(policy)]['entries'][symbol])
                computed.append(row)
                for op in ('IMMA', 'I2F', 'FFMA'):
                    self.assertEqual(row['opcodes'][op], 16777216)
                self.assertEqual(row['opcodes']['LDSM'], 4194304 if policy==0 else 5242880)
                self.assertEqual(row['opcodes']['BAR'], 262144)
            self.assertEqual(computed, saved['rows'])
            self.assertGreater(computed[1]['source_memory_work']['L1 Wavefronts Shared'],
                               computed[0]['source_memory_work']['L1 Wavefronts Shared'])



if __name__ == '__main__':
    unittest.main()
