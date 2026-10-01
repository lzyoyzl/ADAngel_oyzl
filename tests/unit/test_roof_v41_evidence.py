"""Recompute producer-warp results and their evidence, including negative outcomes."""
import hashlib
import json
from pathlib import Path
import runpy
import statistics
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / 'docs/evidence/a100_o378_roof_v41'
P = E / 'reports/o378_roof_v41'
sys.path.insert(0, str(ROOT / 'scripts'))


def read(path):
    return json.loads(path.read_text())


class V41EvidenceTests(unittest.TestCase):
    def test_trace_pairing_and_unchanged_mse(self):
        run = E / 'runs/o378_roof_v41_trace24'
        rows = [json.loads(s) for s in (run / 'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 648)
        self.assertEqual(len({r['sample_id'] for r in rows}), 24)
        for r in rows:
            self.assertEqual(len(r['raw_ms']), 200)
            self.assertTrue(all(x > 0 for x in r['raw_ms']))
            self.assertEqual(statistics.median(r['raw_ms']), r['summary']['median_ms'])
            self.assertTrue(r['bitwise_equal_current_best'])
            self.assertEqual(r['mse_vs_current_best'], 0)
        aggregate = runpy.run_path(str(ROOT / 'scripts/benchmark_roof_producer_probe.py'))['summary']
        result = aggregate(rows)
        self.assertEqual(result, read(run / 'summary.json')['records'])
        for r in result:
            self.assertEqual(r['records'], 72)
            self.assertGreater(r['cv_failed_records'], 0)
            if r['producer_mode']:
                self.assertLess(r['paired_speedup_ci95'][1], 1)

    def test_native_isa_and_spill_tradeoff(self):
        audit = runpy.run_path(str(ROOT / 'scripts/audit_roof_producer_probe.py'))['audit']
        result = audit(P, P / 'best_controls.sass')
        self.assertTrue(result['passed'])
        self.assertTrue(all(result['control_encoded_sass_matches_best'].values()))
        self.assertEqual(len(result['entries']), 6)
        self.assertEqual(result['entries'], read(P / 'audit_portable.json')['entries'])
        for r in result['entries']:
            self.assertTrue(r['native_u4_s4'] and r['native_s4_s4'])
            self.assertFalse(r['int8_mma'])
            if r['producer_mode'] == 1:
                self.assertEqual(r['opcode_counts'].get('LDL', 0), 0)
                self.assertEqual(r['opcode_counts'].get('STL', 0), 0)
            elif r['producer_mode'] == 2:
                self.assertGreater(r['opcode_counts'].get('LDL', 0), 0)
                self.assertGreater(r['opcode_counts'].get('STL', 0), 0)
        self.assertEqual((P / 'producer_1_build.log').read_text().count(
            '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads'), 2)

    def test_preflight_safety_and_resources(self):
        for mode in ('preflight', 'racecheck', 'memcheck', 'synccheck'):
            validation = read(E / f'runs/o378_roof_v41_{mode}/validation.json')
            self.assertTrue(validation['passed'])
            self.assertEqual(len(validation['checks']), 180)
            self.assertEqual({r['shape'][2] for r in validation['checks']}, {128, 256, 384, 640, 4096})
            self.assertEqual(validation['extension_sha256'],
                'fe9c2b18d89787231bf988b704a29d2041edcfdad558c98acee3f5b2195b366f')
            for r in validation['checks']:
                self.assertTrue(r['finite_fp32'] and r['bitwise_equal_best'])
                resource = r['probe_resources']
                policy = r['policy']
                self.assertEqual(resource['cta_tile'], [64, 128, 128])
                self.assertEqual(resource['consumer_warp_layout'], [2, 2])
                self.assertEqual(resource['accumulators_per_thread'], 64)
                self.assertEqual(resource['threads'], 128 if policy == 0 else 160)
                self.assertEqual(resource['active_blocks_per_sm'], 2 if policy == 1 else 3)
                self.assertEqual(resource['active_consumer_warps_per_sm'], 8 if policy == 1 else 12)
                self.assertEqual(resource['registers_per_thread'], 128 if policy == 2 else 168)
                if policy == 1:
                    self.assertEqual(resource['local_size_bytes'], 0)
            if mode != 'preflight':
                marker = ('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'
                          if mode == 'racecheck' else 'ERROR SUMMARY: 0 errors')
                self.assertIn(marker, (P / f'{mode}.log').read_text())

    def test_ncu_identity_and_recomputed_bottleneck(self):
        analyze = runpy.run_path(str(ROOT / 'scripts/profile_roof_producer_probe.py'))['analyze_profile']
        for variant in ('o3', 'o7'):
            saved = read(P / f'ncu/ncu_{variant}_analysis.json')
            for source in saved['sources']:
                self.assertEqual(hashlib.sha256((E / source['file']).read_bytes()).hexdigest(), source['sha256'])
            computed = []
            for policy in (0, 1):
                prefix = P / f'ncu/ncu_{variant}_p{policy}'
                raw = Path(str(prefix) + '_raw.csv').read_text(encoding='utf-8-sig')
                source = Path(str(prefix) + '_source_sass.csv').read_text(encoding='utf-8-sig')
                run = E / f'runs/o378_roof_v41_ncu_{variant}_p{policy}'
                rows = [json.loads(s) for s in (run / 'results.jsonl').read_text().splitlines()]
                resources = next(r['probe_resources'] for r in rows if r['producer_mode'] == policy)
                row = analyze(raw, source, variant, policy, resources)
                computed.append(row)
                for op in ('IMMA', 'I2F', 'FFMA'):
                    self.assertEqual(row['opcodes'][op], 16777216)
            self.assertEqual(computed, saved['rows'])
            control, candidate = computed
            self.assertEqual(candidate['opcodes']['LDSM'], control['opcodes']['LDSM'])
            self.assertEqual(candidate['opcodes']['BAR'], control['opcodes']['BAR'] * 5 // 2)
            self.assertEqual(candidate['source_memory_work']['L2 Theoretical Sectors Local'], 0)
            self.assertLess(candidate['eligible_warps'], control['eligible_warps'])
            self.assertLess(candidate['issue_active_percent'], control['issue_active_percent'])
            self.assertGreater(candidate['pc_sampling']['reason_share_percent']['barrier'],
                               control['pc_sampling']['reason_share_percent']['barrier'])
            self.assertEqual(candidate['optimistic_fixed_work_lower_bound_ms'],
                             control['optimistic_fixed_work_lower_bound_ms'])


if __name__ == '__main__':
    unittest.main()
