"""Recompute v40's negative geometry result; do not hide failed-CV records."""
import csv
import hashlib
import io
import json
from pathlib import Path
import runpy
import statistics
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / 'docs/evidence/a100_o378_roof_v40'
P = E / 'reports/o378_roof_v40'
sys.path.insert(0, str(ROOT / 'scripts'))


def read(path):
    return json.loads(path.read_text())


class V40EvidenceTests(unittest.TestCase):
    def test_full_trace_pairing_and_mse(self):
        run = E / 'runs/o378_roof_v40_trace24'
        rows = [json.loads(s) for s in (run / 'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 432)
        self.assertEqual(len({r['sample_id'] for r in rows}), 24)
        for r in rows:
            self.assertEqual(len(r['raw_ms']), 200)
            self.assertTrue(all(x > 0 for x in r['raw_ms']))
            self.assertEqual(statistics.median(r['raw_ms']), r['summary']['median_ms'])
            self.assertTrue(r['bitwise_equal_current_best'])
            self.assertEqual(r['mse_vs_current_best'], 0)
        aggregate = runpy.run_path(str(ROOT / 'scripts/benchmark_roof_warp_probe.py'))['summary']
        result = aggregate(rows)
        self.assertEqual(result, read(run / 'summary.json')['records'])
        for r in result:
            self.assertEqual(r['records'], 72)
            self.assertGreater(r['cv_failed_records'], 0)
            if r['warp_geometry'] == 1:
                self.assertLess(r['paired_speedup_ci95'][1], 1)

    def test_same_entry_isa_control_and_zero_spill_candidate(self):
        audit = runpy.run_path(str(ROOT / 'scripts/audit_roof_warp_probe.py'))['audit']
        result = audit(P, P / 'best_controls.sass')
        self.assertTrue(result['passed'])
        self.assertTrue(all(result['control_encoded_sass_matches_best'].values()))
        self.assertEqual(len(result['entries']), 4)
        self.assertEqual(result['entries'], read(P / 'audit_portable.json')['entries'])
        for r in result['entries']:
            self.assertTrue(r['native_u4_s4'] and r['native_s4_s4'])
            self.assertFalse(r['int8_mma'])
            if r['warp_geometry'] == 1:
                self.assertEqual(r['opcode_counts'].get('LDL', 0), 0)
                self.assertEqual(r['opcode_counts'].get('STL', 0), 0)
        self.assertEqual((P / 'warp_1_build.log').read_text().count(
            '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads'), 2)

    def test_validation_resources_and_finite_safety_scope(self):
        validation = read(E / 'runs/o378_roof_v40_preflight/validation.json')
        self.assertTrue(validation['passed'])
        self.assertEqual(len(validation['checks']), 120)
        self.assertEqual({r['shape'][2] for r in validation['checks']}, {128, 256, 384, 640, 4096})
        for r in validation['checks']:
            self.assertTrue(r['finite_fp32'] and r['bitwise_equal_best'])
            resource = r['probe_resources']
            self.assertEqual(resource['cta_tile'], [64, 128, 128])
            self.assertEqual(resource['threads'], 128 if r['policy'] == 0 else 256)
            self.assertEqual(resource['active_warps_per_sm'], 12 if r['policy'] == 0 else 16)
            if r['policy'] == 1:
                self.assertEqual(resource['local_size_bytes'], 0)
                self.assertEqual(resource['registers_per_thread'], 120 if r['variant'] == 'o3' else 128)
        for mode in ('memcheck', 'synccheck'):
            self.assertIn('ERROR SUMMARY: 0 errors', (P / f'{mode}.log').read_text())
            run = E / f'runs/o378_roof_v40_{mode}'
            env = read(run / 'environment.json')
            self.assertEqual(env['args']['samples'], 1)
            self.assertEqual(env['extension_sha256'],
                'fe9c2b18d89787231bf988b704a29d2041edcfdad558c98acee3f5b2195b366f')
            rows = [json.loads(s) for s in (run / 'results.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(r['bitwise_equal_current_best'] for r in rows))

    def test_ncu_identity_and_recomputed_work(self):
        analyze = runpy.run_path(str(ROOT / 'scripts/analyze_roof_scale_ncu.py'))['analyze']
        for variant in ('o3', 'o7'):
            saved = read(P / f'ncu/ncu_{variant}_analysis.json')
            for source in saved['sources']:
                self.assertEqual(hashlib.sha256((E / source['file']).read_bytes()).hexdigest(), source['sha256'])
            computed = []
            for geometry in (0, 1):
                prefix = P / f'ncu/ncu_{variant}_g{geometry}'
                raw = Path(str(prefix) + '_raw.csv').read_text(encoding='utf-8-sig')
                source = Path(str(prefix) + '_source_sass.csv').read_text(encoding='utf-8-sig')
                tune, symbol = (54, 'adangel_roof_warp_o3') if variant == 'o3' else (59, 'adangel_roof_warp_o78')
                row = analyze(raw, source, tune, variant, True, True, expected_symbol=symbol)
                block = int(list(csv.DictReader(io.StringIO(raw)))[1]['launch__block_size'])
                self.assertEqual(block, 128 if geometry == 0 else 256)
                row.update(threads=block, reference_math_tune=row.pop('tune'),
                           warp_geometry=geometry, warp_layout=[2, 2 if geometry == 0 else 4])
                computed.append(row)
                for op in ('IMMA', 'I2F', 'FFMA'):
                    self.assertEqual(row['opcodes'][op], 16777216)
            self.assertEqual(computed, saved['rows'])
            control, candidate = computed
            self.assertEqual(candidate['opcodes']['LDSM'], control['opcodes']['LDSM'] * 3 // 2)
            self.assertGreater(candidate['dynamic_instructions'], control['dynamic_instructions'])
            self.assertGreater(candidate['source_memory_work']['L1 Wavefronts Shared'],
                               control['source_memory_work']['L1 Wavefronts Shared'])
            self.assertGreater(candidate['mio_stall_per_issue'], control['mio_stall_per_issue'])
            self.assertGreater(candidate['optimistic_fixed_work_lower_bound_ms'],
                               control['optimistic_fixed_work_lower_bound_ms'])


if __name__ == '__main__':
    unittest.main()
