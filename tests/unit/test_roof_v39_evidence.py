import json
from pathlib import Path
import runpy
import statistics
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
EVIDENCE=ROOT/'docs/evidence/a100_o378_roof_v39'
sys.path.insert(0,str(ROOT/'scripts'))


class V39EvidenceTests(unittest.TestCase):
    def test_full_trace_pairing_and_mse(self):
        run=EVIDENCE/'runs/o378_roof_v39_trace24'
        rows=[json.loads(s) for s in (run/'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),432)
        self.assertEqual(len({r['sample_id'] for r in rows}),24)
        for row in rows:
            self.assertEqual(len(row['raw_ms']),200)
            self.assertEqual(statistics.median(row['raw_ms']),row['summary']['median_ms'])
            self.assertTrue(row['bitwise_equal_current_best'])
            self.assertEqual(row['mse_vs_current_best'],0.0)
            self.assertTrue(all(x>0 for x in row['raw_ms']))
        aggregate=runpy.run_path(str(ROOT/'scripts/benchmark_roof_recompose_probe.py'))['summary']
        result=aggregate(rows,(0,1))
        self.assertEqual(result,json.loads((run/'summary.json').read_text())['records'])
        for r in result:
            self.assertLessEqual(r['paired_speedup_ci95'][0],1.0)
            self.assertGreaterEqual(r['paired_speedup_ci95'][1],1.0)
            self.assertGreater(r['cv_failed_records'],0)

    def test_reaudits_exact_controls_and_duplicate_candidate(self):
        directory=EVIDENCE/'reports/o378_roof_v39'
        audit=runpy.run_path(str(ROOT/'scripts/audit_roof_recompose_probe.py'))['audit']
        result=audit(directory,directory/'best_controls.sass')
        self.assertTrue(result['passed'])
        self.assertTrue(result['policy2_encoded_identical_to_policy1'])
        self.assertEqual(len(result['entries']),6)
        for r in result['entries']:
            self.assertTrue(r['opcode_counts_equal_control'])
            self.assertFalse(r['int8_mma'])
            self.assertEqual(r['opcode_counts']['I2F'],64)
            self.assertEqual(r['opcode_counts']['IMMA'],64)
        archived=json.loads((directory/'audit_portable.json').read_text())
        self.assertEqual(result['entries'],archived['entries'])

    def test_finite_validation_and_safety_scope(self):
        validation=json.loads((EVIDENCE/'runs/o378_roof_v39_preflight/validation.json').read_text())
        self.assertTrue(validation['passed'])
        self.assertEqual(len(validation['checks']),180)
        self.assertEqual({r['shape'][2] for r in validation['checks']},{128,256,384,640,4096})
        for mode in ('memcheck','synccheck'):
            self.assertIn('ERROR SUMMARY: 0 errors',
                (EVIDENCE/f'reports/o378_roof_v39/{mode}.log').read_text())
            run=EVIDENCE/f'runs/o378_roof_v39_{mode}'
            env=json.loads((run/'environment.json').read_text())
            self.assertEqual(env['args']['samples'],1)
            self.assertEqual(env['extension_sha256'],'fe9c2b18d89787231bf988b704a29d2041edcfdad558c98acee3f5b2195b366f')
            rows=[json.loads(s) for s in (run/'results.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),6)
            self.assertTrue(all(r['bitwise_equal_current_best'] for r in rows))


if __name__=='__main__': unittest.main()
