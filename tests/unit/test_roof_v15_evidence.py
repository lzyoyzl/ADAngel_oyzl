"""Archived evidence scope checks: synthetic/safety, not formal24 acceptance."""
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'docs/evidence/a100_o378_roof_v15'


class ReusePipelineEvidence(unittest.TestCase):
    def test_audit_and_exact_controls(self):
        report=E/'reports/o378_roof_v15'
        audit=json.loads((report/'audit/audit.json').read_text())
        self.assertTrue(audit['passed'])
        self.assertEqual(len(audit['functions']),63)
        for name,count in (('baseline_codegen.json',12),('candidate_codegen.json',57)):
            comp=json.loads((report/name).read_text())
            self.assertTrue(comp['passed'])
            self.assertEqual(len(comp['unchanged']),count)
        env=json.loads((E/'runs/o378_roof_v15_screen/environment.json').read_text())
        self.assertEqual(env['binary_sha256'],audit['binary_sha256'])

    def test_numerical_and_safety_scope(self):
        for name,count in (('screen',468),('memcheck',252),('synccheck',252)):
            validation=json.loads((E/f'runs/o378_roof_v15_{name}/validation.json').read_text())
            self.assertTrue(validation['passed'])
            self.assertEqual(len(validation['checks']),count)
            self.assertTrue(all(r['bitwise_equal'] and r['mse_vs_production']==0 for r in validation['checks']))
        for name in ('memcheck','synccheck'):
            self.assertIn('ERROR SUMMARY: 0 errors',(E/f'reports/o378_roof_v15/{name}.log').read_text())
        self.assertIn('0 hazards displayed (0 errors, 0 warnings)',(E/'reports/o378_roof_v15/racecheck.log').read_text())

    def test_screen_is_balanced_and_all_raw_values_retained(self):
        rows=[json.loads(x) for x in (E/'runs/o378_roof_v15_screen/results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),150)
        for v in ('o3','o7','o8'):
            for t in (-1,6,21,22,23):
                rs=[r for r in rows if r['variant']==v and r['tune']==t]
                self.assertEqual(len(rs),10)
                self.assertEqual(sorted(r['order_position'] for r in rs),sorted(list(range(5))*2))
        self.assertTrue(all(len(r['raw_ms'])==200 and r['bitwise_equal_production'] for r in rows))
        self.assertGreater(sum(r['summary']['cv_percent']>=3 for r in rows),0)


if __name__=='__main__': unittest.main()
