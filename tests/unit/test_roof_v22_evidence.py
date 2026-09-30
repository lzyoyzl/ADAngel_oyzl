"""Recorded static-tree results: faster than36, still slower than22/23."""
import json
from pathlib import Path
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'docs/evidence/a100_o378_roof_v22'
R=E/'reports/o378_roof_v22'


def read(path): return json.loads(path.read_text())


class StaticTreeEvidence(unittest.TestCase):
    def test_audit_and_safety_coverage(self):
        audit=read(R/'audit/audit.json')
        self.assertTrue(audit['passed'])
        self.assertEqual(len(audit['functions']),108)
        for name,count in (('production_codegen',12),('candidate_codegen',102)):
            result=read(R/f'{name}.json')
            self.assertTrue(result['passed'])
            self.assertEqual(result['old_symbols'],count)
        self.assertEqual(len(read(E/'runs/o378_roof_v22_preflight/validation.json')['checks']),702)
        for tool in ('memcheck','synccheck'):
            self.assertIn('ERROR SUMMARY: 0 errors',(R/f'{tool}.log').read_text())
            self.assertEqual(len(read(E/f'runs/o378_roof_v22_{tool}/validation.json')['checks']),252)
        self.assertIn('0 hazards displayed (0 errors, 0 warnings)',(R/'racecheck.log').read_text())

    def test_all24_mse_and_performance_decision(self):
        summary=read(E/'runs/o378_roof_v22_trace24/summary.json')
        self.assertTrue(summary['all_24_samples'] and summary['mse_regression_passed'])
        rows=[json.loads(x) for x in (E/'runs/o378_roof_v22_trace24/results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),432)
        self.assertTrue(all(r['semantic_tolerance_passed'] and r['mse_regression_passed'] for r in rows))
        self.assertTrue(all(r['bitwise_equal_dynamic_tree'] for r in rows if r['tune']==37))
        for t in (37,38):
            for ref in (22,23,36):
                for r in read(R/f'trace24_t{t}_vs{ref}.json')['rows']:
                    if ref!=36 and ((r['variant']=='o3') != (ref==22)): continue
                    self.assertEqual((r['samples'],r['rounds']),(24,1))
                    if ref==36:
                        self.assertGreater(r['paired_speedup_ci95'][0],1)
                    else:
                        self.assertLess(r['paired_speedup_ci95'][1],1)
                    self.assertLess(r['median_candidate_mse_vs_production'],1e-12)
        smoke=read(E/'runs/o378_roof_v22_four_smoke/summary.json')
        self.assertFalse(smoke['all_24_samples'])
        self.assertTrue(smoke['all_four_modes_completed'] and smoke['mse_regression_passed'])

    def test_ncu_zero_local_does_not_imply_low_register_usage(self):
        analyze=runpy.run_path(str(ROOT/'scripts/analyze_roof_scale_ncu.py'))['analyze']
        totals={22:120840192,23:116252672,36:272498688,37:134266880,38:131104768}
        stored=read(R/'ncu_analysis.json')
        for t,total in totals.items():
            actual=analyze((R/f'ncu_o7_t{t}_raw.csv').read_text(),
                           (R/f'ncu_o7_t{t}_source_sass.csv').read_text(),t,'o7',True,True)
            self.assertEqual(actual,next(r for r in stored['rows'] if r['tune']==t))
            self.assertEqual(actual['dynamic_instructions'],total)
            if t in (37,38):
                self.assertEqual(actual['opcodes']['BRA'],532480)
                self.assertEqual(actual['opcodes'].get('FFMA',0),0)
                self.assertEqual(actual['registers_per_thread'],255)
                self.assertEqual(actual['max_ctas_per_sm_from_launch_limits'],2)
            if t==38:
                self.assertEqual(actual['source_memory_work']['L2 Theoretical Sectors Local'],0)
                self.assertEqual(actual['opcodes'].get('LDL',0)+actual['opcodes'].get('STL',0),0)


if __name__=='__main__': unittest.main()
