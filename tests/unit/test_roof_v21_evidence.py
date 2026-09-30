"""Reconcile tree negative results; do not relabel numerical drift as failure."""
import json
from pathlib import Path
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'docs/evidence/a100_o378_roof_v21'


def read(path): return json.loads((E/path).read_text())


class TreeEvidence(unittest.TestCase):
    def test_safety_and_unchanged_old_binary_functions(self):
        audit=read('reports/o378_roof_v21/audit/audit.json')
        self.assertTrue(audit['passed']); self.assertEqual(len(audit['functions']),102)
        for name,count in (('production_codegen',12),('candidate_codegen',93)):
            result=read(f'reports/o378_roof_v21/{name}.json')
            self.assertTrue(result['passed']); self.assertEqual(result['old_symbols'],count)
        self.assertEqual(len(read('runs/o378_roof_v21_tree_preflight/validation.json')['checks']),702)
        for tool in ('memcheck','synccheck'):
            self.assertIn('ERROR SUMMARY: 0 errors',(E/f'reports/o378_roof_v21/{tool}.log').read_text())
            self.assertEqual(len(read(f'runs/o378_roof_v21_{tool}/validation.json')['checks']),378)
        self.assertIn('0 hazards displayed (0 errors, 0 warnings)',(E/'reports/o378_roof_v21/racecheck.log').read_text())

    def test_all24_numeric_and_negative_performance(self):
        summary=read('runs/o378_roof_v21_tree_trace24/summary.json')
        self.assertTrue(summary['all_24_samples'] and summary['mse_regression_passed'])
        rows=[json.loads(x) for x in (E/'runs/o378_roof_v21_tree_trace24/results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),432)
        self.assertTrue(all(r['semantic_tolerance_passed'] and r['mse_regression_passed'] for r in rows))
        for t in (34,35,36):
            for ref in (22,23):
                data=read(f'reports/o378_roof_v21/trace24_t{t}_vs{ref}.json')
                for r in data['rows']:
                    if (r['variant']=='o3') != (ref==22): continue
                    self.assertEqual((r['samples'],r['rounds']),(24,1))
                    self.assertLess(r['paired_speedup_ci95'][1],1)
                    self.assertLess(r['median_candidate_mse_vs_production'],1e-12)
        smoke=read('runs/o378_roof_v21_tree_four_smoke/summary.json')
        self.assertFalse(smoke['all_24_samples']); self.assertTrue(smoke['all_four_modes_completed'])
        self.assertTrue(smoke['mse_regression_passed'])

    def test_ncu_tree_counts_and_rescaled_dram_units(self):
        analyze=runpy.run_path(str(ROOT/'scripts/analyze_roof_scale_ncu.py'))['analyze']
        expected={22:120840192,23:116252672,34:148570112,35:171433984,36:272498688}
        stored=read('reports/o378_roof_v21/ncu_analysis.json')
        for t,total in expected.items():
            p=E/'reports/o378_roof_v21'
            actual=analyze((p/f'ncu_o7_t{t}_raw.csv').read_text(),
                           (p/f'ncu_o7_t{t}_source_sass.csv').read_text(),t,'o7',True,True)
            self.assertEqual(actual['dynamic_instructions'],total)
            self.assertEqual(actual,next(r for r in stored['rows'] if r['tune']==t))
            if t in (34,35):
                self.assertGreater(actual['dram_write_bytes'],3e9)
                self.assertEqual(actual['binding_modeled_resources'],['dram_observed_bytes_at_spec_bw'])
            if t==36:
                self.assertEqual(actual['binding_modeled_resources'],['all_instruction_issue'])
            if t>=34: self.assertEqual(actual['opcodes'].get('FFMA',0),0)


if __name__=='__main__': unittest.main()
