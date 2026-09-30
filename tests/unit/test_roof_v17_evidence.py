"""Preflight/synthetic/NCU scope only; do not imply completed24 acceptance."""
import collections
import json
from pathlib import Path
import re
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'docs/evidence/a100_o378_roof_v17'
P=E/'reports/o378_roof_v17'
ANALYZE=runpy.run_path(str(ROOT/'scripts/analyze_roof_scale_ncu.py'))['analyze']


class ReductionBudgetEvidence(unittest.TestCase):
    def test_new_entries_have_no_spill_and_old_code_is_preserved(self):
        audit=json.loads((P/'audit/audit.json').read_text())
        self.assertTrue(audit['passed'])
        self.assertEqual(len(audit['functions']),75)
        new=[f for f in audit['functions'] if re.search(r'ELi2[67]EE',f['symbol'])]
        self.assertEqual(len(new),6)
        for f in new:
            self.assertTrue(f['strict_passed'])
            self.assertEqual((f['instruction_counts']['LDL'],f['instruction_counts']['STL']),(0,0))
            self.assertIn('STACK:0',f['resource'])
            self.assertIn('LOCAL:0',f['resource'])
        for name,count in (('baseline_codegen.json',12),('candidate_codegen.json',69)):
            d=json.loads((P/name).read_text())
            self.assertTrue(d['passed'])
            self.assertEqual(len(d['unchanged']),count)

    def test_safety_and_balanced_synthetic_results(self):
        for name,count in (('budget_screen',432),('memcheck',144),('synccheck',144)):
            d=json.loads((E/f'runs/o378_roof_v17_{name}/validation.json').read_text())
            self.assertTrue(d['passed'])
            self.assertEqual(len(d['checks']),count)
        for name in ('memcheck','synccheck'):
            self.assertIn('ERROR SUMMARY: 0 errors',(P/f'{name}.log').read_text())
        self.assertIn('0 hazards displayed (0 errors, 0 warnings)',(P/'racecheck.log').read_text())
        rows=[json.loads(x) for x in (E/'runs/o378_roof_v17_budget_screen/results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),108)
        for v in ('o3','o7','o8'):
            for t in (-1,6,24,25,26,27):
                rs=[r for r in rows if r['variant']==v and r['tune']==t]
                self.assertEqual(collections.Counter(r['order_position'] for r in rs),dict.fromkeys(range(6),1))
        for r in rows:
            self.assertTrue(r['semantic_tolerance_passed'])
            self.assertEqual(len(r['raw_ms']),200)
            if r['tune'] in (26,27):
                self.assertEqual(r['kernel']['max_resident_blocks_per_sm'],1)
                self.assertEqual(r['kernel']['local_bytes_per_thread'],0)
                self.assertGreater(r['kernel']['registers_per_thread'],128)

    def test_zero_spill_is_not_sufficient_and_reuse_is_a_separate_strategy(self):
        for t,instructions,local in ((6,127238144,10485760),(23,116252672,4063232),
                                      (26,125272064,0),(27,138248192,0)):
            r=ANALYZE((P/f'ncu_o7_t{t}_raw.csv').read_text(),
                      (P/f'ncu_o7_t{t}_source_sass.csv').read_text(),t,'o7',True,True)
            self.assertEqual(r['dynamic_instructions'],instructions)
            self.assertEqual(r['source_memory_work']['L2 Theoretical Sectors Local'],local)
            if t==23:
                self.assertEqual(r['max_ctas_per_sm_from_launch_limits'],3)
                self.assertEqual(r['binding_modeled_resources'],['mma','i2f'])
                self.assertAlmostEqual(r['optimistic_fixed_work_lower_bound_ms'],.22034693984764905)
                # Attribute copy replay instead of falsely blaming ldmatrix.
                self.assertEqual(r['source_memory_work_by_opcode']['LDGSTS']['L1 Wavefronts Shared Excessive'],8388608)
                self.assertEqual(r['source_memory_work_by_opcode']['LDSM']['L1 Wavefronts Shared Excessive'],0)
            if t in (26,27):
                self.assertEqual(r['max_ctas_per_sm_from_launch_limits'],1)
                self.assertLess(r['issue_active_percent'],40)
                self.assertGreater(r['ncu_duration_ms'],.55)
                self.assertEqual(r['source_memory_work_omitted_zero_columns'],['L2 Theoretical Sectors Local'])


if __name__=='__main__': unittest.main()
