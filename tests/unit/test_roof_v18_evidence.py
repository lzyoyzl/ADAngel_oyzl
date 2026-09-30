"""Keep the negative residency result reproducible, without promoting a smoke test."""
import collections
import json
from pathlib import Path
import re
import runpy
import statistics
import unittest

ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'docs/evidence/a100_o378_roof_v18'
P=E/'reports/o378_roof_v18'
ANALYZE=runpy.run_path(str(ROOT/'scripts/analyze_roof_scale_ncu.py'))['analyze']


class ResidencyBudgetEvidence(unittest.TestCase):
    def test_audit_and_safety_with_explicit_spills(self):
        audit=json.loads((P/'audit/audit.json').read_text())
        self.assertTrue(audit['passed'])
        self.assertEqual(len(audit['functions']),78)
        new=[f for f in audit['functions'] if re.search(r'ELi28EE',f['symbol'])]
        self.assertEqual(len(new),3)
        for f in new:
            self.assertIn('REG:128',f['resource'])
            self.assertGreater(f['instruction_counts']['LDL'],0)
            self.assertGreater(f['instruction_counts']['STL'],0)
            self.assertNotIn('STACK:0',f['resource'])
        for name,count in (('baseline_codegen',12),('candidate_codegen',75)):
            r=json.loads((P/(name+'.json')).read_text())
            self.assertTrue(r['passed'])
            self.assertEqual(len(r['unchanged']),count)
        for suffix,count in (('budget_screen',522),('memcheck',126),('synccheck',126)):
            r=json.loads((E/f'runs/o378_roof_v18_{suffix}/validation.json').read_text())
            self.assertTrue(r['passed'])
            self.assertEqual(len(r['checks']),count)
        for name in ('memcheck','synccheck'):
            self.assertIn('ERROR SUMMARY: 0 errors',(P/(name+'.log')).read_text())
        self.assertIn('0 hazards displayed (0 errors, 0 warnings)',(P/'racecheck.log').read_text())

    def test_balanced_screen_and_real_smoke_are_not_full_trace(self):
        d=E/'runs/o378_roof_v18_budget_screen'
        rows=[json.loads(x) for x in (d/'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),75)
        for v in ('o3','o7','o8'):
            for t in (-1,6,22,23,28):
                rs=[r for r in rows if (r['variant'],r['tune'])==(v,t)]
                self.assertEqual(collections.Counter(r['order_position'] for r in rs),dict.fromkeys(range(5),1))
                for r in rs:
                    self.assertTrue(r['bitwise_equal_production'])
                    self.assertEqual(len(r['raw_ms']),200)
                    if t==28:
                        self.assertEqual(r['kernel']['max_resident_blocks_per_sm'],4)
            old=[r['summary']['median_ms'] for r in rows if (r['variant'],r['tune'])==(v,22)]
            new=[r['summary']['median_ms'] for r in rows if (r['variant'],r['tune'])==(v,28)]
            self.assertGreater(statistics.median(new),statistics.median(old)*1.3)
        d=E/'runs/o378_roof_v18_budget_four_smoke'
        s=json.loads((d/'summary.json').read_text())
        self.assertTrue(s['correctness_passed'] and s['all_four_modes_completed'])
        self.assertFalse(s['all_24_samples'])
        rs=[json.loads(x) for x in (d/'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rs),108)
        for r in rs:
            self.assertTrue(r['bitwise_equal_production'])
            self.assertEqual(r['mse_vs_production'],0)
        for v in ('o3','o7','o8'):
            self.assertEqual(len({r['mse_vs_paired_fp16'] for r in rs if r['variant']==v}),1)

    def test_ncu_recomputes_larger_capacity_cost(self):
        r=ANALYZE((P/'ncu_o7_t28_raw.csv').read_text(),
                  (P/'ncu_o7_t28_source_sass.csv').read_text(),28,'o7',True,True)
        self.assertEqual(r['dynamic_instructions'],138256384)
        self.assertEqual(r['source_memory_work']['L2 Theoretical Sectors Local'],61079552)
        self.assertEqual(r['source_memory_work_by_opcode']['LDGSTS']['L1 Wavefronts Shared Excessive'],8388608)
        self.assertEqual(r['source_memory_work_by_opcode']['LDSM']['L1 Wavefronts Shared Excessive'],0)
        self.assertEqual(r['max_ctas_per_sm_from_launch_limits'],4)
        self.assertEqual(r['binding_modeled_resources'],['l1tex_data_wavefront_capacity'])
        self.assertGreater(r['optimistic_fixed_work_lower_bound_ms'],.32)
        self.assertLess(r['issue_active_percent'],37)


if __name__=='__main__': unittest.main()
