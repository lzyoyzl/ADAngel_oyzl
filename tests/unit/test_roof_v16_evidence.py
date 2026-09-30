"""Reconcile archived opt-in rounding, paired timing, safety and NCU evidence."""
import collections
import json
from pathlib import Path
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'docs/evidence/a100_o378_roof_v16'
P=E/'reports/o378_roof_v16'
ANALYZE=runpy.run_path(str(ROOT/'scripts/analyze_roof_scale_ncu.py'))['analyze']
COMPARE=runpy.run_path(str(ROOT/'scripts/compare_roof_trace_candidates.py'))['compare']


class ReductionEvidence(unittest.TestCase):
    def test_audit_controls_and_safety(self):
        audit=json.loads((P/'audit/audit.json').read_text())
        self.assertTrue(audit['passed'])
        self.assertEqual(len(audit['functions']),69)
        for name,count in (('baseline_codegen.json',12),('candidate_codegen.json',63)):
            result=json.loads((P/name).read_text())
            self.assertTrue(result['passed'])
            self.assertEqual(len(result['unchanged']),count)
        for suffix,count in (('reduction_screen',288),('memcheck',144),('synccheck',144)):
            validation=json.loads((E/f'runs/o378_roof_v16_{suffix}/validation.json').read_text())
            self.assertTrue(validation['passed'])
            self.assertEqual(len(validation['checks']),count)
        for name in ('memcheck','synccheck'):
            self.assertIn('ERROR SUMMARY: 0 errors',(P/f'{name}.log').read_text())
        self.assertIn('0 hazards displayed (0 errors, 0 warnings)',(P/'racecheck.log').read_text())

    def test_real_trace_recomputes_mse_without_bitwise_claim(self):
        d=E/'runs/o378_roof_v16_reduction_trace24'
        env=json.loads((d/'environment.json').read_text())
        self.assertTrue(env['args']['allow_reassociation'])
        self.assertEqual(env['binary_sha256'],json.loads((P/'audit/audit.json').read_text())['binary_sha256'])
        rows=[json.loads(x) for x in (d/'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),1152)
        for v in ('o3','o7','o8'):
            for t in (-1,6,24,25):
                group=[r for r in rows if r['variant']==v and r['tune']==t]
                self.assertEqual(collections.Counter(r['order_position'] for r in group),dict.fromkeys(range(4),24))
        for row in rows:
            self.assertEqual(len(row['raw_ms']),200)
            self.assertTrue(row['semantic_tolerance_passed'] and row['mse_regression_passed'])
            if row['tune'] in (24,25):
                self.assertTrue(row['fp32_reassociated'])
                # Reassociation need not change every possible input; exact
                # small/power-of-two sums can still be bitwise identical.
                self.assertGreaterEqual(row['mse_vs_production'],0)
                if row['bitwise_equal_production']:
                    self.assertEqual(row['mse_vs_production'],0)
                for key,value in row['baseline_mse'].items():
                    self.assertLessEqual(abs(row[key]-value),1e-12+1e-5*abs(value))
            else:
                self.assertTrue(row['bitwise_equal_production'])
        self.assertTrue(any(not r['bitwise_equal_production'] for r in rows if r['tune']==24))
        self.assertTrue(any(not r['bitwise_equal_production'] for r in rows if r['tune']==25))
        for t in (24,25):
            comparison=COMPARE(rows,6,t,24,4,['o3','o7','o8'],['compute_only'],True)
            for result in comparison:
                self.assertLess(result['paired_speedup_ci95'][1],1)
                self.assertGreater(result['cv'][str(t)]['any_stage_failed'],0)
        smoke=E/'runs/o378_roof_v16_reduction_four_smoke'
        summary=json.loads((smoke/'summary.json').read_text())
        self.assertTrue(summary['all_four_modes_completed'] and summary['mse_regression_passed'])
        self.assertFalse(summary['all_24_samples'])
        self.assertEqual(len((smoke/'results.jsonl').read_text().splitlines()),192)

    def test_ncu_work_is_not_reduced_by_shorter_dependency_chains(self):
        for tune,total,local,adds in ((6,127238144,10485760,0),(24,137019392,31064064,524288),
                                     (25,168574976,82575360,1572864)):
            raw=(P/f'ncu_o7_t{tune}_raw.csv').read_text()
            sass=(P/f'ncu_o7_t{tune}_source_sass.csv').read_text()
            result=ANALYZE(raw,sass,tune,'o7',True,True)
            self.assertEqual(result['dynamic_instructions'],total)
            self.assertEqual(result['source_memory_work']['L2 Theoretical Sectors Local'],local)
            self.assertEqual(result['source_memory_work']['L1 Wavefronts Shared Excessive'],0)
            self.assertEqual(result['opcodes'].get('FADD',0),adds)
            for opcode in ('IMMA','I2F','FFMA','FMUL'):
                self.assertEqual(result['opcodes'][opcode],16777216)
            bounds=result['resource_service_lower_bounds_ms_at_1410']
            self.assertAlmostEqual(bounds['fp32_scale_and_accumulate_subset'],
                (2*16777216+adds)*32/(108*64*1410000))
            self.assertEqual(result['binding_modeled_resources'],['l1tex_data_wavefront_capacity'])
            if tune in (24,25):
                with self.assertRaisesRegex(ValueError,'final FP32 additions'):
                    ANALYZE(raw,sass.replace('FADD','FMOV'),tune,'o7')


if __name__=='__main__': unittest.main()
