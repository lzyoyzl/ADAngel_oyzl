"""Reconcile the negative M32 experiment without re-running GPU kernels."""
import json
from pathlib import Path
import runpy
import unittest

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'docs/evidence/a100_o378_roof_v32'
REPORTS=ROOT/'reports/o378_roof_v32'
BINARY='8fefd9b69edd574e3d6004754153e5c0d13c6499c18141f4173b13797648c0d0'


@unittest.skipUnless((REPORTS/'audit/audit.json').exists(),'completed archive not present')
class M32EvidenceTests(unittest.TestCase):
    def load(self,path): return json.loads((ROOT/path).read_text())

    def test_native_int4_zero_spill_and_codegen_disclosure(self):
        x=self.load('reports/o378_roof_v32/audit/audit.json')
        self.assertTrue(x['passed']);self.assertEqual(x['binary_sha256'],BINARY)
        self.assertEqual(len(x['functions']),146)
        for tune in (57,58):
            f=next(f for f in x['functions'] if f'Li{tune}E' in f['symbol'])
            self.assertTrue(f['strict_passed']);self.assertTrue(all(f['checks'].values()))
            self.assertIn('REG:128',f['resource'])
        for name,count in (('production',12),('candidate',144)):
            x=self.load(f'reports/o378_roof_v32/{name}_codegen.json')
            self.assertTrue(x['passed']);self.assertEqual(len(x['unchanged']),count)
        x=self.load('reports/o378_roof_v32/all_sm80_codegen.json')
        self.assertEqual(len(x['changed']),4)
        self.assertTrue(all('mixed_binary' in s for s in x['changed']))

    def test_geometry_guards_and_finite_safety(self):
        for name in ('geometry','geometry_memcheck','geometry_synccheck'):
            x=self.load(f'reports/o378_roof_v32/{name}.json')
            self.assertTrue(x['passed']);self.assertEqual(len(x['checks']),64)
            self.assertEqual(x['binary_sha256'],BINARY)
            for r in x['checks']:
                self.assertTrue(r['bitwise_equal_padded_control'])
                self.assertTrue(r['semantic_tolerance_passed'] and r['nondefault_stream'])
        for name in ('memcheck','synccheck'):
            self.assertIn('ERROR SUMMARY: 0 errors',(REPORTS/f'{name}.log').read_text())
        self.assertIn('0 hazards displayed (0 errors, 0 warnings)',(REPORTS/'racecheck.log').read_text())
        x=self.load('runs/o378_roof_v32_preflight/validation.json')
        self.assertTrue(x['passed']);self.assertEqual(len(x['checks']),384)
        self.assertTrue(all(r['bitwise_equal_production'] for r in x['checks']))
        x=self.load('reports/o378_roof_v32/guard_checks.json')
        self.assertEqual(len(x['checks']),26);self.assertTrue(all(r['rejected'] for r in x['checks']))
        self.assertTrue(self.load('runs/o378_roof_v32_mixed_regression/validation.json')['passed'])

    def test_real_trace_negative_result_and_mse_no_filtering(self):
        rows=[json.loads(l) for l in (ROOT/'runs/o378_roof_v32_trace24/results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),240)
        compare=runpy.run_path(str(REPO/'scripts/compare_roof_trace_candidates.py'))['compare']
        for r in rows:
            self.assertTrue(r['bitwise_equal_production']);self.assertEqual(r['mse_vs_production'],0)
            self.assertEqual(len(r['raw_ms']),200)
        for tune in (57,58):
            x=self.load(f'reports/o378_roof_v32/trace24_{tune}_vs56.json')
            self.assertEqual(x['source_binary_sha256'],BINARY)
            result=compare(rows,56,tune,24,1,['o7','o8'],['compute_only'])
            self.assertEqual(result,x['rows'])
            for r in result:
                self.assertLess(r['paired_speedup_ci95'][1],1)
                self.assertGreater(r['cv'][str(tune)]['selected_stage_failed'],0)

    def test_ncu_accounts_for_increased_work(self):
        analyze=runpy.run_path(str(REPO/'scripts/analyze_roof_scale_ncu.py'))['analyze']
        for variant in ('o7','o8'):
            rows=self.load(f'reports/o378_roof_v32/ncu_{variant}_analysis.json')['rows']
            indexed={r['tune']:r for r in rows}
            for r in rows:
                stem=f'ncu_{variant}_t{r["tune"]}'
                got=analyze((REPORTS/f'{stem}_raw.csv').read_text(),
                            (REPORTS/f'{stem}_source_sass.csv').read_text(),r['tune'],variant,True,True)
                self.assertEqual(got,r)
                for op in ('IMMA','I2F','FMUL','FFMA'): self.assertEqual(r['opcodes'][op],16777216)
            for tune in (57,58):
                self.assertEqual(indexed[tune]['source_memory_work']['L2 Theoretical Sectors Local'],0)
                self.assertEqual(indexed[tune]['opcodes']['LDSM']/indexed[56]['opcodes']['LDSM'],1.5)
                self.assertGreater(indexed[tune]['dynamic_instructions'],indexed[56]['dynamic_instructions'])
                self.assertGreater(indexed[tune]['optimistic_fixed_work_lower_bound_ms'],indexed[56]['optimistic_fixed_work_lower_bound_ms'])


if __name__=='__main__': unittest.main()
