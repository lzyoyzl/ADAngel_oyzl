"""Recheck retained v59 results; these CPU checks are not new GPU measurements."""
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
EVIDENCE=ROOT/'docs/evidence/a100_o378_roof_v59'


class FactorEvidenceTests(unittest.TestCase):
    def test_raw_events_and_mse(self):
        from benchmark_a100_o1 import stats
        from benchmark_roof_fullk_integer_probe import summary
        for name,count in (('screen',4),('trace24',24)):
            d=EVIDENCE/f'runs/o378_roof_v59_{name}'
            rows=[json.loads(line) for line in (d/'results.jsonl').read_text().splitlines()]
            result=json.loads((d/'summary.json').read_text())
            self.assertEqual(len(rows),count*3*2)
            self.assertEqual(summary(rows),result['records'])
            self.assertFalse(result['production_default_changed'])
            self.assertTrue(result['factor_async'] and result['no_filtering'])
            for r in rows:
                self.assertEqual(len(r['raw_ms']),200)
                for key,value in stats(r['raw_ms']).items():
                    self.assertAlmostEqual(value,r['summary'][key],places=12)
                self.assertTrue(r['factor_async'])
                self.assertEqual(r['fullk_integer'],r['executed_policy'])
                self.assertTrue(r['guard']['safe'])
                self.assertLessEqual(r['guard']['max_abs_prefix_bound'],2**31-1)
                self.assertEqual(r['guard']['metadata_bytes'],33*4096*4)
                self.assertEqual(r['guard']['metadata_layout'],'int32_factors_and_anchor_codes')
                self.assertTrue(r['bitwise_equal_current_best'])
                self.assertEqual(r['mse_vs_current_best'],0)
                self.assertEqual(r['probe_resources']['registers_per_thread'],168)
                self.assertEqual(r['probe_resources']['active_blocks_per_sm'],3)
            env=json.loads((d/'environment.json').read_text())
            self.assertEqual(env['extension_sha256'],
                '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462')

    def test_exact_control_and_isa(self):
        from compare_a100_codegen import compare
        from probe_roof_fullk_integer_codegen import static_entries,PATTERN
        from probe_roof_factor_async_codegen import generated_header
        d=EVIDENCE/'reports/o378_roof_v59'
        previous=ROOT/'docs/evidence/a100_o378_roof_v55b/reports/o378_roof_v55b/fullk_integer_0.sass'
        result=json.loads((d/'codegen.json').read_text())
        for i in (0,1):
            sass=(d/f'factor_async_{i}.sass').read_text()
            self.assertEqual(static_entries(sass),result['variants'][str(i)]['entries'])
            pattern=PATTERN if i==0 else r'^adangel_roof_fullk_integer_o78$'
            self.assertTrue(compare(previous.read_text(),sass,pattern)['passed'])
            for e in static_entries(sass).values():
                self.assertTrue(e['native_u4_s4'] and e['native_s4_s4'] and e['all_copies_bypass_l1'])
                self.assertFalse(e['int8_mma'])
        self.assertEqual((d/'o3_factor_async_generated.cuh').read_text(),
                         generated_header((ROOT/'csrc/sm80/o3_fullk_integer_probe.cuh').read_text()))

    def test_safety_and_metadata_checks(self):
        for name in ('validation','memcheck','synccheck','racecheck'):
            p=EVIDENCE/f'runs/o378_roof_v59_{name}/validation.json'
            result=json.loads(p.read_text())
            self.assertTrue(result['passed'] and result['metadata_values_checked'])
            self.assertEqual(result['count'],36)
            self.assertEqual(len(result['cache_checks']),2)
            self.assertEqual(max(r['shape'][0] for r in result['checks']),128)
            self.assertEqual(max(r['shape'][1] for r in result['checks']),256)
            if name!='validation':
                log=(EVIDENCE/f'reports/o378_roof_v59_{name}.log').read_text()
                self.assertIn('0 errors',log)
                if name=='racecheck': self.assertIn('0 warnings',log)

    def test_missing_results_not_accepted(self):
        from benchmark_roof_fullk_integer_probe import summary
        rows=[json.loads(line) for line in
              (EVIDENCE/'runs/o378_roof_v59_screen/results.jsonl').read_text().splitlines()]
        for bad in (rows[:-1],rows+[rows[0]],
                    [{**rows[0],'output_close_current_best':False}]+rows[1:]):
            with self.assertRaises(ValueError): summary(bad)


if __name__=='__main__': unittest.main()
