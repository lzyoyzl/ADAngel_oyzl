from pathlib import Path
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[2]
ANALYZE=runpy.run_path(str(ROOT/'scripts/analyze_roof_scale_ncu.py'))['analyze']
EVIDENCE=ROOT/'docs/evidence/a100_o378_roof_v4/reports/o378_roof_v4'


class RoofScaleNcuTests(unittest.TestCase):
    def test_counts_reconcile_with_raw(self):
        for tune,total in ((6,127238144),(11,135364608)):
            raw=(EVIDENCE/f'ncu_o7_t{tune}_raw.csv').read_text()
            sass=(EVIDENCE/f'ncu_o7_t{tune}_source_sass.csv').read_text()
            result=ANALYZE(raw,sass,tune)
            self.assertEqual(result['dynamic_instructions'],total)
            self.assertEqual(result['opcodes']['I2F'],16777216)
            self.assertGreater(result['l1_service_ms_at_1410'],0.27)
            self.assertLess(result['l1_service_ms_at_1410'],0.30)
            with self.assertRaises(ValueError):
                ANALYZE(raw,sass,12)


if __name__=='__main__': unittest.main()
