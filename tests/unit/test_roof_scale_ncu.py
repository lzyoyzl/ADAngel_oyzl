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
            self.assertEqual(result['source_memory_work']['L2 Theoretical Sectors Global Excessive'],0)
            self.assertGreater(result['l1_service_ms_at_1410'],0.27)
            self.assertLess(result['l1_service_ms_at_1410'],0.30)
            with self.assertRaises(ValueError):
                ANALYZE(raw,sass,12)
            with self.assertRaises(ValueError):
                ANALYZE(raw,sass,tune,'o3')

    def test_async_candidate_identity_is_not_power2(self):
        # CPU parser fixture only; replacing names here is not GPU evidence.
        raw=(EVIDENCE/'ncu_o7_t6_raw.csv').read_text().replace('<1, 0, 6>','<1, 0, 14>')
        sass=(EVIDENCE/'ncu_o7_t6_source_sass.csv').read_text().replace('(int)6>','(int)14>')
        result=ANALYZE(raw,sass,14,'o7')
        self.assertEqual(result['opcodes']['FMUL'],16777216)
        with self.assertRaises(ValueError):
            ANALYZE(raw,sass,13,'o7')

    def test_archived_o3_layout_counters(self):
        evidence=ROOT/'docs/evidence/a100_o378_roof_v6/reports/o378_roof_v6'
        for tune,total,sectors in ((6,117268480,8126464),(13,118800384,0)):
            result=ANALYZE((evidence/f'ncu_o3_t{tune}_raw.csv').read_text(),
                           (evidence/f'ncu_o3_t{tune}_source_sass.csv').read_text(),tune,'o3')
            self.assertEqual(result['dynamic_instructions'],total)
            self.assertEqual(result['source_memory_work']['L2 Theoretical Sectors Global Excessive'],sectors)
            self.assertEqual(result['source_memory_work']['L1 Wavefronts Shared Excessive'],0)

    def test_archived_async_scale_counters(self):
        evidence=ROOT/'docs/evidence/a100_o378_roof_v7/reports/o378_roof_v7'
        result=ANALYZE((evidence/'ncu_o7_t14_raw.csv').read_text(),
                       (evidence/'ncu_o7_t14_source_sass.csv').read_text(),14,'o7')
        self.assertEqual(result['dynamic_instructions'],128270336)
        self.assertEqual(result['opcodes']['FMUL'],16777216)
        self.assertEqual(result['source_memory_work']['L1 Wavefronts Shared Excessive'],860159)
        self.assertEqual(result['source_memory_work']['L2 Theoretical Sectors Global Excessive'],491520)


if __name__=='__main__': unittest.main()
