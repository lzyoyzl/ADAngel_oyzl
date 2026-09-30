from pathlib import Path
import csv
import io
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[2]
ANALYZE=runpy.run_path(str(ROOT/'scripts/analyze_roof_scale_ncu.py'))['analyze']
EVIDENCE=ROOT/'docs/evidence/a100_o378_roof_v4/reports/o378_roof_v4'


class RoofScaleNcuTests(unittest.TestCase):
    def test_missing_local_column_only_allowed_without_local_instructions(self):
        # Parser fixture, not profiling evidence. Preserve executed instruction
        # totals while substituting nonlocal opcodes so the omission is valid.
        raw=(EVIDENCE/'ncu_o7_t6_raw.csv').read_text()
        sass=(EVIDENCE/'ncu_o7_t6_source_sass.csv').read_text()
        lines=list(csv.reader(io.StringIO(sass)))
        column=lines[1].index('L2 Theoretical Sectors Local')
        for row in lines[1:]:
            row.pop(column)
        output=io.StringIO();csv.writer(output).writerows(lines)
        missing=output.getvalue()
        with self.assertRaisesRegex(ValueError,'missing required memory counters'):
            ANALYZE(raw,missing,6,'o7')
        no_local=missing.replace('LDL','LDG').replace('STL','STG')
        result=ANALYZE(raw,no_local,6,'o7')
        self.assertEqual(result['source_memory_work']['L2 Theoretical Sectors Local'],0)
        self.assertEqual(result['source_memory_work_omitted_zero_columns'],['L2 Theoretical Sectors Local'])
        with self.assertRaises(ValueError):
            ANALYZE(raw,no_local.replace('L1 Wavefronts Shared Excessive','missing'),6,'o7')

    def test_archived_full_warp_copy_tradeoff(self):
        evidence=ROOT/'docs/evidence/a100_o378_roof_v8/reports/o378_roof_v8'
        result=ANALYZE((evidence/'ncu_o7_t15_raw.csv').read_text(),
                       (evidence/'ncu_o7_t15_source_sass.csv').read_text(),15,'o7')
        self.assertEqual(result['dynamic_instructions'],124698624)
        self.assertEqual(result['source_memory_work']['L2 Theoretical Sectors Local'],12582912)
        self.assertEqual(result['opcodes']['FMUL'],16777216)
        self.assertEqual(result['registers_per_thread'],127)
        self.assertEqual(result['registers_per_thread_allocated'],128)
        self.assertAlmostEqual(result['dynamic_shared_bytes'],68608)
        self.assertAlmostEqual(result['allocated_shared_bytes_including_driver'],69632)
        self.assertEqual(result['max_ctas_per_sm_from_launch_limits'],2)
        self.assertLess(result['achieved_occupancy_percent'],25)

    def test_three_stage_identity_and_memory_units(self):
        # Parser fixture only: renaming an archived kernel is NOT GPU evidence.
        raw=(EVIDENCE/'ncu_o7_t6_raw.csv').read_text()
        sass=(EVIDENCE/'ncu_o7_t6_source_sass.csv').read_text()
        for tune in (16,17,18,19):
            new_raw=raw.replace('<1, 0, 6>',f'<1, 0, {tune}>')
            new_sass=sass.replace('(int)6>',f'(int){tune}>')
            result=ANALYZE(new_raw,new_sass,tune,'o7')
            self.assertEqual(result['opcodes']['IMMA'],16777216)
            self.assertEqual(result['opcodes']['FMUL'],16777216)
            # Resource values remain the fixture's two-stage values, not a
            # fabricated claim of three-CTA residency for the new candidate.
            self.assertEqual(result['max_ctas_per_sm_from_launch_limits'],2)
            with self.assertRaises(ValueError):
                ANALYZE(new_raw.replace('Kbyte/block','KiB/block'),new_sass,tune,'o7')
            with self.assertRaises(ValueError):
                ANALYZE(new_raw,sass,tune,'o7')

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

    def test_archived_three_stage_tradeoff(self):
        evidence=ROOT/'docs/evidence/a100_o378_roof_v12/reports/o378_roof_v12'
        for variant,total,local in (('o3',166494208,80871424),('o7',141000704,8650752)):
            result=ANALYZE((evidence/f'ncu_{variant}_t17_raw.csv').read_text(),
                           (evidence/f'ncu_{variant}_t17_source_sass.csv').read_text(),17,variant)
            self.assertEqual(result['dynamic_instructions'],total)
            self.assertEqual(result['registers_per_thread'],80)
            self.assertEqual(result['max_ctas_per_sm_from_launch_limits'],3)
            self.assertEqual(result['source_memory_work']['L2 Theoretical Sectors Local'],local)
            self.assertEqual(result['source_memory_work']['L1 Wavefronts Shared Excessive'],8388608)
            by_op=result['source_memory_work_by_opcode']
            for name,total_work in result['source_memory_work'].items():
                self.assertEqual(sum(work[name] for work in by_op.values()),total_work)
            # Extra copy work is not evidence of ldmatrix bank conflicts.
            self.assertEqual(by_op['LDGSTS']['L1 Wavefronts Shared Excessive'],8388608)
            self.assertEqual(by_op['LDSM']['L1 Wavefronts Shared Excessive'],0)
            self.assertEqual(by_op['LDL']['L2 Theoretical Sectors Local']+
                             by_op['STL']['L2 Theoretical Sectors Local'],local)


if __name__=='__main__': unittest.main()
