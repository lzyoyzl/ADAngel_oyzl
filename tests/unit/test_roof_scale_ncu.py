from pathlib import Path
import csv
import io
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[2]
MODULE=runpy.run_path(str(ROOT/'scripts/analyze_roof_scale_ncu.py'))
ANALYZE=MODULE['analyze']
EVIDENCE=ROOT/'docs/evidence/a100_o378_roof_v4/reports/o378_roof_v4'


class RoofScaleNcuTests(unittest.TestCase):
    def test_archived_pc_wait_evidence_is_reconciled_not_time_fraction(self):
        evidence=ROOT/'docs/evidence/a100_o378_roof_v14/reports/o378_roof_v14'
        for variant,tune,total,wait in (('o3',6,13548,3029),('o3',21,18486,7975),
                                       ('o7',6,13756,2855),('o7',21,17272,7519)):
            result=ANALYZE((evidence/f'ncu_{variant}_t{tune}_raw.csv').read_text(),
                           (evidence/f'ncu_{variant}_t{tune}_source_sass.csv').read_text(),
                           tune,variant,False,True)['pc_sampling']
            self.assertEqual(result['not_issued_samples'],total)
            self.assertEqual(result['reason_samples']['wait'],wait)
            self.assertAlmostEqual(sum(result['reason_share_percent'].values()),100)
            self.assertAlmostEqual(result['reason_share_percent']['wait'],100*wait/total)
            self.assertEqual(sum(sum(v.values()) for v in result['samples_by_consumer_opcode'].values()),total)
            self.assertIn('not runtime fractions',result['interpretation'])
            self.assertIn('preceding_instructions',result['top_wait_consumer_pcs'][0])

    def test_pc_sampling_rejects_missing_negative_and_inconsistent_counts(self):
        evidence=ROOT/'docs/evidence/a100_o378_roof_v14/reports/o378_roof_v14'
        rows=list(csv.DictReader((evidence/'ncu_o7_t21_source_sass.csv').read_text().splitlines()[1:]))
        analyze=MODULE['pc_stall_summary']
        with self.assertRaisesRegex(ValueError,'empty PC'):
            analyze([])
        with self.assertRaisesRegex(ValueError,'missing PC sampling'):
            analyze([{k:v for k,v in rows[0].items() if k!='stall_wait (Not Issued)'}])
        for value,error in (('-1','negative PC'),('999','reason/total mismatch')):
            with self.assertRaisesRegex(ValueError,error):
                analyze([dict(rows[0],**{'stall_wait (Not Issued)':value})])

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
        for tune in (16,17,18,19,45,46):
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

    def test_archived_relaxed_register_budget(self):
        evidence=ROOT/'docs/evidence/a100_o378_roof_v13/reports/o378_roof_v13'
        for variant,total,local in (('o3',125894656,131072),('o7',128892928,0)):
            result=ANALYZE((evidence/f'ncu_{variant}_t19_raw.csv').read_text(),
                           (evidence/f'ncu_{variant}_t19_source_sass.csv').read_text(),19,variant,resource_model=True)
            bounds=result['resource_service_lower_bounds_ms_at_1410']
            self.assertAlmostEqual(bounds['mma'],0.22034693984764905)
            self.assertEqual(bounds['mma'],bounds['i2f'])
            self.assertAlmostEqual(result['optimistic_fixed_work_lower_bound_ms'],max(bounds.values()))
            self.assertLess(result['optimistic_fixed_work_lower_bound_ms'],sum(bounds.values()))
            self.assertAlmostEqual(result['shared_capacity_crosscheck_wavefronts_per_sm_cycle'],1,places=5)
            self.assertEqual(result['dynamic_instructions'],total)
            self.assertEqual(result['registers_per_thread'],128)
            self.assertEqual(result['max_ctas_per_sm_from_launch_limits'],2)
            self.assertEqual(result['source_memory_work']['L2 Theoretical Sectors Local'],local)
            self.assertEqual(result['source_memory_work_by_opcode']['LDGSTS']['L1 Wavefronts Shared Excessive'],8388608)
            self.assertEqual(result['source_memory_work_by_opcode']['LDSM']['L1 Wavefronts Shared'],25165824)
            self.assertEqual(result['source_memory_work_omitted_zero_columns'],
                             ['L2 Theoretical Sectors Local'] if variant=='o7' else [])

    def test_archived_warp_reuse_reduces_work_not_latency_by_same_factor(self):
        evidence=ROOT/'docs/evidence/a100_o378_roof_v14/reports/o378_roof_v14'
        for variant,total,local in (('o3',109887488,5242880),('o7',113737728,3145728)):
            result=ANALYZE((evidence/f'ncu_{variant}_t21_raw.csv').read_text(),
                           (evidence/f'ncu_{variant}_t21_source_sass.csv').read_text(),21,variant,True)
            self.assertEqual(result['dynamic_instructions'],total)
            self.assertEqual(result['source_memory_work']['L2 Theoretical Sectors Local'],local)
            self.assertEqual(result['source_memory_work_by_opcode']['LDSM']['L1 Wavefronts Shared'],16777216)
            self.assertEqual(result['binding_modeled_resources'],['mma','i2f'])
            self.assertAlmostEqual(result['optimistic_fixed_work_lower_bound_ms'],0.22034693984764905)
            self.assertLess(result['achieved_occupancy_percent'],12.5)
            self.assertLess(result['eligible_warps'],0.55)
            self.assertGreater(result['ncu_duration_ms'],2*result['optimistic_fixed_work_lower_bound_ms'])

    def test_archived_fragment_tree_trades_registers_for_more_work(self):
        evidence=ROOT/'docs/evidence/a100_o378_roof_v23/reports/o378_roof_v23'
        results={t:ANALYZE((evidence/f'ncu_o7_t{t}_raw.csv').read_text(),
                          (evidence/f'ncu_o7_t{t}_source_sass.csv').read_text(),
                          t,'o7',True,True) for t in (23,38,39,40)}
        self.assertEqual(results[38]['registers_per_thread'],255)
        for tune,total in ((39,144392192),(40,146104320)):
            row=results[tune]
            self.assertEqual(row['registers_per_thread'],168)
            self.assertEqual(row['max_ctas_per_sm_from_launch_limits'],3)
            self.assertEqual(row['dynamic_instructions'],total)
            self.assertEqual(row['opcodes']['LDSM'],6291456)
            self.assertEqual(row['opcodes']['LDSM'],results[23]['opcodes']['LDSM']*1.5)
            self.assertEqual(row['opcodes']['IMMA'],results[23]['opcodes']['IMMA'])
            self.assertEqual(row['opcodes']['I2F'],16777216)
            self.assertEqual(row['source_memory_work']['L2 Theoretical Sectors Local'],0)
            self.assertEqual(row['binding_modeled_resources'],['l1tex_data_wavefront_capacity'])
            self.assertGreater(row['optimistic_fixed_work_lower_bound_ms'],
                               results[23]['optimistic_fixed_work_lower_bound_ms'])
            self.assertLess(row['eligible_warps'],results[23]['eligible_warps'])
            # Counters establish increased work, not a sum of runtime fractions.
            self.assertIn('not runtime fractions',row['pc_sampling']['interpretation'])

    def test_resource_model_requires_full_counters_and_correct_units(self):
        raw=(EVIDENCE/'ncu_o7_t6_raw.csv').read_text()
        sass=(EVIDENCE/'ncu_o7_t6_source_sass.csv').read_text()
        with self.assertRaises(KeyError):
            ANALYZE(raw,sass,6,'o7',True)  # Older partial export: not zero-filled.
        evidence=ROOT/'docs/evidence/a100_o378_roof_v14/reports/o378_roof_v14'
        raw=(evidence/'ncu_o7_t21_raw.csv').read_text()
        sass=(evidence/'ncu_o7_t21_source_sass.csv').read_text()
        with self.assertRaises(ValueError):
            ANALYZE(raw.replace('Mbyte','MiB'),sass,21,'o7',True)


if __name__=='__main__': unittest.main()
