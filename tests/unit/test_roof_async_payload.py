"""Exact FP32 scale copies share the existing G128 asynchronous ring."""
from pathlib import Path
import runpy
import unittest

ROOT = Path(__file__).resolve().parents[2]


class AsyncPayloadTests(unittest.TestCase):
    def test_body_is_unchanged_except_compile_time_copy_selection(self):
        old = (ROOT/'csrc/sm80/o3_grouped_payload_candidate.cuh').read_text()
        new = (ROOT/'csrc/sm80/o78_async_payload_candidate.cuh').read_text()
        old = old.replace('// Candidates41/42: G128-major global payload, same22/23 compute and shared layout.',
                          '// Candidates55/56: exact41/42 arithmetic, asynchronous G128 FP32 scale copies.')
        old = old.replace('o3_grouped_payload_experiment', 'o78_async_payload_experiment')
        old = old.replace('!PhasePair && !Cached && !AsyncScale);',
                          '!PhasePair && !Cached && AsyncScale);\n'
                          '  static_assert(DualScale && GroupMajorScale && !ActivationPower2 && !PrebiasActivationScale && !CombinedScalePanels);')
        self.assertEqual(new, old)

    def test_copy_ownership_alignment_and_ring(self):
        for rows, tile in ((64,64), (128,64), (128,128), (256,128)):
            for groups in (1,2,3,5,32):
                for stages in (2,3):
                    for group in range(groups):
                        for block in range(rows//tile):
                            covered = []
                            for lane in range(tile//4):
                                source = 4*(group*rows+block*tile+lane*4)
                                dest = 4*((group%stages)*tile+lane*4)
                                self.assertEqual(source%16, 0)
                                self.assertEqual(dest%16, 0)
                                covered.extend(range(lane*4, lane*4+4))
                            self.assertEqual(covered, list(range(tile)))

    def test_launch_and_host_guards(self):
        tu = (ROOT/'csrc/sm80/roof_async_payload.cu').read_text()
        self.assertIn('__launch_bounds__(128,3)', tu)
        self.assertIn('DualScale && !Fast && (Tune==55 || Tune==56)', tu)
        self.assertIn('Stages=Tune==55?2:3', tu)
        self.assertIn('true,true,6,false,false,true,false,Stages>', tu)
        core = (ROOT/'csrc/sm80/roof_candidates.cuh').read_text()
        self.assertIn('case 55:return roof_config_for<41>(dual);', core)
        self.assertIn('case 56:return roof_config_for<42>(dual);', core)
        self.assertIn('reinterpret_cast<uintptr_t>(as.data_ptr())%16==0', core)
        self.assertIn('reinterpret_cast<uintptr_t>(ws.data_ptr())%16==0', core)
        self.assertIn('check_roof_async_scale_alignment(tune,as,ws)', core)
        mixed = (ROOT/'csrc/sm80/mixed_benchmark.cuh').read_text()
        self.assertIn('check_roof_async_scale_alignment(roof_tune,ca->scale,cw->scale)', mixed)
        self.assertIn('meta["scale_copy_alignment_bytes"]=16', core)
        self.assertIn('meta["async_scale_reference_tune"]=tune==55?41:42', core)
        self.assertIn('roof_async_payload.cu', (ROOT/'setup.py').read_text())

    def test_instruction_audit_does_not_confuse_ldg_and_ldgsts(self):
        audit = runpy.run_path(str(ROOT/'scripts/audit_a100_o1.py'))
        f = audit['roof_async_payload_checks']
        ptx = 'cp.async.commit_group; cp.async.wait_group 1; cp.async.wait_group 0;'
        sass = 'LDGSTS.E.BYPASS.128; BAR.SYNC 0x0; DEPBAR.LE SB0, 0x1; DEPBAR.LE SB0, 0x0;'
        for tune in (55,56):
            symbol = f'adangel_sm80_roof_candidateILb1ELb0ELi{tune}EE'
            self.assertTrue(all(f(symbol,ptx,sass).values()))
            self.assertFalse(all(f(symbol,ptx,sass+' LDG.E R0,[R2];').values()))
            self.assertFalse(all(f(symbol,ptx,sass+' STS [R0],R2;').values()))
            self.assertFalse(all(f(symbol,'',sass).values()))
            if tune==56:
                self.assertTrue(all(audit['roof_pipeline_checks'](symbol,ptx,sass).values()))
                self.assertFalse(all(audit['roof_pipeline_checks'](symbol,'cp.async.wait_group 1;',sass).values()))
        self.assertEqual(f('adangel_sm80_roof_candidateILb1ELb0ELi42EE',ptx,sass), {})

    def test_arithmetic_work_and_bitwise_policy(self):
        parser = runpy.run_path(str(ROOT/'scripts/analyze_roof_scale_ncu.py'))
        for tune in (55,56):
            parser['validate_arithmetic_work'](dict(IMMA=16777216,I2F=16777216,
                                                   FFMA=16777216,FMUL=16777216),tune,False)
        for name in ('benchmark_a100_roof_candidates.py','benchmark_a100_roof_trace.py'):
            source = (ROOT/'scripts'/name).read_text()
            self.assertIn('+(55,56)',source)
            self.assertIn('any(t in (14,15,55,56) for t in args.tunes)',source)
        compare = (ROOT/'scripts/compare_roof_trace_candidates.py').read_text()
        self.assertIn("not r.get('async_scale_metadata_verified')",compare)
        payload = (ROOT/'scripts/roof_payload_validation.py').read_text()
        self.assertIn("not meta['fp32_reassociated']",payload)
        self.assertIn("meta['conversion_kernels_per_operand']==2",payload)


if __name__ == '__main__':
    unittest.main()
