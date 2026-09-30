"""The exact scale shortcut must not silently substitute magic-bias I2F."""
from pathlib import Path
import runpy
import unittest

AUDIT=runpy.run_path(str(Path(__file__).resolve().parents[2]/'scripts/audit_a100_o1.py'))


class RoofScaleAuditTests(unittest.TestCase):
    def test_paired_ring_has_its_own_wait_and_barrier_contract(self):
        symbol='adangel_sm80_roof_candidateILb1ELb0ELi29EE'
        ptx='cp.async.commit_group; cp.async.wait_group 0; bar.sync 0;'
        sass='DEPBAR.LE SB0, 0x0; BAR.SYNC 0x0;'
        check=AUDIT['roof_paired_pipeline_checks']
        self.assertTrue(all(check(symbol,ptx,sass).values()))
        self.assertFalse(all(check(symbol,ptx.replace('wait_group 0','wait_group 1'),sass).values()))
        self.assertFalse(all(check(symbol,ptx,sass.replace('BAR.SYNC','NOP')).values()))
        self.assertEqual(check(symbol.replace('Li29','Li28'),'',''),{})
        self.assertEqual(AUDIT['roof_pipeline_checks'](symbol,ptx,sass),{})

    def test_reduction_final_add_is_required_for_all_opt_in_variants(self):
        for tune in (24,25,26,27):
            symbol=f'adangel_sm80_roof_candidateILb1ELb0ELi{tune}EE'
            check=AUDIT['roof_reduction_checks']
            self.assertTrue(all(check(symbol,dict(I2F=64,FFMA=64,FADD=32)).values()))
            self.assertFalse(all(check(symbol,dict(I2F=64,FFMA=64)).values()))
        self.assertEqual(AUDIT['roof_reduction_checks']('adangel_sm80_roof_candidateILb1ELb0ELi23EE',{}),{})

    def test_three_stage_waits_are_both_required_in_same_entry(self):
        check=AUDIT['roof_pipeline_checks']
        for tune in (16,17,18,19,23):
            symbol=f'adangel_sm80_roof_candidateILb1ELb0ELi{tune}EE'
            ptx='cp.async.wait_group 1; cp.async.wait_group 0;'
            sass='DEPBAR.LE SB0, 0x1; DEPBAR.LE SB0, 0x0;'
            self.assertTrue(all(check(symbol,ptx,sass).values()))
            self.assertFalse(all(check(symbol,ptx,sass.replace('0x1','0x0')).values()))
            self.assertFalse(all(check(symbol,ptx.replace('1;','0;'),sass).values()))
        self.assertEqual(check('adangel_sm80_roof_candidateILb1ELb0ELi14EE','',''),{})
        self.assertEqual(check('adangel_sm80_roof_candidateILb1ELb0ELi22EE','',''),{})

    def test_fast_specialization(self):
        symbol='_ZN40_GLOBAL_adangel_sm80_roof_candidateILb1ELb1ELi11EEEv'
        checks=AUDIT['roof_scale_checks'](symbol,dict(FMUL=0,I2F=32,FFMA=32))
        self.assertEqual(len(checks),3)
        self.assertTrue(all(checks.values()))
        self.assertFalse(all(AUDIT['roof_scale_checks'](symbol,dict(FMUL=1,I2F=32,FFMA=32)).values()))
        self.assertFalse(all(AUDIT['roof_scale_checks'](symbol,dict(FMUL=0,I2F=0,FFMA=32)).values()))
        self.assertFalse(all(AUDIT['roof_scale_checks'](symbol,{}).values()))
        self.assertTrue(all(AUDIT['roof_scale_checks'](symbol.replace('Li11','Li12'),dict(I2F=32,FFMA=32)).values()))

    def test_fallback_and_other_instances_unaffected(self):
        for arguments in ('Lb1ELb0ELi11EE','Lb0ELb1ELi11EE','Lb1ELb0ELi6EE'):
            self.assertEqual(AUDIT['roof_scale_checks']('adangel_sm80_roof_candidateI'+arguments,{}),{})


if __name__=='__main__':
    unittest.main()
