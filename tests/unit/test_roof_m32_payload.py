"""Local-only prototype contract, not GPU performance acceptance."""
from pathlib import Path
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[2]


class M32PayloadTests(unittest.TestCase):
    def test_prepared_only_host_grid_and_metadata(self):
        core=(ROOT/'csrc/sm80/roof_candidates.cuh').read_text()
        self.assertIn('roof_m32_payload(tune)?32:64',core)
        self.assertIn('m64%tile_m==0',core)
        self.assertIn('m64/tile_m<=65535',core)
        self.assertIn('dim3(n/cfg.n,m/tile_m)',core)
        self.assertIn('std::vector<int>{tile_m,cfg.n,cfg.k}',core)
        self.assertIn('tile_m*cfg.n/cfg.threads',core)
        self.assertIn('roof_m32_payload(tune)?1:',core)
        self.assertIn('m32_payload_shared_bytes(tune)',core)
        self.assertIn('roof_m32_payload.cu',(ROOT/'setup.py').read_text())
        mixed=(ROOT/'csrc/sm80/mixed_benchmark.cuh').read_text()
        self.assertIn('TORCH_CHECK(!roof_m32_payload(roof_tune)',mixed)
        self.assertIn('prepared-core only',(ROOT/'scripts/benchmark_a100_roof_trace.py').read_text())

    def test_m32_async_instruction_audit(self):
        audit=runpy.run_path(str(ROOT/'scripts/audit_a100_o1.py'))
        for tune in (57,58):
            symbol=f'adangel_sm80_roof_candidateILb1ELb0ELi{tune}EE'
            ptx='cp.async.commit_group; cp.async.wait_group 1; cp.async.wait_group 0;'
            sass='LDGSTS.E.BYPASS.128; BAR.SYNC 0x0; DEPBAR.LE SB0, 0x1; DEPBAR.LE SB0, 0x0;'
            self.assertTrue(all(audit['roof_async_payload_checks'](symbol,ptx,sass).values()))
            self.assertFalse(all(audit['roof_async_payload_checks'](symbol,ptx,sass+' LDG.E R0,[R2];').values()))
            if tune==58:
                self.assertTrue(audit['roof_pipeline_checks'](symbol,ptx,sass))
                self.assertFalse(all(audit['roof_pipeline_checks'](symbol,'cp.async.wait_group 1;',sass).values()))

    def test_only_geometry_changes_in_device_body(self):
        old=(ROOT/'csrc/sm80/o78_async_payload_candidate.cuh').read_text()
        new=(ROOT/'csrc/sm80/o78_m32_payload_candidate.cuh').read_text()
        old=old.replace('// Candidates55/56: exact41/42 arithmetic, asynchronous G128 FP32 scale copies.',
                        '// Candidates57/58: M32 x N128, 1x4 warps, same56 asynchronous scales and FP32 order.')
        old=old.replace('o78_async_payload_experiment','o78_m32_payload_experiment')
        old=old.replace('static constexpr int WM=2;','static constexpr int WM=1;')
        old=old.replace('static_assert(M==64 && WN==2 && K==128);',
                        'static_assert(M==32 && WN==4 && K==128);')
        self.assertEqual(new,old)

    def test_launch_budget_and_unchanged_math_options(self):
        tu=(ROOT/'csrc/sm80/roof_m32_payload.cu').read_text()
        self.assertIn('__launch_bounds__(128,4)',tu)
        self.assertIn('DualScale && !Fast && (Tune==57 || Tune==58)',tu)
        self.assertIn('Stages=Tune==57?2:3',tu)
        self.assertIn('o3_body<32,128,128,false,false,false,4,false,true,false,true,true,true,false,',tu)
        self.assertIn('true,true,6,false,false,true,false,Stages>',tu)

    def test_scale_vector_coverage_and_payload_cost_not_free(self):
        for rows,tile in ((32,32),(96,32),(128,128),(256,128)):
            for groups in (1,3,5,32):
                for group in range(groups):
                    for block in range(rows//tile):
                        covered=[]
                        for thread in range(tile//4):
                            start=group*rows+block*tile+4*thread
                            self.assertEqual((start*4)%16,0)
                            covered.extend(range(start,start+4))
                        self.assertEqual(covered,list(range(group*rows+block*tile,group*rows+(block+1)*tile)))
        old_per_cta=2*64*64+128*64
        new_per_cta=2*32*64+128*64
        self.assertEqual(2*new_per_cta/old_per_cta,1.5)
        self.assertEqual(32*128//128,32)


if __name__=='__main__': unittest.main()
