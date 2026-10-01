"""Bounded unsigned address arithmetic must not change the numerical algorithm."""
from pathlib import Path
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[2]


class UnsignedPayloadTests(unittest.TestCase):
    def test_device_diff_is_only_dimension_types(self):
        old=(ROOT/'csrc/sm80/o78_async_payload_candidate.cuh').read_text()
        new=(ROOT/'csrc/sm80/o78_unsigned_payload_candidate.cuh').read_text()
        old=old.replace('// Candidates55/56: exact41/42 arithmetic, asynchronous G128 FP32 scale copies.',
                        '// Candidates59/60: unsigned, host-bounded addressing; exact55/56 FP32 math.')
        old=old.replace('o78_async_payload_experiment','o78_unsigned_payload_experiment')
        old=old.replace('const uint8_t* ws,int m,int k,','const uint8_t* ws,uint32_t m,uint32_t k,')
        old=old.replace('const float* grouped_as=nullptr,int total_n=0)',
                        'const float* grouped_as=nullptr,uint32_t total_n=0)')
        old=old.replace('float* y,int m,int n,int k)',
                        'float* y,uint32_t m,uint32_t n,uint32_t k)')
        self.assertEqual(new,old)

    def test_address_bounds_at_legal_extremes(self):
        # No allocation needed: verify extrema of monotone pointer expressions.
        limit=2147483647
        shapes=((64,128,128),(4096,4096,4096),(64,128,16777088),
                (4194240,128,128),(64,8388480,128),(32768,65408,128))
        for m,n,k in shapes:
            self.assertTrue(m%64==n%128==k%128==0)
            self.assertTrue(max(m*k,n*k,m*n)<=limit)
            self.assertLessEqual(m//64,65535)
            self.assertLessEqual(n//128,65535)
            for rows,tile in ((m,64),(n,128)):
                for group in (0,k//128-1):
                    for block in (0,rows//tile-1):
                        payload=group*rows*64+(block*tile+tile-1)*64+48
                        scale=group*rows+block*tile+tile-4
                        self.assertLessEqual(payload+16,rows*k//2)
                        self.assertLessEqual(scale+4,rows*k//128)
                        self.assertLessEqual(payload+16,limit)
                        self.assertEqual((scale*4)%16,0)
                        if rows==m:
                            self.assertLessEqual(payload+16+m*k//2,m*k)
            self.assertLessEqual((m-1)*n+n-1,limit)
            # FP32 pointer scaling is pointer arithmetic, not a uint32 byte offset.
            self.assertLess((m*n-1)*4,2**63)

    def test_host_guard_metadata_and_route(self):
        core=(ROOT/'csrc/sm80/roof_candidates.cuh').read_text()
        self.assertIn('m64*k64<=2147483647LL && n64*k64<=2147483647LL && m64*n64<=2147483647LL',core)
        self.assertIn('m64/tile_m<=65535 && n64/128<=65535',core)
        self.assertIn('meta["dimension_addressing"]="host_bounded_uint32"',core)
        self.assertIn('case 59:return roof_config_for<41>(dual);',core)
        self.assertIn('case 60:return roof_config_for<42>(dual);',core)
        self.assertLess(core.index('if(roof_unsigned_payload(tune)) {',core.index('auto select_roof_kernel')),
                        core.index('if(roof_async_payload(tune)) {',core.index('auto select_roof_kernel')))
        self.assertIn('TORCH_CHECK(!roof_unsigned_payload(roof_tune)',
                      (ROOT/'csrc/sm80/mixed_benchmark.cuh').read_text())
        tu=(ROOT/'csrc/sm80/roof_unsigned_payload.cu').read_text()
        self.assertIn('__launch_bounds__(128,3)',tu)
        self.assertIn('DualScale && !Fast && (Tune==59 || Tune==60)',tu)
        self.assertIn('true,true,6,false,false,true,false,Stages>',tu)
        self.assertIn('roof_unsigned_payload.cu',(ROOT/'setup.py').read_text())

    def test_audit_requires_same_native_pipeline(self):
        a=runpy.run_path(str(ROOT/'scripts/audit_a100_o1.py'))
        ptx='cp.async.commit_group; cp.async.wait_group 1; cp.async.wait_group 0;'
        sass='LDGSTS.E.BYPASS.128; BAR.SYNC 0x0; DEPBAR.LE SB0, 0x1; DEPBAR.LE SB0, 0x0;'
        for tune in (59,60):
            symbol=f'adangel_sm80_roof_candidateILb1ELb0ELi{tune}EE'
            self.assertTrue(all(a['roof_async_payload_checks'](symbol,ptx,sass).values()))
            self.assertFalse(all(a['roof_async_payload_checks'](symbol,ptx,sass+' LDG.E R0,[R2];').values()))
            if tune==60:
                self.assertTrue(a['roof_pipeline_checks'](symbol,ptx,sass))
                self.assertFalse(all(a['roof_pipeline_checks'](symbol,'cp.async.wait_group 1;',sass).values()))


if __name__=='__main__': unittest.main()
