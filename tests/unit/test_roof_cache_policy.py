"""Cache experiments keep the exact existing math/pipeline device bodies."""
from pathlib import Path
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[2]


class CachePolicyTests(unittest.TestCase):
    def test_only_new_tu_changes_copy_policy(self):
        new=(ROOT/'csrc/sm80/roof_cache_policy.cu').read_text()
        self.assertIn('cp.async.ca.shared.global',new)
        self.assertNotIn('cp.async.cg.',new)
        self.assertIn('#include "o3_row_scale_epilogue_candidate.cuh"',new)
        self.assertIn('#include "o78_unsigned_payload_candidate.cuh"',new)
        self.assertIn('__launch_bounds__(128,3)',new)
        self.assertIn('false,true,6,false,false,false,false,3>',new)
        self.assertIn('true,true,6,false,false,true,false,2>',new)
        for name in ('roof_epilogue_scale_layout.cu','roof_unsigned_payload.cu'):
            old=(ROOT/'csrc/sm80'/name).read_text()
            self.assertIn('cp.async.cg.shared.global',old)
            self.assertNotIn('cp.async.ca.',old)
        self.assertNotIn('Magic=true',new)
        self.assertIn('roof_cache_policy.cu',(ROOT/'setup.py').read_text())

    def test_host_routes_are_guarded_and_internal(self):
        core=(ROOT/'csrc/sm80/roof_candidates.cuh').read_text()
        self.assertIn('case 61:return roof_config_for<42>(dual);',core)
        self.assertIn('case 62:return roof_config_for<41>(dual);',core)
        self.assertIn('meta["cache_policy_reference_tune"]=tune==61?54:59;',core)
        for name in ('o1_o3.cu','mixed_benchmark.cuh'):
            self.assertIn('roof_tune!=61 && roof_tune!=62',(ROOT/'csrc/sm80'/name).read_text())
        trace=(ROOT/'scripts/benchmark_a100_roof_trace.py').read_text()
        self.assertIn('args.all_modes and any(t in (61,62)',trace)
        self.assertIn('for_61_bitwise_to_54',trace)
        verify=(ROOT/'scripts/roof_reduction_validation.py').read_text()
        self.assertIn('54 if tune==61 else tune-2',verify)
        self.assertIn('torch.equal(y.view(torch.int32),row_scale_baseline.view(torch.int32))',verify)

    def test_cache_audit_fails_closed(self):
        audit=runpy.run_path(str(ROOT/'scripts/audit_a100_o1.py'))
        check=audit['roof_cache_policy_checks']
        for dual,tune in ((0,61),(1,62)):
            symbol=f'adangel_sm80_roof_candidateILb{dual}ELb0ELi{tune}EE'
            ptx='cp.async.ca.shared.global [%r0], [%rd0], 16;'
            sass='LDGSTS.E.128 [R2], [R4.64];'
            self.assertTrue(all(check(symbol,ptx,sass).values()))
            self.assertFalse(all(check(symbol,ptx.replace('.ca.','.cg.'),sass).values()))
            self.assertFalse(all(check(symbol,ptx,sass.replace('.E.','.E.BYPASS.')).values()))
            self.assertFalse(all(check(symbol,ptx,'').values()))
        self.assertEqual(check('adangel_sm80_roof_candidateILb1ELb0ELi59EE','',''),{})


if __name__=='__main__': unittest.main()
