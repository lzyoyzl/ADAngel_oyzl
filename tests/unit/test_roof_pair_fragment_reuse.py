"""v52 isolates two-group fragment lifetimes without expanding the math scope."""
from pathlib import Path
import runpy
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))


class FragmentReuseTests(unittest.TestCase):
    def test_single_b_group_and_register_partials_for_both_m_atoms(self):
        code=(ROOT/'csrc/sm80/o3_pair_fragment_reuse_probe.cuh').read_text()
        self.assertNotIn('b10',code)
        self.assertNotIn('b11',code)
        self.assertIn('cute::_4{},cute::_2{},cute::_4{}',code)
        self.assertIn('first(vi,mi,ni)=pl(vi,ni)+16*ph(vi,ni)',code)
        self.assertIn('load_b(slot+1,nb,b00,b01)',code)
        self.assertIn('if constexpr(ReuseA) load_a(slot+1,a00,a01,h00,h01)',code)
        self.assertIn('integer_group(mi,a10,a11,h10,h11,b00,b01,pl,ph)',code)
        self.assertIn('sizeof(Storage)==68608',code)
        self.assertIn('cp.async.wait_group 0',code)
        self.assertEqual(code.count('__syncthreads()'),1)
        self.assertNotIn('s.partial',code)

    def test_guard_both_candidates_and_all_preflight_policies(self):
        driver=(ROOT/'scripts/benchmark_roof_pair_fragment_reuse_probe.py').read_text()
        for text in ('executed=3 if size in (1,2)', 'difference<=13', 'weight_scale._version',
                     'range_check_wall_ms=elapsed', 'self.guard_tensors[key]=weight_scale'):
            self.assertIn(text,driver)
        validation=(ROOT/'scripts/validate_roof_pair_fragment_reuse_probe.py').read_text()
        self.assertIn("for policy in ((1,2) if pattern=='subnormal' else (0,1,2))",validation)
        self.assertIn('for requested in (1,2):',validation)
        self.assertIn('assert driver.last_policy==3',validation)
        self.assertIn('requested_policy=requested',validation)

    def test_no_production_change_and_original_native_mma(self):
        wrapper=(ROOT/'csrc/sm80/roof_pair_fragment_reuse_probe.cu').read_text()
        self.assertIn('body<ADANGEL_PAIR_FRAGMENT_REUSE==2>',wrapper)
        self.assertIn('o3_pair_alignment_experiment::body',wrapper)
        code=(ROOT/'csrc/sm80/o3_pair_fragment_reuse_probe.cuh').read_text()
        for atom in ('SM80_16x8x64_S32U4S4S32_TN','SM80_16x8x64_S32S4S4S32_TN'):
            self.assertIn(atom,code)
        for path in ('setup.py','csrc/sm80/roof_candidates.cuh'):
            self.assertNotIn('pair_fragment_reuse_probe',(ROOT/path).read_text())

    def test_three_policy_summary_and_filtered_profile_subset(self):
        fn=runpy.run_path(str(ROOT/'scripts/benchmark_roof_pair_fragment_reuse_probe.py'))['summary']
        rows=[dict(sample_id='sample',variant='o3',round=r,pair_fragment_reuse=p,
            summary=dict(median_ms=p+1,cv_percent=9),mse_vs_o0=.125+p*1e-8,
            current_best_mse_vs_o0=.125,bitwise_equal_current_best=p==0,
            mse_vs_current_best=p*1e-12,max_abs_difference_current_best=p*1e-5,
            finite_fp32=True,output_close_current_best=True) for r in range(3) for p in (0,1,2)]
        result=fn(rows)
        self.assertEqual([r['paired_speedup'] for r in result],[1,.5,1/3])
        self.assertEqual([r['cv_failed_records'] for r in result],[3,3,3])
        self.assertEqual(len(fn([r for r in rows if r['pair_fragment_reuse'] in (0,2)],(0,2))),2)
        with self.assertRaises(ValueError): fn(rows[:-1])
        with self.assertRaises(ValueError): fn(rows,(0,1,1))
        rows[1]['mse_vs_o0']=.2
        with self.assertRaises(ValueError): fn(rows)


if __name__=='__main__': unittest.main()
