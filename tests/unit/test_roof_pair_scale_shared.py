"""v51 moves alignment decisions out of per-output math, not out of accounting."""
from pathlib import Path
import runpy
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))


class SharedPairTests(unittest.TestCase):
    def test_fast_finish_has_no_guard_or_code_decode(self):
        code=(ROOT/'csrc/sm80/o3_pair_scale_shared_probe.cuh').read_text()
        finish=code.split('const int p0=first(vi,ni)')[1].split('o1_static_for<0,decltype(cute::size(acc))')[0]
        for token in ('if(', 'decode(', 'anchor=c0', 'd0<=13', 's.codes'):
            self.assertNotIn(token,finish)
        self.assertIn('p0*s.factor0[slot/2][col]+p1*s.factor1[slot/2][col]',finish)
        self.assertIn('s.anchor_scale[slot/2][col]',finish)
        self.assertIn('if constexpr(First)',code)
        self.assertIn('sizeof(Storage)==68608',code)
        self.assertIn('SM80_16x8x64_S32U4S4S32_TN',code)
        self.assertIn('SM80_16x8x64_S32S4S4S32_TN',code)

    def test_exact_parameters_for_all_safe_encoded_pairs(self):
        for c0 in range(255):
            for c1 in range(max(0,c0-13),min(254,c0+13)+1):
                h=min(c0,c1);f0=1<<(c0-h);f1=1<<(c1-h)
                self.assertLessEqual(131072*(f0+f1),2**31-1)
                for p0,p1 in ((131072,131072),(-131072,131072),(-131072,-131072),(1,-1)):
                    self.assertEqual((p0*f0+p1*f1)*(1<<h),p0*(1<<c0)+p1*(1<<c1))

    def test_scope_and_fallback(self):
        wrapper=(ROOT/'csrc/sm80/roof_pair_scale_shared_probe.cu').read_text()
        self.assertIn('void adangel_roof_pair_scale_shared_fallback',wrapper)
        self.assertIn('o3_pair_alignment_experiment::body',wrapper)
        driver=(ROOT/'scripts/benchmark_roof_pair_scale_shared_probe.py').read_text()
        for text in ('difference<=13','weight_scale._version',"UE8M0 code 255 is invalid",
                     'self.guard_tensors[key]=weight_scale',"range_check_wall_ms=elapsed",
                     "executed=2 if size==1 and not self.guard_metadata['safe_all_pairs'] else size",
                     'self.handles[executed]',"choices=['o3'],default=['o3']"):
            self.assertIn(text,driver)
        validation=(ROOT/'scripts/validate_roof_pair_scale_shared_probe.py').read_text()
        self.assertIn("pattern in ('fallback14','fallback15') and g>1",validation)
        self.assertIn('safe_then_unsafe_fallback=True',validation)
        self.assertIn('invalid_after_cached_guard_rejected=True',validation)
        for file in ('setup.py','csrc/sm80/roof_candidates.cuh'):
            self.assertNotIn('pair_scale_shared_probe',(ROOT/file).read_text())

    def test_summary_preserves_failing_cv_and_mse_gate(self):
        fn=runpy.run_path(str(ROOT/'scripts/benchmark_roof_pair_scale_shared_probe.py'))['summary']
        rows=[dict(sample_id='x',variant='o3',round=r,pair_scale_shared=p,
            summary=dict(median_ms=1+p,cv_percent=9),mse_vs_o0=.125+p*1e-8,
            current_best_mse_vs_o0=.125,bitwise_equal_current_best=p==0,
            mse_vs_current_best=p*1e-12,max_abs_difference_current_best=p*1e-5,
            finite_fp32=True,output_close_current_best=True) for r in range(3) for p in (0,1)]
        out=fn(rows)
        self.assertEqual([r['paired_speedup'] for r in out],[1,.5])
        self.assertEqual([r['cv_failed_records'] for r in out],[3,3])
        with self.assertRaises(ValueError): fn(rows[:-1])
        rows[1]['mse_vs_o0']=.14
        with self.assertRaises(ValueError): fn(rows)


if __name__=='__main__': unittest.main()
