"""v50 O3-only paired-integer scope, overflow proof and experiment accounting."""
from pathlib import Path
import runpy
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))


class PairAlignmentTests(unittest.TestCase):
    def test_guard_covers_signed_extrema(self):
        p=131072
        for delta in range(14):
            for p0 in (-p,-1,0,1,p):
                for p1 in (-p,-1,0,1,p):
                    q=p0*(1<<delta)+p1
                    self.assertLessEqual(abs(q),2**31-1)
                    self.assertEqual(q,p0*2**delta+p1)
        self.assertGreater(p*((1<<14)+1),2**31-1)
        code=(ROOT/'csrc/sm80/o3_pair_alignment_probe.cuh').read_text()
        self.assertIn('if(d0<=13 && d1<=13)',code)
        self.assertIn('p0*int(1u<<d0)+p1*int(1u<<d1)',code)
        self.assertNotIn('p0<<',code)
        self.assertIn('if(second) acc',code)

    def test_four_slot_pair_lifetime(self):
        for groups in (1,2,3,5,32):
            slots={0:0}
            if groups>1: slots[1]=1
            visited=[]
            for group in range(0,groups,2):
                current=group%4
                self.assertEqual(slots[current],group)
                if group+1<groups: self.assertEqual(slots[current+1],group+1)
                # Barrier precedes overwriting a previous pair's slots.
                for future in (group+2,group+3):
                    if future<groups:
                        self.assertNotIn(future%4,(current,current+1))
                        slots[future%4]=future
                visited.extend(range(group,min(group+2,groups)))
            self.assertEqual(visited,list(range(groups)))

    def test_native_instructions_and_no_default_edit(self):
        code=(ROOT/'csrc/sm80/o3_pair_alignment_probe.cuh').read_text()
        for text in ('SM80_16x8x64_S32U4S4S32_TN','SM80_16x8x64_S32S4S4S32_TN',
                     'partition_C','__syncthreads()','cp.async.wait_group 0','sizeof(Storage)==67584'):
            self.assertIn(text,code)
        for p in ('setup.py','csrc/sm80/roof_candidates.cuh'):
            self.assertNotIn('pair_alignment_probe',(ROOT/p).read_text())
        driver=(ROOT/'scripts/benchmark_roof_pair_alignment_probe.py').read_text()
        self.assertIn("choices=['o3'],default=['o3']",driver)
        self.assertIn('UE8M0 code 255 is invalid',driver)

    def test_numeric_changes_allowed_but_errors_not_hidden(self):
        summarize=runpy.run_path(str(ROOT/'scripts/benchmark_roof_pair_alignment_probe.py'))['summary']
        rows=[dict(sample_id='x',variant='o3',round=r,pair_alignment=p,
            summary=dict(median_ms=1+p,cv_percent=9),mse_vs_o0=.125+p*1e-8,
            current_best_mse_vs_o0=.125,bitwise_equal_current_best=p==0,
            mse_vs_current_best=p*1e-12,max_abs_difference_current_best=p*1e-5,
            finite_fp32=True,output_close_current_best=True) for r in range(3) for p in (0,1)]
        out=summarize(rows)
        self.assertEqual([r['paired_speedup'] for r in out],[1,.5])
        self.assertEqual([r['cv_failed_records'] for r in out],[3,3])
        self.assertEqual(out[1]['bitwise_equal_records'],0)
        with self.assertRaises(ValueError): summarize(rows[:-1])
        with self.assertRaises(ValueError): summarize(rows+[rows[0]])
        rows[1]['mse_vs_o0']=.14
        with self.assertRaises(ValueError): summarize(rows)

    def test_preflight_has_tail_and_fallback(self):
        script=(ROOT/'scripts/validate_roof_pair_alignment_probe.py').read_text()
        for s in ('128),(64,128,256),(64,128,384)',"'guard13'","'fallback14'","'fallback15'","'subnormal'",
                  'reference_fp64','torch.cuda.Stream()'):
            self.assertIn(s,script)

    def test_ncu_math_accounting_does_not_relax_old_checks(self):
        from analyze_roof_scale_ncu import validate_arithmetic_work, validate_paired_integer_work
        pair=dict(IMMA=16777216,I2F=8388608,FFMA=8388608,FMUL=524288)
        validate_paired_integer_work(pair)
        with self.assertRaises(ValueError): validate_arithmetic_work(pair,54,False)
        old=dict(pair,I2F=16777216,FFMA=16777216)
        validate_arithmetic_work(old,54,False)
        with self.assertRaises(ValueError): validate_paired_integer_work(old)


if __name__=='__main__': unittest.main()
