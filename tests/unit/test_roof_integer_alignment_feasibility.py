"""CPU-only overflow bounds; these tests do not implement an alternative GEMM."""
import itertools
from pathlib import Path
import runpy
import unittest

S=runpy.run_path(str(Path(__file__).resolve().parents[2]/'scripts/inspect_o3_integer_alignment_feasibility.py'))


class IntegerAlignmentFeasibilityTests(unittest.TestCase):
    def test_boundary_and_zero_code(self):
        bound=S['aligned_integer_bound'];p=S['PARTIAL_BOUND']
        self.assertEqual(bound([0,0]),2*p)
        self.assertEqual(bound([127,140]),p*(1+2**13))
        self.assertLessEqual(bound([127,140]),2**31-1)
        self.assertGreater(bound([127,141]),2**31-1)
        self.assertEqual(bound([0,254]),p*(1+2**254))
        self.assertEqual(bound([254]*32),32*p)
        for codes in ([],[255],[True],[1.0],[-1]):
            with self.assertRaises(ValueError): bound(codes)

    def test_triangle_bound_covers_signed_extrema_and_prefixes(self):
        p=S['PARTIAL_BOUND'];bound=S['aligned_integer_bound']
        for codes in ([110,110],[110,114],[112,115,110,113]):
            anchor=min(codes)
            for signs in itertools.product((-1,1),repeat=len(codes)):
                total=0
                for code,sign in zip(codes,signs):
                    total+=sign*p*2**(code-anchor)
                    self.assertLessEqual(abs(total),bound(codes))

    def test_histogram_counts_and_invalid_shapes(self):
        inspect=S['inspect_scale_rows']
        r=inspect([[100,100,100,114],[0,13,254,254]],2)
        self.assertEqual(r['windows'],4)
        self.assertEqual(r['int32_safe_windows'],3)
        self.assertEqual(r['all_windows_safe_columns'],1)
        self.assertEqual(r['exponent_spread_histogram'],{'0':2,'13':1,'14':1})
        self.assertEqual(r['exact_fp32_integer_bound_windows'],2)
        self.assertEqual(r['zero_code_windows'],1)
        for values,window in (([],2),([[1]],2),([[1,2],[1]],2),([[1,2]],3)):
            with self.assertRaises(ValueError): inspect(values,window)

    def test_aggregate_recomputes_coverage(self):
        inspect=S['inspect_scale_rows'];aggregate=S['aggregate']
        samples=[dict(windows=[inspect([[120]*32],w) for w in S['WINDOWS']]),
                 dict(windows=[inspect([[120,121]*16],w) for w in S['WINDOWS']])]
        r=aggregate(samples,2)
        self.assertEqual((r['samples'],r['windows'],r['int32_safe_windows']),(2,32,32))
        self.assertEqual(r['exponent_spread_histogram'],{'0':16,'1':16})
        with self.assertRaises(ValueError): aggregate([{'windows':[]}],2)


if __name__=='__main__': unittest.main()
