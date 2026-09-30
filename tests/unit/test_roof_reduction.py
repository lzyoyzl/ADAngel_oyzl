"""CPU/source policy gates, not substitutes for SM80 numerical/perf tests."""
import importlib.util
from pathlib import Path
import struct
import unittest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('reduction_policy',ROOT/'scripts/roof_reduction_validation.py')
policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)


class ReductionTest(unittest.TestCase):
    def test_mse_gate_is_explicit_and_not_bitwise(self):
        self.assertTrue(policy.mse_regression_ok(.00500000001,.005))
        self.assertFalse(policy.mse_regression_ok(.006,.005))
        self.assertFalse(policy.mse_regression_ok(float('nan'),.005))
        for name in ('benchmark_a100_roof_trace.py','benchmark_a100_roof_candidates.py'):
            text=(ROOT/'scripts'/name).read_text()
            self.assertIn('--allow-reassociation',text)
            self.assertIn('and not args.allow_reassociation',text)
        body=(ROOT/'scripts/roof_reduction_validation.py').read_text()
        self.assertIn('if not reassociated and changed:',body)
        self.assertIn('mse_vs_semantic_fp64',body)

    def test_every_g128_contributes_once_to_static_chain(self):
        for count in (2,4,6,32):
            for chains in (2,4):
                members=[[] for _ in range(chains)]
                for stage in range(count//2):
                    for group in (0,1):
                        chain=group if chains==2 or stage%2==0 else group+2
                        self.assertEqual(chain,(stage*2+group)%chains)
                        members[chain].append(stage*2+group)
                self.assertEqual(sorted(g for c in members for g in c),list(range(count)))
                self.assertTrue(all(c==sorted(c) for c in members))

    def test_association_really_can_change_fp32(self):
        def f(x): return struct.unpack('f',struct.pack('f',x))[0]
        terms=[16777216.,1.,-16777216.,1.]
        ordered=0.
        for term in terms: ordered=f(ordered+term)
        independent=f(f(terms[0]+terms[2])+f(terms[1]+terms[3]))
        self.assertEqual((ordered,independent),(1.,2.))

    def test_candidate_is_isolated_and_only_changes_reduction(self):
        body=(ROOT/'csrc/sm80/o3_reduction_candidate.cuh').read_text()
        self.assertNotIn('__global__',body)
        self.assertIn('cute::append(cute::shape(low),cute::Int<Chains>{})',body)
        self.assertIn('auto acc=chain_acc(cute::_,cute::_,cute::_,chain)',body)
        self.assertIn('const int partial=pl(vi)+16*ph(vi)',body)
        self.assertIn('__fmul_rn(row,column)',body)
        self.assertIn('__fmaf_rn(float(partial),scale,acc(vi,mi,full_ni))',body)
        self.assertIn('__fadd_rn(__fadd_rn(c0(i),c1(i)),__fadd_rn(c2(i),c3(i)))',body)
        roof=(ROOT/'csrc/sm80/roof_candidates.cuh').read_text()
        self.assertLess(roof.index('if(tune>=24)'),roof.index('if(tune>=22)'))
        self.assertIn('select_reduction_kernel',roof)
        self.assertNotIn('#include "o3_reduction_candidate.cuh"',roof)
        self.assertIn('__launch_bounds__(256,2)',(ROOT/'csrc/sm80/roof_reduction.cu').read_text())
        for name in ('roof_candidates.cuh','o1_o3.cu','mixed_benchmark.cuh'):
            self.assertIn('fp32_reassociated',(ROOT/'csrc/sm80'/name).read_text())


if __name__=='__main__': unittest.main()
