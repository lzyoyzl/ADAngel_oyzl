"""Static phase scheduling keeps the dynamic tree's mathematical order."""
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]


class StaticTreeTest(unittest.TestCase):
    def test_visit_order_slot_and_tail_coverage(self):
        for groups in range(1,66):
            for window in (2,4):
                dynamic=[(g,g%2,g%window) for g in range(groups)]
                static=[(base+p,(base+p)%2,p) for base in range(0,groups,window)
                        for p in range(window) if base+p<groups]
                self.assertEqual(static,dynamic)

    def test_independent_source_compile_time_phase_and_exact_math(self):
        source=(ROOT/'csrc/sm80/o3_static_eager_tree_candidate.cuh').read_text()
        finish=source.split('auto finish=',1)[1].split('if constexpr(RoofTune&2)',1)[0]
        self.assertNotIn('stage%4',finish)
        for phase in (0,1,2): self.assertIn(f'if constexpr(decltype(phase)::value=={phase})',finish)
        self.assertIn('const float product=__fmul_rn(float(partial),scale)',finish)
        self.assertNotIn('__fmaf_rn',finish)
        self.assertIn('if(stage<k/K) process_stage(stage,stage%Stages,phase)',source)
        self.assertIn('const int tail=(k/K)%Window',source)
        self.assertIn('if constexpr(Window==2) remaining=leaf(i)',source)
        old=(ROOT/'csrc/sm80/o3_eager_tree_candidate.cuh').read_text()
        self.assertIn('if(stage%4==0)',old)
        wrapper=(ROOT/'csrc/sm80/roof_static_eager_tree.cu').read_text()
        self.assertIn('__launch_bounds__(128,1)',wrapper)
        self.assertIn('static_assert(Tune==37 || Tune==38)',wrapper)
        self.assertIn('"csrc/sm80/roof_static_eager_tree.cu"',(ROOT/'setup.py').read_text())

    def test_opt_in_and_bitwise_dynamic_tree_guard(self):
        policy=(ROOT/'scripts/roof_reduction_validation.py').read_text()
        self.assertIn('tree_baseline is None',policy)
        self.assertIn('torch.equal(y.view(torch.int32),tree_baseline.view(torch.int32))',policy)
        for name in ('roof_candidates.cuh','o1_o3.cu','mixed_benchmark.cuh'):
            self.assertIn('meta["compile_time_reduction_phase"]=',(ROOT/'csrc/sm80'/name).read_text())
        compare=(ROOT/'scripts/compare_roof_trace_candidates.py').read_text()
        self.assertIn("r['tune']==37 and not r.get('bitwise_equal_dynamic_tree')",compare)


if __name__=='__main__': unittest.main()
