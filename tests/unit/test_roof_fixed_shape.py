"""Compile-time shape isolation contracts; GPU correctness/performance separate."""
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]


class FixedShapeContract(unittest.TestCase):
    def test_same_body_only_dimensions_specialized(self):
        old=(ROOT/'csrc/sm80/roof_reuse_pipeline.cu').read_text()
        new=(ROOT/'csrc/sm80/roof_fixed_shape.cu').read_text()
        start='o3_reuse_pipeline_experiment::o3_body<'
        old_call=old[old.index(start):old.index(';',old.index(start))]
        new_call=new[new.index(start):new.index(';',new.index(start))]
        self.assertEqual(old_call.replace('(a,w,as,ws,y,m,n,k)',
                                         '(a,w,as,ws,y,4096,4096,4096)'),new_call)
        self.assertIn('__launch_bounds__(128,3)',new)
        self.assertIn('Stages=Tune==32?2:3',new)
        self.assertIn('#include "o3_reuse_pipeline_candidate.cuh"',new)
        self.assertIn('"csrc/sm80/roof_fixed_shape.cu"',(ROOT/'setup.py').read_text())

    def test_all_host_entries_guard_and_label_shape(self):
        host=(ROOT/'csrc/sm80/roof_candidates.cuh').read_text()
        self.assertIn('(tune!=32 && tune!=33) || (m==4096 && n==4096 && k==4096)',host)
        self.assertIn('check_roof_fixed_shape(tune,m64,n64,k64)',host)
        self.assertLess(host.index('select_fixed_shape_kernel(dual,fast,tune)'),
                        host.index('select_reduction_budget_kernel(dual,fast,tune)'))
        self.assertNotIn('ROOF_PICK(32)',host)
        self.assertNotIn('ROOF_PICK(33)',host)
        for name in ('o1_o3.cu','mixed_benchmark.cuh'):
            source=(ROOT/'csrc/sm80'/name).read_text()
            self.assertIn('check_roof_fixed_shape(roof_tune,m,n,k)',source)
            self.assertIn('meta["compile_time_shape"]',source)
            self.assertIn('meta["fp32_reassociated"]=roof_tune>=24 && roof_tune<=27',source)

    def test_gpu_validation_includes_full_shape_and_rejections(self):
        source=(ROOT/'scripts/benchmark_a100_roof_candidates.py').read_text()
        self.assertIn('shapes += [(4096,4096,4096)]',source)
        self.assertIn("'fixed4096 candidate32/33 requires' not in str(error)",source)
        self.assertIn('rejected_shapes.append',source)
        self.assertIn('torch.equal(y.view(torch.int32), expected.view(torch.int32))',source)
        self.assertIn('33)EE',(ROOT/'scripts/audit_a100_o1.py').read_text())


if __name__=='__main__': unittest.main()
