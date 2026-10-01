"""Opt-in v54 contracts; hardware correctness is separately tested on A100."""
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]


class VectorPipelineTests(unittest.TestCase):
    def test_fixed_opt_in_and_untouched_timing(self):
        cu=(ROOT/'csrc/sm80/mixed_benchmark.cuh').read_text()
        self.assertIn('conversion_impl<=5',cu)
        self.assertIn('conversion_impl==0 || (roof_tune==59',cu)
        self.assertIn('vector_mixed_fixed(kind',cu)
        self.assertIn('16-byte aligned payloads',cu)
        self.assertLess(cu.index('16-byte aligned payloads'),cu.index('for(int i=0;i<warmup'))
        self.assertIn('totals[j]=elapsed(e.start,e.end)',cu)
        self.assertIn('sum_of_batched_stage_samples',cu)
        self.assertIn('conversion_impl==5?16:0',cu)
        self.assertIn('meta["gemm_math_changed"]=false',cu)

    def test_shared_kernel_and_native_sm80_only(self):
        setup=(ROOT/'setup.py').read_text()
        self.assertGreater(setup.index('"csrc/sm80/roof_vector_conversion.cu"'),setup.index('if target == "sm80"'))
        for name in ('roof_vector_conversion.cu','roof_vector_conversion_probe.cu'):
            self.assertIn('#include "roof_vector_conversion_impl.cuh"',(ROOT/'csrc/sm80'/name).read_text())
        vector=(ROOT/'csrc/sm80/roof_vector_conversion.cu').read_text()
        self.assertIn('GroupedSourceKind::K,16',vector)
        self.assertNotIn('cudaMalloc',vector)

    def test_paired_best_control_and_validation_scripts(self):
        for name in ('audit','validate','benchmark'):
            ast.parse((ROOT/f'scripts/{name}_vector_conversion_pipeline.py').read_text())
        runner=(ROOT/'scripts/benchmark_vector_conversion_pipeline.py').read_text()
        self.assertIn('default=[4,5]',runner)
        self.assertIn("r['round'],4)",runner)
        self.assertIn('incomplete paired vector/scalar coverage',runner)
        self.assertIn('single', (ROOT/'csrc/sm80/mixed_benchmark.cuh').read_text())
        validator=(ROOT/'scripts/validate_vector_conversion_pipeline.py').read_text()
        self.assertIn('torch.cuda.stream(stream)',validator)
        self.assertIn('misaligned_payload',validator)
        self.assertIn('semantic_tolerance_passed=True',validator)


if __name__=='__main__':unittest.main()
