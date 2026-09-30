"""Layout candidate contracts; CPU index proofs do not replace GPU validation."""
from pathlib import Path
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[2]


class PayloadLayoutTests(unittest.TestCase):
    def test_vector_permutation_is_bijective_and_preserves_planes(self):
        for planes in (1,2):
            for rows in (64,128,192):
                for groups in (1,2,3,5,32):
                    indices=[]
                    for i in range(planes*rows*groups*4):
                        span=rows*groups*4
                        plane,local=divmod(i,span)
                        group=local//(rows*4)
                        row=local//4%rows
                        col=local%4
                        src=(plane*rows+row)*groups*4+group*4+col
                        self.assertEqual(src//span,plane)
                        self.assertEqual((src%span)//(groups*4),row)
                        indices.append(src)
                    self.assertEqual(sorted(indices),list(range(planes*rows*groups*4)))

    def test_compute_body_only_changes_global_address_layout(self):
        old=(ROOT/'csrc/sm80/o3_reuse_pipeline_candidate.cuh').read_text().splitlines()[1:]
        new=(ROOT/'csrc/sm80/o3_grouped_payload_candidate.cuh').read_text().splitlines()[1:]
        new='\n'.join(new).replace('o3_grouped_payload_experiment','o3_reuse_pipeline_experiment')
        new=new.replace('a+stage*m*C::Bytes+(blockIdx.y*M+row)*C::Bytes+col',
                        'a+(blockIdx.y*M+row)*(k/2)+stage*C::Bytes+col')
        new=new.replace('w+stage*total_n*C::Bytes+(blockIdx.x*N+row)*C::Bytes+col',
                        'w+(blockIdx.x*N+row)*(k/2)+stage*C::Bytes+col')
        self.assertEqual(new,'\n'.join(old))

    def test_reorder_in_both_conversion_paths_and_explicit_extra_traffic(self):
        for file in ('o1_o3.cu','mixed_benchmark.cuh'):
            source=(ROOT/'csrc/sm80'/file).read_text()
            weight=source[source.index('auto cvw='):source.index('auto cva=')]
            activation=source[source.index('auto cva='):source.index('auto gemm=')]
            self.assertIn('roof_pack_payload',weight)
            self.assertIn('roof_pack_payload',activation)
            self.assertNotIn('at::empty',weight+activation)
            self.assertIn('packed_activation_g128_major',source)
            self.assertIn('packed_weight_g128_major',source)
        fn=runpy.run_path(str(ROOT/'scripts/roof_payload_validation.py'))['payload_reorder_bytes_for_stage']
        meta=dict(weight_payload_reorder_traffic_bytes=4096**2,
                  activation_payload_reorder_traffic_bytes=2*4096**2)
        self.assertEqual(fn(meta,'conversion_only','total'),3*4096**2)
        self.assertEqual(fn(meta,'cold','weight_conversion'),4096**2)
        self.assertEqual(fn(meta,'steady_state','activation_conversion'),2*4096**2)
        self.assertEqual(fn(meta,'cold','total'),0)
        self.assertEqual(fn(meta,'compute_only','gemm'),0)
        self.assertEqual(fn({},'conversion_only','total'),0)

    def test_odd_groups_and_native_three_stage_audit_remain_enabled(self):
        audit=runpy.run_path(str(ROOT/'scripts/audit_a100_o1.py'))
        check=audit['roof_pipeline_checks']('adangel_sm80_roof_candidateILb1ELb0ELi42EE',
            'cp.async.wait_group 1; cp.async.wait_group 0;',
            'DEPBAR.LE SB0, 0x1; DEPBAR.LE SB0, 0x0;')
        self.assertTrue(check and all(check.values()))
        for script in ('benchmark_a100_roof_candidates.py','benchmark_a100_roof_trace.py'):
            self.assertIn('verify_grouped_payload',(ROOT/'scripts'/script).read_text())


if __name__=='__main__': unittest.main()
