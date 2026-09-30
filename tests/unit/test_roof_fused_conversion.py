"""CPU index/host-path contracts; numerical acceptance still requires SM80."""
from pathlib import Path
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[2]


class FusedConversionTests(unittest.TestCase):
    def test_byte_destinations_are_bijective_for_tail_rows_and_odd_groups(self):
        for rows in (1,3,64,129):
            for groups in (1,2,3,5,32):
                dest=[]
                for pair in range(rows*groups*64):
                    row=pair//(groups*64);group=pair//64%groups
                    index=(group*rows+row)*64+pair%64
                    inverse=(index//64%rows)*groups*64+index//(rows*64)*64+index%64
                    self.assertEqual(inverse,pair)
                    dest.append(index)
                self.assertEqual(sorted(dest),list(range(len(dest))))

    def test_fused_variants_reuse_existing_gemm(self):
        text=(ROOT/'csrc/sm80/roof_candidates.cuh').read_text()
        self.assertIn('case 43:return roof_config_for<41>(dual);',text)
        self.assertIn('case 44:return roof_config_for<42>(dual);',text)
        self.assertIn('roof_fused_payload(tune)?tune-2:tune',text)
        self.assertIn('candidate43/44 require source-format four-mode API',text)
        self.assertNotIn('ROOF_PICK(43)',text)
        self.assertNotIn('ROOF_PICK(44)',text)
        fused=(ROOT/'csrc/sm80/roof_fused_conversion.cu').read_text()
        self.assertNotIn('mma.sync',fused)
        self.assertNotIn('cp.async',fused)

    def test_fused_paths_return_before_old_conversion_and_repack(self):
        for name in ('o1_o3.cu','mixed_benchmark.cuh'):
            text=(ROOT/'csrc/sm80'/name).read_text()
            for start,end in (('auto cvw=','auto cva='),('auto cva=','auto gemm=')):
                body=text[text.index(start):text.index(end)]
                self.assertLess(body.index('roof_fused_payload'),body.index('roof_pack_payload'))
                self.assertLess(body.index('return;'),body.index('roof_pack_payload'))
                self.assertNotIn('at::empty',body)
            self.assertIn('diagnostic_inverse_layout_after_timing',text)
            self.assertGreater(text.index('diagnostic_inverse_layout_after_timing'),text.index('timings["total"]'))

    def test_logical_conversion_bytes_do_not_charge_removed_reorder(self):
        fn=runpy.run_path(str(ROOT/'scripts/roof_payload_validation.py'))['payload_reorder_bytes_for_stage']
        for mode,stage in (('conversion_only','total'),('cold','weight_conversion'),('steady_state','activation_conversion')):
            self.assertEqual(fn({'activation_payload_reorder_traffic_bytes':0,'weight_payload_reorder_traffic_bytes':0},mode,stage),0)
        text=(ROOT/'scripts/benchmark_a100_roof_trace.py').read_text()
        self.assertIn('fused conversion candidates43/44 require --all-modes',text)
        self.assertIn('count=conversion_bytes',text)  # Source read + target write are NOT free.

    def test_fused_tu_is_sm80_only(self):
        setup=(ROOT/'setup.py').read_text()
        position=setup.index('"csrc/sm80/roof_fused_conversion.cu"')
        self.assertGreater(position,setup.index('if target == "sm80":'))
        self.assertNotIn('roof_fused_conversion',(ROOT/'csrc/sm120/conversion.cu').read_text())


if __name__=='__main__': unittest.main()
