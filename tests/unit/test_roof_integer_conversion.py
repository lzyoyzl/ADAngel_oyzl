"""CPU contracts for the independent v34 CUDA conversion probe."""
from pathlib import Path
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[2]
P=runpy.run_path(str(ROOT/'scripts/probe_integer_fixed_conversion.py'))


class IntegerConversionTests(unittest.TestCase):
    def test_nv4_register_lut_is_exact(self):
        for code in range(16):
            value=(0x64322100>>(4*(code&7)))&15
            q=-value if code&8 else value
            self.assertEqual(q,P['integer_fixed'](code,'e2m1'))

    def test_tiled_traversal_exactly_once_and_no_padding_reads(self):
        for rows in (1,3,4,7,64,129):
            for groups in (1,2,3,5,32):
                src=[];dst=[];scales=[]
                for block_y in range(groups):
                    for block_x in range((rows+3)//4):
                        for lane in range(256):
                            row=block_x*4+lane//64
                            if row>=rows: continue
                            i=lane%64
                            src.append((row*groups+block_y)*64+i)
                            dst.append((block_y*rows+row)*64+i)
                            if i==0: scales.append(block_y*rows+row)
                self.assertEqual(sorted(src),list(range(rows*groups*64)))
                self.assertEqual(sorted(dst),list(range(rows*groups*64)))
                self.assertEqual(sorted(scales),list(range(rows*groups)))

    def test_no_payload_float_round_or_production_dispatch(self):
        cu=(ROOT/'csrc/sm80/roof_integer_conversion.cu').read_text()
        self.assertNotIn('__float2int_rn',cu)
        self.assertIn('payload_nv4(byte&15)',cu)
        self.assertIn('__fmul_rn(ue8m0_scale(scale[group]),4.f)',cu)
        self.assertIn('value=__fmul_rn(value,.25f)',cu)
        self.assertNotIn('mma.sync',cu)
        host=(ROOT/'csrc/sm80/mixed_conversion_probe.cuh').read_text()
        self.assertIn('implementation!=3 || s.k/128<=65535',host)
        self.assertIn('MixedSource s(source)',host)
        self.assertIn('roof_pack_payload(d.packed,packed',host)
        self.assertLess(host.index('std::vector<ConversionPair>'),host.index('for(int j=0;j<warmup'))
        self.assertLess(host.index('at::empty'),host.index('auto convert='))
        self.assertIn('/inner',host)
        self.assertNotIn('integer_mixed_fixed',(ROOT/'csrc/sm80/mixed_benchmark.cuh').read_text())
        setup=(ROOT/'setup.py').read_text()
        self.assertGreater(setup.index('csrc/sm80/roof_integer_conversion.cu'),setup.index('if target == "sm80":'))


if __name__=='__main__': unittest.main()
