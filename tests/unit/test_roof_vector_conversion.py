"""v53 mapping and isolation contracts; GPU tests validate actual CUDA values."""
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]


class VectorConversionTests(unittest.TestCase):
    def test_each_element_and_scale_written_once(self):
        for elements in (8,16):
            lanes=128//elements;tile=256//lanes
            for rows in (1,3,7,16,17,32,33,129):
                for groups in (1,3,5):
                    src=[];dst=[];sc=[]
                    for g in range(groups):
                        for bx in range((rows+tile-1)//tile):
                            for tid in range(256):
                                row=bx*tile+tid//lanes;i=tid%lanes
                                if row>=rows:continue
                                src.extend((row*groups+g)*128+i*elements+j for j in range(elements))
                                dst.extend((g*rows+row)*128+i*elements+j for j in range(elements))
                                if i==0:sc.append(g*rows+row)
                    self.assertEqual(sorted(src),list(range(rows*groups*128)))
                    self.assertEqual(sorted(dst),list(range(rows*groups*128)))
                    self.assertEqual(sorted(sc),list(range(rows*groups)))

    def test_signed_nibbles_reconstruct_all_int8(self):
        for x in range(-128,128):
            low=x&15; high=(x>>4)&15
            signed_high=high-16 if high&8 else high
            self.assertEqual(low+16*signed_high,x)

    def test_isolated_scalar_best_and_ordered_scales(self):
        cu=(ROOT/'csrc/sm80/roof_vector_conversion_probe.cu').read_text()
        self.assertIn('if(kind==Kind::Mx8)',cu)
        self.assertIn('integer_mixed_fixed(kind,true',cu)
        self.assertIn('value=__fmul_rn(e4m3(c),tensor_scale[0])',cu)
        self.assertIn('value=__fmul_rn(value,.25f)',cu)
        self.assertIn('times[r]=elapsed/inner',cu)
        self.assertLess(cu.index('Events event;'),cu.index('for(int r=0;r<repeats'))
        self.assertNotIn('mma.sync',cu)
        self.assertNotIn('roof_vector_conversion_probe.cu',(ROOT/'setup.py').read_text())
        host=(ROOT/'scripts/vector_conversion_probe.py').read_text()
        for guard in ('mf.validate_source(source)','int(source[\'scale\'].max()) > 252','effective scale overflow'):
            self.assertIn(guard,host)

    def test_script_syntax_and_paired_rotation(self):
        for name in ('build','audit','validate','benchmark'):
            ast.parse((ROOT/f'scripts/{name}_vector_conversion_probe.py').read_text())
        text=(ROOT/'scripts/benchmark_vector_conversion_probe.py').read_text()
        self.assertIn('incomplete paired coverage',text)
        self.assertIn('library_sha256',text)
        self.assertIn('output_bitwise_current_best=True',text)
        self.assertIn('mse_vs_paired_fp16=mse(y,fp)',text)


if __name__=='__main__':unittest.main()
