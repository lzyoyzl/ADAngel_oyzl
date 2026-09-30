"""Exact scale multiplication identity; does not replace native GPU validation."""
import math
import struct
import unittest


def f32(bits):
    return struct.unpack('<f',struct.pack('<I',bits))[0]


class Power2ScaleExactTest(unittest.TestCase):
    def test_all_normal_exponent_pairs_and_boundary_mantissas(self):
        checked=0
        for ae in range(1,255):
            for we in range(1,255):
                result_exp=ae+we-127
                if not 1<=result_exp<=254:
                    continue  # native guard must use generic path here
                for mantissa in (0,1,0x3fffff,0x7ffffe,0x7fffff):
                    a_bits=ae<<23;w_bits=(we<<23)|mantissa
                    # FP64 multiplication by an exact power2 preserves all
                    # 24 significand bits and exponent range of this corpus.
                    expected=struct.pack('<f',f32(a_bits)*f32(w_bits))
                    actual=struct.pack('<I',w_bits+a_bits-0x3f800000)
                    self.assertEqual(actual,expected)
                    self.assertTrue(math.isfinite(f32(w_bits+a_bits-0x3f800000)))
                    checked+=1
        self.assertGreater(checked,200000)


if __name__=='__main__': unittest.main()
