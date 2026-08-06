import math
import random
import unittest

from adangel.quantization.mxfp4 import (
    E2M1_DECODE,
    E2M1_TO_INT8_BASE,
    choose_ue8m0_scale_code,
    decode_e2m1,
    decode_ue8m0,
    dequantize_mxfp4_block,
    e2m1_to_int8_base,
    encode_e2m1,
    pack_e2m1,
    pack_mma_scale_fragments,
    quantize_mxfp4_block,
    unpack_e2m1,
)


class TestMXFP4Codec(unittest.TestCase):
    def test_all_sixteen_encodings(self):
        for code, expected in enumerate(E2M1_DECODE):
            self.assertEqual(decode_e2m1(code), expected)

    def test_round_ties_to_even(self):
        cases = {0.25: 0, 0.75: 2, 1.25: 2, 1.75: 4, 2.5: 4, 3.5: 6, 5.0: 6}
        for value, code in cases.items():
            self.assertEqual(encode_e2m1(value), code)
            self.assertEqual(encode_e2m1(-value), code | 8)

    def test_saturation_and_zero(self):
        self.assertEqual(encode_e2m1(100.0), 7)
        self.assertEqual(encode_e2m1(-100.0), 15)
        self.assertEqual(encode_e2m1(-0.0), 0)

    def test_exact_o1_mapping(self):
        self.assertEqual(tuple(e2m1_to_int8_base(i) for i in range(16)), E2M1_TO_INT8_BASE)

    def test_pack_order(self):
        codes = list(range(16))
        self.assertEqual(unpack_e2m1(pack_e2m1(codes)), codes)
        self.assertEqual(pack_e2m1([1, 2]), bytes([0x21]))

    def test_ue8m0_normal_zero_overflow_underflow(self):
        self.assertEqual(decode_ue8m0(127), 1.0)
        self.assertEqual(choose_ue8m0_scale_code(0.0), 127)
        self.assertEqual(choose_ue8m0_scale_code(4.0), 127)
        self.assertEqual(choose_ue8m0_scale_code(8.0), 128)
        self.assertEqual(choose_ue8m0_scale_code(math.ldexp(1.0, 200)), 254)
        self.assertEqual(choose_ue8m0_scale_code(math.ldexp(1.0, -200)), 0)
        self.assertTrue(math.isnan(decode_ue8m0(255)))

    def test_mma_scale_fragments_are_separate(self):
        sfa = [[10 * row, 10 * row + 1] for row in range(16)]
        sfb = [[200 + col for col in range(8)], [220 + col for col in range(8)]]
        scale_a, scale_b = pack_mma_scale_fragments(sfa, sfb)
        for q in range(8):
            self.assertEqual(scale_a[4 * q] & 0xFFFF, sfa[q][0] | (sfa[q][1] << 8))
            self.assertEqual(
                scale_a[4 * q + 1] & 0xFFFF,
                sfa[q + 8][0] | (sfa[q + 8][1] << 8),
            )
            self.assertEqual(scale_b[4 * q] & 0xFFFF, sfb[0][q] | (sfb[1][q] << 8))
            self.assertEqual(scale_b[4 * q + 1], 0)
            self.assertEqual(scale_a[4 * q + 2], 0)
            self.assertEqual(scale_b[4 * q + 3], 0)

    def test_block_edge_patterns(self):
        patterns = [
            [0.0] * 32,
            [6.0] * 32,
            [6.0 if index % 2 else -6.0 for index in range(32)],
            [1.0e-40] * 32,
            [1.0e30] * 32,
        ]
        rng = random.Random(9)
        patterns.append([rng.uniform(-10.0, 10.0) for _ in range(32)])
        for values in patterns:
            codes, scale = quantize_mxfp4_block(values)
            decoded = dequantize_mxfp4_block(codes, scale)
            self.assertEqual(len(codes), 32)
            self.assertTrue(all(0 <= code <= 15 for code in codes))
            self.assertTrue(all(math.isfinite(value) for value in decoded))
        self.assertEqual(quantize_mxfp4_block([0.0] * 32)[1], 127)

    def test_nan_inf_rejected(self):
        with self.assertRaises(ValueError):
            quantize_mxfp4_block([math.nan] + [0.0] * 31)
        with self.assertRaises(ValueError):
            quantize_mxfp4_block([math.inf] + [0.0] * 31)


if __name__ == "__main__":
    unittest.main()
