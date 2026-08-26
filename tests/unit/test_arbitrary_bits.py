import unittest

import torch

from adangel.quantization.arbitrary_bits import (
    A8_BIT_WEIGHTS,
    Q4_BIT_WEIGHTS,
    int8_bitplanes_scalar,
    pack_twos_complement_bitplanes,
    q4_bitplanes_scalar,
    reconstruct_from_bitplanes,
    split_int8_scalar,
    split_int8_to_packed_int4,
    unpack_split_int4,
    unpack_twos_complement_bitplanes,
)
from adangel.quantization.mxfp4 import (
    E2M1_TO_Q4,
    e2m1_to_q4,
    mxfp4_to_q4_packed,
    unpack_int4_tensor,
)
from adangel.reference import run_o3_reference, run_o4_reference
from adangel.trace.schema import PreparedInputs, validate_prepared


class TestArbitraryBitSemantics(unittest.TestCase):
    def test_int8_split_is_exact_for_all_values(self):
        for value in range(-128, 128):
            low, high = split_int8_scalar(value)
            self.assertTrue(0 <= low <= 15)
            self.assertTrue(-8 <= high <= 7)
            self.assertEqual(low + 16 * high, value)

    def test_twos_complement_bit_weights_are_exact(self):
        for value in range(-128, 128):
            self.assertEqual(
                reconstruct_from_bitplanes(int8_bitplanes_scalar(value), A8_BIT_WEIGHTS),
                value,
            )
        for value in range(-8, 8):
            self.assertEqual(
                reconstruct_from_bitplanes(q4_bitplanes_scalar(value), Q4_BIT_WEIGHTS),
                value,
            )

    def test_q4_mapping_and_nibble_order(self):
        self.assertEqual(tuple(e2m1_to_q4(code) for code in range(16)), E2M1_TO_Q4)
        codes = torch.arange(16, dtype=torch.uint8).reshape(1, 16)
        packed_e2m1 = codes[:, 0::2] | (codes[:, 1::2] << 4)
        converted = unpack_int4_tensor(mxfp4_to_q4_packed(packed_e2m1))
        self.assertEqual(converted.tolist()[0], list(E2M1_TO_Q4))

    def test_tensor_split_and_bitplane_round_trip(self):
        values = torch.arange(-128, 128, dtype=torch.int16).to(torch.int8).reshape(2, 128)
        packed = split_int8_to_packed_int4(values)
        low, high = unpack_split_int4(packed, rows=2)
        reconstructed = low.to(torch.int16) + 16 * high.to(torch.int16)
        torch.testing.assert_close(reconstructed.to(torch.int8), values, rtol=0, atol=0)

        words = pack_twos_complement_bitplanes(values, 8)
        self.assertEqual(tuple(words.shape), (8, 2, 4))
        torch.testing.assert_close(
            unpack_twos_complement_bitplanes(words, 8), values, rtol=0, atol=0
        )

        q4 = torch.arange(-8, 8, dtype=torch.int8).repeat(2, 8)
        q4_words = pack_twos_complement_bitplanes(q4, 4)
        self.assertEqual(tuple(q4_words.shape), (4, 2, 4))
        torch.testing.assert_close(
            unpack_twos_complement_bitplanes(q4_words, 4), q4, rtol=0, atol=0
        )

    def test_o3_o4_match_direct_g128_integer_reference(self):
        generator = torch.Generator().manual_seed(20250805)
        m, n, k = 5, 7, 128
        a = torch.randint(-127, 128, (m, k), dtype=torch.int8, generator=generator)
        a_scale = torch.rand(m, dtype=torch.float32, generator=generator) + 0.25
        q4 = torch.randint(-8, 8, (n, k), dtype=torch.int8, generator=generator)
        q4_raw = q4.to(torch.int16) & 0xF
        w_q4 = (
            q4_raw[:, 0::2].to(torch.uint8)
            | (q4_raw[:, 1::2].to(torch.uint8) << 4)
        ).contiguous()
        w_scale_g128 = torch.randint(124, 131, (n, 1), dtype=torch.uint8, generator=generator)

        # K32 tensors are part of the common schema but are not consumed by O3/O4.
        inputs = PreparedInputs(
            "small",
            a.contiguous(),
            a_scale.contiguous(),
            torch.zeros((n, k // 2), dtype=torch.uint8),
            torch.full((n, k // 32), 127, dtype=torch.uint8),
            torch.zeros((n, k // 2), dtype=torch.uint8),
            w_scale_g128.contiguous(),
            w_q4,
        )
        validate_prepared(inputs, require_arbitrary_bits=True)
        scales = torch.pow(2.0, w_scale_g128.to(torch.int32) - 127)
        expected = (a.to(torch.int32) @ q4.to(torch.int32).T).to(torch.float32)
        expected *= a_scale[:, None] * scales[:, 0][None, :]

        o3 = run_o3_reference(inputs)
        o4 = run_o4_reference(inputs)
        torch.testing.assert_close(o3["output"], expected, rtol=0, atol=0)
        torch.testing.assert_close(o4["output"], expected, rtol=0, atol=0)
        torch.testing.assert_close(o3["output"], o4["output"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
