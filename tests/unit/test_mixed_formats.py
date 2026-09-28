"""Codec contracts, independent of the provenance chosen for formal O5/O6 data."""
import copy
import math
import random
import unittest

from adangel.quantization import mixed_formats as mf
from adangel.quantization.arbitrary_bits import split_int8_scalar
from adangel.quantization.mxfp4 import E2M1_DECODE

try:
    import torch
except ImportError:
    torch = None


class TestMixedScalar(unittest.TestCase):
    def test_known_codebooks(self):
        self.assertEqual(mf.BOOKS["e2m1"], (0, .5, 1, 1.5, 2, 3, 4, 6))
        self.assertEqual(mf.BOOKS["e2m3"][:16], tuple(i / 8 for i in range(16)))
        self.assertEqual(mf.BOOKS["e2m3"][-1], 7.5)
        self.assertEqual(mf.BOOKS["e4m3"][1], 2 ** -9)
        self.assertEqual(mf.BOOKS["e4m3"][126], 448)
        self.assertEqual(mf.BOOKS["e6m2"][0], 2 ** -48)
        self.assertEqual(mf.BOOKS["e6m2"][192], 1)
        self.assertEqual(mf.BOOKS["e6m2"][254], 49152)

    def test_every_finite_code_and_midpoint(self):
        for kind, levels in mf.BOOKS.items():
            for sign in ((1, -1) if kind in mf.SIGN_BITS else (1,)):
                sign_bit = mf.SIGN_BITS.get(kind, 0) if sign < 0 else 0
                for code, value in enumerate(levels):
                    self.assertEqual(mf.decode_scalar(code | sign_bit, kind), sign * value)
                    # Only exact negative zero is canonicalized by the encoder.
                    expected = code | (sign_bit if value else 0)
                    self.assertEqual(mf.encode_scalar(sign * value, kind), expected)
                for code, (a, b) in enumerate(zip(levels, levels[1:])):
                    midpoint = (a + b) / 2
                    expected = (code + (code & 1)) | sign_bit
                    self.assertEqual(mf.encode_scalar(sign * midpoint, kind), expected)
                    self.assertEqual(mf.encode_scalar(sign * math.nextafter(midpoint, 0), kind),
                                     code | sign_bit)
                    self.assertEqual(mf.encode_scalar(sign * math.nextafter(midpoint, math.inf), kind),
                                     (code + 1) | sign_bit)

    def test_invalid_codes_and_values(self):
        for kind in mf.BOOKS:
            for value in (math.nan, math.inf, -math.inf):
                with self.assertRaises(ValueError):
                    mf.encode_scalar(value, kind)
            with self.assertRaises(ValueError):
                mf.decode_scalar(-1, kind)
        for kind, code in (("e2m1", 16), ("e2m3", 64), ("e4m3", 127),
                           ("e4m3", 255), ("e6m2", 255)):
            with self.assertRaises(ValueError):
                mf.decode_scalar(code, kind)

    def test_fixed_e2m1_matches_o3(self):
        expected = (0, 0, 1, 2, 2, 3, 4, 6, 0, 0, -1, -2, -2, -3, -4, -6)
        self.assertEqual(tuple(mf.fixed_scalar(v, 4, 0) for v in E2M1_DECODE), expected)

    def test_fixed_range_scale_and_sign_extension(self):
        for kind, bits, fractional in mf.FORMATS.values():
            limit = 2 ** (bits - 1) - 1
            for value in mf.BOOKS[kind]:
                for sign in (-1, 1):
                    q = mf.fixed_scalar(sign * value, bits, fractional)
                    self.assertLessEqual(abs(q), limit)
                    self.assertLessEqual(abs(q * 2.0 ** -fractional - sign * value),
                                         .5 * 2.0 ** -fractional)
                    lo, hi = split_int8_scalar(q)
                    self.assertEqual(lo + 16 * hi, q)
        self.assertEqual(mf.fixed_scalar(448, 8, -2), 112)
        self.assertEqual(mf.fixed_scalar(-7.5, 6, 2), -30)
        self.assertEqual((-30) & 255, 226)  # INT6 100010 -> INT8 11100010.
        self.assertEqual(mf.fixed_scalar(.125, 6, 2), 0)
        self.assertEqual(mf.fixed_scalar(.375, 6, 2), 2)
        self.assertEqual(mf.fixed_scalar(1e6, 4, 0), 7)
        self.assertEqual(mf.fixed_scalar(-1e6, 4, 0), -7)

    def test_hif4_known_blocks(self):
        codes, scale, e8, e4 = mf.hif4_block_scalar([0] * 128)
        self.assertEqual((codes, scale, e8, e4), ([0] * 128, 0, [0] * 16, [0] * 32))
        codes, scale, e8, e4 = mf.hif4_block_scalar([7, -7] * 64)
        self.assertEqual(codes, [7, 15] * 64)
        self.assertEqual((scale, e8, e4), (192, [1] * 16, [1] * 32))
        # Force scale=1 but expose the leaf half-up tie at 0.125.
        values = [0.125] * 120 + [7] * 8
        codes, scale, e8, e4 = mf.hif4_block_scalar(values)
        self.assertEqual(codes[:120], [1] * 120)
        self.assertEqual(scale, 192)
        self.assertEqual(e8, [0] * 15 + [1])
        self.assertEqual(e4, [0] * 30 + [1, 1])


@unittest.skipIf(torch is None, "requires PyTorch; scalar tests still run without it")
class TestMixedTorch(unittest.TestCase):
    def devices(self):
        return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    def test_encoder_all_midpoints_against_scalar(self):
        for device in self.devices():
            for kind, levels in mf.BOOKS.items():
                mids = [(a + b) / 2 for a, b in zip(levels, levels[1:])]
                values = list(levels) + mids
                if kind != "e6m2":
                    values += [-v for v in values]
                x = torch.tensor(values, dtype=torch.float32, device=device)
                encoded = mf._encode_tensor(x, kind)
                self.assertEqual(encoded.cpu().tolist(), [mf.encode_scalar(v, kind) for v in values])
                decoded = mf._decode_tensor(encoded, kind)
                self.assertEqual(decoded.cpu().tolist(),
                                 [mf.decode_scalar(c, kind) for c in encoded.cpu().tolist()])

    def test_e4m3_against_torch_native_float8(self):
        for device in self.devices():
            levels = mf.BOOKS["e4m3"]
            values = list(levels) + [(a + b) / 2 for a, b in zip(levels, levels[1:])]
            values += [-v for v in values if v != 0]
            x = torch.tensor(values, dtype=torch.float32, device=device)
            expected = x.to(torch.float8_e4m3fn).view(torch.uint8)
            self.assertTrue(torch.equal(mf._encode_tensor(x, "e4m3"), expected))

    def test_nibble_bit_and_split_storage(self):
        from adangel.quantization.arbitrary_bits import unpack_split_int4
        for device in self.devices():
            nibble = torch.arange(16, dtype=torch.uint8, device=device).reshape(2, 8)
            packed = mf._pack_nibbles(nibble)
            self.assertEqual(packed.cpu().flatten().tolist(), [0x10, 0x32, 0x54, 0x76,
                                                               0x98, 0xba, 0xdc, 0xfe])
            self.assertTrue(torch.equal(mf._unpack_nibbles(packed), nibble))
            bits = (torch.arange(128, device=device).reshape(2, 4, 16) % 3 == 0).byte()
            self.assertTrue(torch.equal(mf._unpack_bits(mf._pack_bits(bits)), bits))
            w = mf.quantize_source(torch.linspace(-7, 7, 256, device=device).reshape(2, 128), "hif4_g128")
            a = mf.quantize_source(torch.linspace(-8, 8, 384, device=device).reshape(3, 128),
                                   "nvstyle_fp6_e2m3_g128")
            prepared = mf.prepare_integer_reference(w, a)
            low, high = unpack_split_int4(prepared["A_split"], 3)
            aq, asc = mf.to_fixed_reference(a)
            wq, ws = mf.to_fixed_reference(w)
            self.assertTrue(torch.equal(low.short() + 16 * high.short(), aq.short()))
            raw_w = mf._unpack_nibbles(prepared["W_q4"]).short()
            self.assertTrue(torch.equal(torch.where(raw_w >= 8, raw_w - 16, raw_w), wq.short()))
            self.assertTrue(torch.equal(prepared["A_scale"], asc))
            self.assertTrue(torch.equal(prepared["W_scale"], ws))

    def test_hif4_matches_scalar_with_nonuniform_micro_exponents(self):
        rng = random.Random(5606)
        blocks = [[0] * 128, [7, -7] * 64, [.125] * 120 + [7] * 8]
        blocks += [[rng.uniform(-1, 1) * 2 ** (i // 4 % 9 - 5) for i in range(128)]
                   for _ in range(5)]
        for device in self.devices():
            x = torch.tensor(blocks, dtype=torch.float32, device=device).reshape(2, 512)
            source = mf.quantize_source(x, "hif4_g128")
            codes = mf._unpack_nibbles(source["payload"]).reshape(-1, 128).cpu().tolist()
            e8 = mf._unpack_bits(source["micro8"]).reshape(-1, 16).cpu().tolist()
            e4 = mf._unpack_bits(source["micro4"]).reshape(-1, 32).cpu().tolist()
            scales = source["scale"].flatten().cpu().tolist()
            for index, values in enumerate(blocks):
                self.assertEqual((codes[index], scales[index], e8[index], e4[index]),
                                 mf.hif4_block_scalar(values))

    def test_mxfp8_upward_scale_boundary(self):
        for device in self.devices():
            # Just above 448 needs scale=2, not scale=1; zero block uses 1.
            maxima = torch.tensor([0., 224., 448., 449., 896.], device=device)
            source = mf.quantize_source(maxima[:, None].expand(-1, 128), "mxfp8_e4m3_g128")
            self.assertEqual(source["scale"].flatten().cpu().tolist(), [127, 126, 127, 128, 128])
            _, fixed_scale = mf.to_fixed_reference(source)
            self.assertEqual(fixed_scale.flatten().cpu().tolist(), [4., 2., 4., 8., 8.])

    def test_nv_two_level_scale_and_underflow(self):
        for device in self.devices():
            for fmt, max_local in (("nvfp4_g128", 6), ("nvstyle_fp6_e2m3_g128", 7.5)):
                x = torch.tensor([max_local * 448, 1e-7, 0.], device=device)
                source = mf.quantize_source(x[:, None].expand(-1, 128), fmt)
                self.assertEqual(source["tensor_scale"].item(), 1.)
                self.assertEqual(source["scale"].flatten().cpu().tolist(), [126, 0, 0])
                self.assertEqual(source["payload"][1:].count_nonzero().item(), 0)
                self.assertEqual(mf.dequantize_source(source)[0, 0].item(), max_local * 448)

    def test_source_and_fixed_reconstruction_bounds(self):
        for device in self.devices():
            torch.manual_seed(5605)
            x = torch.randn((3, 512), device=device) * 3
            for fmt in mf.FORMATS:
                source = mf.quantize_source(x, fmt)
                mf.validate_source(source)
                local, scale = mf.source_components(source)
                q, effective = mf.to_fixed_reference(source)
                _, bits, fractional = mf.FORMATS[fmt]
                self.assertEqual(q.dtype, torch.int8)
                self.assertTrue(q.is_contiguous())
                self.assertLessEqual(q.abs().max().item(), 2 ** (bits - 1) - 1)
                expected_q = torch.tensor([mf.fixed_scalar(v, bits, fractional)
                                           for v in local.flatten().cpu().tolist()], device=device)
                self.assertTrue(torch.equal(q.flatten().long(), expected_q))
                self.assertTrue(torch.equal(effective, scale * 2.0 ** -fractional))
                decoded = mf.dequantize_source(source).reshape(3, 4, 128)
                reconstructed = q.reshape(3, 4, 128).float() * effective[..., None]
                bound = .5 * effective[..., None] + 2e-6
                self.assertTrue(torch.all((reconstructed - decoded).abs() <= bound))
                zero = mf.quantize_source(torch.zeros_like(x), fmt)
                self.assertEqual(mf.dequantize_source(zero).count_nonzero().item(), 0)
                self.assertEqual(mf.to_fixed_reference(zero)[0].count_nonzero().item(), 0)

    def test_invalid_source_contract(self):
        for fmt in mf.FORMATS:
            source = mf.quantize_source(torch.ones((2, 256)), fmt)
            for field, value in (("version", 99), ("group_size", 32), ("shape", [2, 255]),
                                 ("payload", source["payload"].float()),
                                 ("scale", torch.full_like(source["scale"], 255))):
                bad = dict(source, **{field: value})
                with self.assertRaises(ValueError, msg=f"{fmt}/{field}"):
                    mf.validate_source(bad)
            if "tensor_scale" in source:
                for value in (0., -1., math.inf, math.nan):
                    with self.assertRaises(ValueError):
                        mf.validate_source(dict(source, tensor_scale=torch.tensor([value])))
            if fmt == "mxfp8_e4m3_g128":
                bad = copy.deepcopy(source)
                bad["payload"][0, 0] = 127
                with self.assertRaises(ValueError):
                    mf.validate_source(bad)
                bad = copy.deepcopy(source)
                bad["scale"].fill_(254)  # UE8M0 finite; compensating *4 overflows.
                with self.assertRaises(ValueError):
                    mf.to_fixed_reference(bad)
            if fmt == "nvstyle_fp6_e2m3_g128":
                bad = copy.deepcopy(source)
                bad["payload"][0, 0] = 64
                with self.assertRaises(ValueError):
                    mf.validate_source(bad)
        for x in (torch.ones(2, 127), torch.ones(2, 128, dtype=torch.int8),
                  torch.full((2, 128), math.inf), torch.full((2, 128), math.nan)):
            with self.assertRaises(ValueError):
                mf.quantize_source(x, "nvfp4_g128")

    def test_mixed_pair_contract(self):
        w = mf.quantize_source(torch.ones(2, 128), "nvfp4_g128")
        a = mf.quantize_source(torch.ones(3, 128), "mxfp8_e4m3_g128")
        mf.prepare_integer_reference(w, a)
        for bad in (w, mf.quantize_source(torch.ones(3, 128), "nvstyle_fp6_e2m3_g128"),
                    mf.quantize_source(torch.ones(3, 256), "mxfp8_e4m3_g128")):
            with self.assertRaises(ValueError):
                mf.prepare_integer_reference(w, bad)


if __name__ == "__main__":
    unittest.main()
