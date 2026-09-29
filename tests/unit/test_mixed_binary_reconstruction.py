import unittest


def horner(a, w, bits):
    value = 0
    for ab in range(bits-1, -1, -1):
        partial = -(((a >> ab) & 1) & ((w >> 3) & 1))
        for wb in (2, 1, 0):
            partial = 2*partial + (((a >> ab) & 1) & ((w >> wb) & 1))
        value = -partial if ab == bits-1 else 2*value + partial
    return value


class BinaryReconstructionTests(unittest.TestCase):
    def test_all_q8_q4_and_q6_q4_values(self):
        # Includes -128/-32/-8 beyond current source formats' normal maxima.
        for bits in (6, 8):
            for a in range(-(1 << (bits-1)), 1 << (bits-1)):
                for w in range(-8, 8):
                    self.assertEqual(horner(a, w, bits), a*w)

    def test_group_sums_and_intermediate_bound(self):
        import random
        rng = random.Random(20260929)
        for bits in (6, 8):
            for _ in range(32):
                a = [rng.randrange(-(1 << (bits-1)), 1 << (bits-1)) for _ in range(128)]
                w = [rng.randrange(-8, 8) for _ in range(128)]
                value = sum(horner(x, y, bits) for x, y in zip(a, w))
                self.assertEqual(value, sum(x*y for x, y in zip(a, w)))
                self.assertLessEqual(abs(value), 128*128*8)


if __name__ == "__main__":
    unittest.main()
