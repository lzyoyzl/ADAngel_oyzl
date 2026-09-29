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
    def test_k256_shared_swizzle_preserves_cpasync_chunks_and_fragment_words(self):
        # Address model only; CUDA correctness/sanitizer/NCU remain required.
        for rows in (64, 128):
            physical = {}
            for row in range(rows):
                addresses = [row*8+(word^(row&4)) for word in range(8)]
                self.assertEqual(set(addresses), set(range(row*8, row*8+8)))
                for chunk in (0, 4):
                    start = row*8+(chunk^(row&4))
                    self.assertEqual(start % 4, 0)
                    self.assertEqual(addresses[chunk:chunk+4], list(range(start,start+4)))
                for word, address in enumerate(addresses):
                    physical[address] = (row,word)
            for base in range(0,rows,8):
                for g in (0,1):
                    banks = []
                    for lane in range(32):
                        row, word = base+lane//4, g*4+lane%4
                        address = row*8+(word^(row&4))
                        self.assertEqual(physical[address], (row,word))
                        banks.append(address % 32)
                    self.assertEqual(len(set(banks)),32)

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
