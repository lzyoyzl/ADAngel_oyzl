from pathlib import Path
import random
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from roof_gpu_factor_probe import fast_guard_flag
from o3_fullk_probe import guard_columns,PARTIAL_BOUND,INT32_MAX


class GPUFactorProofTests(unittest.TestCase):
    def test_integer_threshold(self):
        self.assertEqual(INT32_MAX//PARTIAL_BOUND,16383)
        self.assertLessEqual(32*(1<<14),2**32-1)
        # sum=16383 is safe, one additional unit is unsafe.
        self.assertLessEqual(16383*PARTIAL_BOUND,INT32_MAX)
        self.assertGreater(16384*PARTIAL_BOUND,INT32_MAX)

    def test_clamping_does_not_change_decision(self):
        rng=random.Random(60)
        columns=[[rng.randrange(1,255) for _ in range(32)] for _ in range(1000)]
        columns += [[116]+[116+rng.randrange(14) for _ in range(31)] for _ in range(1000)]
        columns += [[116]*31+[116+d] for d in range(20)]
        for c in columns:
            safe=guard_columns([c])[1]['safe']
            self.assertEqual(fast_guard_flag(c)==0,safe)
        self.assertEqual(fast_guard_flag([127]*32),0)
        self.assertTrue(fast_guard_flag([0]*32)&4)
        self.assertTrue(fast_guard_flag([255]*32)&2)


if __name__=='__main__':unittest.main()
