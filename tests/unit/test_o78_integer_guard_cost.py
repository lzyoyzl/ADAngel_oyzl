import json
from pathlib import Path
import random
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from analyze_o78_integer_guard_cost import ceil_sqrt,rooted_ctas,work_budget,analyze,LIMIT


class GuardCostTests(unittest.TestCase):
    def test_integer_root_bounds(self):
        rng=random.Random(65)
        values=list(range(1000))+[2**64-1,2**128,2**128+1]
        values += [rng.randrange(2**80) for _ in range(1000)]
        for n in values:
            r=ceil_sqrt(n)
            self.assertGreaterEqual(r*r,n)
            if r:self.assertLess((r-1)*(r-1),n)
        for n in (-1,1.0):
            with self.assertRaises(ValueError):ceil_sqrt(n)

    def test_guard_conservative_at_threshold_and_factor_overflow(self):
        a=dict(norm2=[LIMIT**2,0],factor_max=[1,1])
        w=dict(norm2=[1,2],factor_max=[1,1])
        self.assertEqual(rooted_ctas(a,w,1,1)['unsafe_cta_coordinates'],[[0,1]])
        a['factor_max'][1]=2**32
        self.assertEqual(rooted_ctas(a,w,1,1)['unsafe_cta_coordinates'],[[0,1],[1,0],[1,1]])

    def test_archived_analysis_reproduces_all_values(self):
        expected=json.loads((ROOT/'docs/evidence/a100_o378_roof_v65/guard_cost.json').read_text())
        result=analyze()
        self.assertEqual(result,expected)
        self.assertEqual(len(result['cases']),48)
        self.assertEqual(result['additional_rejected_ctas'],0)
        for row in result['work_budget']:
            self.assertEqual(row['hypothetical_reduction_before_epilogue_and_guard'],16777216)
            self.assertEqual(row['unchanged_IMMA_warp_instructions'],16777216)
            self.assertEqual(row['final_I2F_warp_instructions'],524288)
