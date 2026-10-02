import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('o3_fullk_probe',ROOT/'scripts/o3_fullk_probe.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class FullKTests(unittest.TestCase):
    def test_guard_and_signed_prefixes(self):
        columns=[[116+i%5 for i in range(32)],[121]*32]
        anchors,guard=module.guard_columns(columns)
        self.assertEqual(anchors,[116,121]); self.assertTrue(guard['safe'])
        import random
        r=random.Random(55)
        for codes,h in zip(columns,anchors):
            factors=[1<<(v-h) for v in codes]
            bound=module.PARTIAL_BOUND*sum(factors)
            for _ in range(100):
                total=0
                for f in factors:
                    total+=r.randint(-module.PARTIAL_BOUND,module.PARTIAL_BOUND)*f
                    self.assertLessEqual(abs(total),bound)
    def test_fallback_and_invalid(self):
        for codes in ([116]*31+[131],[1,254]*16,[116]*2,[0]*32):
            self.assertFalse(module.guard_columns([codes])[1]['safe'])
        for columns in ([],[[]],[[255]*32],[[-1]*32]):
            with self.assertRaises(ValueError): module.guard_columns(columns)
    def test_no_default_or_second_accumulator(self):
        source=(ROOT/'csrc/sm80/o3_fullk_integer_probe.cuh').read_text()
        self.assertIn('make_fragment_like<int>',source)
        self.assertNotIn('make_fragment_like<float>',source)
        self.assertNotIn('__fmaf_rn',source)
        self.assertIn('partial*s.factor[slot][col]',source)
        self.assertNotIn('fullk_integer', (ROOT/'setup.py').read_text())


if __name__=='__main__': unittest.main()
