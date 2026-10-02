import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from analyze_roof_scale_ncu import analyze, validate_arithmetic_work, validate_fullk_integer_work


class FullKNCUAccountingTests(unittest.TestCase):
    def test_exact_integer_work(self):
        counts = dict(IMMA=16777216, I2F=524288, FMUL=1048576)
        validate_fullk_integer_work(counts)
        for key, value in (('IMMA', 0), ('I2F', 16777216), ('FMUL', 524288), ('FFMA', 1)):
            with self.assertRaises(ValueError):
                validate_fullk_integer_work({**counts, key: value})

    def test_control_and_fullk_are_not_interchangeable(self):
        control = dict(IMMA=16777216, I2F=16777216, FFMA=16777216, FMUL=524288)
        validate_arithmetic_work(control, 54, False)
        with self.assertRaises(ValueError):
            validate_fullk_integer_work(control)
        with self.assertRaises(ValueError):
            analyze('', '', 59, 'o7', fullk_integer=True)
        with self.assertRaises(ValueError):
            analyze('', '', 54, 'o3', fullk_integer=True,
                    expected_symbol='adangel_sm80_roof_candidate')


if __name__ == '__main__':
    unittest.main()
