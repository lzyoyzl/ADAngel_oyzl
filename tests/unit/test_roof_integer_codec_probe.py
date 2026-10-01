"""Candidate design proof only; existing native conversion is not replaced."""
from fractions import Fraction
from pathlib import Path
import runpy
import unittest

ROOT = Path(__file__).resolve().parents[2]
PROBE = runpy.run_path(str(ROOT / 'scripts/probe_integer_fixed_conversion.py'))


class IntegerCodecProbeTests(unittest.TestCase):
    def test_rne_including_even_odd_ties(self):
        for value in range(32):
            for shift in range(-12, 5):
                exact = Fraction(value * (1 << shift)) if shift >= 0 else Fraction(value, 1 << -shift)
                self.assertEqual(PROBE['round_unsigned_pow2'](value, shift), round(exact))

    def test_all_finite_codes_match_existing_reference(self):
        x = PROBE['probe']()
        self.assertTrue(x['passed'])
        self.assertEqual(x['checks'], 398)
        self.assertEqual([r['checked'] for r in x['rows']], [16, 254, 64, 64])
        self.assertEqual([r['max_abs_integer'] for r in x['rows']], [6, 112, 7, 30])
        self.assertFalse(x['performance_measured'])
        self.assertFalse(x['changes_scale'])

    def test_nan_rejected_and_signed_zero_canonicalized(self):
        for code in (127, 255):
            with self.assertRaises(ValueError): PROBE['integer_fixed'](code, 'e4m3')
        for kind, sign in (('e2m1', 8), ('e2m3', 32), ('e4m3', 128), ('s1p2', 8)):
            self.assertEqual(PROBE['integer_fixed'](0, kind), 0)
            self.assertEqual(PROBE['integer_fixed'](sign, kind), 0)


if __name__ == '__main__': unittest.main()
