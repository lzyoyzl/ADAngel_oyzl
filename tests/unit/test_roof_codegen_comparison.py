from pathlib import Path
import importlib.util
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('codegen', ROOT/'scripts/compare_a100_codegen.py')
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def dump(name='adangel_sm80_split_grouped_major', word='0000000000000001'):
    return f'Function : {name}\n /*0000*/ IMMA; /* 0x{word} */\n /* 0x0000000000000000 */\n'


class CodegenComparisonTest(unittest.TestCase):
    def test_exact_words_ignore_display_labels(self):
        self.assertTrue(MOD.compare(dump(), dump().replace('IMMA;', 'changed label;'))['passed'])
        self.assertFalse(MOD.compare(dump(), dump(word='0000000000000002'))['passed'])

    def test_missing_and_duplicate_are_not_success(self):
        self.assertFalse(MOD.compare(dump() + dump('adangel_sm80_o3_swizzled_bound2'), dump())['passed'])
        with self.assertRaises(ValueError):
            MOD.compare(dump(), dump() + dump())
        with self.assertRaises(ValueError):
            MOD.compare(dump(), 'no matching SASS')


if __name__ == '__main__':
    unittest.main()
