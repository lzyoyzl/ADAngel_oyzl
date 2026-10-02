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

    def test_archived_capture_reproduces_all_metrics(self):
        from analyze_roof_fullk_ncu import analyze_capture
        report = ROOT / 'docs/evidence/a100_o378_roof_v58_ncu/reports/o378_roof_v58_ncu'
        saved = json.loads((report / 'analysis.json').read_text())
        for policy, label in enumerate(('control', 'fullk')):
            row = analyze_capture((report / f'{label}_raw.csv').read_text(),
                                  (report / f'{label}_source_sass.csv').read_text(), policy)
            self.assertEqual(row, saved['rows'][policy])
        old, new = saved['rows']
        self.assertEqual(old['opcodes']['IMMA'], new['opcodes']['IMMA'])
        self.assertEqual(old['opcodes']['I2F'], 32 * new['opcodes']['I2F'])
        self.assertLess(new['dynamic_instructions'], old['dynamic_instructions'])
        self.assertGreater(new['opcodes']['IMAD'], 2 * old['opcodes']['IMAD'])
        self.assertEqual(old['optimistic_fixed_work_lower_bound_ms'],
                         new['optimistic_fixed_work_lower_bound_ms'])

    def test_capture_provenance_and_single_sample_scope(self):
        root = ROOT / 'docs/evidence/a100_o378_roof_v58_ncu'
        for label in ('control', 'fullk'):
            run = root / f'runs/o378_roof_v58_ncu_{label}'
            env = json.loads((run / 'environment.json').read_text())
            self.assertEqual(env['args']['warmup'], 50)
            self.assertEqual(env['args']['repeats'], 1)
            self.assertEqual(env['args']['samples'], 1)
            self.assertEqual(env['args']['policies'], [0, 1])
            rows = [json.loads(line) for line in (run / 'results.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual([r['executed_policy'] for r in rows], [0, 1])
            for row in rows:
                self.assertEqual(row['sample_id'], 'layer_00_q_proj')
                self.assertTrue(row['finite_fp32'])
                self.assertTrue(row['bitwise_equal_current_best'])
                self.assertEqual(row['mse_vs_current_best'], 0.0)
                self.assertTrue(row['guard']['safe'])


if __name__ == '__main__':
    unittest.main()
