"""Keep the short best-candidate report tied to same-run raw evidence."""
import json
from pathlib import Path
import statistics
import unittest

ROOT = Path(__file__).resolve().parents[2]


class BestReportTest(unittest.TestCase):
    def test_compute_and_mse_reconcile(self):
        doc = (ROOT / 'docs/o3_o7_o8_current_best.md').read_text()
        summary = json.loads((ROOT / 'docs/evidence/a100_o378_roof_v19/runs/'
                              'o378_roof_v19_finish_trace24/summary.json').read_text())
        self.assertTrue(summary['all_24_samples'])
        self.assertEqual(summary['numerical_policy'], 'bitwise')
        for variant, tune in (('o3', 22), ('o7', 23), ('o8', 23)):
            rows = [r for r in summary['records'] if r['variant'] == variant]
            baseline = next(r for r in rows if r['tune'] == -1)
            candidate = next(r for r in rows if r['tune'] == tune)
            self.assertEqual(candidate['records'], 120)
            for row in (baseline, candidate):
                self.assertIn(f"{row['median_ms']:.6f}", doc)
            speed = candidate['paired_speedup_median']
            self.assertIn(f'{(speed - 1) * 100:.2f}%', doc)
            self.assertIn(f'{(1 - 1 / speed) * 100:.2f}%', doc)
            mse = [r['mse_vs_paired_fp16'] for r in candidate['per_sample']]
            self.assertEqual(mse, [r['mse_vs_paired_fp16'] for r in baseline['per_sample']])
            self.assertIn(f'{statistics.median(mse):.12f}', doc)
            self.assertIn(f'{statistics.mean(mse):.12f}', doc)
            self.assertIn(f"{candidate['cv_failed_records']}/120", doc)

    def test_separate_fourmode_source_and_no_default_promotion(self):
        doc = (ROOT / 'docs/o3_o7_o8_current_best.md').read_text()
        for name in ('o3_four24_t22_vsprod', 'o78_four24_t23_vsprod'):
            report = json.loads((ROOT / 'docs/evidence/a100_o378_roof_v17/reports/'
                                 f'o378_roof_v17/{name}.json').read_text())
            for row in report['rows']:
                self.assertIn(f"{row['candidate_median_ms']:.6f}", doc)
                if row['mode'] != 'compute_only':
                    self.assertIn(f"{row['reference_median_ms']:.6f}", doc)
        self.assertIn('正式默认尚未切换', doc)
        self.assertIn('不是v19同一轮数据', doc)
        self.assertIn('不等同于所有阶段均满足CV<3%', doc)


if __name__ == '__main__':
    unittest.main()
