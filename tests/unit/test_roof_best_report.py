"""Keep the best-candidate report tied to same-run raw evidence."""
import json
from pathlib import Path
import statistics
import unittest

ROOT=Path(__file__).resolve().parents[2]
EVIDENCE=ROOT/'docs/evidence/a100_o378_roof_v24'
O3_EVIDENCE=ROOT/'docs/evidence/a100_o378_roof_v29'


@unittest.skipUnless((EVIDENCE/'runs/o378_roof_v24_trace24/summary.json').exists(),
                     'v24 archive has not yet been imported')
class BestReportTest(unittest.TestCase):
    def test_compute_and_mse_reconcile(self):
        doc=(ROOT/'docs/o3_o7_o8_current_best.md').read_text()
        summary=json.loads((EVIDENCE/'runs/o378_roof_v24_trace24/summary.json').read_text())
        self.assertTrue(summary['all_24_samples'])
        self.assertEqual(summary['numerical_policy'],'bitwise')
        for variant,old,tune in (('o7',23,42),('o8',23,42)):
            rows=[r for r in summary['records'] if r['variant']==variant]
            baseline=next(r for r in rows if r['tune']==-1)
            previous=next(r for r in rows if r['tune']==old)
            candidate=next(r for r in rows if r['tune']==tune)
            self.assertEqual(candidate['records'],120)
            for row in (baseline,previous,candidate):
                self.assertIn(f"{row['median_ms']:.6f}",doc)
            comparison=json.loads((EVIDENCE/f'reports/o378_roof_v24/trace24_t{tune}_vs{old}.json').read_text())
            paired=next(r for r in comparison['rows'] if r['variant']==variant)
            self.assertIn(f"{(paired['paired_speedup_median']-1)*100:.2f}%",doc)
            mse=[r['mse_vs_paired_fp16'] for r in candidate['per_sample']]
            self.assertEqual(mse,[r['mse_vs_paired_fp16'] for r in baseline['per_sample']])
            self.assertIn(f'{statistics.median(mse):.15f}',doc)
            self.assertIn(f'{statistics.mean(mse):.15f}',doc)
            self.assertIn(f"{candidate['cv_failed_records']}/120",doc)

    def test_new_o3_evidence_and_changed_rounding_are_reported(self):
        doc=(ROOT/'docs/o3_o7_o8_current_best.md').read_text()
        for reference in (-1,41):
            x=json.loads((O3_EVIDENCE/f'reports/o378_roof_v29/trace24_52_vs{reference}.json').read_text())['rows'][0]
            self.assertEqual((x['samples'],x['rounds']),(24,5))
            self.assertIn(f"{x['reference_median_ms']:.6f}",doc)
            self.assertIn(f"{x['candidate_median_ms']:.6f}",doc)
            self.assertIn(f"{(x['paired_speedup_median']-1)*100:.2f}%",doc)
            mse=[r['mse_vs_paired_fp16'] for r in x['per_sample']]
            self.assertIn(f'{statistics.median(mse):.15f}',doc)
            self.assertIn(f'{statistics.mean(mse):.15f}',doc)
            self.assertTrue(all(r['candidate_mse_vs_production']>0 for r in x['per_sample']))
            self.assertIn(f"{x['cv']['52']['selected_stage_failed']}/120",doc)
        self.assertIn('不能再声称新版O3与旧版逐位相同',doc)
        self.assertIn('O3来自本轮v29，O7/O8沿用v24',doc)

    def test_fourmode_sources_and_no_default_promotion(self):
        doc=(ROOT/'docs/o3_o7_o8_current_best.md').read_text()
        paths=(EVIDENCE/'reports/o378_roof_v24/o78_four24_t42_vs23.json',
               O3_EVIDENCE/'reports/o378_roof_v29/four24_52_vs41.json')
        for path in paths:
            report=json.loads(path.read_text())
            for row in report['rows']:
                self.assertEqual(row['rounds'],1)
                self.assertEqual(row['samples'],24)
                if row['mode']=='compute_only': continue
                self.assertIn(f"{row['candidate_median_ms']:.6f}",doc)
                self.assertIn(f"{row['reference_median_ms']:.6f}",doc)
        self.assertIn('正式默认尚未切换',doc)
        self.assertIn('另外的24样本、单轮四模式测量',doc)
        self.assertIn('不等同于严格全阶段CV<3%',doc)
        self.assertNotIn('待完成',doc)


if __name__=='__main__': unittest.main()
