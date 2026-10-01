"""Keep the best-candidate report tied to same-run raw evidence."""
import json
from pathlib import Path
import statistics
import unittest

ROOT=Path(__file__).resolve().parents[2]
EVIDENCE=ROOT/'docs/evidence/a100_o378_roof_v33'
FOUR_EVIDENCE=ROOT/'docs/evidence/a100_o378_roof_v35'
O3_EVIDENCE=ROOT/'docs/evidence/a100_o378_roof_v30'


@unittest.skipUnless((EVIDENCE/'runs/o378_roof_v33_trace24_r5/summary.json').exists(),
                     'v33 archive has not yet been imported')
class BestReportTest(unittest.TestCase):
    def test_compute_and_mse_reconcile(self):
        doc=(ROOT/'docs/o3_o7_o8_current_best.md').read_text()
        summary=json.loads((EVIDENCE/'runs/o378_roof_v33_trace24_r5/summary.json').read_text())
        self.assertTrue(summary['all_24_samples'])
        self.assertEqual(summary['numerical_policy'],'bitwise')
        for variant,old,tune in (('o7',56,59),('o8',56,59)):
            rows=[r for r in summary['records'] if r['variant']==variant]
            baseline=next(r for r in rows if r['tune']==-1)
            previous=next(r for r in rows if r['tune']==old)
            candidate=next(r for r in rows if r['tune']==tune)
            self.assertEqual(candidate['records'],120)
            for row in (baseline,previous,candidate):
                self.assertIn(f"{row['median_ms']:.6f}",doc)
            comparison=json.loads((EVIDENCE/f'reports/o378_roof_v33/trace24_r5_{tune}_vs{old}.json').read_text())
            paired=next(r for r in comparison['rows'] if r['variant']==variant)
            self.assertIn(f"{(paired['paired_speedup_median']-1)*100:.2f}%",doc)
            mse=[r['mse_vs_paired_fp16'] for r in candidate['per_sample']]
            self.assertEqual(mse,[r['mse_vs_paired_fp16'] for r in baseline['per_sample']])
            self.assertIn(f'{statistics.median(mse):.15f}',doc)
            self.assertIn(f'{statistics.mean(mse):.15f}',doc)
            self.assertIn(f"{candidate['cv_failed_records']}/120",doc)

    def test_new_o3_evidence_and_changed_rounding_are_reported(self):
        doc=(ROOT/'docs/o3_o7_o8_current_best.md').read_text()
        for reference in (-1,52):
            x=json.loads((O3_EVIDENCE/f'reports/o378_roof_v30/trace24_54_vs{reference}.json').read_text())['rows'][0]
            self.assertEqual((x['samples'],x['rounds']),(24,5))
            self.assertIn(f"{x['reference_median_ms']:.6f}",doc)
            self.assertIn(f"{x['candidate_median_ms']:.6f}",doc)
            self.assertIn(f"{(x['paired_speedup_median']-1)*100:.2f}%",doc)
            mse=[r['mse_vs_paired_fp16'] for r in x['per_sample']]
            self.assertIn(f'{statistics.median(mse):.15f}',doc)
            self.assertIn(f'{statistics.mean(mse):.15f}',doc)
            self.assertTrue(all(r['candidate_mse_vs_production']>0 for r in x['per_sample']))
            self.assertIn(f"{x['cv']['54']['selected_stage_failed']}/120",doc)
        self.assertIn('新版O3与原正式并非逐位相同，但与上一版52逐位相同',doc)
        self.assertIn('O3沿用v30，O7/O8来自v33',doc)

    def test_fourmode_sources_and_no_default_promotion(self):
        doc=(ROOT/'docs/o3_o7_o8_current_best.md').read_text()
        paths=(O3_EVIDENCE/'reports/o378_roof_v30/four24_54_vs52.json',)
        for path in paths:
            report=json.loads(path.read_text())
            for row in report['rows']:
                self.assertEqual(row['rounds'],1)
                self.assertEqual(row['samples'],24)
                if row['mode']=='compute_only': continue
                self.assertIn(f"{row['candidate_median_ms']:.6f}",doc)
                self.assertIn(f"{row['reference_median_ms']:.6f}",doc)
        # Latest conversion/end-to-end results use the v35 same-GEMM 0/4 pair,
        # not the older v33 comparison between GEMM candidates 56 and 59.
        latest=json.loads((FOUR_EVIDENCE/'runs/o378_roof_v35_four24/summary.json').read_text())
        self.assertEqual(len(latest),16)
        self.assertEqual({r['implementation'] for r in latest},{0,4})
        for row in latest:
            self.assertEqual(row['samples'],24)
            self.assertEqual(row['records'],24)
            self.assertIn(f"{row['median_ms']:.6f}",doc)
            if row['implementation']==4 and row['mode']!='compute_only':
                self.assertIn(f"{(row['paired_speedup']-1)*100:.2f}%",doc)
        self.assertIn('O7/O8为最新v35',doc)
        self.assertIn('GEMM机器码完全相同',doc)
        self.assertIn('正式默认尚未切换',doc)
        self.assertIn('另外的24样本、单轮四模式测量',doc)
        self.assertIn('不等同于严格全阶段CV<3%',doc)
        self.assertNotIn('待完成',doc)


if __name__=='__main__': unittest.main()
