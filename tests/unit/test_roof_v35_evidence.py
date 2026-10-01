"""Recompute archived four-mode conversion integration, including negative/noisy data."""
import json
from pathlib import Path
import runpy
import statistics
import struct
import unittest

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'docs/evidence/a100_o378_roof_v35'
REPORTS=ROOT/'reports/o378_roof_v35'
BINARY='273b846dd55ebca0028f556c60d653a49d35b7756685bb8707c05e30ae2270af'


@unittest.skipUnless((ROOT/'runs/o378_roof_v35_four24/summary.json').exists(),'completed archive not present')
class ConversionPipelineEvidenceTests(unittest.TestCase):
    def load(self,path): return json.loads((ROOT/path).read_text())

    def test_correctness_safety_and_unchanged_mma(self):
        for name,count in (('preflight',288),('memcheck',96),('synccheck',96)):
            j=self.load(f'runs/o378_roof_v35_{name}/validation.json')
            self.assertTrue(j['passed']);self.assertEqual(j['binary_sha256'],BINARY)
            self.assertEqual(len(j['checks']),count);self.assertEqual(len(j['rejected']),4)
            self.assertTrue(all(r['bitwise'] and r['semantic_tolerance_passed'] for r in j['checks']))
        for name in ('memcheck','synccheck'):
            self.assertIn('ERROR SUMMARY: 0 errors',(REPORTS/f'{name}.log').read_text())
        for name,count in (('production',12),('candidate',148)):
            j=self.load(f'reports/o378_roof_v35/{name}_codegen.json')
            self.assertTrue(j['passed']);self.assertEqual(len(j['unchanged']),count)
        j=self.load('reports/o378_roof_v35/all_sm80_codegen.json')
        self.assertFalse(j['passed']);self.assertEqual(len(j['changed']),3)
        self.assertTrue(all('mixed_binary' in s for s in j['changed']))
        self.assertTrue(self.load('runs/o378_roof_v35_mixed_regression/validation.json')['passed'])
        for name,count in (('mma_audit',148),('conversion_audit',8)):
            j=self.load(f'reports/o378_roof_v35/{name}/audit.json')
            self.assertTrue(j['passed']);self.assertEqual(j['binary_sha256'],BINARY)
            self.assertEqual(len(j['functions']),count)

    def test_native_timing_contract_and_bitwise_mse(self):
        prefix=ROOT/'runs/o378_roof_v35_four24'
        rows=[json.loads(line) for line in (prefix/'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),384)
        env=json.loads((prefix/'environment.json').read_text())
        self.assertEqual(env['binary_sha256'],BINARY);self.assertFalse(env['gemm_math_changed'])
        self.assertEqual(len({r['sample_id'] for r in rows}),24)
        self.assertEqual({r['implementation'] for r in rows},{0,4})
        for r in rows:
            self.assertTrue(r['bitwise_equal_current_best']);self.assertEqual(r['mse_vs_current_best'],0.)
            for stage,raw in r['raw_ms'].items():
                self.assertEqual(len(raw),200)
                self.assertEqual(r['stage_timing_inner_repeats'][stage],
                    100 if 'conversion' in stage or r['mode']=='conversion_only' else 1)
                self.assertEqual(statistics.median(raw),r['stage_summaries'][stage]['median_ms'])
            if r['mode']=='conversion_only':
                self.assertEqual(r['total_timing'],'sum_of_batched_stage_samples')
                self.assertEqual(r['raw_ms']['total'],[struct.unpack('f',struct.pack('f',w+a))[0]
                    for w,a in zip(r['raw_ms']['weight_conversion'],r['raw_ms']['activation_conversion'])])
            else: self.assertEqual(r['total_timing'],'single_execution_cuda_event')
            self.assertEqual(r['weight_cached'],r['mode'] not in ('cold','conversion_only'))
            self.assertEqual(r['activation_prepared'],r['mode']=='compute_only')
            meta=r['kernel'];self.assertEqual(meta['roof_tune'],59)
            if r['implementation']==4:
                self.assertEqual(meta['weight_conversion_impl'],3)
                self.assertEqual(meta['activation_conversion_impl'],1 if r['variant']=='o7' else 3)
                self.assertTrue(meta['payload_reorder_fused']);self.assertFalse(meta['gemm_math_changed'])
                self.assertEqual(meta['conversion_kernels_per_operand'],1)
        for v,expected in (('o7',.0055361724266658075),('o8',.004411084948644645)):
            vals=[r['mse_vs_paired_fp16'] for r in rows if r['variant']==v and r['implementation']==4 and r['mode']=='compute_only']
            self.assertAlmostEqual(statistics.median(vals),expected,places=14)

    def test_paired_statistics_recomputed(self):
        prefix=ROOT/'runs/o378_roof_v35_four24'
        rows=[json.loads(line) for line in (prefix/'results.jsonl').read_text().splitlines()]
        index={(r['sample_id'],r['variant'],r['mode'],r['implementation']):r for r in rows}
        self.assertEqual(len(index),384)
        ci=runpy.run_path(str(REPO/'python/adangel/benchmark/metrics.py'))['bootstrap_median_ci']
        for s in json.loads((prefix/'summary.json').read_text()):
            selected=sorted([r for r in rows if (r['variant'],r['mode'],r['implementation'])==
                (s['variant'],s['mode'],s['implementation'])],key=lambda r:r['sample_id'])
            ratios=[index[(r['sample_id'],r['variant'],r['mode'],0)]['summary']['median_ms']/r['summary']['median_ms'] for r in selected]
            self.assertEqual(s['samples'],24);self.assertEqual(s['records'],24)
            self.assertEqual(s['median_ms'],statistics.median(r['summary']['median_ms'] for r in selected))
            self.assertEqual(s['paired_speedup'],statistics.median(ratios))
            self.assertEqual(s['paired_speedup_ci95'],list(ci(ratios,10000,.95,20261001)))
            self.assertEqual(s['selected_cv_failed_records'],sum(r['summary']['cv_percent']>=3 for r in selected))
            self.assertEqual(s['any_stage_cv_failed_records'],sum(any(x['cv_percent']>=3 for x in r['stage_summaries'].values()) for r in selected))


if __name__=='__main__': unittest.main()
