"""Validate archived v36 same-GEMM O3 conversion comparisons, no outlier filtering."""
import json
from pathlib import Path
import runpy
import statistics
import struct
import unittest

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'docs/evidence/a100_o378_roof_v36'
BINARY='04612afa2fecd53215260a1d3aea334279c1feef35d95dc44c00abc0d6d6ffab'


@unittest.skipUnless((ROOT/'runs/o378_roof_v36_four24/summary.json').exists(),'complete archive not present')
class O3ConversionEvidenceTests(unittest.TestCase):
    def load(self,path): return json.loads((ROOT/path).read_text())

    def test_correctness_safety_and_codegen(self):
        for name,count in (('preflight',96),('memcheck',32),('synccheck',32)):
            j=self.load(f'runs/o378_roof_v36_{name}/validation.json')
            self.assertTrue(j['passed']);self.assertEqual(j['binary_sha256'],BINARY)
            self.assertEqual(len(j['checks']),count);self.assertEqual(len(j['rejected']),8)
            self.assertTrue(all(r['output_bitwise'] and r['payload_bitwise'] and r['scale_bitwise']
                and r['semantic_tolerance_passed'] and r['nondefault_stream'] for r in j['checks']))
        for name in ('memcheck','synccheck'):
            self.assertIn('ERROR SUMMARY: 0 errors',(ROOT/f'reports/o378_roof_v36/{name}.log').read_text())
        for name,count in (('production',12),('candidate',148)):
            j=self.load(f'reports/o378_roof_v36/{name}_codegen.json')
            self.assertTrue(j['passed']);self.assertEqual(len(j['unchanged']),count)
        j=self.load('reports/o378_roof_v36/all_sm80_codegen.json')
        self.assertFalse(j['passed']);self.assertEqual(len(j['changed']),2)
        self.assertTrue(all('mixed_binary' in s for s in j['changed']))
        self.assertTrue(self.load('runs/o378_roof_v36_mixed_regression/validation.json')['passed'])
        for name,count in (('mma_audit',148),('conversion_audit',4)):
            j=self.load(f'reports/o378_roof_v36/{name}/audit.json')
            self.assertTrue(j['passed']);self.assertEqual(j['binary_sha256'],BINARY)
            self.assertEqual(len(j['functions']),count)

    def test_event_contract_and_mse(self):
        prefix=ROOT/'runs/o378_roof_v36_four24'
        rows=[json.loads(line) for line in (prefix/'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),288)
        env=json.loads((prefix/'environment.json').read_text())
        self.assertEqual(env['binary_sha256'],BINARY);self.assertFalse(env['gemm_math_changed'])
        self.assertEqual(len({r['sample_id'] for r in rows}),24)
        self.assertEqual({r['implementation'] for r in rows},{0,1,2})
        for r in rows:
            self.assertEqual(r['paired_reference'],'o0')
            self.assertTrue(r['bitwise_equal_current_best']);self.assertEqual(r['mse_vs_current_best'],0.)
            for stage,raw in r['raw_ms'].items():
                self.assertEqual(len(raw),200)
                self.assertEqual(statistics.median(raw),r['stage_summaries'][stage]['median_ms'])
                self.assertEqual(r['stage_timing_inner_repeats'][stage],
                    100 if 'conversion' in stage or r['mode']=='conversion_only' else 1)
            if r['mode']=='conversion_only':
                self.assertEqual(r['total_timing'],'sum_of_batched_stage_samples')
                self.assertEqual(r['raw_ms']['total'],[struct.unpack('f',struct.pack('f',w+a))[0]
                    for w,a in zip(r['raw_ms']['weight_conversion'],r['raw_ms']['activation_conversion'])])
            else: self.assertEqual(r['total_timing'],'single_execution_cuda_event')
            self.assertEqual(r['weight_cached'],r['mode'] not in ('cold','conversion_only'))
            self.assertEqual(r['activation_prepared'],r['mode']=='compute_only')
            meta=r['kernel'];self.assertEqual(meta['roof_tune'],54)
            self.assertEqual(meta['conversion_candidate'],r['implementation'])
            self.assertEqual(meta['weight_conversion_kernels'],1 if r['implementation'] else 3)
            self.assertEqual(meta['activation_conversion_kernels'],1 if r['implementation'] else 2)
        values=[r['mse_vs_paired_fp16'] for r in rows if r['implementation']==2 and r['mode']=='compute_only']
        self.assertAlmostEqual(statistics.median(values),.006653010287409885,places=14)

    def test_paired_statistics_all_records(self):
        prefix=ROOT/'runs/o378_roof_v36_four24'
        rows=[json.loads(line) for line in (prefix/'results.jsonl').read_text().splitlines()]
        index={(r['sample_id'],r['mode'],r['implementation']):r for r in rows}
        self.assertEqual(len(index),288)
        ci=runpy.run_path(str(REPO/'python/adangel/benchmark/metrics.py'))['bootstrap_median_ci']
        for s in json.loads((prefix/'summary.json').read_text()):
            group=sorted([r for r in rows if (r['mode'],r['implementation'])==(s['mode'],s['implementation'])],key=lambda r:r['sample_id'])
            ratio=[index[(r['sample_id'],r['mode'],0)]['summary']['median_ms']/r['summary']['median_ms'] for r in group]
            self.assertEqual(s['samples'],24);self.assertEqual(s['records'],24)
            self.assertEqual(s['median_ms'],statistics.median(r['summary']['median_ms'] for r in group))
            self.assertEqual(s['paired_speedup'],statistics.median(ratio))
            self.assertEqual(s['paired_speedup_ci95'],list(ci(ratio,10000,.95,20261001)))
            self.assertEqual(s['selected_cv_failed_records'],sum(r['summary']['cv_percent']>=3 for r in group))
            self.assertEqual(s['any_stage_cv_failed_records'],sum(any(x['cv_percent']>=3 for x in r['stage_summaries'].values()) for r in group))

    def test_long_warmup_diagnostic_preserves_all_samples(self):
        # Separate run, never overwrite/filter the originally configured50 run.
        prefix=ROOT/'runs/o378_roof_v36_four24_warmup500'
        original=self.load('runs/o378_roof_v36_four24/environment.json')
        env=json.loads((prefix/'environment.json').read_text())
        self.assertEqual(original['args']['warmup'],50);self.assertEqual(env['args']['warmup'],500)
        self.assertEqual(env['binary_sha256'],BINARY)
        for name in ('prepared_manifest_sha256','raw_manifest_sha256'):
            self.assertEqual(env[name],original[name])
        rows=[json.loads(line) for line in (prefix/'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),288)
        self.assertEqual(len({r['sample_id'] for r in rows}),24)
        self.assertTrue(all(r['bitwise_equal_current_best'] and r['mse_vs_current_best']==0 for r in rows))
        index={(r['sample_id'],r['mode'],r['implementation']):r for r in rows}
        self.assertEqual(len(index),288)
        ci=runpy.run_path(str(REPO/'python/adangel/benchmark/metrics.py'))['bootstrap_median_ci']
        for s in json.loads((prefix/'summary.json').read_text()):
            group=sorted([r for r in rows if (r['mode'],r['implementation'])==(s['mode'],s['implementation'])],key=lambda r:r['sample_id'])
            ratio=[index[(r['sample_id'],r['mode'],0)]['summary']['median_ms']/r['summary']['median_ms'] for r in group]
            self.assertEqual(s['records'],24)
            self.assertTrue(all(len(v)==200 for r in group for v in r['raw_ms'].values()))
            self.assertEqual(s['median_ms'],statistics.median(r['summary']['median_ms'] for r in group))
            self.assertEqual(s['paired_speedup'],statistics.median(ratio))
            self.assertEqual(s['paired_speedup_ci95'],list(ci(ratio,10000,.95,20261001)))
            self.assertEqual(s['selected_cv_failed_records'],sum(r['summary']['cv_percent']>=3 for r in group))
            self.assertEqual(s['any_stage_cv_failed_records'],sum(any(x['cv_percent']>=3 for x in r['stage_summaries'].values()) for r in group))


if __name__=='__main__': unittest.main()
