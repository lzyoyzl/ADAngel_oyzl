"""Recorded v25 numerical/four-mode evidence; never hides performance regressions."""
import json
import statistics
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]/'docs/evidence/a100_o378_roof_v25'
REPORTS=ROOT/'reports/o378_roof_v25'
BINARY='23530dba211d16393b91f9287aecdb9e402c94af39cd23dee960d1b19e2a6a3c'


@unittest.skipUnless((REPORTS/'preflight.log').exists(),'preflight archive not present')
class V25EvidenceTests(unittest.TestCase):
    def load(self,path): return json.loads((ROOT/path).read_text())

    def test_bitwise_synthetic_and_finite_codec_coverage(self):
        d=self.load('runs/o378_roof_v25_preflight/validation.json')
        self.assertTrue(d['passed'])
        self.assertEqual(d['binary_sha256'],BINARY)
        self.assertEqual(len(d['checks']),267)
        modes=[c for c in d['checks'] if c['type']=='four_mode']
        self.assertEqual(len(modes),216)
        self.assertEqual({c['tune'] for c in modes},{43,44})
        self.assertTrue(all(c['bitwise'] and c['mse_vs_previous']==0 for c in modes))
        self.assertEqual(sum(c['type']=='invalid_scale' for c in d['checks']),4)

    def test_gemm_codegen_unchanged_but_broad_exception_preserved(self):
        for name,count in (('production_codegen.json',12),('candidate_codegen.json',120)):
            d=json.loads((REPORTS/name).read_text())
            self.assertTrue(d['passed'])
            self.assertEqual(len(d['unchanged']),count)
        d=json.loads((REPORTS/'all_sm80_codegen.json').read_text())
        self.assertFalse(d['passed'])
        self.assertFalse(d['missing'])
        self.assertEqual(len(d['changed']),3)
        self.assertTrue(all('mixed_binary' in name for name in d['changed']))
        self.assertEqual(len(d['unchanged']),243)

    def test_native_int4_and_finite_sanitizer_checks(self):
        d=json.loads((REPORTS/'audit/audit.json').read_text())
        self.assertTrue(d['passed'])
        self.assertEqual(d['binary_sha256'],BINARY)
        self.assertEqual(len(d['functions']),120)
        for f in d['functions']:
            self.assertTrue(all(f['checks'][key] for key in ('sass_u4s4','sass_s4s4','sass_no_int8','ptx_u4s4','ptx_s4s4')))
        for kind in ('memcheck','synccheck'):
            self.assertIn('ERROR SUMMARY: 0 errors',(REPORTS/f'{kind}.log').read_text())
            d=self.load(f'runs/o378_roof_v25_{kind}/validation.json')
            self.assertTrue(d['passed'])
            self.assertEqual(len(d['checks']),123)

    def test_smoke_is_one_sample_with_no_hidden_reorder_traffic(self):
        env=self.load('runs/o378_roof_v25_four_smoke/environment.json')
        self.assertEqual(env['binary_sha256'],BINARY)
        self.assertEqual(env['args']['samples'],1)
        self.assertEqual(env['args']['repeats'],20)
        rows=[json.loads(line) for line in (ROOT/'runs/o378_roof_v25_four_smoke/results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),60)
        for r in rows:
            self.assertTrue(r['bitwise_equal_production'])
            if r['tune'] not in (43,44): continue
            self.assertTrue(r['payload_layout_bitwise_verified'])
            self.assertEqual(r['kernel']['gemm_tune'],r['tune']-2)
            self.assertTrue(r['kernel']['payload_reorder_fused'])
            self.assertEqual(r['kernel']['activation_payload_reorder_traffic_bytes'],0)
            self.assertEqual(r['kernel']['weight_payload_reorder_traffic_bytes'],0)
            self.assertEqual(r['mse_vs_production'],0)

    def test_o3_complete_four_modes_and_no_mse_change(self):
        run=ROOT/'runs/o378_roof_v25_o3_four24'
        if not (run/'summary.json').exists(): self.skipTest('O3 complete archive not present')
        env=json.loads((run/'environment.json').read_text())
        self.assertEqual(env['binary_sha256'],BINARY)
        self.assertEqual(env['args']['samples'],24)
        self.assertEqual((env['args']['warmup'],env['args']['repeats'],env['args']['inner']),(50,200,100))
        rows=[json.loads(line) for line in (run/'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),384)
        self.assertEqual(len({r['sample_id'] for r in rows}),24)
        for tune in (-1,22,41,43):
            for mode in ('conversion_only','compute_only','cold','steady_state'):
                group=[r for r in rows if (r['tune'],r['mode'])==(tune,mode)]
                self.assertEqual(len(group),24)
                self.assertTrue(all(r['bitwise_equal_production'] and r['mse_vs_production']==0 for r in group))
                self.assertTrue(all(len(r['raw_ms'])==200 for r in group))
        report=json.loads((REPORTS/'o3_four24_43_vs41.json').read_text())
        doc=(ROOT/'README.md').read_text()
        for r in report['rows']:
            self.assertIn(f"{r['candidate_median_ms']:.6f}",doc)
            self.assertIn(f"{r['reference_median_ms']:.6f}",doc)
            a=sorted([x for x in rows if (x['tune'],x['mode'])==(41,r['mode'])],key=lambda x:x['sample_id'])
            b=sorted([x for x in rows if (x['tune'],x['mode'])==(43,r['mode'])],key=lambda x:x['sample_id'])
            speed=statistics.median(x['summary']['median_ms']/y['summary']['median_ms'] for x,y in zip(a,b))
            self.assertEqual(speed,r['paired_speedup_median'])
            self.assertEqual(r['cv']['43']['selected_stage_failed'],sum(y['summary']['cv_percent']>=3 for y in b))

    def test_o78_complete_and_negative_o8_result_is_retained(self):
        run=ROOT/'runs/o378_roof_v25_o78_four24'
        if not (run/'summary.json').exists(): self.skipTest('O7/O8 complete archive not present')
        env=json.loads((run/'environment.json').read_text())
        self.assertEqual(env['binary_sha256'],BINARY)
        self.assertEqual((env['args']['samples'],env['args']['rounds']),(24,1))
        self.assertEqual((env['args']['warmup'],env['args']['repeats'],env['args']['inner']),(50,200,100))
        rows=[json.loads(line) for line in (run/'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),768)
        self.assertTrue(all(r['bitwise_equal_production'] and r['mse_vs_production']==0 for r in rows))
        report=json.loads((REPORTS/'o78_four24_44_vs42.json').read_text())
        doc=(ROOT/'README.md').read_text()
        for r in report['rows']:
            self.assertIn(f"{r['candidate_median_ms']:.6f}",doc)
            self.assertIn(f"{r['reference_median_ms']:.6f}",doc)
            for tune,key in ((42,'reference_median_ms'),(44,'candidate_median_ms')):
                group=[x for x in rows if (x['variant'],x['mode'],x['tune'])==(r['variant'],r['mode'],tune)]
                self.assertEqual(len(group),24)
                self.assertEqual(statistics.median(x['summary']['median_ms'] for x in group),r[key])
                self.assertEqual(sum(x['summary']['cv_percent']>=3 for x in group),r['cv'][str(tune)]['selected_stage_failed'])
                self.assertTrue(all(len(x['raw_ms'])==200 for x in group))
            if r['variant']=='o8' and r['mode'] in ('conversion_only','cold'):
                self.assertLess(r['paired_speedup_ci95'][1],1)
                self.assertGreater(r['candidate_median_ms'],r['reference_median_ms'])


if __name__=='__main__': unittest.main()
