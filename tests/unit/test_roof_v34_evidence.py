"""Recompute archived v34 conversion results; no GPU rerun required."""
import json
from pathlib import Path
import runpy
import statistics
import unittest

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'docs/evidence/a100_o378_roof_v34'
REPORTS=ROOT/'reports/o378_roof_v34'
BINARY='605fd27299aa203f7fbcaaa309d509ceba0dcd01b2d4942c447cd7e9c46ffd36'


@unittest.skipUnless((ROOT/'runs/o378_roof_v34_trace24/summary.json').exists(),'completed archive not present')
class IntegerConversionEvidenceTests(unittest.TestCase):
    def load(self,path): return json.loads((ROOT/path).read_text())

    def test_finite_codes_and_invalid_inputs(self):
        for name,count in (('preflight',524),('memcheck',108),('synccheck',108)):
            j=self.load(f'runs/o378_roof_v34_{name}/validation.json')
            self.assertTrue(j['passed']);self.assertEqual(j['binary_sha256'],BINARY)
            self.assertEqual(len(j['checks']),count);self.assertEqual(len(j['rejected']),24)
            self.assertTrue(all(x['bitwise'] and x['nondefault_stream'] for x in j['checks']))
        for name in ('memcheck','synccheck'):
            self.assertIn('ERROR SUMMARY: 0 errors',(REPORTS/f'{name}.log').read_text())

    def test_codegen_audit_discloses_control_and_nontarget_changes(self):
        x=self.load('reports/o378_roof_v34/conversion_audit_v2/audit.json')
        self.assertTrue(x['passed']);self.assertEqual(len(x['functions']),8)
        self.assertEqual(x['binary_sha256'],BINARY)
        for r in x['functions']:
            self.assertTrue(r['passed']);self.assertIn('STACK:0',r['resource'])
            self.assertEqual(r['no_float_to_integer'],r['tiled'])
        # Preserve the original over-broad audit failure, don't erase it.
        self.assertFalse(self.load('reports/o378_roof_v34/conversion_audit/audit.json')['passed'])
        for name,count in (('production',12),('candidate',148)):
            x=self.load(f'reports/o378_roof_v34/{name}_codegen.json')
            self.assertTrue(x['passed']);self.assertEqual(len(x['unchanged']),count)
        x=self.load('reports/o378_roof_v34/all_sm80_codegen.json')
        self.assertFalse(x['passed']);self.assertEqual(len(x['changed']),4)
        self.assertTrue(all('mixed_binary' in symbol for symbol in x['changed']))
        self.assertTrue(self.load('runs/o378_roof_v34_mixed_regression/validation.json')['passed'])
        x=self.load('reports/o378_roof_v34/mma_audit/audit.json')
        self.assertTrue(x['passed']);self.assertEqual(len(x['functions']),148)

    def test_trace24_pairing_and_output_mse(self):
        prefix=ROOT/'runs/o378_roof_v34_trace24'
        env=json.loads((prefix/'environment.json').read_text())
        self.assertEqual(env['binary_sha256'],BINARY)
        self.assertFalse(env['gemm_performance_measured']);self.assertFalse(env['formal_dispatch_changed'])
        rows=[json.loads(s) for s in (prefix/'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),384)
        self.assertEqual(len({r['sample_id'] for r in rows}),24)
        self.assertEqual(len({(r['sample_id'],r['format'],r['round'],r['implementation']) for r in rows}),384)
        for r in rows:
            self.assertEqual(len(r['timings_ms']),200);self.assertEqual(r['inner'],100)
            self.assertTrue(r['bitwise_payload_and_scale'])
            self.assertAlmostEqual(statistics.median(r['timings_ms']),r['summary']['median_ms'])
        index={(r['sample_id'],r['format'],r['implementation']):r for r in rows}
        for s in json.loads((prefix/'summary.json').read_text()):
            selected=[r for r in rows if (r['format'],r['implementation'])==(s['format'],s['implementation'])]
            self.assertEqual(s['samples'],24)
            self.assertEqual(s['median_ms'],statistics.median(r['summary']['median_ms'] for r in selected))
            ratios=[index[(r['sample_id'],r['format'],0)]['summary']['median_ms']/r['summary']['median_ms'] for r in selected]
            self.assertEqual(s['paired_speedup'],statistics.median(ratios))
            self.assertEqual(s['cv_failed_records'],sum(r['summary']['cv_percent']>=3 for r in selected))
        mse=[json.loads(s) for s in (prefix/'output_mse.jsonl').read_text().splitlines()]
        self.assertEqual(len(mse),192)
        for r in mse:
            self.assertTrue(r['output_bitwise_current_best']);self.assertEqual(r['mse_vs_current_best'],0.)
        for v,median in (('o7',.0055361724266658075),('o8',.004411084948644645)):
            vals=[r['mse_vs_paired_fp16'] for r in mse if r['variant']==v and r['implementation']==3]
            self.assertAlmostEqual(statistics.median(vals),median,places=14)


if __name__=='__main__': unittest.main()
