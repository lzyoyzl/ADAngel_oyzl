"""Reconcile archived fixed-shape evidence; never invent full-trace coverage."""
import json
from pathlib import Path
import statistics
import unittest

ROOT=Path(__file__).resolve().parents[2]/'docs/evidence/a100_o378_roof_v20'


def read(path): return json.loads((ROOT/path).read_text())


class FixedShapeEvidence(unittest.TestCase):
    def test_full_shape_checks_rejections_and_safety(self):
        data=read('runs/o378_roof_v20_fixed_screen/validation.json')
        self.assertTrue(data['passed']);self.assertEqual(len(data['checks']),414)
        new=[r for r in data['checks'] if r['tune'] in (32,33)]
        self.assertEqual(len(new),36)
        self.assertTrue(all(r['shape']==[4096]*3 and r['bitwise_equal_production'] for r in new))
        self.assertEqual(len(data['rejected_shapes']),42)
        for tool in ('memcheck','synccheck'):
            self.assertIn('ERROR SUMMARY: 0 errors',(ROOT/f'reports/o378_roof_v20/{tool}.log').read_text())
            self.assertEqual(len(read(f'runs/o378_roof_v20_{tool}/validation.json')['checks']),36)

    def test_codegen_and_non_adoption_screen(self):
        audit=read('reports/o378_roof_v20/audit/audit.json')
        self.assertTrue(audit['passed']);self.assertEqual(len(audit['functions']),93)
        for name,count in (('production_codegen',12),('old_candidate_codegen',87)):
            data=read(f'reports/o378_roof_v20/{name}.json')
            self.assertTrue(data['passed']);self.assertEqual(data['old_symbols'],count)
        rows=[json.loads(x) for x in (ROOT/'runs/o378_roof_v20_fixed_screen/results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),75)
        for variant,speed in (('o3',.9866666755364057),('o7',.9906191744559003),('o8',.9852125639463891)):
            index={(r['round'],r['tune']):r for r in rows if r['variant']==variant}
            actual=statistics.median(index[i,22]['summary']['median_ms']/index[i,32]['summary']['median_ms'] for i in range(5))
            self.assertAlmostEqual(actual,speed)

    def test_four_mode_smoke_not_full_trace(self):
        summary=read('runs/o378_roof_v20_fixed_four_smoke/summary.json')
        self.assertTrue(summary['correctness_passed']);self.assertFalse(summary['all_24_samples'])
        rows=[json.loads(x) for x in (ROOT/'runs/o378_roof_v20_fixed_four_smoke/results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),36)
        self.assertEqual(len({r['sample_id'] for r in rows}),1)
        self.assertTrue(all(r['bitwise_equal_production'] and r['mse_vs_production']==0 for r in rows))
        for variant in ('o3','o7','o8'):
            self.assertEqual(len({r['mse_vs_paired_fp16'] for r in rows if r['variant']==variant}),1)


if __name__=='__main__': unittest.main()
