"""Check archived coverage and paired arithmetic, not substitute for GPU tests."""
import collections
import json
from pathlib import Path
import statistics
import unittest

ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'docs/evidence/a100_o378_roof_v14'


class WarpReuseEvidence(unittest.TestCase):
    def test_trace_coverage_order_mse_and_no_filtering(self):
        rows=[json.loads(x) for x in (E/'runs/o378_roof_v14_trace24/results.jsonl').read_text().splitlines()]
        index={(r['sample_id'],r['variant'],r['round'],r['tune']):r for r in rows}
        self.assertEqual(len(rows),768)
        self.assertEqual(len(index),len(rows))
        self.assertEqual(len({r['sample_id'] for r in rows}),24)
        self.assertEqual({r['mode'] for r in rows},{'compute_only'})
        self.assertTrue(all(r['bitwise_equal_production'] and r['mse_vs_production']==0 for r in rows))
        for v in ('o7','o8'):
            by_sample=collections.defaultdict(list)
            for r in rows:
                if r['variant']==v:
                    by_sample[r['sample_id']].append(r)
            for group in by_sample.values():
                self.assertEqual(len(group),16)
                self.assertEqual(len({(r['mse_vs_o0'],r['mse_vs_paired_fp16']) for r in group}),1)
                for tune in (-1,6,14,21):
                    self.assertEqual(sorted(r['order_position'] for r in group if r['tune']==tune),[0,1,2,3])
        for v,expected in (('o7',1.009183724594082),('o8',1.0096551191967615)):
            ratios=[]
            for sid in sorted({r['sample_id'] for r in rows}):
                ratios.append(statistics.median(index[sid,v,r,14]['summary']['median_ms']/
                                               index[sid,v,r,21]['summary']['median_ms'] for r in range(4)))
            self.assertAlmostEqual(statistics.median(ratios),expected)
        self.assertGreater(sum(r['summary']['cv_percent']>=3 for r in rows),0)

    def test_preserved_controls_and_audit_are_same_binary(self):
        report=E/'reports/o378_roof_v14'
        audit=json.loads((report/'audit/audit.json').read_text())
        env=json.loads((E/'runs/o378_roof_v14_trace24/environment.json').read_text())
        self.assertEqual(audit['binary_sha256'],env['binary_sha256'])
        self.assertEqual(len(audit['functions']),57)
        self.assertTrue(audit['passed'])
        for name,count in (('baseline_codegen.json',12),('candidate_codegen.json',51)):
            codegen=json.loads((report/name).read_text())
            self.assertTrue(codegen['passed'])
            self.assertEqual(codegen['old_symbols'],count)
            self.assertEqual(len(codegen['unchanged']),count)


if __name__=='__main__': unittest.main()
