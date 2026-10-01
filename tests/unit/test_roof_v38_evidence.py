import json
from pathlib import Path
import runpy
import statistics
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
EVIDENCE=ROOT/'docs/evidence/a100_o378_roof_v38'
sys.path.insert(0,str(ROOT/'scripts'))


class V38EvidenceTests(unittest.TestCase):
    def test_full_trace_recomputes_with_no_filtering(self):
        run=EVIDENCE/'runs/o378_roof_v38_trace24'
        rows=[json.loads(s) for s in (run/'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),648)
        self.assertEqual(len({r['sample_id'] for r in rows}),24)
        self.assertTrue(all(len(r['raw_ms'])==200 and r['bitwise_equal_current_best'] for r in rows))
        for row in rows:
            self.assertEqual(statistics.median(row['raw_ms']),row['summary']['median_ms'])
            self.assertTrue(all(x>0 for x in row['raw_ms']))
        summarize=runpy.run_path(str(ROOT/'scripts/benchmark_roof_l2_probe.py'))['summary']
        result=summarize(rows)
        self.assertEqual(result,json.loads((run/'summary.json').read_text())['records'])
        for row in result:
            lo,hi=row['paired_speedup_ci95']
            self.assertLessEqual(lo,1.0);self.assertGreaterEqual(hi,1.0)
            self.assertGreater(row['cv_failed_records'],0)

    def test_encoded_controls_native_int4_and_only_ltc_change(self):
        directory=EVIDENCE/'reports/o378_roof_v38'
        check=runpy.run_path(str(ROOT/'scripts/audit_roof_l2_probe.py'))['audit']
        result=check(directory,directory/'best_controls.sass')
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['entries']),6)
        archived=json.loads((directory/'hint_audit_portable.json').read_text())
        self.assertEqual(result['entries'],archived['entries'])
        self.assertEqual(result['control_encoded_sass_matches_best'],archived['control_encoded_sass_matches_best'])

    def test_finite_safety_scope_and_preserved_extension(self):
        for mode in ('memcheck','synccheck'):
            log=(EVIDENCE/f'reports/o378_roof_v38/{mode}.log').read_text()
            self.assertIn('ERROR SUMMARY: 0 errors',log)
            run=EVIDENCE/f'runs/o378_roof_v38_{mode}'
            env=json.loads((run/'environment.json').read_text())
            self.assertEqual(env['args']['samples'],1)
            self.assertEqual(env['extension_sha256'],'fe9c2b18d89787231bf988b704a29d2041edcfdad558c98acee3f5b2195b366f')
            rows=[json.loads(s) for s in (run/'results.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),9)
            self.assertTrue(all(r['bitwise_equal_current_best'] for r in rows))


if __name__=='__main__': unittest.main()
