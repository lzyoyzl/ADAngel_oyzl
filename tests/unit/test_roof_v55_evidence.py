"""Recompute O3 full-K paired screens from retained, unfiltered raw samples."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))


class FullKEvidenceTests(unittest.TestCase):
    def test_full_trace24(self):
        from benchmark_a100_o1 import stats
        from benchmark_roof_fullk_integer_probe import summary
        run=ROOT/'docs/evidence/a100_o378_roof_v55b/runs/o378_roof_v55b_trace24'
        rows=[json.loads(s) for s in (run/'results.jsonl').read_text().splitlines()]
        saved=json.loads((run/'summary.json').read_text())
        self.assertEqual(len(rows),144)
        self.assertEqual(len({r['sample_id'] for r in rows}),24)
        self.assertEqual(summary(rows),saved['records'])
        for r in rows:
            self.assertEqual(len(r['raw_ms']),200)
            for key,value in stats(r['raw_ms']).items():
                self.assertAlmostEqual(value,r['summary'][key],delta=max(1e-14,abs(value)*1e-12))
            self.assertEqual(r['executed_policy'],r['fullk_integer'])
            self.assertTrue(r['bitwise_equal_current_best'])
            self.assertEqual(r['mse_vs_current_best'],0)
            self.assertTrue(r['guard']['safe'])
        self.assertEqual(saved['records'][1]['median_mse'],0.006653010287409885)
        self.assertEqual(saved['records'][1]['mean_mse'],0.007578847013302749)
        self.assertEqual([r['cv_failed_records'] for r in saved['records']],[35,37])

    def test_sanitizers(self):
        d=ROOT/'docs/evidence/a100_o378_roof_v55b/reports/o378_roof_v55b'
        for tool in ('memcheck','synccheck','racecheck'):
            result=json.loads((d/tool/'validation.json').read_text())
            self.assertTrue(result['passed']);self.assertEqual(result['count'],36)
            log=(d/f'{tool}.log').read_text()
            self.assertIn('0 errors',log)
            if tool=='racecheck': self.assertIn('0 warnings',log)

    def test_screens(self):
        from benchmark_a100_o1 import stats
        from benchmark_roof_fullk_integer_probe import summary
        for name in ('v55','v55b'):
            base=ROOT/f'docs/evidence/a100_o378_roof_{name}'
            run=base/f'runs/o378_roof_{name}_screen'
            rows=[json.loads(s) for s in (run/'results.jsonl').read_text().splitlines()]
            saved=json.loads((run/'summary.json').read_text())
            self.assertEqual(len(rows),24)
            self.assertEqual({r['sample_id'] for r in rows},
                {f'layer_00_{p}_proj' for p in ('q','k','v','o')})
            self.assertEqual(summary(rows),saved['records'])
            self.assertFalse(saved['production_default_changed'])
            for r in rows:
                self.assertEqual(len(r['raw_ms']),200)
                for key,value in stats(r['raw_ms']).items():
                    self.assertAlmostEqual(value,r['summary'][key],delta=max(1e-14,abs(value)*1e-12))
                self.assertEqual(r['executed_policy'],r['fullk_integer'])
                self.assertTrue(r['guard']['safe'])
                self.assertLessEqual(r['guard']['max_abs_prefix_bound'],2**31-1)
                self.assertEqual(r['probe_resources']['active_blocks_per_sm'],3)
                self.assertTrue(r['output_close_current_best'])
                self.assertTrue(r['bitwise_equal_current_best'])
                self.assertEqual(r['mse_vs_current_best'],0)
            env=json.loads((run/'environment.json').read_text())
            self.assertEqual(env['extension_sha256'],
                '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462')

    def test_audits_and_preflight(self):
        from compare_a100_codegen import instructions
        from probe_roof_fullk_integer_codegen import PATTERN,static_entries
        olddir=ROOT/'docs/evidence/a100_o378_roof_v54/reports/o378_roof_v54'
        payload=gzip.decompress((olddir/'after.sass.gz').read_bytes()).decode()
        old=instructions(payload,r'adangel_sm80_roof_candidateILb[01]ELb0ELi(?:54|59)EE')
        reference={('adangel_roof_fullk_integer_o3' if 'ELi54EE' in s else 'adangel_roof_fullk_integer_o78'):v
                   for s,v in old.items()}
        for name in ('v55','v55b'):
            d=ROOT/f'docs/evidence/a100_o378_roof_{name}/reports/o378_roof_{name}'
            audit=json.loads((d/'audit.json').read_text())
            self.assertTrue(audit['passed'])
            control=instructions((d/'fullk_integer_0.sass').read_text(),PATTERN)
            self.assertEqual(reference,control)
            for policy in (0,1):
                sass=(d/f'fullk_integer_{policy}.sass').read_text()
                entries=static_entries(sass)
                words=instructions(sass,PATTERN)
                self.assertEqual(words['adangel_roof_fullk_integer_o78'],reference['adangel_roof_fullk_integer_o78'])
                for e in entries.values():
                    self.assertTrue(e['native_u4_s4'] and e['native_s4_s4'] and e['all_copies_bypass_l1'])
                    self.assertFalse(e['int8_mma'])
            for item in audit['sources']:
                src=Path(item['file'])
                local=d/src.name
                raw=gzip.decompress((olddir/'after.sass.gz').read_bytes()) if src.name=='after.sass' else local.read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(),item['sha256'])
            preflight=json.loads((d/'preflight/validation.json').read_text())
            self.assertTrue(preflight['passed']);self.assertEqual(preflight['count'],36)
            self.assertEqual(len(preflight['cache_checks']),2)


if __name__=='__main__': unittest.main()
