"""Recompute v49 code-equivalence evidence; do not invent performance records."""
from pathlib import Path
import hashlib
import itertools
import json
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'docs/evidence/a100_o378_roof_v49'
REPORT=BASE/'reports/o378_roof_v49'
sys.path.insert(0,str(ROOT/'scripts'))


class V49EvidenceTests(unittest.TestCase):
    def test_recompute_full_encoded_sass_and_hashes(self):
        from audit_roof_b_lookahead_probe import audit
        result=audit(REPORT,REPORT/'best_controls.sass')
        self.assertTrue(result['passed'])
        self.assertTrue(all(result['control_encoded_sass_matches_best'].values()))
        saved=json.loads((REPORT/'audit.json').read_text())
        self.assertEqual(result['entries'],saved['entries'])
        self.assertEqual(len(result['entries']),6)
        for row in result['entries']:
            self.assertTrue(row['encoded_identical_to_control'])
            self.assertEqual(row['instructions'],944 if row['symbol'].endswith('_o3') else 880)
            self.assertEqual(row['opcode_counts']['IMMA'],64)
            self.assertEqual(row['opcode_counts']['I2F'],64)
        for item in saved['sources']:
            path=BASE/item['file']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),item['sha256'])

    def test_codegen_and_resource_identity(self):
        from compare_a100_codegen import compare
        from probe_roof_b_lookahead_codegen import PATTERN,static_entries
        manifest=json.loads((REPORT/'codegen.json').read_text())
        self.assertEqual(manifest['source_commit'],'425538371efd91777060cfa6f6de5a0b5ab8e065')
        self.assertTrue(manifest['host_stream_mapping_passed'])
        control=(REPORT/'b_lookahead_0.sass').read_text()
        for policy in (0,1,2):
            text=(REPORT/f'b_lookahead_{policy}.sass').read_text()
            self.assertEqual(static_entries(text),manifest['variants'][str(policy)]['entries'])
            if policy:
                compared=compare(control,text,PATTERN)
                self.assertTrue(compared['passed'])
                self.assertEqual(compared,manifest['variants'][str(policy)]['encoded_comparison_to_control'])
            build=(REPORT/f'b_lookahead_{policy}_build.log').read_text()
            self.assertEqual(build.count('Used 168 registers'),2)
            self.assertIn('16 bytes stack frame, 12 bytes spill stores, 12 bytes spill loads',build)
            self.assertIn('8 bytes stack frame, 8 bytes spill stores, 8 bytes spill loads',build)

    def test_gpu_finite_coverage_without_performance_claims(self):
        d=json.loads((BASE/'runs/o378_roof_v49_preflight/validation.json').read_text())
        self.assertTrue(d['passed'])
        self.assertEqual(d['count'],180)
        expected=set(itertools.product(('o3','o7','o8'),
            ((64,128,128),(64,128,256),(64,128,384),(128,256,640),(64,128,4096)),
            ('random','zero','extrema','zero_scale'),(0,1,2)))
        actual={(r['variant'],tuple(r['shape']),r['pattern'],r['policy']) for r in d['checks']}
        self.assertEqual(actual,expected)
        self.assertEqual(len(d['checks']),len(actual))
        for r in d['checks']:
            self.assertTrue(r['bitwise_equal_best'] and r['finite_fp32'])
            p=r['probe_resources']
            self.assertEqual((p['registers_per_thread'],p['active_blocks_per_sm'],p['threads']),(168,3,128))
            self.assertEqual((p['streamed_n_columns'],p['independent_n_atoms']),(64,4))
            self.assertEqual(p['shared_memory_bytes'],50688 if r['variant']=='o3' else 34304)
        self.assertEqual(d['extension_sha256'],'fe9c2b18d89787231bf988b704a29d2041edcfdad558c98acee3f5b2195b366f')
        self.assertFalse(list(BASE.rglob('results.jsonl')))
        self.assertFalse(list(BASE.rglob('ncu_*analysis.json')))


if __name__=='__main__': unittest.main()
