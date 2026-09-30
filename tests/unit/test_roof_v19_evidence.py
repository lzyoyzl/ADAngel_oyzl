"""Reconcile pipeline/scheduling diagnostics without claiming trace acceptance."""
import collections
import itertools
import json
from pathlib import Path
import re
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'docs/evidence/a100_o378_roof_v19'
P=E/'reports/o378_roof_v19'
ANALYZE=runpy.run_path(str(ROOT/'scripts/analyze_roof_scale_ncu.py'))['analyze']
COMPARE=runpy.run_path(str(ROOT/'scripts/compare_roof_trace_candidates.py'))['compare']


class CopyAndFinishEvidence(unittest.TestCase):
    def test_trace24_does_not_promote_a_small_win_over_the_wrong_control(self):
        d=E/'runs/o378_roof_v19_finish_trace24'
        env=json.loads((d/'environment.json').read_text())
        summary=json.loads((d/'summary.json').read_text())
        rows=[json.loads(x) for x in (d/'results.jsonl').read_text().splitlines()]
        self.assertEqual(env['binary_sha256'],json.loads((P/'audit/audit.json').read_text())['binary_sha256'])
        self.assertTrue(summary['all_24_samples'] and summary['correctness_passed'] and summary['no_filtering'])
        self.assertFalse(summary['all_four_modes_completed'])
        self.assertEqual((env['args']['samples'],env['args']['rounds'],env['args']['warmup'],
                          env['args']['repeats']),(24,5,50,200))
        self.assertEqual(len(rows),1800)
        ids={r['sample_id'] for r in rows}
        variants=['o3','o7','o8'];tunes=[-1,6,22,23,30]
        self.assertEqual(len(ids),24)
        self.assertEqual({(r['sample_id'],r['variant'],r['round'],r['tune']) for r in rows},
                         set(itertools.product(ids,variants,range(5),tunes)))
        for r in rows:
            self.assertEqual(r['mode'],'compute_only')
            self.assertTrue(r['bitwise_equal_production'])
            self.assertEqual(r['mse_vs_production'],0)
            self.assertEqual(len(r['raw_ms']),200)
        for v,t in itertools.product(variants,tunes):
            rs=[r for r in rows if (r['variant'],r['tune'])==(v,t)]
            self.assertEqual(collections.Counter(r['order_position'] for r in rs),dict.fromkeys(range(5),24))
        for ref,label in ((-1,'prod'),(22,'22'),(23,'23')):
            calculated=COMPARE(rows,ref,30,24,5,variants,['compute_only'])
            self.assertEqual(calculated,json.loads((P/f'trace24_t30_vs{label}.json').read_text())['rows'])
            for r in calculated:
                lo,hi=r['paired_speedup_ci95']
                if ref==22 and r['variant']=='o3':
                    self.assertLess(lo,1);self.assertGreater(hi,1)
                elif ref==23 and r['variant']!='o3':
                    self.assertLess(hi,1)
                else:
                    self.assertGreater(lo,1)
                self.assertGreater(r['cv']['30']['any_stage_failed'],0)

    def test_original_instructions_and_safety_are_preserved(self):
        audit=json.loads((P/'audit/audit.json').read_text())
        self.assertTrue(audit['passed'])
        self.assertEqual(len(audit['functions']),87)
        new=[f for f in audit['functions'] if re.search(r'ELi(?:29|30|31)EE',f['symbol'])]
        self.assertEqual(len(new),9)
        for f in new:
            self.assertIn('REG:168',f['resource'])
            self.assertGreater(f['instruction_counts']['IMMA'],0)
            self.assertGreater(f['instruction_counts']['LDGSTS'],0)
        for name,count in (('baseline_codegen',12),('candidate_codegen',78)):
            r=json.loads((P/(name+'.json')).read_text())
            self.assertTrue(r['passed'])
            self.assertEqual(len(r['unchanged']),count)
        for name,count in (('copy_finish_screen',774),('memcheck',378),('synccheck',378)):
            r=json.loads((E/f'runs/o378_roof_v19_{name}/validation.json').read_text())
            self.assertTrue(r['passed'])
            self.assertEqual(len(r['checks']),count)
        for name in ('memcheck','synccheck'):
            self.assertIn('ERROR SUMMARY: 0 errors',(P/(name+'.log')).read_text())
        self.assertIn('0 hazards displayed (0 errors, 0 warnings)',(P/'racecheck.log').read_text())

    def test_synthetic_order_is_balanced_and_new_metadata_is_explicit(self):
        d=E/'runs/o378_roof_v19_copy_finish_screen'
        rows=[json.loads(x) for x in (d/'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),147)
        for v in ('o3','o7','o8'):
            for t in (-1,6,22,23,29,30,31):
                rs=[r for r in rows if (r['variant'],r['tune'])==(v,t)]
                self.assertEqual(collections.Counter(r['order_position'] for r in rs),dict.fromkeys(range(7),1))
                for r in rs:
                    self.assertTrue(r['bitwise_equal_production'])
                    self.assertEqual(len(r['raw_ms']),200)
                    self.assertEqual(r['kernel']['interleaved_mma_finish'],t in (30,31))
                    self.assertEqual(r['kernel']['paired_g128_copy'],t==29)
                    self.assertFalse(r['kernel']['fp32_reassociated'])
                    if t in (29,30,31):
                        self.assertEqual(r['kernel']['max_resident_blocks_per_sm'],3)

    def test_measured_copy_work_refutes_simple_bank_prediction(self):
        results={}
        for t in (22,23,29,30,31):
            r=ANALYZE((P/f'ncu_o7_t{t}_raw.csv').read_text(),
                      (P/f'ncu_o7_t{t}_source_sass.csv').read_text(),t,'o7',True,True)
            self.assertEqual(r['source_memory_work_by_opcode']['LDSM']['L1 Wavefronts Shared Excessive'],0)
            self.assertEqual(r['max_ctas_per_sm_from_launch_limits'],3)
            self.assertEqual(r['opcodes']['IMMA'],16777216)
            self.assertEqual(r['opcodes']['I2F'],16777216)
            self.assertEqual(r['opcodes']['FFMA'],16777216)
            results[t]=r
        self.assertEqual(results[29]['source_memory_work_by_opcode']['LDGSTS']['L1 Wavefronts Shared Excessive'],8365748)
        self.assertGreater(results[29]['opcodes']['IMAD'],2*results[22]['opcodes']['IMAD'])
        self.assertEqual(results[29]['binding_modeled_resources'],['all_instruction_issue'])
        self.assertGreater(results[29]['dynamic_instructions'],results[22]['dynamic_instructions'])
        self.assertLess(results[31]['source_memory_work']['L2 Theoretical Sectors Local'],
                        results[23]['source_memory_work']['L2 Theoretical Sectors Local'])
        self.assertGreater(results[31]['ncu_duration_ms'],results[23]['ncu_duration_ms'])
        self.assertLess(results[30]['dynamic_instructions'],results[22]['dynamic_instructions'])
        for t in (22,23,30,31):
            self.assertAlmostEqual(results[t]['optimistic_fixed_work_lower_bound_ms'],.22034693984764905)

    def test_all_modes_smoke_includes_rejected_candidates_but_is_not_24_samples(self):
        d=E/'runs/o378_roof_v19_copy_finish_four_smoke'
        s=json.loads((d/'summary.json').read_text())
        self.assertTrue(s['all_four_modes_completed'] and s['correctness_passed'])
        self.assertFalse(s['all_24_samples'])
        rs=[json.loads(x) for x in (d/'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rs),60)
        self.assertEqual({(r['variant'],r['mode'],r['tune']) for r in rs},
                         set(itertools.product(['o3','o7','o8'],
                             ['conversion_only','compute_only','cold','steady_state'],[-1,22,29,30,31])))
        for r in rs:
            self.assertTrue(r['bitwise_equal_production'])
            self.assertEqual(r['mse_vs_production'],0)
        for v in ('o3','o7','o8'):
            self.assertEqual(len({r['mse_vs_paired_fp16'] for r in rs if r['variant']==v}),1)


if __name__=='__main__': unittest.main()
