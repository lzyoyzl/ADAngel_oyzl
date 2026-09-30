"""Reconcile preflight, full24 compute and NCU; full24 four-mode is separate."""
import collections
import itertools
import json
import math
from pathlib import Path
import re
import runpy
import statistics
import unittest

ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'docs/evidence/a100_o378_roof_v17'
P=E/'reports/o378_roof_v17'
ANALYZE=runpy.run_path(str(ROOT/'scripts/analyze_roof_scale_ncu.py'))['analyze']
COMPARE=runpy.run_path(str(ROOT/'scripts/compare_roof_trace_candidates.py'))['compare']


class ReductionBudgetEvidence(unittest.TestCase):
    def test_o78_four_modes_are_complete_balanced_and_bitwise(self):
        d=E/'runs/o378_roof_v17_o78_four24'
        env=json.loads((d/'environment.json').read_text())
        summary=json.loads((d/'summary.json').read_text())
        rows=[json.loads(x) for x in (d/'results.jsonl').read_text().splitlines()]
        modes=['conversion_only','compute_only','cold','steady_state']
        variants=['o7','o8'];tunes=[-1,21,23]
        self.assertEqual(env['binary_sha256'],json.loads((P/'audit/audit.json').read_text())['binary_sha256'])
        self.assertEqual((env['args']['samples'],env['args']['rounds'],env['args']['warmup'],
                          env['args']['repeats'],env['args']['inner']),(24,3,50,200,100))
        self.assertEqual((env['args']['variants'],env['args']['tunes']),(variants,tunes))
        self.assertTrue(summary['all_24_samples'] and summary['all_four_modes_completed']
                        and summary['correctness_passed'] and summary['no_filtering'])
        self.assertEqual(summary['numerical_policy'],'bitwise')
        ids={r['sample_id'] for r in rows}
        self.assertEqual(len(ids),24)
        self.assertEqual(len(rows),1728)
        self.assertEqual({(r['sample_id'],r['variant'],r['mode'],r['round'],r['tune']) for r in rows},
                         set(itertools.product(ids,variants,modes,range(3),tunes)))
        for v,mode,tune in itertools.product(variants,modes,tunes):
            selected=[r for r in rows if (r['variant'],r['mode'],r['tune'])==(v,mode,tune)]
            self.assertEqual(collections.Counter(r['order_position'] for r in selected),dict.fromkeys(range(3),24))
        for r in rows:
            self.assertTrue(r['bitwise_equal_production'])
            self.assertEqual(r['mse_vs_production'],0)
            self.assertEqual(r['conversion_inner_repeats'],100)
            self.assertEqual(r['total_timing'],'sum_of_batched_stage_samples' if r['mode']=='conversion_only'
                             else 'single_execution_cuda_event')
            for stage,values in r['stage_timings_ms'].items():
                self.assertEqual(len(values),200)
                self.assertTrue(all(math.isfinite(x) and x>0 for x in values))
                self.assertAlmostEqual(statistics.median(values),r['stage_summaries'][stage]['median_ms'],places=12)
                self.assertEqual(r['stage_timing_inner_repeats'][stage],100 if
                                 stage.endswith('_conversion') or r['mode']=='conversion_only' else 1)
        for ref,label in ((-1,'prod'),(21,'21')):
            calculated=COMPARE(rows,ref,23,24,3,variants,modes)
            stored=json.loads((P/f'o78_four24_t23_vs{label}.json').read_text())['rows']
            self.assertEqual(calculated,stored)
            for r in calculated:
                self.assertEqual(r['cv']['23']['records'],72)
                if r['mode']!='conversion_only':
                    self.assertGreater(r['paired_speedup_ci95'][0],1)
                    self.assertGreater(r['cv']['23']['any_stage_failed'],0)

    def test_complete_trace_is_balanced_and_all_mse_are_recomputed(self):
        d=E/'runs/o378_roof_v17_trace24'
        env=json.loads((d/'environment.json').read_text())
        summary=json.loads((d/'summary.json').read_text())
        self.assertEqual(env['binary_sha256'],json.loads((P/'audit/audit.json').read_text())['binary_sha256'])
        self.assertTrue(summary['all_24_samples'] and summary['correctness_passed'])
        self.assertFalse(summary['all_four_modes_completed'])
        rows=[json.loads(x) for x in (d/'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),3528)
        for v in ('o3','o7','o8'):
            for t in (-1,6,21,22,23,26,27):
                rs=[r for r in rows if r['variant']==v and r['tune']==t]
                self.assertEqual(collections.Counter(r['order_position'] for r in rs),dict.fromkeys(range(7),24))
        for r in rows:
            self.assertEqual(len(r['raw_ms']),200)
            self.assertTrue(r['semantic_tolerance_passed'] and r['mse_regression_passed'])
            for key,value in r['baseline_mse'].items():
                self.assertLessEqual(abs(r[key]-value),1e-12+1e-5*abs(value))
            if r['tune']<24:
                self.assertTrue(r['bitwise_equal_production'])
                self.assertEqual(r['mse_vs_production'],0)
        for ref,tune in ((6,22),(21,23),(6,26),(6,27)):
            calculated=COMPARE(rows,ref,tune,24,7,['o3','o7','o8'],['compute_only'],True)
            stored=json.loads((P/f'trace24_t{tune}_vs{ref}.json').read_text())['rows']
            self.assertEqual(calculated,stored)
            for r in calculated:
                if tune in (26,27) or (tune==23 and r['variant']=='o3'):
                    self.assertLess(r['paired_speedup_ci95'][1],1)
                else:
                    self.assertGreater(r['paired_speedup_ci95'][0],1)
                self.assertGreater(r['cv'][str(tune)]['any_stage_failed'],0)
        smoke=E/'runs/o378_roof_v17_reduction_four_smoke'
        s=json.loads((smoke/'summary.json').read_text())
        self.assertTrue(s['all_four_modes_completed'] and s['mse_regression_passed'])
        self.assertFalse(s['all_24_samples'])
        self.assertEqual(len((smoke/'results.jsonl').read_text().splitlines()),108)

    def test_o3_reuse_saves_load_work_but_not_proportional_runtime(self):
        results=[]
        for tune,instructions in ((6,117268480),(22,112279552)):
            r=ANALYZE((P/f'ncu_o3_t{tune}_raw.csv').read_text(),
                      (P/f'ncu_o3_t{tune}_source_sass.csv').read_text(),tune,'o3',True,True)
            self.assertEqual(r['dynamic_instructions'],instructions)
            self.assertEqual(r['source_memory_work']['L2 Theoretical Sectors Local'],6291456)
            self.assertEqual(r['source_memory_work_by_opcode']['LDSM']['L1 Wavefronts Shared Excessive'],0)
            results.append(r)
        baseline,candidate=results
        self.assertEqual(candidate['source_memory_work_by_opcode']['LDGSTS']['L1 Wavefronts Shared Excessive'],8388608)
        self.assertEqual(candidate['source_memory_work_by_opcode']['LDSM']['L1 Wavefronts Shared'],16777216)
        self.assertEqual(baseline['source_memory_work_by_opcode']['LDSM']['L1 Wavefronts Shared'],25165824)
        self.assertLess(candidate['issue_active_percent'],baseline['issue_active_percent'])
        self.assertEqual(candidate['binding_modeled_resources'],['mma','i2f'])

    def test_new_entries_have_no_spill_and_old_code_is_preserved(self):
        audit=json.loads((P/'audit/audit.json').read_text())
        self.assertTrue(audit['passed'])
        self.assertEqual(len(audit['functions']),75)
        new=[f for f in audit['functions'] if re.search(r'ELi2[67]EE',f['symbol'])]
        self.assertEqual(len(new),6)
        for f in new:
            self.assertTrue(f['strict_passed'])
            self.assertEqual((f['instruction_counts']['LDL'],f['instruction_counts']['STL']),(0,0))
            self.assertIn('STACK:0',f['resource'])
            self.assertIn('LOCAL:0',f['resource'])
        for name,count in (('baseline_codegen.json',12),('candidate_codegen.json',69)):
            d=json.loads((P/name).read_text())
            self.assertTrue(d['passed'])
            self.assertEqual(len(d['unchanged']),count)

    def test_safety_and_balanced_synthetic_results(self):
        for name,count in (('budget_screen',432),('memcheck',144),('synccheck',144)):
            d=json.loads((E/f'runs/o378_roof_v17_{name}/validation.json').read_text())
            self.assertTrue(d['passed'])
            self.assertEqual(len(d['checks']),count)
        for name in ('memcheck','synccheck'):
            self.assertIn('ERROR SUMMARY: 0 errors',(P/f'{name}.log').read_text())
        self.assertIn('0 hazards displayed (0 errors, 0 warnings)',(P/'racecheck.log').read_text())
        rows=[json.loads(x) for x in (E/'runs/o378_roof_v17_budget_screen/results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),108)
        for v in ('o3','o7','o8'):
            for t in (-1,6,24,25,26,27):
                rs=[r for r in rows if r['variant']==v and r['tune']==t]
                self.assertEqual(collections.Counter(r['order_position'] for r in rs),dict.fromkeys(range(6),1))
        for r in rows:
            self.assertTrue(r['semantic_tolerance_passed'])
            self.assertEqual(len(r['raw_ms']),200)
            if r['tune'] in (26,27):
                self.assertEqual(r['kernel']['max_resident_blocks_per_sm'],1)
                self.assertEqual(r['kernel']['local_bytes_per_thread'],0)
                self.assertGreater(r['kernel']['registers_per_thread'],128)

    def test_zero_spill_is_not_sufficient_and_reuse_is_a_separate_strategy(self):
        for t,instructions,local in ((6,127238144,10485760),(23,116252672,4063232),
                                      (26,125272064,0),(27,138248192,0)):
            r=ANALYZE((P/f'ncu_o7_t{t}_raw.csv').read_text(),
                      (P/f'ncu_o7_t{t}_source_sass.csv').read_text(),t,'o7',True,True)
            self.assertEqual(r['dynamic_instructions'],instructions)
            self.assertEqual(r['source_memory_work']['L2 Theoretical Sectors Local'],local)
            if t==23:
                self.assertEqual(r['max_ctas_per_sm_from_launch_limits'],3)
                self.assertEqual(r['binding_modeled_resources'],['mma','i2f'])
                self.assertAlmostEqual(r['optimistic_fixed_work_lower_bound_ms'],.22034693984764905)
                # Attribute copy replay instead of falsely blaming ldmatrix.
                self.assertEqual(r['source_memory_work_by_opcode']['LDGSTS']['L1 Wavefronts Shared Excessive'],8388608)
                self.assertEqual(r['source_memory_work_by_opcode']['LDSM']['L1 Wavefronts Shared Excessive'],0)
            if t in (26,27):
                self.assertEqual(r['max_ctas_per_sm_from_launch_limits'],1)
                self.assertLess(r['issue_active_percent'],40)
                self.assertGreater(r['ncu_duration_ms'],.55)
                self.assertEqual(r['source_memory_work_omitted_zero_columns'],['L2 Theoretical Sectors Local'])


if __name__=='__main__': unittest.main()
