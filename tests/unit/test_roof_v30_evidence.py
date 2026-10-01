"""Reconcile v30 evidence; these CPU checks do not repeat GPU experiments."""
import json
from pathlib import Path
import runpy
import unittest

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'docs/evidence/a100_o378_roof_v30'
REPORTS=ROOT/'reports/o378_roof_v30'
BINARY='9d7ee7d29b1b6a990bb9eefb4a700614051a44515c5dbdf8a4f32fe9a843db6a'


@unittest.skipUnless((REPORTS/'audit/audit.json').exists(),'completed archive not present')
class ScaleLayoutEvidenceTests(unittest.TestCase):
    def load(self,path): return json.loads((ROOT/path).read_text())

    def records(self,run):
        return [json.loads(line) for line in (ROOT/'runs'/run/'results.jsonl').read_text().splitlines()]

    def test_native_int4_and_old_target_machine_code(self):
        x=self.load('reports/o378_roof_v30/audit/audit.json')
        self.assertTrue(x['passed']);self.assertEqual(x['binary_sha256'],BINARY)
        self.assertEqual(len(x['functions']),142)
        for tune in (53,54):
            f=next(f for f in x['functions'] if f'Li{tune}E' in f['symbol'])
            for key in ('sass_u4s4','sass_s4s4','sass_no_int8','sass_async'):
                self.assertTrue(f['checks'][key])
            self.assertIn('REG:168',f['resource'])
            self.assertEqual(f['strict_passed'],tune==53)
        for name,count in (('production_codegen.json',12),('candidate_codegen.json',140)):
            x=self.load('reports/o378_roof_v30/'+name)
            self.assertTrue(x['passed']);self.assertEqual(len(x['unchanged']),count)
        x=self.load('reports/o378_roof_v30/all_sm80_codegen.json')
        self.assertFalse(x['passed']);self.assertEqual(x['old_symbols'],272)
        self.assertEqual(len(x['changed']),4)
        self.assertTrue(all('mixed_binary' in s for s in x['changed']))

    def test_finite_safety_scope_and_guards(self):
        for kind,count in (('preflight',192),('memcheck',84),('synccheck',84)):
            x=self.load(f'runs/o378_roof_v30_{kind}/validation.json')
            self.assertTrue(x['passed']);self.assertEqual(len(x['checks']),count)
            self.assertTrue(all(r['semantic_tolerance_passed'] for r in x['checks']))
            if kind!='preflight':
                self.assertIn('ERROR SUMMARY: 0 errors',(REPORTS/f'{kind}.log').read_text())
        self.assertIn('0 hazards displayed (0 errors, 0 warnings)',(REPORTS/'racecheck.log').read_text())
        x=self.load('reports/o378_roof_v30/guard_checks.json')
        self.assertEqual(x['binary_sha256'],BINARY)
        self.assertEqual(len(x['checks']),20);self.assertTrue(all(r['rejected'] for r in x['checks']))
        x=self.load('runs/o378_roof_v30_mixed_regression/validation.json')
        self.assertTrue(x['passed']);self.assertEqual(len(x['binary_gemm_checks']),480)

    def test_real_trace_layout_bitwise_and_mse(self):
        rows=self.records('o378_roof_v30_trace24')
        self.assertEqual(len(rows),600)
        self.assertEqual({r['variant'] for r in rows},{'o3'})
        index={(r['sample_id'],r['round'],r['tune']):r for r in rows}
        self.assertEqual(len(index),600)
        for t in (-1,51,52,53,54):
            group=[r for r in rows if r['tune']==t]
            self.assertEqual(len(group),120)
            self.assertEqual(len({r['sample_id'] for r in group}),24)
            self.assertEqual({r['round'] for r in group},set(range(5)))
            for r in group:
                self.assertEqual(len(r['raw_ms']),200)
                self.assertTrue(r['mse_regression_passed'] and r['semantic_tolerance_passed'])
                if t not in (53,54): continue
                self.assertTrue(r['weight_scale_layout_bitwise_verified'])
                self.assertTrue(r['bitwise_equal_row_scale_baseline'])
                self.assertTrue(r['payload_layout_bitwise_verified'])
                old=index[r['sample_id'],r['round'],t-2]
                self.assertEqual(r['mse_vs_o0'],old['mse_vs_o0'])
                self.assertEqual(r['kernel']['weight_scale_reorder_bytes'],262144)
                self.assertEqual(r['kernel']['cross_group_accumulator_dtype'],'fp32')

    def test_paired_small_gain_not_cross_run_median_ratio(self):
        compare=runpy.run_path(str(REPO/'scripts/compare_roof_trace_candidates.py'))['compare']
        rows=self.records('o378_roof_v30_trace24')
        for t in (53,54):
            x=self.load(f'reports/o378_roof_v30/trace24_{t}_vs52.json')
            self.assertEqual(x['source_binary_sha256'],BINARY)
            self.assertEqual(compare(rows,52,t,24,5,['o3'],['compute_only'],True),x['rows'])
            ci=x['rows'][0]['paired_speedup_ci95']
            if t==54:
                self.assertGreater(ci[0],1);self.assertLess(ci[1],1.01)
            else:
                self.assertLess(ci[0],1);self.assertGreater(ci[1],1)
            self.assertGreater(x['rows'][0]['cv'][str(t)]['selected_stage_failed'],0)

    def test_four_modes_count_new_scale_reorder(self):
        run='o378_roof_v30_four24'
        rows=self.records(run);self.assertEqual(len(rows),288)
        env=self.load(f'runs/{run}/environment.json')
        self.assertEqual(env['binary_sha256'],BINARY)
        self.assertEqual((env['args']['warmup'],env['args']['repeats'],env['args']['inner']),(50,200,100))
        modes=('conversion_only','compute_only','cold','steady_state')
        for t in (-1,52,54):
            for mode in modes:
                group=[r for r in rows if (r['tune'],r['mode'])==(t,mode)]
                self.assertEqual(len(group),24)
                for r in group:
                    self.assertTrue(r['mse_regression_passed'])
                    self.assertEqual(r['total_timing'],'sum_of_batched_stage_samples' if mode=='conversion_only'
                                     else 'single_execution_cuda_event')
                    if t!=54:continue
                    self.assertTrue(r['bitwise_equal_row_scale_baseline'])
                    k=r['kernel'];self.assertEqual(k['weight_scale_reorder_bytes'],262144)
                    self.assertEqual((k['activation_conversion_kernels'],k['weight_conversion_kernels']),(2,3))
                    self.assertEqual(k['activation_payload_reorder_traffic_bytes'],33554432)
                    self.assertEqual(k['weight_payload_reorder_traffic_bytes'],16777216)
        compare=runpy.run_path(str(REPO/'scripts/compare_roof_trace_candidates.py'))['compare']
        x=self.load('reports/o378_roof_v30/four24_54_vs52.json')
        self.assertEqual(compare(rows,52,54,24,1,['o3'],modes,True),x['rows'])

    def test_ncu_memory_layout_not_mma_work_change(self):
        analyze=runpy.run_path(str(REPO/'scripts/analyze_roof_scale_ncu.py'))['analyze']
        x=self.load('reports/o378_roof_v30/ncu_analysis.json')
        self.assertEqual(len(x['rows']),4)
        for row in x['rows']:
            t=row['tune']
            raw=(REPORTS/f'ncu_o3_t{t}_raw.csv').read_text(encoding='utf-8-sig')
            sass=(REPORTS/f'ncu_o3_t{t}_source_sass.csv').read_text(encoding='utf-8-sig')
            self.assertEqual(analyze(raw,sass,t,'o3',True,True),row)
            for op in ('IMMA','I2F','FFMA'):self.assertEqual(row['opcodes'][op],16777216)
            self.assertEqual(row['source_memory_work']['L1 Wavefronts Shared'],29622272)
            self.assertEqual(row['source_memory_work']['L2 Theoretical Sectors Global Excessive'],
                             0 if t in (53,54) else 8126464)
            self.assertAlmostEqual(row['optimistic_fixed_work_lower_bound_ms'],0.22034693984764905)


if __name__=='__main__':unittest.main()
