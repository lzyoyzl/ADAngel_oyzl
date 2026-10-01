"""Reconcile archived v31 data; does not repeat GPU measurements."""
import json
from pathlib import Path
import runpy
import unittest

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'docs/evidence/a100_o378_roof_v31'
REPORTS=ROOT/'reports/o378_roof_v31'
BINARY='35cb37f9dd2219f73236299767defd10a9322d722d134649d1adc2632ee20fa1'


@unittest.skipUnless((REPORTS/'audit/audit.json').exists(),'completed archive not present')
class AsyncScaleEvidenceTests(unittest.TestCase):
    def load(self,path): return json.loads((ROOT/path).read_text())

    def records(self,run):
        return [json.loads(line) for line in (ROOT/'runs'/run/'results.jsonl').read_text().splitlines()]

    def test_native_int4_and_old_target_code(self):
        x=self.load('reports/o378_roof_v31/audit/audit.json')
        self.assertTrue(x['passed']);self.assertEqual(x['binary_sha256'],BINARY)
        self.assertEqual(len(x['functions']),144)
        for tune in (55,56):
            f=next(f for f in x['functions'] if f'Li{tune}E' in f['symbol'])
            for key in ('sass_u4s4','sass_s4s4','sass_no_int8','sass_async',
                        'async_scales_no_scalar_ldg','async_scales_no_scalar_sts',
                        'async_scales_ptx_wait','async_scales_ptx_commit','async_scales_cta_barrier'):
                self.assertTrue(f['checks'][key])
            self.assertIn('REG:168',f['resource'])
            self.assertFalse(f['strict_passed'])
        for name,count in (('production',12),('candidate',142)):
            x=self.load(f'reports/o378_roof_v31/{name}_codegen.json')
            self.assertTrue(x['passed']);self.assertEqual(len(x['unchanged']),count)
        x=self.load('reports/o378_roof_v31/all_sm80_codegen.json')
        self.assertFalse(x['passed']);self.assertEqual(x['old_symbols'],274)
        self.assertEqual(len(x['changed']),5)
        self.assertTrue(all('mixed_binary' in s for s in x['changed']))

    def test_finite_safety_and_guard_scope(self):
        for kind,count in (('preflight',384),('memcheck',168),('synccheck',168)):
            x=self.load(f'runs/o378_roof_v31_{kind}/validation.json')
            self.assertTrue(x['passed']);self.assertEqual(len(x['checks']),count)
            self.assertTrue(all(r['bitwise_equal_production'] for r in x['checks']))
            if kind!='preflight': self.assertIn('ERROR SUMMARY: 0 errors',(REPORTS/f'{kind}.log').read_text())
        self.assertIn('0 hazards displayed (0 errors, 0 warnings)',(REPORTS/'racecheck.log').read_text())
        x=self.load('reports/o378_roof_v31/guard_checks.json')
        self.assertEqual(x['binary_sha256'],BINARY)
        self.assertEqual(len(x['checks']),26);self.assertTrue(all(r['rejected'] for r in x['checks']))
        x=self.load('runs/o378_roof_v31_mixed_regression/validation.json')
        self.assertTrue(x['passed']);self.assertEqual(len(x['binary_gemm_checks']),480)

    def test_balanced_trace_and_bitwise_mse(self):
        for run,rounds,tunes in (('trace24',1,(-1,41,42,55,56)),('trace24_r5',5,(-1,42,56))):
            rows=self.records('o378_roof_v31_'+run)
            self.assertEqual(len(rows),24*2*rounds*len(tunes))
            self.assertEqual(len({(r['sample_id'],r['variant'],r['tune'],r['round']) for r in rows}),len(rows))
            for variant in ('o7','o8'):
                samples={r['sample_id'] for r in rows if r['variant']==variant}
                self.assertEqual(len(samples),24)
                for sid in samples:
                    group=[r for r in rows if r['variant']==variant and r['sample_id']==sid]
                    self.assertEqual(len({r['mse_vs_paired_fp16'] for r in group}),1)
            for r in rows:
                self.assertTrue(r['bitwise_equal_production']);self.assertEqual(r['mse_vs_production'],0)
                self.assertEqual(len(r['raw_ms']),200)
                if r['tune'] not in (55,56): continue
                self.assertTrue(r['async_scale_metadata_verified'] and r['payload_layout_bitwise_verified'])
                self.assertFalse(r['kernel']['fp32_reassociated'])
                self.assertEqual(r['kernel']['scale_copy_alignment_bytes'],16)
                self.assertEqual(r['kernel']['conversion_kernels_per_operand'],2)

    def test_paired_gain_reproduces_without_cv_filtering(self):
        compare=runpy.run_path(str(REPO/'scripts/compare_roof_trace_candidates.py'))['compare']
        rows=self.records('o378_roof_v31_trace24_r5')
        for ref in (-1,42):
            x=self.load(f'reports/o378_roof_v31/trace24_r5_56_vs{ref}.json')
            self.assertEqual(x['source_binary_sha256'],BINARY)
            result=compare(rows,ref,56,24,5,['o7','o8'],['compute_only'])
            self.assertEqual(result,x['rows'])
            for r in result:
                self.assertGreater(r['paired_speedup_ci95'][0],1)
                self.assertGreater(r['cv']['56']['selected_stage_failed'],0)

    def test_four_mode_contract(self):
        rows=self.records('o378_roof_v31_four24')
        self.assertEqual(len(rows),576)
        self.assertEqual({r['mode'] for r in rows},{'conversion_only','compute_only','cold','steady_state'})
        for r in rows:
            self.assertTrue(r['bitwise_equal_production']);self.assertEqual(r['mse_vs_production'],0)
            self.assertEqual(r['conversion_inner_repeats'],100)
            for name,inner in r['stage_timing_inner_repeats'].items():
                self.assertEqual(inner,100 if 'conversion' in name or r['mode']=='conversion_only' else 1)
            self.assertEqual(r['total_timing'],'sum_of_batched_stage_samples' if r['mode']=='conversion_only' else 'single_execution_cuda_event')
            if r['tune']==56:
                self.assertEqual(r['kernel']['activation_payload_reorder_traffic_bytes'],33554432)
                self.assertEqual(r['kernel']['weight_payload_reorder_traffic_bytes'],16777216)

    def test_ncu_reconciles_math_and_async_supply(self):
        analyze=runpy.run_path(str(REPO/'scripts/analyze_roof_scale_ncu.py'))['analyze']
        for variant,tunes in (('o7',(41,42,55,56)),('o8',(42,56))):
            archived=json.loads((REPORTS/f'ncu_analysis_{variant}.json').read_text())['rows']
            actual=[]
            for tune in tunes:
                r=analyze((REPORTS/f'ncu_{variant}_t{tune}_raw.csv').read_text(),
                          (REPORTS/f'ncu_{variant}_t{tune}_source_sass.csv').read_text(),
                          tune,variant,True,True)
                actual.append(r)
                for op in ('IMMA','I2F','FMUL','FFMA'):
                    self.assertEqual(r['opcodes'][op],16777216)
                self.assertEqual(r['registers_per_thread'],168)
                self.assertEqual(r['max_ctas_per_sm_from_launch_limits'],3)
                self.assertAlmostEqual(r['optimistic_fixed_work_lower_bound_ms'],.22034693984764905)
                self.assertAlmostEqual(sum(r['pc_sampling']['reason_share_percent'].values()),100)
                if tune in (55,56):
                    self.assertNotIn('LDG',r['opcodes']);self.assertNotIn('STS',r['opcodes'])
                    self.assertEqual(r['opcodes']['LDGSTS'],2621440)
                    self.assertGreater(r['source_memory_work']['L1 Wavefronts Shared Excessive'],0)
            self.assertEqual(actual,archived)
            old=next(r for r in actual if r['tune']==42)
            new=next(r for r in actual if r['tune']==56)
            self.assertEqual((old['dynamic_instructions'],new['dynamic_instructions']),
                             (117547008,118767616))
            self.assertGreater(new['eligible_warps'],old['eligible_warps'])
            self.assertLess(new['pc_sampling']['reason_share_percent']['long_sb'],
                            old['pc_sampling']['reason_share_percent']['long_sb'])


if __name__=='__main__': unittest.main()
