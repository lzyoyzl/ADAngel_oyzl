"""Archived unsigned-address candidate evidence; no GPU rerun here."""
import json
from pathlib import Path
import runpy
import unittest

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'docs/evidence/a100_o378_roof_v33'
REPORTS=ROOT/'reports/o378_roof_v33'
BINARY='402c9ebd54670fef538389d7c72adea7d2f0abbec65c2185f92f071e1052690f'


@unittest.skipUnless((REPORTS/'audit/audit.json').exists(),'completed archive not present')
class UnsignedEvidenceTests(unittest.TestCase):
    def load(self,path): return json.loads((ROOT/path).read_text())

    def test_audit_preserves_native_math_and_discloses_spills(self):
        x=self.load('reports/o378_roof_v33/audit/audit.json')
        self.assertTrue(x['passed']);self.assertEqual(x['binary_sha256'],BINARY)
        self.assertEqual(len(x['functions']),148)
        for tune in (59,60):
            f=next(f for f in x['functions'] if f'Li{tune}E' in f['symbol'])
            self.assertTrue(f['passed']);self.assertFalse(f['strict_passed'])
            self.assertIn('REG:168',f['resource'])
            self.assertIn('STACK:8' if tune==59 else 'STACK:16',f['resource'])
            for key in ('sass_u4s4','sass_s4s4','sass_no_int8','sass_async',
                        'async_scales_no_scalar_ldg','async_scales_no_scalar_sts'):
                self.assertTrue(f['checks'][key])
        for name,count in (('production',12),('candidate',146)):
            x=self.load(f'reports/o378_roof_v33/{name}_codegen.json')
            self.assertTrue(x['passed']);self.assertEqual(len(x['unchanged']),count)
        x=self.load('reports/o378_roof_v33/all_sm80_codegen.json')
        self.assertEqual(len(x['changed']),3)
        self.assertTrue(all('mixed_binary' in s for s in x['changed']))

    def test_finite_correctness_and_safety(self):
        for name,count in (('preflight',384),('memcheck',168),('synccheck',168)):
            x=self.load(f'runs/o378_roof_v33_{name}/validation.json')
            self.assertTrue(x['passed']);self.assertEqual(len(x['checks']),count)
            for r in x['checks']:
                self.assertTrue(r['bitwise_equal_production'])
                if name=='preflight': self.assertTrue(r['semantic_tolerance_passed'])
                if r['tune'] in (59,60):
                    self.assertEqual(r['kernel']['dimension_addressing'],'host_bounded_uint32')
                    self.assertFalse(r['kernel']['fp32_reassociated'])
        for name in ('memcheck','synccheck'):
            self.assertIn('ERROR SUMMARY: 0 errors',(REPORTS/f'{name}.log').read_text())
        self.assertIn('0 hazards displayed (0 errors, 0 warnings)',(REPORTS/'racecheck.log').read_text())
        x=self.load('reports/o378_roof_v33/guard_checks.json')
        self.assertEqual(len(x['checks']),26);self.assertTrue(all(r['rejected'] for r in x['checks']))
        self.assertTrue(self.load('runs/o378_roof_v33_mixed_regression/validation.json')['passed'])

    def test_trace_coverage_bitwise_and_paired_statistics(self):
        compare=runpy.run_path(str(REPO/'scripts/compare_roof_trace_candidates.py'))['compare']
        for run,rounds,tunes,candidates in (('trace24',1,(-1,55,56,59,60),(59,60)),
                                            ('trace24_r5',5,(-1,56,59),(59,))):
            rows=[json.loads(l) for l in (ROOT/f'runs/o378_roof_v33_{run}/results.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),24*2*rounds*len(tunes))
            for r in rows:
                self.assertTrue(r['bitwise_equal_production']);self.assertEqual(r['mse_vs_production'],0)
                self.assertEqual(len(r['raw_ms']),200)
            for t in candidates:
                x=self.load(f'reports/o378_roof_v33/{run}_{t}_vs56.json')
                self.assertEqual(x['source_binary_sha256'],BINARY)
                self.assertEqual(compare(rows,56,t,24,rounds,['o7','o8'],['compute_only']),x['rows'])

    def test_ncu_reconciles_unchanged_math(self):
        analyze=runpy.run_path(str(REPO/'scripts/analyze_roof_scale_ncu.py'))['analyze']
        for v in ('o7','o8'):
            x=self.load(f'reports/o378_roof_v33/ncu_{v}_analysis.json')
            self.assertEqual({r['tune'] for r in x['rows']},{56,59,60})
            for r in x['rows']:
                stem=f'ncu_{v}_t{r["tune"]}'
                got=analyze((REPORTS/f'{stem}_raw.csv').read_text(),
                            (REPORTS/f'{stem}_source_sass.csv').read_text(),r['tune'],v,True,True)
                self.assertEqual(got,r)
                for op in ('IMMA','I2F','FMUL','FFMA'): self.assertEqual(r['opcodes'][op],16777216)
                self.assertEqual(r['max_ctas_per_sm_from_launch_limits'],3)


@unittest.skipUnless((ROOT/'runs/o378_roof_v33_four24/summary.json').exists(),
                     'completed four-mode archive not present')
class UnsignedFourModeEvidenceTests(unittest.TestCase):
    def load(self,path): return json.loads((ROOT/path).read_text())

    def test_rebuild_codegen_and_non_target_regression(self):
        prefix='reports/o378_roof_v33_four'
        for name,count in (('production',12),('candidate',148)):
            x=self.load(f'{prefix}/{name}_codegen.json')
            self.assertTrue(x['passed']);self.assertEqual(len(x['unchanged']),count)
        x=self.load(f'{prefix}/all_sm80_codegen.json')
        self.assertFalse(x['passed']);self.assertEqual(len(x['changed']),3)
        self.assertTrue(all('mixed_binary' in s for s in x['changed']))
        x=self.load(f'{prefix}/audit/audit.json')
        self.assertTrue(x['passed'])
        self.assertEqual(x['binary_sha256'],
            '02e6433d560f18344d8db2c1d5a46e6c9423837c5559f140cc0ba60af14a24c7')
        reg=self.load('runs/o378_roof_v33_four_mixed_regression/validation.json')
        self.assertTrue(reg['passed']);self.assertEqual(len(reg['binary_gemm_checks']),480)

    def test_full_mode_statistics_and_timing_contract(self):
        run='runs/o378_roof_v33_four24'
        rows=[json.loads(l) for l in (ROOT/run/'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),576)
        modes=['conversion_only','compute_only','cold','steady_state']
        self.assertEqual({r['mode'] for r in rows},set(modes))
        for r in rows:
            self.assertTrue(r['bitwise_equal_production']);self.assertEqual(r['mse_vs_production'],0)
            self.assertEqual(len(r['raw_ms']),200)
            self.assertEqual(r['conversion_inner_repeats'],100)
            for name,inner in r['stage_timing_inner_repeats'].items():
                self.assertEqual(inner,100 if 'conversion' in name or r['mode']=='conversion_only' else 1)
            self.assertEqual(r['total_timing'],'sum_of_batched_stage_samples' if r['mode']=='conversion_only' else 'single_execution_cuda_event')
            if r['tune']==59:
                meta=r['kernel']
                self.assertEqual(meta['activation_payload_reorder_traffic_bytes'],33554432)
                self.assertEqual(meta['weight_payload_reorder_traffic_bytes'],16777216)
                self.assertEqual(meta['pipeline_stages'],2)
                self.assertEqual(meta['dimension_addressing'],'host_bounded_uint32')
        compare=runpy.run_path(str(REPO/'scripts/compare_roof_trace_candidates.py'))['compare']
        for ref in (-1,56):
            x=self.load(f'reports/o378_roof_v33_four/four24_59_vs{ref}.json')
            self.assertEqual(compare(rows,ref,59,24,1,['o7','o8'],modes),x['rows'])


if __name__=='__main__': unittest.main()
