"""Reconcile archived static-ring measurements; never substitute CPU GPU results."""
import hashlib
import json
from pathlib import Path
import statistics
import unittest

ROOT=Path(__file__).resolve().parents[2]/'docs/evidence/a100_o378_roof_v27'
REPORTS=ROOT/'reports/o378_roof_v27'
BINARY='8a53615ff6f60db20e97e35cb01ecd501a1543564821d5f89efbb6e23984952a'


@unittest.skipUnless((REPORTS/'audit/audit.json').exists(),'completed archive not present')
class StaticRingEvidenceTests(unittest.TestCase):
    def load(self,path): return json.loads((ROOT/path).read_text())

    def trace(self):
        return [json.loads(line) for line in
                (ROOT/'runs/o378_roof_v27_trace24/results.jsonl').read_text().splitlines()]

    def test_native_int4_preserves_old_targets_but_is_not_spill_free(self):
        d=json.loads((REPORTS/'audit/audit.json').read_text())
        self.assertTrue(d['passed'])
        self.assertEqual(d['binary_sha256'],BINARY)
        self.assertEqual(len(d['functions']),132)
        new=[f for f in d['functions'] if any(f'Li{t}E' in f['symbol'] for t in (47,48))]
        self.assertEqual(len(new),6)
        for f in new:
            for key in ('sass_u4s4','sass_s4s4','sass_no_int8','sass_async'):
                self.assertTrue(f['checks'][key])
            self.assertFalse(f['strict_passed'])
            self.assertIn('sass_no_local',f['warnings'])
            self.assertIn('resource_no_stack',f['warnings'])
            self.assertIn('REG:168',f['resource'])
        for name,count in (('production_codegen.json',12),('candidate_codegen.json',126)):
            d=json.loads((REPORTS/name).read_text())
            self.assertTrue(d['passed'])
            self.assertEqual(len(d['unchanged']),count)
        d=json.loads((REPORTS/'all_sm80_codegen.json').read_text())
        self.assertFalse(d['passed'])
        self.assertFalse(d['missing'])
        self.assertEqual(len(d['changed']),2)
        self.assertTrue(all('mixed_binary' in s for s in d['changed']))

    def test_preflight_and_finite_safety_coverage(self):
        for kind,count in (('preflight',576),('memcheck',252),('synccheck',252)):
            d=self.load(f'runs/o378_roof_v27_{kind}/validation.json')
            self.assertTrue(d['passed'])
            self.assertEqual(len(d['checks']),count)
            self.assertTrue(all(r['bitwise_equal_production'] and r['mse_vs_production']==0
                                for r in d['checks']))
            self.assertEqual(self.load(f'runs/o378_roof_v27_{kind}/environment.json')['binary_sha256'],BINARY)
            if kind!='preflight':
                self.assertIn('ERROR SUMMARY: 0 errors',(REPORTS/f'{kind}.log').read_text())
        self.assertIn('0 hazards displayed (0 errors, 0 warnings)',(REPORTS/'racecheck.log').read_text())
        d=self.load('runs/o378_roof_v27_mixed_regression/validation.json')
        self.assertTrue(d['passed'])
        self.assertEqual(d['binary_sha256'],BINARY)
        self.assertEqual(len(d['binary_gemm_checks']),480)

    def test_real_trace_coverage_metadata_and_mse(self):
        env=self.load('runs/o378_roof_v27_trace24/environment.json')
        self.assertEqual(env['binary_sha256'],BINARY)
        self.assertEqual((env['args']['samples'],env['args']['rounds']), (24,1))
        self.assertEqual((env['args']['warmup'],env['args']['repeats']), (50,200))
        rows=self.trace()
        self.assertEqual(len(rows),360)
        for variant in ('o3','o7','o8'):
            for tune in (-1,41,42,47,48):
                group=[r for r in rows if (r['variant'],r['tune'])==(variant,tune)]
                self.assertEqual(len(group),24)
                self.assertTrue(all(r['mode']=='compute_only' and len(r['raw_ms'])==200 for r in group))
            for sid in {r['sample_id'] for r in rows}:
                group=[r for r in rows if (r['variant'],r['sample_id'])==(variant,sid)]
                self.assertEqual(len({r['mse_vs_paired_fp16'] for r in group}),1)
        for r in rows:
            self.assertTrue(r['bitwise_equal_production'])
            self.assertEqual(r['mse_vs_production'],0)
            if r['tune'] not in (47,48): continue
            self.assertTrue(r['payload_layout_bitwise_verified'])
            k=r['kernel']
            self.assertEqual(k['cta_tile'],[64,128,128])
            self.assertTrue(k['compile_time_ring_slots'])
            self.assertEqual(k['pipeline_stages'],2 if r['tune']==47 else 3)
            self.assertEqual(k['accumulators_per_thread'],64)
            self.assertEqual(k['max_resident_blocks_per_sm'],3)
            self.assertFalse(k['fp32_reassociated'])

    def test_paired_results_retain_cv_failures_and_confirm_negative_result(self):
        rows=self.trace()
        for reference in (41,42):
            for candidate in (47,48):
                report=json.loads((REPORTS/f'trace24_{candidate}_vs{reference}.json').read_text())
                self.assertEqual(report['source_binary_sha256'],BINARY)
                for d in report['rows']:
                    groups=[]
                    for tune in (reference,candidate):
                        group=sorted([r for r in rows if (r['variant'],r['tune'])==(d['variant'],tune)],
                                     key=lambda r:r['sample_id'])
                        self.assertEqual(len(group),24)
                        self.assertEqual(d['cv'][str(tune)]['selected_stage_failed'],
                                         sum(r['summary']['cv_percent']>=3 for r in group))
                        groups.append(group)
                    ratio=statistics.median(a['summary']['median_ms']/b['summary']['median_ms']
                                            for a,b in zip(*groups))
                    self.assertEqual(ratio,d['paired_speedup_median'])
                    if reference==(41 if d['variant']=='o3' else 42):
                        self.assertLess(d['paired_speedup_ci95'][1],1)

    def test_ncu_work_counters_and_eligible_warps_not_just_instruction_count(self):
        d=json.loads((REPORTS/'ncu_analysis.json').read_text())
        for source in d['sources']:
            self.assertEqual(hashlib.sha256((ROOT/source['file']).read_bytes()).hexdigest(),source['sha256'])
        rows={r['tune']:r for r in d['rows']}
        self.assertEqual(set(rows),{41,42,47,48})
        for r in rows.values():
            for op in ('IMMA','I2F','FMUL','FFMA'):
                self.assertEqual(r['opcodes'][op],16777216)
            self.assertEqual(r['opcodes']['LDSM'],4194304)
            self.assertEqual(r['source_memory_work']['L1 Wavefronts Shared'],30801920)
            self.assertEqual(r['source_memory_work']['L1 Wavefronts Shared Excessive'],0)
            self.assertEqual(r['registers_per_thread'],168)
            self.assertEqual(r['max_ctas_per_sm_from_launch_limits'],3)
            self.assertEqual(r['binding_modeled_resources'],['mma','i2f'])
        for old,new in ((41,47),(42,48)):
            self.assertGreater(rows[new]['ncu_duration_ms'],rows[old]['ncu_duration_ms'])
            self.assertLess(rows[new]['eligible_warps'],rows[old]['eligible_warps'])
            self.assertLess(rows[new]['issue_active_percent'],rows[old]['issue_active_percent'])
        self.assertEqual(rows[47]['dynamic_instructions'],114106368)
        self.assertLess(rows[47]['dynamic_instructions'],rows[41]['dynamic_instructions'])
        local='L2 Theoretical Sectors Local'
        self.assertEqual(rows[47]['source_memory_work'][local],3211264)
        self.assertEqual(rows[48]['source_memory_work'][local],7110656)
        self.assertLess(rows[47]['source_memory_work'][local],rows[41]['source_memory_work'][local])
        self.assertGreater(rows[48]['source_memory_work'][local],rows[42]['source_memory_work'][local])


if __name__=='__main__': unittest.main()
