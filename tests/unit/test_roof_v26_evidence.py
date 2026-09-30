"""Check archived N64 evidence, including regressions; no simulated GPU data."""
import json
from pathlib import Path
import statistics
import unittest

ROOT=Path(__file__).resolve().parents[2]/'docs/evidence/a100_o378_roof_v26'
REPORTS=ROOT/'reports/o378_roof_v26'
BINARY='d141eaede944a94c4b0261b44917dda7869ce5c9a1005de5d64fd8551876c734'


@unittest.skipUnless((REPORTS/'audit/audit.json').exists(),'completed archive not present')
class V26EvidenceTests(unittest.TestCase):
    def load(self,path): return json.loads((ROOT/path).read_text())

    def test_native_int4_without_spill_and_preserved_target_codegen(self):
        audit=json.loads((REPORTS/'audit/audit.json').read_text())
        self.assertTrue(audit['passed'])
        self.assertEqual(audit['binary_sha256'],BINARY)
        self.assertEqual(len(audit['functions']),126)
        new=[f for f in audit['functions'] if 'Li45E' in f['symbol'] or 'Li46E' in f['symbol']]
        self.assertEqual(len(new),6)
        self.assertTrue(all(f['strict_passed'] and not f['warnings'] for f in new))
        for f in new:
            for key in ('sass_u4s4','sass_s4s4','sass_no_int8','sass_async',
                        'sass_no_local','resource_no_stack'):
                self.assertTrue(f['checks'][key])
        for name,count in (('production_codegen.json',12),('candidate_codegen.json',120)):
            d=json.loads((REPORTS/name).read_text())
            self.assertTrue(d['passed'])
            self.assertEqual(len(d['unchanged']),count)
        broad=json.loads((REPORTS/'all_sm80_codegen.json').read_text())
        self.assertFalse(broad['passed'])  # Do not hide a non-target codegen change.
        self.assertFalse(broad['missing'])
        self.assertEqual(len(broad['changed']),1)
        self.assertIn('mixed_binary',broad['changed'][0])

    def test_preflight_and_sanitizer_coverage(self):
        for kind,count in (('preflight',576),('memcheck',252),('synccheck',252)):
            d=self.load(f'runs/o378_roof_v26_{kind}/validation.json')
            self.assertTrue(d['passed'])
            self.assertEqual(len(d['checks']),count)
            self.assertTrue(all(x['bitwise_equal_production'] and x['mse_vs_production']==0
                                for x in d['checks']))
            env=self.load(f'runs/o378_roof_v26_{kind}/environment.json')
            self.assertEqual(env['binary_sha256'],BINARY)
            if kind!='preflight':
                self.assertIn('ERROR SUMMARY: 0 errors',(REPORTS/f'{kind}.log').read_text())

    def test_real_trace_complete_and_mse_unchanged(self):
        run=ROOT/'runs/o378_roof_v26_trace24'
        if not (run/'summary.json').exists(): self.skipTest('trace archive incomplete')
        env=json.loads((run/'environment.json').read_text())
        self.assertEqual(env['binary_sha256'],BINARY)
        self.assertEqual((env['args']['samples'],env['args']['rounds']), (24,1))
        self.assertEqual((env['args']['warmup'],env['args']['repeats']), (50,200))
        rows=[json.loads(x) for x in (run/'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),360)
        for variant in ('o3','o7','o8'):
            for tune in (-1,41,42,45,46):
                group=[r for r in rows if (r['variant'],r['tune'])==(variant,tune)]
                self.assertEqual(len(group),24)
                self.assertTrue(all(len(r['raw_ms'])==200 and r['mode']=='compute_only' for r in group))
            for sid in {r['sample_id'] for r in rows}:
                group=[r for r in rows if (r['variant'],r['sample_id'])==(variant,sid)]
                self.assertEqual(len({r['mse_vs_paired_fp16'] for r in group}),1)
        self.assertTrue(all(r['bitwise_equal_production'] and r['mse_vs_production']==0 for r in rows))
        for r in rows:
            if r['tune'] not in (45,46): continue
            self.assertTrue(r['payload_layout_bitwise_verified'])
            k=r['kernel']
            self.assertEqual(k['cta_tile'],[64,64,128])
            self.assertEqual(k['accumulators_per_thread'],32)
            self.assertEqual(k['local_bytes_per_thread'],0)
            self.assertEqual(k['max_resident_blocks_per_sm'],4)

    def test_pair_summaries_match_raw_without_cv_filtering(self):
        run=ROOT/'runs/o378_roof_v26_trace24'
        if not (run/'summary.json').exists(): self.skipTest('trace archive incomplete')
        rows=[json.loads(x) for x in (run/'results.jsonl').read_text().splitlines()]
        for candidate in (45,46):
            for reference in (41,42):
                report=json.loads((REPORTS/f'trace24_{candidate}_vs{reference}.json').read_text())
                for d in report['rows']:
                    groups=[]
                    for tune in (reference,candidate):
                        group=sorted([r for r in rows if (r['variant'],r['tune'])==(d['variant'],tune)],
                                     key=lambda r:r['sample_id'])
                        self.assertEqual(len(group),24)
                        self.assertEqual(d['cv'][str(tune)]['selected_stage_failed'],
                                         sum(r['summary']['cv_percent']>=3 for r in group))
                        groups.append(group)
                    speed=statistics.median(a['summary']['median_ms']/b['summary']['median_ms']
                                            for a,b in zip(*groups))
                    self.assertEqual(speed,d['paired_speedup_median'])

    def test_more_occupancy_does_not_mean_less_work(self):
        d=self.load('ncu_analysis.json')
        rows={r['tune']:r for r in d['rows']}
        self.assertEqual(set(rows),{42,45,46})
        for tune,total in ((45,131973120),(46,137723904)):
            r=rows[tune]
            self.assertEqual(r['dynamic_instructions'],total)
            self.assertEqual(r['source_memory_work']['L2 Theoretical Sectors Local'],0)
            self.assertEqual(r['opcodes']['LDSM'],rows[42]['opcodes']['LDSM']*1.5)
            for op in ('IMMA','I2F','FMUL','FFMA'):
                self.assertEqual(r['opcodes'][op],16777216)
            self.assertEqual(r['source_memory_work']['L1 Wavefronts Shared'],44564480)
            self.assertGreater(r['eligible_warps'],rows[42]['eligible_warps'])
            self.assertGreater(r['ncu_duration_ms'],rows[42]['ncu_duration_ms'])
            self.assertEqual(r['binding_modeled_resources'],['l1tex_data_wavefront_capacity'])
            self.assertGreater(r['optimistic_fixed_work_lower_bound_ms'],
                               rows[42]['optimistic_fixed_work_lower_bound_ms'])
            self.assertLess(r['dram_read_bytes'],rows[42]['dram_read_bytes'])

    def test_auxiliary_regression_and_racecheck_not_omitted(self):
        d=self.load('runs/o378_roof_v26_mixed_regression/validation.json')
        self.assertTrue(d['passed'])
        self.assertEqual(d['binary_sha256'],BINARY)
        self.assertEqual(len(d['binary_gemm_checks']),480)
        self.assertIn('0 hazards displayed (0 errors, 0 warnings)',
                      (REPORTS/'racecheck.log').read_text())


if __name__=='__main__': unittest.main()
