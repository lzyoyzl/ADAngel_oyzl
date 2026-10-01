"""Recompute v50 paired alignment performance, MSE, safety and NCU evidence."""
import hashlib
import json
from pathlib import Path
import runpy
import statistics
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'docs/evidence/a100_o378_roof_v50'
P=E/'reports/o378_roof_v50'
sys.path.insert(0,str(ROOT/'scripts'))


def read(path): return json.loads(path.read_text())


class V50EvidenceTests(unittest.TestCase):
    def test_all_pairs_and_unchanged_measured_mse(self):
        run=E/'runs/o378_roof_v50_trace24'
        rows=[json.loads(s) for s in (run/'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),144)
        self.assertEqual(len({r['sample_id'] for r in rows}),24)
        for row in rows:
            self.assertEqual(len(row['raw_ms']),200)
            self.assertTrue(all(v>0 for v in row['raw_ms']))
            self.assertEqual(statistics.median(row['raw_ms']),row['summary']['median_ms'])
            self.assertTrue(row['bitwise_equal_current_best'])
            self.assertEqual(row['mse_vs_current_best'],0)
            self.assertEqual(row['mse_vs_o0'],row['current_best_mse_vs_o0'])
        fn=runpy.run_path(str(ROOT/'scripts/benchmark_roof_pair_alignment_probe.py'))['summary']
        result=fn(rows)
        self.assertEqual(result,read(run/'summary.json')['records'])
        self.assertEqual([r['cv_failed_records'] for r in result],[67,1])
        self.assertLess(result[1]['paired_speedup_ci95'][1],.4)

    def test_native_isa_and_exact_control(self):
        fn=runpy.run_path(str(ROOT/'scripts/audit_roof_pair_alignment_probe.py'))['audit']
        result=fn(P,P/'best_controls.sass')
        self.assertTrue(result['passed'])
        self.assertTrue(all(result['control_encoded_sass_matches_best'].values()))
        self.assertEqual(len(result['entries']),4)
        self.assertEqual(result['entries'],read(P/'audit.json')['entries'])
        for r in result['entries']:
            self.assertTrue(r['native_u4_s4'] and r['native_s4_s4'] and r['all_copies_bypass_l1'])
            self.assertFalse(r['int8_mma'])

    def test_safety_guard_shapes_and_resources(self):
        for mode in ('preflight_v2','memcheck','synccheck','racecheck'):
            d=read(E/f'runs/o378_roof_v50_{mode}/validation.json')
            self.assertTrue(d['passed'])
            self.assertEqual(len(d['checks']),75)
            self.assertEqual({r['shape'][2] for r in d['checks']},{128,256,384,640,4096})
            self.assertEqual(d['extension_sha256'],'fe9c2b18d89787231bf988b704a29d2041edcfdad558c98acee3f5b2195b366f')
            subnormal=[r for r in d['checks'] if r['pattern']=='subnormal']
            self.assertEqual(len(subnormal),5)
            self.assertTrue(all(r['policy']==1 and r['best_reference']=='fp64_semantic_float' for r in subnormal))
            self.assertEqual({r['pattern'] for r in d['checks']},
                {'random','zero','extrema','zero_scale','guard13','fallback14','fallback15','subnormal'})
            for r in d['checks']:
                self.assertTrue(r['finite_fp32'])
                p=r['policy'];resources=r['probe_resources']
                self.assertEqual(resources['registers_per_thread'],[168,245][p])
                self.assertEqual(resources['active_blocks_per_sm'],[3,2][p])
                self.assertEqual(resources['active_consumer_warps_per_sm'],[12,8][p])
                self.assertEqual(resources['shared_memory_bytes'],[50688,67584][p])
                self.assertEqual(resources['local_size_bytes'],[16,0][p])
                if r['pattern'].startswith('fallback'):
                    self.assertTrue(r['bitwise_equal_best'])
            if mode!='preflight_v2':
                marker='RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if mode=='racecheck' else 'ERROR SUMMARY: 0 errors'
                self.assertIn(marker,(P/f'{mode}.log').read_text())

    def test_ncu_reconciles_saved_counter_work(self):
        fn=runpy.run_path(str(ROOT/'scripts/profile_roof_pair_alignment_probe.py'))['analyze_profile']
        saved=read(P/'ncu_full/ncu_o3_analysis.json');cg=read(P/'codegen.json')
        for source in saved['sources']:
            self.assertEqual(hashlib.sha256((E/source['file']).read_bytes()).hexdigest(),source['sha256'])
        computed=[]
        for policy in (0,1):
            prefix=P/f'ncu_full/ncu_o3_p{policy}'
            raw=Path(str(prefix)+'_raw.csv').read_text(encoding='utf-8-sig')
            sass=Path(str(prefix)+'_source_sass.csv').read_text(encoding='utf-8-sig')
            rows=[json.loads(s) for s in (E/f'runs/o378_roof_v50_ncu_o3_p{policy}/results.jsonl').read_text().splitlines()]
            resources=next(r['probe_resources'] for r in rows if r['pair_alignment']==policy)
            r=fn(raw,sass,'o3',policy,resources,cg['variants'][str(policy)]['entries']['adangel_roof_pair_alignment_o3'])
            self.assertEqual(r['opcodes']['IMMA'],16777216)
            self.assertEqual(r['opcodes']['I2F'],[16777216,8388608][policy])
            self.assertEqual(r['opcodes']['FFMA'],[16777216,8388608][policy])
            self.assertEqual(r['dynamic_instructions'],[99319808,222838784][policy])
            self.assertEqual(r['opcodes']['BAR'],[262144,131072][policy])
            computed.append(r)
        self.assertEqual(computed,saved['rows'])


if __name__=='__main__': unittest.main()
