from pathlib import Path
import random
import json
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from roof_gpu_factor_probe import fast_guard_flag
from o3_fullk_probe import guard_columns,PARTIAL_BOUND,INT32_MAX


class GPUFactorProofTests(unittest.TestCase):
    def test_integer_threshold(self):
        self.assertEqual(INT32_MAX//PARTIAL_BOUND,16383)
        self.assertLessEqual(32*(1<<14),2**32-1)
        # sum=16383 is safe, one additional unit is unsafe.
        self.assertLessEqual(16383*PARTIAL_BOUND,INT32_MAX)
        self.assertGreater(16384*PARTIAL_BOUND,INT32_MAX)

    def test_clamping_does_not_change_decision(self):
        rng=random.Random(60)
        columns=[[rng.randrange(1,255) for _ in range(32)] for _ in range(1000)]
        columns += [[116]+[116+rng.randrange(14) for _ in range(31)] for _ in range(1000)]
        columns += [[116]*31+[116+d] for d in range(20)]
        for c in columns:
            safe=guard_columns([c])[1]['safe']
            self.assertEqual(fast_guard_flag(c)==0,safe)
        self.assertEqual(fast_guard_flag([127]*32),0)
        self.assertTrue(fast_guard_flag([0]*32)&4)
        self.assertTrue(fast_guard_flag([255]*32)&2)


class GPUFactorEvidenceTests(unittest.TestCase):
    root=ROOT/'docs/evidence/a100_o378_roof_v60'

    def test_paired_screen_reproduces(self):
        from benchmark_roof_gpu_factor_probe import summarize,stats
        d=self.root/'runs/o378_roof_v60_screen_fixed'
        rows=[json.loads(s) for s in (d/'results.jsonl').read_text().splitlines()]
        saved=json.loads((d/'summary.json').read_text())
        self.assertEqual(len(rows),48)
        self.assertEqual({r['sample_id'] for r in rows},{f'layer_00_{p}_proj' for p in ('q','k','v','o')})
        self.assertEqual(summarize(rows),saved['records'])
        for r in rows:
            self.assertEqual(len(r['raw_ms']),200)
            for key,value in stats(r['raw_ms']).items():
                self.assertAlmostEqual(value,r['summary'][key],places=12)
            if r['mode']!='prepare_gpu_only':self.assertTrue(r['output_bitwise_best'])
            if r['mode']!='control_compute':
                self.assertTrue(r['metadata_exact']);self.assertEqual(r['guard_status'],0)
        self.assertFalse(saved['production_default_changed'])
        for bad in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(ValueError):summarize(bad)

    def test_unchanged_gemm_and_resource_gate(self):
        d=self.root/'runs/o378_roof_v60_screen/build'
        old=json.loads((ROOT/'docs/evidence/a100_o378_roof_v59/reports/o378_roof_v59/codegen.json').read_text())
        build=json.loads((d/'build.json').read_text())
        self.assertEqual(build['gemm_cubins'],{i:v['cubin_sha256'] for i,v in old['variants'].items()})
        log=(d/'build.log').read_text()
        self.assertIn('Used 48 registers',log)
        self.assertIn('0 bytes spill stores, 0 bytes spill loads',log)
        env=json.loads((self.root/'runs/o378_roof_v60_screen_fixed/environment.json').read_text())
        self.assertEqual(env['extension_sha256'],
            '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462')
        self.assertIn('NOT full Cold',env['scope'])

    def test_scoped_safety_and_invalid_rejection(self):
        for name in ('screen_fixed','memcheck','synccheck','racecheck'):
            r=json.loads((self.root/f'runs/o378_roof_v60_{name}/validation.json').read_text())
            self.assertTrue(r['passed']);self.assertEqual(r['count'],48);self.assertEqual(r['rejections'],12)
            self.assertEqual(max(c['shape'][0] for c in r['checks']),128)
            self.assertEqual(max(c['shape'][1] for c in r['checks']),256)
            for c in r['checks']:
                if c['mode']!='control_compute':
                    self.assertTrue(c['metadata_exact'])
                    self.assertEqual(c['status'],0 if c['guard_safe'] else 1)
            if name!='screen_fixed':
                log=(self.root/f'reports/o378_roof_v60_{name}.log').read_text()
                self.assertIn('0 errors',log)
                if name=='racecheck':self.assertIn('0 warnings',log)


if __name__=='__main__':unittest.main()
