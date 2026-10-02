from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_roof_distributed_copy_codegen import generated_header,HEADERS


class DistributedCopyTests(unittest.TestCase):
    def test_control_unchanged_and_math_preserved(self):
        for name in HEADERS.values():
            source=(ROOT/'csrc/sm80'/name).read_text()
            self.assertEqual(source,generated_header(source,0))
            candidate=generated_header(source,1)
            for marker in ('cute::gemm(', '__fmaf_rn(', '__fmul_rn(',
                           '__syncthreads()', 'cp.async.wait_group',
                           'const int partial=pl(vi)+16*ph(vi);'):
                self.assertEqual(source.count(marker),candidate.count(marker))
            for part in range(4):
                self.assertEqual(candidate.count(f'prefetch_part(cute::Int<{part}>{{}});'),1)
            self.assertIn('if constexpr(CopyPhase<0 || CopyPhase==3)',candidate)
            self.assertIn('if constexpr(CopyPhase<0 || CopyPhase==0)',candidate)

    def test_byte_coverage_no_overlap(self):
        # 128 lanes, each phase issues two 16-byte cp.async payload copies.
        # Verify every packed byte exactly once, separately for low/high/W.
        found={name:[] for name in ('low','high','weight')}
        for phase in range(4):
            for tid in range(128):
                for name,chunk in (((('low',0),('high',0))),
                                  ((('weight',0),('weight',1))),
                                  ((('low',1),('high',1))),
                                  ((('weight',2),('weight',3))))[phase]:
                    start=tid*16+chunk*2048
                    found[name].extend(range(start,start+16))
        for name,length in (('low',4096),('high',4096),('weight',8192)):
            self.assertEqual(sorted(found[name]),list(range(length)))

    def test_stage_commit_wait_ring(self):
        for stages in (2,3):
            for groups in (1,2,3,5,32):
                committed=list(range(min(stages-1,groups)))
                for group in range(groups):
                    self.assertIn(group,committed)
                    pending=min(stages-2,groups-1-group)
                    # Old wait threshold leaves only newer committed groups.
                    self.assertTrue(committed.index(group)<len(committed)-pending)
                    next_group=group+stages-1
                    if next_group<groups:
                        self.assertNotEqual(group%stages,next_group%stages)
                        committed.append(next_group)
                self.assertEqual(committed,list(range(groups)))

    def test_reject_source_drift(self):
        with self.assertRaises(ValueError): generated_header('bad',1)
        with self.assertRaises(ValueError): generated_header('',2)

    def test_driver_preserves_m64_and_separate_partials(self):
        import benchmark_roof_interleaved_merge_probe as harness
        from benchmark_roof_distributed_copy_probe import Driver
        class Fake:
            def __init__(self,lib,cubins,variant,smem):
                self.path=cubins[0];self.resources={0:dict(m128=0,cta_tile=[64,128,128])}
            def run(self,index,*args): return self.path,index
            def close(self): pass
        with patch.object(harness,'M128Driver',Fake):
            driver=Driver('lib',{0:'old',1:'new'},'o7',34304)
            self.assertEqual(driver.run(1),('new',0))
            self.assertEqual(driver.resources[1]['copy_issue_phases'],4)
            self.assertEqual(driver.resources[1]['partial_registers_per_four_n_atoms'],32)
            self.assertNotIn('interleaved_merge',driver.resources[1])
            driver.close()


class DistributedCopyEvidenceTests(unittest.TestCase):
    root=ROOT/'docs/evidence/a100_o378_roof_v64'

    def test_codegen_and_unchanged_control(self):
        from compare_a100_codegen import compare
        from probe_roof_fullk_integer_codegen import static_entries
        directory=self.root/'reports/o378_roof_v64'
        x=json.loads((directory/'codegen.json').read_text())
        previous=ROOT/'docs/evidence/a100_o378_roof_v57/reports/o378_roof_v57/m128_64.sass'
        self.assertTrue(compare(previous.read_text(),(directory/'distributed_copy_0.sass').read_text(),
            r'^adangel_roof_m128_(?:o3|o78)$')['passed'])
        for policy in (0,1):
            entries=static_entries((directory/f'distributed_copy_{policy}.sass').read_text(),
                r'^adangel_roof_m128_(?:o3|o78)$',{'adangel_roof_m128_o3','adangel_roof_m128_o78'})
            self.assertEqual(entries,x['variants'][str(policy)]['entries'])
            for e in entries.values():
                self.assertTrue(e['native_u4_s4'] and e['native_s4_s4'] and e['all_copies_bypass_l1'])
                self.assertFalse(e['int8_mma'])
                self.assertEqual(e['opcode_counts']['IMMA'],64)
                self.assertEqual(e['opcode_counts']['LDSM'],16)
                if policy:
                    self.assertEqual(e['opcode_counts'].get('LDL',0),0)
                    self.assertEqual(e['opcode_counts'].get('STL',0),0)
            for family,name in HEADERS.items():
                self.assertEqual((directory/f'policy_{policy}/{family}_m128_generated.cuh').read_text(),
                    generated_header((ROOT/'csrc/sm80'/name).read_text(),policy))

    def test_raw_pairs_mse_and_statistics(self):
        from benchmark_roof_distributed_copy_probe import summary
        from benchmark_a100_o1 import stats
        directory=self.root/'runs/o378_roof_v64_screen'
        rows=[json.loads(s) for s in (directory/'results.jsonl').read_text().splitlines()]
        saved=json.loads((directory/'summary.json').read_text())
        self.assertEqual(len(rows),72)
        self.assertEqual(summary(rows),saved['records'])
        self.assertFalse(saved['production_default_changed'])
        self.assertEqual({r['sample_id'] for r in rows},{f'layer_00_{p}_proj' for p in ('q','k','v','o')})
        for row in rows:
            self.assertEqual(len(row['raw_ms']),200)
            for key,value in stats(row['raw_ms']).items():
                self.assertAlmostEqual(value,row['summary'][key],places=12)
            self.assertTrue(row['bitwise_equal_current_best'])
            self.assertEqual(row['mse_vs_current_best'],0)
            self.assertNotIn('interleaved_merge',row)
            r=row['probe_resources']
            self.assertEqual(r['cta_tile'],[64,128,128])
            self.assertEqual(r['active_blocks_per_sm'],3)
            self.assertEqual(r['registers_per_thread'],168)
            self.assertEqual(r['partial_registers_per_four_n_atoms'],32)
        env=json.loads((directory/'environment.json').read_text())
        self.assertEqual(env['policy_key'],'distributed_copy')
        self.assertEqual(env['extension_sha256'],
            '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462')

    def test_validation_and_scoped_sanitizers(self):
        for scope in ('preflight','memcheck','synccheck','racecheck','screen'):
            x=json.loads((self.root/f'runs/o378_roof_v64_{scope}/validation.json').read_text())
            self.assertTrue(x['passed'])
            self.assertEqual(x['count'],96)
            checks=x['checks']
            self.assertTrue(all(r['bitwise_equal_best'] and r['finite_fp32'] for r in checks))
            self.assertEqual({tuple(r['shape']) for r in checks},
                {(64,128,128),(64,128,384),(128,256,640),(64,128,4096)})
        for name in ('memcheck','synccheck'):
            self.assertIn('ERROR SUMMARY: 0 errors',
                (self.root/f'reports/o378_roof_v64/{name}.log').read_text())
        self.assertIn('0 hazards displayed (0 errors, 0 warnings)',
            (self.root/'reports/o378_roof_v64/racecheck.log').read_text())
