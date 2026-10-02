from pathlib import Path
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
