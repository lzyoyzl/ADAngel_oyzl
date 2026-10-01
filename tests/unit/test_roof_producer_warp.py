"""Producer ring protocol and source invariants; GPU checks remain mandatory."""
from pathlib import Path
import random
import runpy
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))


class ProducerWarpTests(unittest.TestCase):
    def test_consumer_math_and_output_unchanged(self):
        for old,new in (('o3_row_scale_epilogue_candidate','o3_producer_warp_probe'),
                        ('o78_unsigned_payload_candidate','o78_producer_warp_probe')):
            a=(ROOT/f'csrc/sm80/{old}.cuh').read_text()
            b=(ROOT/f'csrc/sm80/{new}.cuh').read_text()
            start='    auto process_group=[&](auto group) {'
            end='    o1_static_for<0,C::Groups>(process_group);'
            self.assertEqual(a[a.index(start):a.index(end)],b[b.index(start):b.index(end)])
            # The epilogue and output coordinates also retain their original semantics.
            tail='  if constexpr(PhasePair) {'
            self.assertEqual(a[a.index(tail):a.rindex('} // namespace')],
                             b[b.index(tail):b.rindex('} // namespace')])
            self.assertIn('const unsigned copy_tid=threadIdx.x-128u;',b)
            self.assertIn('constexpr int CopyThreads=32;',b)
            self.assertIn('roof_producer_wait(1+int(slot));',b)
            self.assertIn('roof_producer_arrive(4+int(slot));',b)
            self.assertIn('if(stage>=Stages) roof_producer_wait(4+slot);',b)
            producer=b[b.index('  if(threadIdx.x>=128) {'):b.index('  if constexpr(Cached) {',b.index('  if(threadIdx.x>=128) {'))]
            self.assertLess(producer.index('cp.async.wait_group 0;'),producer.index('roof_producer_arrive(1+slot)'))
            self.assertIn('roof_producer_wait(4+stage%Stages);',producer)

    def test_producer_copy_coverage(self):
        # Each lane copies 16B. Both activation planes and B retain complete, unique coverage.
        for rows in (64,128):
            offsets=[tid*16+chunk*32*16 for tid in range(32) for chunk in range(rows*64//512)]
            self.assertEqual(sorted(offsets),list(range(0,rows*64,16)))
        self.assertEqual(sorted(t+32*p for t in range(32) for p in range(4)),list(range(128)))
        source=(ROOT/'csrc/sm80/o3_producer_warp_probe.cuh').read_text()
        self.assertIn('const unsigned scale_tid=copy_tid+32*panel;',source)

    def test_ring_random_interleavings(self):
        # Abstract ready/free phases model four independent consumer warps.
        # This is not a substitute for CUDA memory-model checks or racecheck.
        for stages in (2,3):
            for groups in (1,2,3,5,32):
                for seed in range(16):
                    rng=random.Random(seed)
                    p=0; c=[0]*4; phase=[0]*4
                    ready=[set() for _ in range(groups)]
                    published=[False]*groups; consumed=[set() for _ in range(groups)]
                    slots=[None]*stages; steps=0
                    while p<groups or min(c)<groups:
                        choices=[]
                        if p<groups and (p<stages or len(consumed[p-stages])==4): choices.append(4)
                        for w in range(4):
                            g=c[w]
                            if g<groups and (phase[w]==0 or (published[g] and len(ready[g])==4)):
                                choices.append(w)
                        self.assertTrue(choices,'ring deadlock')
                        w=rng.choice(choices)
                        if w==4:
                            self.assertTrue(p<stages or consumed[p-stages]==set(range(4)))
                            slots[p%stages]=p; published[p]=True; p+=1
                        elif phase[w]==0:
                            ready[c[w]].add(w); phase[w]=1
                        else:
                            g=c[w];self.assertEqual(slots[g%stages],g,'slot overwritten before consumption')
                            consumed[g].add(w); c[w]+=1;phase[w]=0
                        steps+=1;self.assertLessEqual(steps,groups*9)
                    self.assertTrue(all(len(x)==4 for x in consumed))

    def test_driver_and_launch_contract(self):
        source=(ROOT/'csrc/sm80/roof_producer_warp_probe.cu').read_text()
        self.assertIn('ProbeThreads=ADANGEL_PRODUCER_MODE?160:128',source)
        self.assertIn('ProbeMinBlocks=ADANGEL_PRODUCER_MODE==1?2:3',source)
        self.assertIn('barrier.cta.sync %0, 160;',source)
        self.assertIn('barrier.cta.arrive %0, 160;',source)
        self.assertNotIn('barrier.cta.sync.aligned',source)
        self.assertNotIn('roof_producer_warp_probe.cu',(ROOT/'setup.py').read_text())
        driver=(ROOT/'csrc/sm80/roof_producer_warp_driver.cpp').read_text()
        self.assertIn('p->threads!=128 && p->threads!=160',driver)
        self.assertIn('n/128,m/64,1,p->threads,1,1',driver)

    def test_summary_pairing_and_cv(self):
        summary=runpy.run_path(str(ROOT/'scripts/benchmark_roof_producer_probe.py'))['summary']
        rows=[dict(sample_id='x',variant='o3',round=r,producer_mode=p,
            summary=dict(median_ms=1+p,cv_percent=9),mse_vs_o0=.125,mse_vs_paired_fp16=.125,
            bitwise_equal_current_best=True,mse_vs_current_best=0) for r in range(3) for p in (0,1,2)]
        self.assertEqual([r['paired_speedup'] for r in summary(rows)],[1,.5,1/3])
        self.assertTrue(all(r['cv_failed_records']==3 for r in summary(rows)))
        with self.assertRaises(ValueError): summary(rows[:-1])
        with self.assertRaises(ValueError): summary(rows+[rows[0]])

    def test_ncu_launch_selection(self):
        order=runpy.run_path(str(ROOT/'scripts/benchmark_a100_roof_trace.py'))['measurement_order']
        for candidate in (1,2):
            self.assertEqual(order([0,candidate],0,0,0),[0,candidate])
        source=(ROOT/'scripts/profile_roof_producer_probe.py').read_text()
        self.assertIn('skip=50 if policy==0 else 101',source)
        self.assertIn("row['registers_per_thread']!=expected_resources['registers_per_thread']",source)
        self.assertIn("threads!=expected_resources['threads']",source)


if __name__=='__main__': unittest.main()
