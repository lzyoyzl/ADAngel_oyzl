"""CPU address/schedule proof only; CUDA racecheck and NCU remain mandatory."""
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]


def swizzle(offset, base):
    return offset ^ ((offset & (3 << (base+3))) >> 3)


class PairedRingContract(unittest.TestCase):
    def test_ring_never_overwrites_an_unconsumed_group(self):
        for count in (1,2,3,4,5,6,7,8,31,32,33):
            slots={};pending=set();ready=set();consumed=[];copies=[]
            def issue_pair(first):
                self.assertEqual(first%2,0)
                for g in range(first,min(first+2,count)):
                    slot=g%3
                    self.assertTrue(slot not in slots or slots[slot] in consumed)
                    slots[slot]=g;pending.add(g);copies.append(g)
            issue_pair(0)
            for g in range(count):
                # wait_group0 then CTA barrier, before reusing any old slot.
                ready.update(pending);pending.clear()
                self.assertIn(g,ready)
                self.assertEqual(slots[g%3],g)
                if g%2 and g+1<count:
                    issue_pair(g+1)
                    self.assertEqual(slots[g%3],g)
                consumed.append(g)
            self.assertEqual(copies,list(range(count)))
            self.assertEqual(consumed,list(range(count)))
            self.assertFalse(pending)

    def test_pair_scatter_and_cute_nibble_view_agree(self):
        for rows in (64,128):
            for count in (1,3,32):
                kbytes=count*64
                for first in range(0,count,2):
                    destinations={};source_offsets=set()
                    for thread in range(128):
                        for chunk in range(rows*128//(128*16)):
                            off=thread*16+chunk*128*16
                            row,col=divmod(off,128)
                            g=first+col//64
                            if g>=count: continue
                            src=row*kbytes+first*64+col
                            dst=(g&1)*64+swizzle(row*64+col%64,4)
                            self.assertEqual(dst%16,0)
                            self.assertLess(dst+15,rows*64+128)
                            for byte in range(16):
                                key=(g%3,dst+byte)
                                self.assertNotIn(key,destinations)
                                self.assertLess(src+byte,rows*kbytes)
                                destinations[key]=src+byte
                                source_offsets.add(src+byte)
                    expected={row*kbytes+g*64+c for row in range(rows)
                              for g in range(first,min(first+2,count)) for c in range(64)}
                    self.assertEqual(source_offsets,expected)
                    for g in range(first,min(first+2,count)):
                        for row in range(rows):
                            for nibble in range(128):
                                physical=(g&1)*128+swizzle(row*128+nibble,5)
                                self.assertEqual(physical%2,nibble%2)
                                self.assertEqual(destinations[g%3,physical//2],row*kbytes+g*64+nibble//2)

    def test_global_segments_and_destination_bank_model(self):
        # Model for4096^3 aligned data, not a prediction of NCU wavefronts.
        # Adjacent groups rotate64B so an8-lane/128B source span has distinct
        # destination banks even though its halves live in different slots.
        kbytes=4096//2
        for rows in (64,128):
            for first in range(0,32,2):
                for warp in range(4):
                    addresses=[];banks=[]
                    for lane in range(32):
                        off=(warp*32+lane)*16
                        row,col=divmod(off,128)
                        group=first+col//64
                        src=row*kbytes+first*64+col
                        dst=(group%3)*(rows*64+128)+(group&1)*64+swizzle(row*64+col%64,4)
                        addresses.append(src//128)
                        banks.append([(dst//4+i)%32 for i in range(4)])
                    self.assertEqual(len(set(addresses)),4)
                    for start in range(0,32,8):
                        self.assertEqual(sorted(b for lane in banks[start:start+8] for b in lane),list(range(32)))

    def test_math_body_and_old_translation_units_are_unchanged(self):
        old=(ROOT/'csrc/sm80/o3_reuse_pipeline_candidate.cuh').read_text()
        new=(ROOT/'csrc/sm80/o3_paired_pipeline_candidate.cuh').read_text()
        start='    auto process_group='
        end='    o1_static_for<0,C::Groups>(process_group);'
        self.assertEqual(old[old.index(start):old.index(end)],new[new.index(start):new.index(end)])
        self.assertIn('if((stage&1) && stage+1<k/128)',new)
        self.assertIn('o3_prefetch_pair<M,N,Fast,DualScale>(s,stage+1',new)
        stage=new[new.index('auto process_stage='):new.index(start)]
        self.assertLess(stage.index('cp.async.wait_group 0'),stage.index('__syncthreads'))
        self.assertLess(stage.index('__syncthreads'),stage.index('o3_prefetch_pair'))
        self.assertNotIn('o3_prefetch<',new)
        host=(ROOT/'csrc/sm80/roof_candidates.cuh').read_text()
        self.assertIn('paired_pipeline_shared_bytes(dual)',host)
        self.assertNotIn('#include "o3_paired_pipeline_candidate.cuh"',host)
        self.assertNotIn('ROOF_PICK(29)',host)
        self.assertIn('"csrc/sm80/roof_paired_pipeline.cu"',(ROOT/'setup.py').read_text())
        for dual in (False,True):
            size=3*(64*128+128*128//2+3*128+128*4+(64*4 if dual else 0))
            self.assertEqual(size,52608 if dual else 51840)
            self.assertLess(3*(size+1024),164*1024)


if __name__=='__main__': unittest.main()
