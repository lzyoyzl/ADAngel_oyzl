"""Fragment-tree contracts; model coverage is not a GPU racecheck substitute."""
from pathlib import Path
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[2]


class FragmentTreeTest(unittest.TestCase):
    def test_three_slot_pair_schedule_with_odd_tails(self):
        for groups in range(1,66):
            slots=[None]*3
            prefetched=[];consumed=[]
            def put(g,live):
                self.assertNotIn(g%3,live)
                slots[g%3]=g
                prefetched.append(g)
            put(0,set())
            if groups>1: put(1,{0})
            for first in range(0,groups,2):
                pair=list(range(first,min(first+2,groups)))
                self.assertEqual([slots[g%3] for g in pair],pair)
                if first+2<groups: put(first+2,{g%3 for g in pair})
                consumed+=pair
                # The end-of-pair CTA barrier releases both consumed slots.
                if first+3<groups: put(first+3,{(first+2)%3})
            self.assertEqual(prefetched,list(range(groups)))
            self.assertEqual(consumed,list(range(groups)))

    def test_small_leaf_bank_and_explicit_two_stage_math(self):
        source=(ROOT/'csrc/sm80/roof_fragment_tree.cu').read_text()
        self.assertIn('constexpr int SliceN=Tune==39?64:32',source)
        self.assertIn('cute::make_shape(cute::_4{},cute::Int<NAtoms>{})',source)
        self.assertIn('static_assert(NAtoms==SliceN/16)',source)
        self.assertIn('shape_fragment=thr.make_fragment_C(coords)',source)
        self.assertNotIn('make_fragment_like<float>(acc)',source)
        self.assertNotIn('__fmaf_rn',source)
        self.assertIn('const float product=__fmul_rn(float(partial),scale)',source)
        self.assertIn('__fadd_rn(acc(vi,mi,full_ni),__fadd_rn(leaf(vi,ni),product))',source)
        self.assertEqual(source.count('__syncthreads();'),2)
        self.assertIn('cp.async.wait_group 0',source)
        self.assertIn('if(first+3<groups) prefetch',source)
        self.assertIn('"csrc/sm80/roof_fragment_tree.cu"',(ROOT/'setup.py').read_text())

    def test_opt_in_metadata_and_native_isa_audit(self):
        policy=(ROOT/'scripts/roof_reduction_validation.py').read_text()
        self.assertIn('bitwise equal to pair tree38',policy)
        self.assertIn('bitwise_equal_pair_tree',policy)
        for name in ('roof_candidates.cuh','o1_o3.cu','mixed_benchmark.cuh'):
            source=(ROOT/'csrc/sm80'/name).read_text()
            self.assertIn('meta["fragment_local_product_tree"]=',source)
            self.assertIn('meta["leaf_values_per_thread"]=',source)
            self.assertIn('meta["paired_group_pipeline"]=',source)
        audit=runpy.run_path(str(ROOT/'scripts/audit_a100_o1.py'))
        for tune in (39,40):
            symbol=f'adangel_sm80_roof_candidateILb1ELb0ELi{tune}EE'
            self.assertTrue(all(audit['roof_reduction_checks'](symbol,dict(I2F=1,FMUL=1,FADD=1,FFMA=0)).values()))
            self.assertFalse(audit['roof_reduction_checks'](symbol,dict(I2F=1,FMUL=1,FADD=1,FFMA=1))['products_not_contracted'])
            checks=audit['roof_paired_pipeline_checks'](symbol,
                'cp.async.wait_group 0; cp.async.commit_group; bar.sync;',
                'DEPBAR.LE SB0, 0x0; BAR.SYNC;')
            self.assertTrue(checks and all(checks.values()))


if __name__=='__main__': unittest.main()
