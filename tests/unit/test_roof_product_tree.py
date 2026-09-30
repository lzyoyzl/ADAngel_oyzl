"""Tree coverage/order and source isolation. GPU validation is separate."""
import importlib.util
from pathlib import Path
import random
import struct
import unittest

ROOT=Path(__file__).resolve().parents[2]


def f32(x):
    return struct.unpack('f',struct.pack('f',x))[0]


def tree(values, window):
    acc=0.
    for base in range(0,len(values),window):
        work=values[base:base+window]+[0.]*(window-len(values[base:base+window]))
        stride=1
        while stride<window:
            for g in range(0,window,2*stride):
                work[g]=f32(work[g]+work[g+stride])
            stride*=2
        acc=f32(acc+work[0])
    return acc


def eager(values):
    acc=leaf=pair=0.
    for g,value in enumerate(values):
        if g%4==0: leaf=value
        elif g%4==1: pair=f32(leaf+value)
        elif g%4==2: leaf=value
        else: acc=f32(acc+f32(pair+f32(leaf+value)))
    tail=len(values)%4
    if tail: acc=f32(acc+(leaf if tail==1 else pair if tail==2 else f32(pair+leaf)))
    return acc


class ProductTreeTest(unittest.TestCase):
    def test_ncu_work_keeps_actual_predicated_tree_instruction_count(self):
        import runpy
        validate=runpy.run_path(str(ROOT/'scripts/analyze_roof_scale_ncu.py'))['validate_arithmetic_work']
        groups=16777216
        for tune in (34,35,36):
            for fast in (False,True):
                counts=dict(IMMA=groups,I2F=groups,FMUL=groups*(1 if fast else 2),FADD=groups*3)
                validate(counts,tune,fast)
                self.assertEqual(counts['FADD'],groups*3)
                for change in (dict(FFMA=1),dict(FMUL=0),dict(FADD=groups-1),dict(IMMA=groups//2)):
                    with self.assertRaises(ValueError): validate(dict(counts,**change),tune,fast)

    def test_eager_reuses_storage_without_changing_four_term_tree(self):
        rng=random.Random(930)
        for count in (1,2,3,4,5,6,7,8,31,32,33,64):
            for _ in range(100):
                values=[f32(rng.randint(-131072,131072)*2.**rng.randint(-16,3)) for g in range(count)]
                self.assertEqual(eager(values),tree(values,4))

    def test_tree_reads_every_leaf_once(self):
        for window in (4,32):
            for count in (1,2,3,4,5,31,32,33):
                seen=[]
                for base in range(0,count,window):
                    work=[[g] if g<count else [] for g in range(base,base+window)]
                    step=1
                    while step<window:
                        for g in range(0,window,2*step): work[g]+=work[g+step]
                        step*=2
                    seen+=work[0]
                self.assertEqual(seen,list(range(count)))

    def test_independent_products_not_running_fma(self):
        for name in ('o3_product_tree_candidate.cuh','o3_eager_tree_candidate.cuh'):
            source=(ROOT/'csrc/sm80'/name).read_text()
            finish=source.split('auto finish=',1)[1].split('if constexpr(RoofTune&2)',1)[0]
            self.assertIn('const int partial=pl(vi)+16*ph(vi)',finish)
            self.assertIn('const float product=__fmul_rn(float(partial),scale)',finish)
            self.assertNotIn('__fmaf_rn',finish)
            self.assertIn('cp.async.wait_group 0',source)
        eager_source=(ROOT/'csrc/sm80/o3_eager_tree_candidate.cuh').read_text()
        self.assertNotIn('float products[',eager_source)
        self.assertIn('auto leaf=cute::make_fragment_like<float>(low)',eager_source)
        self.assertIn('auto pair=cute::make_fragment_like<float>(low)',eager_source)
        self.assertIn('if((k/K)%4)',eager_source)

    def test_explicit_opt_in_and_audit(self):
        policy=(ROOT/'scripts/roof_reduction_validation.py').read_text()
        self.assertIn('reassociated=tune in (24,25,26,27,34,35,36)',policy)
        self.assertIn('if not reassociated and changed:',policy)
        for name in ('roof_product_tree.cu','roof_eager_tree.cu'):
            wrapper=(ROOT/'csrc/sm80'/name).read_text()
            self.assertIn('__launch_bounds__(128,1)',wrapper)
            self.assertIn('"csrc/sm80/'+name+'"',(ROOT/'setup.py').read_text())
        spec=importlib.util.spec_from_file_location('audit',ROOT/'scripts/audit_a100_o1.py')
        audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
        for tune in (34,35,36):
            checks=audit.roof_reduction_checks('adangel_sm80_roof_candidateILb1ELb0ELi'+str(tune)+'EE',
                dict(I2F=1,FMUL=1,FADD=1,FFMA=0))
            self.assertTrue(checks and all(checks.values()))
            self.assertFalse(audit.roof_reduction_checks('adangel_sm80_roof_candidateILb1ELb0ELi'+str(tune)+'EE',
                dict(I2F=1,FMUL=1,FADD=1,FFMA=1))['products_not_contracted'])


if __name__=='__main__': unittest.main()
