from pathlib import Path
import runpy
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))


class RecomposeTests(unittest.TestCase):
    def test_only_active_recomposition_and_namespace_differ(self):
        for source,target,namespace in (
            ('o3_row_scale_epilogue_candidate','o3_partial_recompose_probe','o3_row_scale_epilogue_experiment'),
            ('o78_unsigned_payload_candidate','o78_partial_recompose_probe','o78_unsigned_payload_experiment')):
            original=(ROOT/f'csrc/sm80/{source}.cuh').read_text()
            actual=(ROOT/f'csrc/sm80/{target}.cuh').read_text().split('\n',1)[1]
            self.assertEqual(original.count('const int partial=pl(vi)+16*ph(vi);'),1)
            expected=original.replace(namespace,namespace+'_v39').replace(
                'const int partial=pl(vi)+16*ph(vi);','const int partial=roof_recompose(pl(vi),ph(vi));')
            self.assertEqual(actual,expected)

    def test_bounded_integer_identity_and_signed_nibbles(self):
        # Exhaust all high-partial values with representative low endpoints,
        # then exhaust all INT8/S4 scalar products. PTX shl is bitwise, not a
        # C++ left shift of a negative signed value.
        for high in range(-8192,8193):
            shifted=((high & 0xffffffff)<<4)&0xffffffff
            if shifted>=0x80000000: shifted-=0x100000000
            for low in (-15360,-1,0,1,15360):
                self.assertEqual(low+shifted,low+16*high)
                self.assertLess(abs(low+16*high),2**31)
        for a in range(-128,128):
            low=a&15;high=a//16
            for w in range(-8,8):
                self.assertEqual(low*w+16*high*w,a*w)

    def test_same_entry_audit_and_probe_only(self):
        check=runpy.run_path(str(ROOT/'scripts/probe_roof_recompose_codegen.py'))['static_entries']
        sass='\n'.join(f'''Function : adangel_roof_recompose_o{name}
/*0000*/ LDGSTS.E.BYPASS.128 [R1], [R2.64];
/*0010*/ IMMA.16832.U4.S4 R4, R5, R6, R7;
/*0020*/ IMMA.16832.S4.S4 R4, R5, R6, R7;
/*0030*/ LEA R1, R2, R3, 4;''' for name in ('3','78'))
        self.assertTrue(all(x['native_u4_s4'] and x['native_s4_s4'] and x['opcode_counts']['LEA']==1 for x in check(sass).values()))
        with self.assertRaises(ValueError): check(sass.replace('recompose_o78','unrelated'))
        self.assertNotIn('roof_recompose_probe.cu',(ROOT/'setup.py').read_text())
        source=(ROOT/'csrc/sm80/roof_recompose_probe.cu').read_text()
        self.assertIn('shl.b32',source)
        self.assertIn('add.s32',source)
        self.assertNotIn('high <<',source)
        self.assertNotIn('__int_as_float',source)

    def test_summary_pairing_keeps_cv_failures(self):
        aggregate=runpy.run_path(str(ROOT/'scripts/benchmark_roof_recompose_probe.py'))['summary']
        rows=[dict(sample_id='x',variant='o7',round=r,recompose_policy=policy,
            summary=dict(median_ms=1.0 if policy==0 else 2.0,cv_percent=9.0),
            mse_vs_o0=.5,mse_vs_paired_fp16=.125,bitwise_equal_current_best=True,mse_vs_current_best=0.0)
            for r in range(3) for policy in (0,1,2)]
        self.assertEqual([r['paired_speedup'] for r in aggregate(rows)],[1.0,.5,.5])
        self.assertTrue(all(r['cv_failed_records']==3 for r in aggregate(rows)))
        with self.assertRaises(ValueError): aggregate(rows[:-1])
        with self.assertRaises(ValueError): aggregate(rows+[rows[0]])
        two=[r for r in rows if r['recompose_policy'] in (0,1)]
        self.assertEqual([r['paired_speedup'] for r in aggregate(two,(0,1))],[1.0,.5])
        with self.assertRaises(ValueError): aggregate(two)


if __name__=='__main__': unittest.main()
