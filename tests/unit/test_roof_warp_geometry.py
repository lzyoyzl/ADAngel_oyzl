from pathlib import Path
import runpy
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))


class WarpGeometryTests(unittest.TestCase):
    def test_source_changes_only_geometry_guard(self):
        for original,new,namespace in (
            ('o3_row_scale_epilogue_candidate','o3_warp_geometry_probe','o3_row_scale_epilogue_experiment'),
            ('o78_unsigned_payload_candidate','o78_warp_geometry_probe','o78_unsigned_payload_experiment')):
            src=(ROOT/f'csrc/sm80/{original}.cuh').read_text()
            self.assertEqual(src.count('static_assert(M==64 && WN==2 && K==128);'),1)
            expected=src.replace(namespace,namespace+'_v40').replace(
                'static_assert(M==64 && WN==2 && K==128);',
                'static_assert(M==64 && (WN==2 || WN==4) && K==128);')
            self.assertEqual((ROOT/f'csrc/sm80/{new}.cuh').read_text().split('\n',1)[1],expected)

    def test_kernel_and_launch_geometry_are_explicit(self):
        source=(ROOT/'csrc/sm80/roof_warp_geometry_probe.cu').read_text()
        self.assertIn('ProbeThreads=64*ProbeWN',source)
        self.assertIn('__launch_bounds__(ProbeThreads,ProbeMinBlocks)',source)
        self.assertIn('sizeof(o3_row_scale_epilogue_experiment_v40',source)
        self.assertIn('sizeof(o78_unsigned_payload_experiment_v40',source)
        driver=(ROOT/'csrc/sm80/roof_warp_geometry_driver.cpp').read_text()
        self.assertIn('CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK',driver)
        self.assertIn('n/128,m/64,1,p->threads,1,1',driver)
        self.assertIn('cuOccupancyMaxActiveBlocksPerMultiprocessor',driver)
        self.assertNotIn('roof_warp_geometry_probe.cu',(ROOT/'setup.py').read_text())
        self.assertNotIn('roof_warp_geometry_driver.cpp',(ROOT/'setup.py').read_text())

    def test_summary_pairing_and_unfiltered_cv(self):
        summary=runpy.run_path(str(ROOT/'scripts/benchmark_roof_warp_probe.py'))['summary']
        rows=[dict(sample_id='x',variant='o7',round=r,warp_geometry=p,
            summary=dict(median_ms=1.0 if p==0 else 2.0,cv_percent=9.0),
            mse_vs_o0=.5,mse_vs_paired_fp16=.125,bitwise_equal_current_best=True,mse_vs_current_best=0.0)
            for r in range(3) for p in (0,1)]
        self.assertEqual([r['paired_speedup'] for r in summary(rows)],[1.0,.5])
        self.assertTrue(all(r['cv_failed_records']==3 for r in summary(rows)))
        with self.assertRaises(ValueError): summary(rows[:-1])
        with self.assertRaises(ValueError): summary(rows+[rows[0]])

    def test_same_entry_int4_requirement(self):
        check=runpy.run_path(str(ROOT/'scripts/probe_roof_warp_codegen.py'))['static_entries']
        sass='\n'.join(f'''Function : adangel_roof_warp_o{name}
/*0000*/ LDGSTS.E.BYPASS.128 [R1], [R2.64];
/*0010*/ IMMA.16864.U4.S4 R4, R5, R6, R7;
/*0020*/ IMMA.16864.S4.S4 R4, R5, R6, R7;''' for name in ('3','78'))
        self.assertTrue(all(r['native_u4_s4'] and r['native_s4_s4'] for r in check(sass).values()))
        with self.assertRaises(ValueError): check(sass.replace('warp_o78','unrelated'))
        self.assertTrue(all(r['int8_mma'] for r in check(sass.replace('.U4.', '.U8.')).values()))


if __name__=='__main__': unittest.main()
