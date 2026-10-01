from pathlib import Path
import runpy
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
CHECK = runpy.run_path(str(ROOT / 'scripts/probe_roof_l2_codegen.py'))['static_entries']


class L2PrefetchTests(unittest.TestCase):
    def test_paired_summary_keeps_outliers_and_rejects_missing_data(self):
        aggregate=runpy.run_path(str(ROOT/'scripts/benchmark_roof_l2_probe.py'))['summary']
        rows=[dict(sample_id='x',variant='o7',round=r,prefetch_bytes=size,
            summary=dict(median_ms=1.0 if size==0 else 2.0,cv_percent=9.0),
            mse_vs_o0=0.5,mse_vs_paired_fp16=0.125,bitwise_equal_current_best=True,mse_vs_current_best=0.0)
            for r in range(3) for size in (0,128,256)]
        result=aggregate(rows)
        self.assertEqual([r['paired_speedup'] for r in result],[1.0,0.5,0.5])
        self.assertTrue(all(r['cv_failed_records']==3 for r in result))
        with self.assertRaises(ValueError): aggregate(rows[:-1])
        with self.assertRaises(ValueError): aggregate(rows+[rows[0]])
        rows[0]=dict(rows[0],mse_vs_paired_fp16=0.25)
        with self.assertRaises(ValueError): aggregate(rows)

    def test_native_event_harness_has_no_python_launch_loop(self):
        driver=(ROOT/'csrc/sm80/roof_probe_driver.cpp').read_text()
        self.assertLess(driver.index('Events events(repeats*2)'),driver.index('cuEventRecord(events.handles[i*2],stream)'))
        self.assertIn('cuLaunchKernel',driver)
        self.assertIn('cuStreamSynchronize(stream)',driver)
        self.assertNotIn('roof_probe_driver.cpp',(ROOT/'setup.py').read_text())
        runner=(ROOT/'scripts/benchmark_roof_l2_probe.py').read_text()
        self.assertIn('torch.equal(y.view(torch.int32),best[\'output\'].view(torch.int32))',runner)
        self.assertIn('measurement_order([0,128,256]',runner)
        self.assertIn('reference_kernel=dict(best[\'kernel\'])',runner)

    def test_probe_is_not_production(self):
        source = (ROOT / 'csrc/sm80/roof_l2_prefetch_probe.cu').read_text()
        self.assertNotIn('roof_l2_prefetch_probe.cu', (ROOT / 'setup.py').read_text())
        for hint in ('cp.async.cg.shared.global [', 'cp.async.cg.shared.global.L2::128B',
                     'cp.async.cg.shared.global.L2::256B'):
            self.assertIn(hint, source)
        self.assertNotIn('cp.async.ca.', source)
        for body in ('o3_row_scale_epilogue_candidate.cuh', 'o78_unsigned_payload_candidate.cuh'):
            self.assertIn(f'#include "{body}"', source)

    def test_same_entry_checks_fail_closed(self):
        sass = '\n'.join(f'''Function : adangel_roof_l2_o{name}
/*0000*/ LDGSTS.E.BYPASS.128 [R1], [R2.64];
/*0010*/ IMMA.16832.U4.S4 R4, R5, R6, R7;
/*0020*/ IMMA.16832.S4.S4 R4, R5, R6, R7;''' for name in ('3', '78'))
        self.assertTrue(all(x['all_copies_bypass_l1'] for x in CHECK(sass).values()))
        self.assertFalse(all(x['all_copies_bypass_l1'] for x in CHECK(sass.replace('.BYPASS', '')).values()))
        self.assertFalse(all(x['native_u4_s4'] for x in CHECK(sass.replace('.U4.', '.U8.')).values()))
        with self.assertRaises(ValueError):
            CHECK(sass.replace('adangel_roof_l2_o78', 'unrelated_kernel'))


if __name__ == '__main__':
    unittest.main()
