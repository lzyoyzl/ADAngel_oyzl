"""v45 isolates dimension knowledge without changing the best G128 math."""
import json
from pathlib import Path
import runpy
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))


class FixedDimsProbeTests(unittest.TestCase):
    def test_only_loop_directives_change_in_headers(self):
        for old, new, ns in (
            ('o3_row_scale_epilogue_candidate', 'o3_fixed_dims_probe', 'o3_row_scale_epilogue_experiment'),
            ('o78_unsigned_payload_candidate', 'o78_fixed_dims_probe', 'o78_unsigned_payload_experiment')):
            a = (ROOT / f'csrc/sm80/{old}.cuh').read_text().strip()
            b = (ROOT / f'csrc/sm80/{new}.cuh').read_text()
            self.assertEqual(b.count('#pragma unroll 1'), 2)
            restored = b.split('\n', 1)[1].replace('    #pragma unroll 1\n', '').replace(ns + '_v45', ns).strip()
            self.assertEqual(a, restored)

    def test_native_fixed_shape_guard_precedes_events_and_launch(self):
        s = (ROOT / 'csrc/sm80/roof_fixed_dims_driver.cpp').read_text()
        guard = s.index('if(m!=4096 || n!=4096 || k!=4096)')
        self.assertLess(guard, s.index('Events events(repeats*2);'))
        self.assertLess(guard, s.index('cuLaunchKernel('))
        self.assertIn('fixed4096 probe requires M=N=K=4096', s)
        for term in ('int64_t(m)*n>INT32_MAX', 'int64_t(m)*k>INT32_MAX', 'int64_t(n)*k>INT32_MAX'):
            self.assertIn(term, s)

    def test_launch_specialization_and_isolation(self):
        s = (ROOT / 'csrc/sm80/roof_fixed_dims_probe.cu').read_text()
        self.assertIn('#if ADANGEL_FIXED_DIMS==2', s)
        self.assertEqual(s.count('ADANGEL_FIXED_DIMS?4096:m,ADANGEL_FIXED_DIMS?4096:n,'), 2)
        self.assertEqual(s.count('ADANGEL_FIXED_DIMS==2?4096:k);'), 2)
        self.assertIn('constexpr int ProbeThreads=128;', s)
        self.assertIn('constexpr int ProbeMinBlocks=3;', s)
        for path in ('setup.py', 'csrc/sm80/roof_candidates.cuh'):
            self.assertNotIn('fixed_dims_probe', (ROOT / path).read_text())

    def test_validation_uses_actual_full_shape_and_rejections(self):
        s = (ROOT / 'scripts/validate_roof_fixed_dims_probe.py').read_text()
        self.assertIn('for m,n,k in ((4096,4096,4096),):', s)
        self.assertIn('semantic=reference_fp64(variant,values)', s)
        self.assertIn('torch.cuda.Stream()', s)
        self.assertIn("driver.lib.roof_probe_benchmark(driver.handles[policy]", s)
        self.assertIn("b'fixed4096' not in driver.lib.roof_probe_error()", s)
        self.assertIn('rejected=rejected', s)
        runner = (ROOT / 'scripts/benchmark_roof_fixed_dims_probe.py').read_text()
        self.assertIn('assert (m,n,k)==(4096,4096,4096)', runner)
        self.assertIn('supported_shape=[4096,4096,4096]', runner)

    def test_paired_summary_retains_failures(self):
        summary = runpy.run_path(str(ROOT / 'scripts/benchmark_roof_fixed_dims_probe.py'))['summary']
        rows = [dict(sample_id='x', variant='o3', round=r, fixed_dims=p,
                     summary=dict(median_ms=1+p, cv_percent=9), mse_vs_o0=.125,
                     mse_vs_paired_fp16=.125, bitwise_equal_current_best=True,
                     mse_vs_current_best=0) for r in range(3) for p in (0, 1, 2)]
        self.assertEqual([r['paired_speedup'] for r in summary(rows)], [1, .5, 1/3])
        self.assertTrue(all(r['cv_failed_records'] == 3 for r in summary(rows)))
        with self.assertRaises(ValueError):
            summary(rows[:-1])
        with self.assertRaises(ValueError):
            summary(rows + [rows[0]])

    def test_ncu_resource_and_static_identity_fixture(self):
        p = ROOT / 'docs/evidence/a100_o378_roof_v44/reports/o378_roof_v44'
        saved = json.loads((p / 'ncu_payload/ncu_o3_analysis.json').read_text())
        resources = saved['rows'][0]['probe_resources']
        static = json.loads((p / 'codegen.json').read_text())['variants']['0']['entries']['adangel_roof_flat_address_o3']
        texts = [(p / f'ncu_payload/ncu_o3_p0_{suffix}.csv').read_text().replace(
            'adangel_roof_flat_address_o3', 'adangel_roof_fixed_dims_o3') for suffix in ('raw', 'source_sass')]
        analyze = runpy.run_path(str(ROOT / 'scripts/profile_roof_fixed_dims_probe.py'))['analyze_profile']
        self.assertEqual(analyze(*texts, 'o3', 0, resources, static)['opcodes'], saved['rows'][0]['opcodes'])
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            analyze(*texts, 'o3', 0, resources, dict(static, instructions=static['instructions']+1))
        with self.assertRaisesRegex(ValueError, 'resources'):
            analyze(*texts, 'o3', 0, dict(resources, threads=160), static)


if __name__ == '__main__':
    unittest.main()
