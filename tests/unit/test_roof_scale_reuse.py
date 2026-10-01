"""v43 contracts; numerical equivalence still requires the GPU checks."""
from pathlib import Path
import runpy
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))


class ScaleReuseTests(unittest.TestCase):
    def test_copy_pipeline_math_and_store_preserved(self):
        for old, new in (('o3_row_scale_epilogue_candidate', 'o3_scale_reuse_probe'),
                         ('o78_unsigned_payload_candidate', 'o78_scale_reuse_probe')):
            a = (ROOT / f'csrc/sm80/{old}.cuh').read_text()
            b = (ROOT / f'csrc/sm80/{new}.cuh').read_text()
            start = 'template<int M,int N,int K,bool Fast,bool Cached,int WN'
            end = 'template<int M,int N,int K,bool Fast,bool Cached=false,bool Magic'
            self.assertEqual(a[a.index(start):a.index(end)], b[b.index(start):b.index(end)])
            for token in ('cute::gemm(', '__fmaf_rn(', '__fmul_rn(', '__syncthreads();',
                          'cp.async.wait_group', 'const int partial=pl(vi)+16*ph(vi);'):
                self.assertEqual(a.count(token), b.count(token))
            tail = '    o1_static_for<0,C::Groups>(process_group);'
            self.assertEqual(a[a.index(tail):a.rindex('} // namespace')],
                             b[b.index(tail):b.rindex('} // namespace')])

    def test_coordinate_based_reuse_scopes(self):
        for name in ('o3', 'o78'):
            source = (ROOT / f'csrc/sm80/{name}_scale_reuse_probe.cuh').read_text()
            start = source.index('// v43: W columns')
            body = source[start:]
            self.assertLess(body.index('full_columns(vi,ni)=s.scales'),
                            body.index('o1_static_for<0,N/SliceN>'))
            self.assertLess(body.index('reused_columns(vi,ni)=s.scales'),
                            body.index('o1_static_for<0,decltype(cute::size<1>(acc))'))
            self.assertIn('coords(vi,cute::_0{},full_ni)', body)
            self.assertIn('ADANGEL_SCALE_REUSE==1 ? reused_columns(vi,ni) : full_columns(vi,full_ni)', body)
        o78 = (ROOT / 'csrc/sm80/o78_scale_reuse_probe.cuh').read_text()
        self.assertIn('if constexpr(ADANGEL_SCALE_REUSE==2) row=group_rows(vi,mi);', o78)
        self.assertIn('coords(vi,mi,cute::_0{})', o78)

    def test_launch_and_production_isolation(self):
        source = (ROOT / 'csrc/sm80/roof_scale_reuse_probe.cu').read_text()
        self.assertIn('constexpr int ProbeThreads=128;', source)
        self.assertIn('constexpr int ProbeMinBlocks=3;', source)
        self.assertIn('#if ADANGEL_SCALE_REUSE==0', source)
        self.assertIn('#include "o3_row_scale_epilogue_candidate.cuh"', source)
        for path in ('setup.py', 'csrc/sm80/roof_candidates.cuh'):
            self.assertNotIn('scale_reuse_probe', (ROOT / path).read_text())

    def test_summary_keeps_failures_and_pairs(self):
        summary = runpy.run_path(str(ROOT / 'scripts/benchmark_roof_scale_reuse_probe.py'))['summary']
        rows = [dict(sample_id='x', variant='o3', round=r, scale_reuse=p,
                     summary=dict(median_ms=1+p, cv_percent=9), mse_vs_o0=.125,
                     mse_vs_paired_fp16=.125, bitwise_equal_current_best=True,
                     mse_vs_current_best=0) for r in range(3) for p in (0, 1, 2)]
        self.assertEqual([r['paired_speedup'] for r in summary(rows)], [1, .5, 1/3])
        self.assertTrue(all(r['cv_failed_records'] == 3 for r in summary(rows)))
        with self.assertRaises(ValueError):
            summary(rows[:-1])
        with self.assertRaises(ValueError):
            summary(rows+[rows[0]])

    def test_profile_identity_and_static_fingerprint(self):
        # Relabel a historical fixture for parser testing, never as new evidence.
        report = ROOT / 'docs/evidence/a100_o378_roof_v42/reports/o378_roof_v42'
        saved = json.loads((report / 'ncu_pos1/ncu_o3_analysis.json').read_text())
        resources = saved['rows'][0]['probe_resources']
        static = json.loads((report / 'codegen.json').read_text())['variants']['0']['entries']['adangel_roof_prefetch_o3']
        texts = [(report / f'ncu_pos1/ncu_o3_p0_{suffix}.csv').read_text().replace(
            'adangel_roof_prefetch_o3', 'adangel_roof_scale_reuse_o3') for suffix in ('raw', 'source_sass')]
        analyze = runpy.run_path(str(ROOT / 'scripts/profile_roof_scale_reuse_probe.py'))['analyze_profile']
        result = analyze(*texts, 'o3', 0, resources, static)
        self.assertEqual(result['opcodes'], saved['rows'][0]['opcodes'])
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            analyze(*texts, 'o3', 0, resources, dict(static, instructions=static['instructions']+1))
        with self.assertRaisesRegex(ValueError, 'resources'):
            analyze(*texts, 'o3', 0, dict(resources, threads=160), static)


if __name__ == '__main__':
    unittest.main()
