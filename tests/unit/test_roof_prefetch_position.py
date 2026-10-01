"""Local contracts for v42; CUDA audit, safety and numerical tests are separate."""
from pathlib import Path
import json
import runpy
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))


class PrefetchPositionTests(unittest.TestCase):
    def test_math_copy_and_epilogue_unchanged(self):
        for old, new in (('o3_row_scale_epilogue_candidate', 'o3_prefetch_position_probe'),
                         ('o78_unsigned_payload_candidate', 'o78_prefetch_position_probe')):
            a = (ROOT / f'csrc/sm80/{old}.cuh').read_text()
            b = (ROOT / f'csrc/sm80/{new}.cuh').read_text()
            start = '    auto process_group=[&](auto group) {'
            end = '    o1_static_for<0,C::Groups>(process_group);'
            body = b[b.index(start):b.index(end)]
            body = body.replace('        if constexpr(ADANGEL_PREFETCH_POSITION==1) prefetch_next();\n', '')
            body = body.replace('          // Uniform compile-time condition: submit once, after N slice0.\n', '')
            body = body.replace('          if constexpr(ADANGEL_PREFETCH_POSITION==2 && decltype(nb)::value==0) prefetch_next();\n', '')
            self.assertEqual(body, a[a.index(start):a.index(end)])
            self.assertEqual(a[a.index(end):a.rindex('} // namespace')],
                             b[b.index(end):b.rindex('} // namespace')])
            copy_start = 'template<int M,int N,int K,bool Fast,bool Cached,int WN'
            copy_end = 'template<int M,int N,int K,bool Fast,bool Cached=false,bool Magic'
            self.assertEqual(a[a.index(copy_start):a.index(copy_end)],
                             b[b.index(copy_start):b.index(copy_end)])
            self.assertEqual(a.count('__syncthreads();'), b.count('__syncthreads();'))
            self.assertEqual(a.count('cp.async.wait_group'), b.count('cp.async.wait_group'))

    def test_prefetch_position_and_once_per_group(self):
        for name in ('o3', 'o78'):
            source = (ROOT / f'csrc/sm80/{name}_prefetch_position_probe.cuh').read_text()
            stage = source[source.index('  auto process_stage='):]
            self.assertLess(stage.index('__syncthreads();'), stage.index('auto prefetch_next='))
            self.assertIn('static_assert(Stream && K==128 && C::Groups==1);', stage)
            self.assertEqual(stage.count('prefetch_next();'), 2)
            self.assertLess(stage.index(')),hd1);'), stage.index('ADANGEL_PREFETCH_POSITION==1'))
            self.assertLess(stage.index('finish(ni,pl,ph);'), stage.index('ADANGEL_PREFETCH_POSITION==2'))
            self.assertIn('decltype(nb)::value==0', stage)
            # Both copy branches target a released slot, not the current one.
            self.assertIn('(s,(stage+2)%Stages,stage+2,a,w,ws,m,k,as,n);', stage)
            self.assertIn('(s,1-slot,stage+1,a,w,ws,m,k,as,n);', stage)

    def test_ring_slot_ownership_for_short_and_odd_k(self):
        # Uniformly moving the submit point within the stage doesn't change
        # ownership. This model is not a substitute for GPU racecheck.
        for stages in (2, 3):
            for groups in range(1, 66):
                for position in (1, 2):
                    owner = {g: g for g in range(min(stages - 1, groups))}
                    submitted = list(owner.values())
                    done = []
                    for g in range(groups):
                        current = g % stages
                        self.assertEqual(owner[current], g)
                        for checkpoint in (1, 2):
                            if checkpoint != position:
                                continue
                            future = g + stages - 1
                            if future < groups:
                                slot = future % stages
                                self.assertNotEqual(current, slot)
                                if slot in owner:
                                    self.assertIn(owner[slot], done)
                                owner[slot] = future
                                submitted.append(future)
                        done.append(g)
                    self.assertEqual(submitted, list(range(groups)))

    def test_launch_and_no_default_change(self):
        source = (ROOT / 'csrc/sm80/roof_prefetch_position_probe.cu').read_text()
        self.assertIn('constexpr int ProbeThreads=128;', source)
        self.assertIn('constexpr int ProbeMinBlocks=3;', source)
        self.assertIn('#if ADANGEL_PREFETCH_POSITION==0', source)
        self.assertIn('#include "o3_row_scale_epilogue_candidate.cuh"', source)
        self.assertNotIn('roof_prefetch_position_probe.cu', (ROOT / 'setup.py').read_text())
        self.assertNotIn('roof_prefetch_position_probe', (ROOT / 'csrc/sm80/roof_candidates.cuh').read_text())

    def test_pairing_all_candidates_and_no_cv_filter(self):
        summary = runpy.run_path(str(ROOT / 'scripts/benchmark_roof_prefetch_probe.py'))['summary']
        rows = [dict(sample_id='x', variant='o3', round=r, prefetch_position=p,
                     summary=dict(median_ms=1+p, cv_percent=9), mse_vs_o0=.125,
                     mse_vs_paired_fp16=.125, bitwise_equal_current_best=True,
                     mse_vs_current_best=0) for r in range(3) for p in (0, 1, 2)]
        self.assertEqual([r['paired_speedup'] for r in summary(rows)], [1, .5, 1/3])
        self.assertTrue(all(r['cv_failed_records'] == 3 for r in summary(rows)))
        with self.assertRaises(ValueError):
            summary(rows[:-1])
        with self.assertRaises(ValueError):
            summary(rows+[rows[0]])

    def test_ncu_parser_fixture_and_launch_selection(self):
        # Parser fixture only: reuse an archived control and rename its symbol.
        # This is not new v42 profiling evidence.
        root = ROOT / 'docs/evidence/a100_o378_roof_v41'
        reports = root / 'reports/o378_roof_v41'
        analyze = runpy.run_path(str(ROOT / 'scripts/profile_roof_prefetch_probe.py'))['analyze_profile']
        saved = json.loads((reports / 'ncu/ncu_o3_analysis.json').read_text())
        resources = saved['rows'][0]['probe_resources']
        codegen = json.loads((reports / 'codegen.json').read_text())
        static = codegen['variants']['0']['entries']['adangel_roof_producer_o3']
        texts = [(reports / f'ncu/ncu_o3_p0_{suffix}.csv').read_text().replace(
            'adangel_roof_producer_o3', 'adangel_roof_prefetch_o3') for suffix in ('raw', 'source_sass')]
        result = analyze(*texts, 'o3', 0, resources, static)
        self.assertEqual(result['opcodes'], saved['rows'][0]['opcodes'])
        wrong = dict(static, instructions=static['instructions']+1)
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            analyze(*texts, 'o3', 0, resources, wrong)
        with self.assertRaisesRegex(ValueError, 'resources'):
            analyze(*texts, 'o3', 0, dict(resources, threads=160), static)
        order = runpy.run_path(str(ROOT / 'scripts/benchmark_a100_roof_trace.py'))['measurement_order']
        for candidate in (1, 2):
            self.assertEqual(order([0, candidate], 0, 0, 0), [0, candidate])
        source = (ROOT / 'scripts/profile_roof_prefetch_probe.py').read_text()
        self.assertIn('skip=50 if policy==0 else 101', source)


if __name__ == '__main__':
    unittest.main()
