"""v49 contracts: B-only double buffering, exact arithmetic and isolated validation."""
from pathlib import Path
import json
import runpy
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))


class BLookaheadTests(unittest.TestCase):
    def test_only_b_copy_schedule_changes(self):
        for old,new,namespace in (
            ('o3_row_scale_epilogue_candidate','o3_b_lookahead_probe','o3_row_scale_epilogue_experiment'),
            ('o78_unsigned_payload_candidate','o78_b_lookahead_probe','o78_unsigned_payload_experiment')):
            a=(ROOT/f'csrc/sm80/{old}.cuh').read_text()
            b=(ROOT/f'csrc/sm80/{new}.cuh').read_text().split('\n',1)[1].replace(namespace+'_v49',namespace)
            extra='''        auto next_br0=cute::make_fragment_like(br0), next_br1=cute::make_fragment_like(br1);
        auto next_sd0=sbc.retile_D(next_br0), next_sd1=sbc.retile_D(next_br1);
        static_assert(SliceN==64 && N==128);
'''
            b=b.replace(extra,'')
            start=b.index('          // Keep the N64 partial window.')
            end=b.index('          o1_static_for<0,decltype(cute::size<1>(acc))::value>',start)
            b=b[:start]+'''          cute::copy(SmallCopy{},sbc.partition_S(slice_b(nb,sub)),sd0);
          cute::copy(SmallCopy{},sbc.partition_S(slice_b(nb,sub+cute::_1{})),sd1);
'''+b[end:]
            b=b.replace('          if constexpr(ADANGEL_B_LOOKAHEAD==2 && decltype(mi)::value==0) prefetch_next_b();\n','')
            self.assertEqual(a,b)

    def test_copy_coverage_and_final_boundary(self):
        for policy in (1,2):
            reads=[];computed=[];next_loaded=False
            for nb in range(2):
                if nb==0: reads.append(nb)
                else: self.assertTrue(next_loaded)
                for mi in range(2):
                    if nb==0 and mi==0 and policy==1:
                        reads.append(nb+1);next_loaded=True
                    computed.append((nb,mi))
                    if nb==0 and mi==0 and policy==2:
                        reads.append(nb+1);next_loaded=True
            self.assertEqual(reads,[0,1])
            self.assertEqual(computed,[(0,0),(0,1),(1,0),(1,1)])
        code=(ROOT/'csrc/sm80/o3_b_lookahead_probe.cuh').read_text()
        self.assertIn('decltype(nb)::value+1<N/SliceN',code)
        self.assertIn('cute::copy(next_br0,br0);cute::copy(next_br1,br1);',code)

    def test_isolation_and_resources(self):
        code=(ROOT/'csrc/sm80/roof_b_lookahead_probe.cu').read_text()
        for value in ('#if ADANGEL_B_LOOKAHEAD==0','ProbeThreads=128','ProbeMinBlocks=3',
                      'Storage)==50688','Storage)==34304'):
            self.assertIn(value,code)
        for path in ('setup.py','csrc/sm80/roof_candidates.cuh'):
            self.assertNotIn('b_lookahead_probe',(ROOT/path).read_text())
        build=(ROOT/'scripts/probe_roof_b_lookahead_codegen.py').read_text()
        self.assertIn('tests/cuda/validate_roof_stream_width_coordinates.cu',build)
        self.assertIn("manifest['host_stream_mapping_passed']=True",build)

    def test_pairs_and_no_outlier_filter(self):
        summarize=runpy.run_path(str(ROOT/'scripts/benchmark_roof_b_lookahead_probe.py'))['summary']
        rows=[dict(sample_id='x',variant='o3',round=r,b_lookahead=p,
                   summary=dict(median_ms=1+p,cv_percent=9),mse_vs_o0=.125,
                   mse_vs_paired_fp16=.125,bitwise_equal_current_best=True,
                   mse_vs_current_best=0) for r in range(3) for p in (0,1,2)]
        self.assertEqual([r['paired_speedup'] for r in summarize(rows)],[1,.5,1/3])
        self.assertTrue(all(r['cv_failed_records']==3 for r in summarize(rows)))
        with self.assertRaises(ValueError): summarize(rows[:-1])
        with self.assertRaises(ValueError): summarize(rows+[rows[0]])

    def test_ncu_identity(self):
        # Historical data only exercises the parser; not v49 measurements.
        p=ROOT/'docs/evidence/a100_o378_roof_v43/reports/o378_roof_v43'
        saved=json.loads((p/'ncu_full/ncu_o3_analysis.json').read_text())
        resources=saved['rows'][0]['probe_resources']
        static=json.loads((p/'codegen.json').read_text())['variants']['0']['entries']['adangel_roof_scale_reuse_o3']
        texts=[(p/f'ncu_full/ncu_o3_p0_{suffix}.csv').read_text().replace(
            'adangel_roof_scale_reuse_o3','adangel_roof_b_lookahead_o3') for suffix in ('raw','source_sass')]
        analyze=runpy.run_path(str(ROOT/'scripts/profile_roof_b_lookahead_probe.py'))['analyze_profile']
        self.assertEqual(analyze(*texts,'o3',0,resources,static)['opcodes'],saved['rows'][0]['opcodes'])
        with self.assertRaisesRegex(ValueError,'fingerprint'):
            analyze(*texts,'o3',0,resources,dict(static,instructions=static['instructions']+1))


if __name__=='__main__': unittest.main()
