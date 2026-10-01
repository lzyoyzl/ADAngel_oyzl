"""v47 geometry contracts; actual CuTe mapping/GPU checks are separate gates."""
from pathlib import Path
import json
import runpy
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))


class WarpAspectTests(unittest.TestCase):
    def test_only_warp_geometry_changes_in_device_headers(self):
        for old,new,namespace in (
            ('o3_row_scale_epilogue_candidate','o3_warp_aspect_probe','o3_row_scale_epilogue_experiment'),
            ('o78_unsigned_payload_candidate','o78_warp_aspect_probe','o78_unsigned_payload_experiment')):
            a=(ROOT/f'csrc/sm80/{old}.cuh').read_text()
            b=(ROOT/f'csrc/sm80/{new}.cuh').read_text().split('\n',1)[1]
            b=b.replace(namespace+'_v47',namespace)
            b=b.replace('static constexpr int WM=4/WN;','static constexpr int WM=2;')
            b=b.replace('static_assert(M==64 && (WN==1 || WN==2 || WN==4) && K==128);',
                        'static_assert(M==64 && WN==2 && K==128);')
            self.assertEqual(a,b)

    def test_fragment_work_model_is_not_a_performance_prediction(self):
        # Logical packed operand bytes loaded into warp fragments per CTA/G128.
        # This is not a dynamic LDSM instruction/wavefront count.
        work=[]
        for wm,wn in ((2,2),(1,4),(4,1)):
            self.assertEqual(wm*wn,4)
            self.assertEqual(64*128//(wm*wn*32),64)
            self.assertEqual((64//(16*wm))*(128//(32*wn))*4*4,64)
            work.append(2*64*64*wn+128*64*wm)
        self.assertEqual(work,[32768,40960,40960])

    def test_coordinate_program_uses_actual_cute_and_checks_unique_ownership(self):
        code=(ROOT/'tests/cuda/validate_roof_warp_aspect_coordinates.cu').read_text()
        for token in ('partition_C(identity)','for(int count:owners) assert(count==1);',
                      'cute::size(coords))==64','check_both<2>();check_both<4>();check_both<1>();',
                      'sizeof(typename C::Storage)==sizeof(typename Reference::Storage)',
                      'row==cute::get<0>(high_coords(value))','C::WM*wn==4'):
            self.assertIn(token,code)
        build=(ROOT/'scripts/probe_roof_warp_aspect_codegen.py').read_text()
        self.assertIn("manifest['host_coordinate_check_passed']=True",build)
        self.assertLess(build.index("'coordinates.log'"),build.index('for size in (0, 1, 2):'))

    def test_launch_and_production_isolation(self):
        source=(ROOT/'csrc/sm80/roof_warp_aspect_probe.cu').read_text()
        for text in ('constexpr int ProbeThreads=128;','constexpr int ProbeMinBlocks=3;',
                     'ADANGEL_WARP_ASPECT==1?4:ADANGEL_WARP_ASPECT==2?1:2',
                     '#if ADANGEL_WARP_ASPECT==0', 'Storage)==50688','Storage)==34304'):
            self.assertIn(text,source)
        for path in ('setup.py','csrc/sm80/roof_candidates.cuh'):
            self.assertNotIn('warp_aspect_probe',(ROOT/path).read_text())

    def test_pairs_and_retained_cv_failures(self):
        summary=runpy.run_path(str(ROOT/'scripts/benchmark_roof_warp_aspect_probe.py'))['summary']
        rows=[dict(sample_id='x',variant='o3',round=r,warp_aspect=p,
                   summary=dict(median_ms=1+p,cv_percent=9),mse_vs_o0=.125,
                   mse_vs_paired_fp16=.125,bitwise_equal_current_best=True,
                   mse_vs_current_best=0) for r in range(3) for p in (0,1,2)]
        self.assertEqual([r['paired_speedup'] for r in summary(rows)],[1,.5,1/3])
        self.assertTrue(all(r['cv_failed_records']==3 for r in summary(rows)))
        with self.assertRaises(ValueError): summary(rows[:-1])
        with self.assertRaises(ValueError): summary(rows+[rows[0]])

    def test_profile_identity_and_fingerprint(self):
        # Relabel historical text solely as a parser fixture, not v47 evidence.
        p=ROOT/'docs/evidence/a100_o378_roof_v43/reports/o378_roof_v43'
        saved=json.loads((p/'ncu_full/ncu_o3_analysis.json').read_text())
        resources=saved['rows'][0]['probe_resources']
        static=json.loads((p/'codegen.json').read_text())['variants']['0']['entries']['adangel_roof_scale_reuse_o3']
        texts=[(p/f'ncu_full/ncu_o3_p0_{suffix}.csv').read_text().replace(
            'adangel_roof_scale_reuse_o3','adangel_roof_warp_aspect_o3') for suffix in ('raw','source_sass')]
        analyze=runpy.run_path(str(ROOT/'scripts/profile_roof_warp_aspect_probe.py'))['analyze_profile']
        self.assertEqual(analyze(*texts,'o3',0,resources,static)['opcodes'],saved['rows'][0]['opcodes'])
        with self.assertRaisesRegex(ValueError,'fingerprint'):
            analyze(*texts,'o3',0,resources,dict(static,instructions=static['instructions']+1))
        with self.assertRaisesRegex(ValueError,'resources'):
            analyze(*texts,'o3',0,dict(resources,threads=160),static)


if __name__=='__main__': unittest.main()
