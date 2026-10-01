"""v48 geometry contracts; actual CuTe mapping/GPU checks are separate gates."""
from pathlib import Path
import json
import runpy
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))


class StreamWidthTests(unittest.TestCase):
    def test_only_stream_width_changes_in_device_headers(self):
        for old,new,namespace in (
            ('o3_row_scale_epilogue_candidate','o3_stream_width_probe','o3_row_scale_epilogue_experiment'),
            ('o78_unsigned_payload_candidate','o78_stream_width_probe','o78_unsigned_payload_experiment')):
            a=(ROOT/f'csrc/sm80/{old}.cuh').read_text()
            b=(ROOT/f'csrc/sm80/{new}.cuh').read_text().split('\n',1)[1]
            b=b.replace(namespace+'_v48',namespace)
            b=b.replace('constexpr int SliceN=ADANGEL_STREAM_WIDTH==1?32:128;',
                        'constexpr int SliceN=WN*((RoofTune&4)?32:16);')
            self.assertEqual(a,b)

    def test_atom_coverage_and_unchanged_fragment_work(self):
        for width in (32,64,128):
            atoms=width//16
            slices=128//width
            self.assertEqual(atoms*slices,8)
            self.assertEqual(4*2*atoms*slices,64)
            # 2 M atoms * 2 routes * 2 K64 halves * 8 total N atoms.
            self.assertEqual(2*2*2*atoms*slices,64)
            self.assertEqual(2*atoms*slices,16)  # B registers, each K64 half
        source=(ROOT/'scripts/benchmark_roof_stream_width_probe.py').read_text()
        self.assertIn('streamed_n_columns=[64,32,128][size]',source)
        self.assertIn('independent_n_atoms=[4,2,8][size]',source)

    def test_coordinate_program_compares_streams_to_whole_B(self):
        code=(ROOT/'tests/cuda/validate_roof_stream_width_coordinates.cu').read_text()
        for token in ('partition_B(whole)','partition_B(tile)',
                      'original(vi,nb*(Width/16)+ni,ki)',
                      'check_both<64>();check_both<32>();check_both<128>();',
                      'cute::get<0>(x)==cute::get<0>(y) && cute::get<1>(x)==cute::get<1>(y)'):
            self.assertIn(token,code)
        build=(ROOT/'scripts/probe_roof_stream_width_codegen.py').read_text()
        self.assertIn("manifest['host_stream_mapping_passed']=True",build)
        self.assertLess(build.index("'coordinates.log'"),build.index('for size in (0, 1, 2):'))

    def test_launch_and_production_isolation(self):
        source=(ROOT/'csrc/sm80/roof_stream_width_probe.cu').read_text()
        for text in ('constexpr int ProbeThreads=128;','constexpr int ProbeMinBlocks=3;',
                     'constexpr int ProbeWN=2;', '#if ADANGEL_STREAM_WIDTH==0',
                     'Storage)==50688','Storage)==34304'):
            self.assertIn(text,source)
        for path in ('setup.py','csrc/sm80/roof_candidates.cuh'):
            self.assertNotIn('stream_width_probe',(ROOT/path).read_text())

    def test_pairs_and_retained_cv_failures(self):
        summary=runpy.run_path(str(ROOT/'scripts/benchmark_roof_stream_width_probe.py'))['summary']
        rows=[dict(sample_id='x',variant='o3',round=r,stream_width=p,
                   summary=dict(median_ms=1+p,cv_percent=9),mse_vs_o0=.125,
                   mse_vs_paired_fp16=.125,bitwise_equal_current_best=True,
                   mse_vs_current_best=0) for r in range(3) for p in (0,1,2)]
        self.assertEqual([r['paired_speedup'] for r in summary(rows)],[1,.5,1/3])
        self.assertTrue(all(r['cv_failed_records']==3 for r in summary(rows)))
        with self.assertRaises(ValueError): summary(rows[:-1])
        with self.assertRaises(ValueError): summary(rows+[rows[0]])

    def test_profile_identity_and_fingerprint(self):
        # Relabel historical text solely as a parser fixture, not v48 evidence.
        p=ROOT/'docs/evidence/a100_o378_roof_v43/reports/o378_roof_v43'
        saved=json.loads((p/'ncu_full/ncu_o3_analysis.json').read_text())
        resources=saved['rows'][0]['probe_resources']
        static=json.loads((p/'codegen.json').read_text())['variants']['0']['entries']['adangel_roof_scale_reuse_o3']
        texts=[(p/f'ncu_full/ncu_o3_p0_{suffix}.csv').read_text().replace(
            'adangel_roof_scale_reuse_o3','adangel_roof_stream_width_o3') for suffix in ('raw','source_sass')]
        analyze=runpy.run_path(str(ROOT/'scripts/profile_roof_stream_width_probe.py'))['analyze_profile']
        self.assertEqual(analyze(*texts,'o3',0,resources,static)['opcodes'],saved['rows'][0]['opcodes'])
        with self.assertRaisesRegex(ValueError,'fingerprint'):
            analyze(*texts,'o3',0,resources,dict(static,instructions=static['instructions']+1))
        with self.assertRaisesRegex(ValueError,'resources'):
            analyze(*texts,'o3',0,dict(resources,threads=160),static)


if __name__=='__main__': unittest.main()
