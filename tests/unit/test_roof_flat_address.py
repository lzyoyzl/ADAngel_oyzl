"""v44 bounded offset algebra; GPU correctness/audit remain separate gates."""
from pathlib import Path
import json
import runpy
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))


class FlatAddressTests(unittest.TestCase):
    def test_complete_relative_offsets_and_bounds(self):
        shapes = ((64,128,128), (4096,4096,4096), (64,128,16777088),
                  (65536,128,32640), (64,8388480,128), (128,256,640))
        for m,n,k in shapes:
            self.assertLessEqual(max(m*n,m*k,n*k), 2**31-1)
            self.assertLessEqual(max(m//64,n//128), 65535)
            for rows,tile in ((m,64),(n,128)):
                for block in set((0,1 if rows>tile else 0,rows//tile-1)):
                    for g in set((0,1 if k>128 else 0,k//128-1)):
                        for off in range(0,tile*64,16):
                            row,col=divmod(off,64)
                            old=g*rows*64+(block*tile+row)*64+col
                            flat=(g*rows+block*tile)*64+off
                            self.assertEqual(old,flat)
                            self.assertLessEqual(flat+16,rows*k//2)
                            self.assertLess(flat,2**31)
                            if tile==64:
                                high=flat+m*(k//2)
                                self.assertLessEqual(high+16,m*k)
                                self.assertLess(high,2**31)
                        for first in range(0,tile,4):
                            index=g*rows+block*tile+first
                            self.assertLessEqual(index+4,rows*(k//128))
                            self.assertLess((index+4)*4,2**31)

    def test_driver_enforces_guards(self):
        source=(ROOT/'csrc/sm80/roof_producer_warp_driver.cpp').read_text()
        for expression in ('int64_t(m)*n>INT32_MAX','int64_t(m)*k>INT32_MAX',
                           'int64_t(n)*k>INT32_MAX','m%64','n%128','k%128'):
            self.assertIn(expression,source)

    def test_compute_pipeline_and_copy_count_unchanged(self):
        for old,new in (('o3_row_scale_epilogue_candidate','o3_flat_address_probe'),
                        ('o78_unsigned_payload_candidate','o78_flat_address_probe')):
            a=(ROOT/f'csrc/sm80/{old}.cuh').read_text()
            b=(ROOT/f'csrc/sm80/{new}.cuh').read_text()
            start='template<int M,int N,int K,bool Fast,bool Cached=false,bool Magic'
            self.assertEqual(a[a.index(start):a.rindex('} // namespace')],
                             b[b.index(start):b.rindex('} // namespace')])
            for token in ('copy16(', 'cp.async.commit_group;', '__syncthreads();'):
                self.assertEqual(a.count(token),b.count(token))
            self.assertIn('a+high_offset',b)
            self.assertIn('const uint32_t offset=(uint32_t(stage)*uint32_t(m)+blockIdx.y*M)*C::Bytes+off;',b)
            self.assertIn('if constexpr(ADANGEL_FLAT_ADDRESS==2)',b)

    def test_launch_and_production_isolation(self):
        source=(ROOT/'csrc/sm80/roof_flat_address_probe.cu').read_text()
        self.assertIn('constexpr int ProbeThreads=128;',source)
        self.assertIn('constexpr int ProbeMinBlocks=3;',source)
        self.assertIn('#if ADANGEL_FLAT_ADDRESS==0',source)
        for path in ('setup.py','csrc/sm80/roof_candidates.cuh'):
            self.assertNotIn('flat_address_probe',(ROOT/path).read_text())

    def test_pairs_and_retained_cv_failures(self):
        summary=runpy.run_path(str(ROOT/'scripts/benchmark_roof_flat_address_probe.py'))['summary']
        rows=[dict(sample_id='x',variant='o3',round=r,flat_address=p,
                   summary=dict(median_ms=1+p,cv_percent=9),mse_vs_o0=.125,
                   mse_vs_paired_fp16=.125,bitwise_equal_current_best=True,
                   mse_vs_current_best=0) for r in range(3) for p in (0,1,2)]
        self.assertEqual([r['paired_speedup'] for r in summary(rows)],[1,.5,1/3])
        self.assertTrue(all(r['cv_failed_records']==3 for r in summary(rows)))
        with self.assertRaises(ValueError): summary(rows[:-1])
        with self.assertRaises(ValueError): summary(rows+[rows[0]])

    def test_profile_identity_and_fingerprint(self):
        # Historical data relabeled only for parser fixture testing.
        p=ROOT/'docs/evidence/a100_o378_roof_v43/reports/o378_roof_v43'
        saved=json.loads((p/'ncu_full/ncu_o3_analysis.json').read_text())
        resources=saved['rows'][0]['probe_resources']
        static=json.loads((p/'codegen.json').read_text())['variants']['0']['entries']['adangel_roof_scale_reuse_o3']
        texts=[(p/f'ncu_full/ncu_o3_p0_{suffix}.csv').read_text().replace(
            'adangel_roof_scale_reuse_o3','adangel_roof_flat_address_o3') for suffix in ('raw','source_sass')]
        analyze=runpy.run_path(str(ROOT/'scripts/profile_roof_flat_address_probe.py'))['analyze_profile']
        self.assertEqual(analyze(*texts,'o3',0,resources,static)['opcodes'],saved['rows'][0]['opcodes'])
        with self.assertRaisesRegex(ValueError,'fingerprint'):
            analyze(*texts,'o3',0,resources,dict(static,instructions=static['instructions']+1))
        with self.assertRaisesRegex(ValueError,'resources'):
            analyze(*texts,'o3',0,dict(resources,threads=160),static)


if __name__=='__main__': unittest.main()
