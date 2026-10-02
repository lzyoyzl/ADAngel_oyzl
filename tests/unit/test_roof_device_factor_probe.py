from pathlib import Path
import json
import statistics
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from roof_device_factor_probe import MODES


class DeviceFactorContractTests(unittest.TestCase):
    def test_no_host_sync_in_device_action(self):
        s=(ROOT/'csrc/sm80/roof_device_factor_driver.cpp').read_text()
        action=s.split('auto action=[&]() {',1)[1].split('for(int i=0;i<warmup',1)[0]
        self.assertNotIn('cuStreamSynchronize',action)
        self.assertNotIn('validate_status()',action)
        self.assertIn('if(mode==3) prepare();',action)
        self.assertIn('&ws,&meta,&status,&y',action)
        self.assertLess(s.index('if(mode!=0) validate_status();'),s.index('cuEventElapsedTime(times+i'))

    def test_modes_are_not_full_end_to_end(self):
        self.assertEqual(len(MODES),4)
        self.assertEqual(MODES[-1],'prepare_device_compute')
        self.assertTrue(all('cold' not in m for m in MODES))
        s=(ROOT/'scripts/benchmark_roof_device_factor_probe.py').read_text()
        self.assertIn('mixed_tiles',s)
        self.assertIn('post_mutation_recoveries=6',s)
        self.assertIn('NOT full Cold',s)


class DeviceFactorEvidenceTests(unittest.TestCase):
    root=ROOT/'docs/evidence/a100_o378_roof_v61'

    def test_paired_raw_results(self):
        from benchmark_roof_device_factor_probe import summarize,stats
        for run,count in (('screen',4),('trace24',24)):
            d=self.root/f'runs/o378_roof_v61_{run}'
            rows=[json.loads(s) for s in (d/'results.jsonl').read_text().splitlines()]
            saved=json.loads((d/'summary.json').read_text())
            self.assertEqual(len(rows),count*3*4)
            self.assertEqual(len({r['sample_id'] for r in rows}),count)
            self.assertEqual(summarize(rows),saved['records'])
            for r in rows:
                self.assertEqual(len(r['raw_ms']),200)
                self.assertTrue(r['output_bitwise_best'])
                if r['mode']!='control_compute':
                    self.assertTrue(r['metadata_exact']);self.assertEqual(r['guard_status'],0)
                for k,v in stats(r['raw_ms']).items():self.assertAlmostEqual(v,r['summary'][k],places=12)
            self.assertFalse(saved['production_default_changed'])
            if run=='trace24':
                from compare_roof_trace_candidates import metrics
                index={(r['sample_id'],r['round'],r['mode']):r for r in rows}
                ids=sorted({r['sample_id'] for r in rows})
                speed=[statistics.median(index[s,i,'cached_v59_compute']['summary']['median_ms']/
                       index[s,i,'device_cached_compute']['summary']['median_ms'] for i in range(3)) for s in ids]
                self.assertAlmostEqual(statistics.median(speed),1.0044150825961613)
                lo,hi=metrics.bootstrap_median_ci(speed,10000,.95,61)
                self.assertAlmostEqual(lo,1.0011135617817586)
                self.assertAlmostEqual(hi,1.0065933928164454)
            for bad in (rows[:-1],rows+[rows[0]]):
                with self.assertRaises(ValueError):summarize(bad)

    def test_isa_resource_and_unchanged_extension(self):
        d=self.root/'reports/o378_roof_v61'
        audit=json.loads((d/'codegen.json').read_text())
        self.assertTrue(audit['o78_sentinel']['passed'])
        e=audit['entries']['adangel_roof_device_factor_o3']
        self.assertTrue(e['native_s4_s4'] and e['native_u4_s4'] and e['all_copies_bypass_l1'])
        self.assertFalse(e['int8_mma'])
        resources=json.loads((self.root/'runs/o378_roof_v61_trace24/resources.json').read_text())
        for r in resources.values():
            self.assertEqual(r['registers_per_thread'],168)
            self.assertEqual(r['active_blocks_per_sm'],3)
        self.assertEqual(resources['device']['local_size_bytes'],32)
        env=json.loads((self.root/'runs/o378_roof_v61_trace24/environment.json').read_text())
        self.assertEqual(env['device_cubin_sha256'],audit['cubin_sha256'])
        self.assertEqual(env['extension_sha256'],'94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462')
        self.assertIn('NOT full Cold',env['scope'])

    def test_safety_and_mixed_tile_fallback(self):
        for run in ('screen','trace24','memcheck','synccheck','racecheck'):
            r=json.loads((self.root/f'runs/o378_roof_v61_{run}/validation.json').read_text())
            self.assertTrue(r['passed']);self.assertEqual(r['count'],56)
            self.assertEqual(r['rejections'],12);self.assertEqual(r['post_mutation_recoveries'],6)
            mixed=[c for c in r['checks'] if c['pattern']=='mixed_tiles' and c['shape'][1]==256]
            self.assertEqual(len(mixed),4)
            for c in mixed:
                self.assertFalse(c['guard_safe']);self.assertTrue(c['finite_fp32'])
            if run in ('memcheck','synccheck','racecheck'):
                log=(self.root/f'reports/o378_roof_v61_{run}.log').read_text()
                self.assertIn('0 errors',log)
                if run=='racecheck':self.assertIn('0 warnings',log)


if __name__=='__main__':unittest.main()
