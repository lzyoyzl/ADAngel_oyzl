from pathlib import Path
import json
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from roof_full_pipeline_probe import MODES,stage_contract


class FullPipelineContractTests(unittest.TestCase):
    def test_dual_timing_contract(self):
        for mode in MODES:
            counts=stage_contract(mode,100)
            self.assertEqual('weight_conversion' in counts,mode in ('conversion_only','cold'))
            self.assertEqual('activation_conversion' in counts,mode!='compute_only')
            self.assertEqual('gemm' in counts,mode!='conversion_only')
            self.assertEqual(counts['total'],100 if mode=='conversion_only' else 1)
        for mode,inner in (('unknown',100),('cold',0)):
            with self.assertRaises(ValueError):stage_contract(mode,inner)

    def test_guard_cost_is_weight_cost(self):
        s=(ROOT/'csrc/sm80/roof_full_pipeline_driver.cpp').read_text()
        weight=s.split('auto cvw=[&]() {',1)[1].split('auto cva=',1)[0]
        self.assertIn('if(policy)',weight);self.assertIn('&gws,&meta,&status,&n',weight)
        timed=s.split('for(int i=0;i<repeats;++i) {',1)[1].split('auto batch=',1)[0]
        self.assertNotIn('cuMemcpyDtoH',timed)
        self.assertIn('if(weight) cvw();',timed)
        self.assertIn('if(activation) cva();',timed)
        self.assertIn('times[3*repeats+i]=times[i]+times[repeats+i]',s)
        self.assertIn('times[stage*repeats+i]/=inner',s)


class FullPipelineEvidenceTests(unittest.TestCase):
    root=ROOT/'docs/evidence/a100_o378_roof_v62'

    def test_four_modes_and_raw_statistics(self):
        from benchmark_roof_full_pipeline import summarize,timing_check,stats
        for name,samples in (('screen',4),('trace24',24)):
            d=self.root/f'runs/o378_roof_v62_{name}'
            rows=[json.loads(s) for s in (d/'results.jsonl').read_text().splitlines()]
            saved=json.loads((d/'summary.json').read_text())
            self.assertEqual(len(rows),samples*8)
            self.assertEqual(summarize(rows),saved['records'])
            for r in rows:
                self.assertTrue(r['payload_bitwise'] and r['bitwise_equal_current_best'])
                self.assertEqual(r['mse_vs_current_best'],0)
                self.assertEqual(r['stage_timing_inner_repeats'],stage_contract(r['mode'],100))
                timing_check(dict(r,timings_ms=r['raw_ms']),r['mode'],100,200,r['implementation'])
                self.assertEqual(r['weight_cached'],r['mode'] not in ('cold','conversion_only'))
                for s,raw in r['raw_ms'].items():
                    self.assertEqual(len(raw),200)
                    for k,v in stats(raw).items():self.assertAlmostEqual(v,r['stage_summaries'][s][k],places=12)
            self.assertFalse(saved['production_default_changed'])
            for bad in ([],rows[:-1],rows+[rows[0]]):
                with self.assertRaises(ValueError):summarize(bad)

    def test_unchanged_conversion_gemm_and_extension(self):
        from compare_a100_codegen import compare
        d=self.root/'runs/o378_roof_v62_trace24'
        result=compare((ROOT/'docs/evidence/a100_o378_roof_v36/reports/o378_roof_v36/conversion_audit/conversion_kernels.sass').read_text(),
                       (d/'build/conversion.sass').read_text(),r'adangel_sm80_o3_tiled_conversionILb[01]ELb1EE')
        self.assertTrue(result['passed']);self.assertEqual(result['old_symbols'],2)
        self.assertTrue(json.loads((self.root/'reports/o378_roof_v62_conversion_compare.json').read_text())['passed'])
        b=json.loads((d/'build/four_build.json').read_text())
        old=json.loads((ROOT/'docs/evidence/a100_o378_roof_v61/reports/o378_roof_v61/codegen.json').read_text())
        self.assertEqual(b['core']['device_cubin_sha256'],old['cubin_sha256'])
        env=json.loads((d/'environment.json').read_text())
        self.assertEqual(env['extension_sha256'],'94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462')

    def test_scoped_safety(self):
        for name in ('screen','trace24','memcheck','synccheck','racecheck'):
            v=json.loads((self.root/f'runs/o378_roof_v62_{name}/validation.json').read_text())
            self.assertTrue(v['passed']);self.assertEqual(len(v['checks']),96)
            self.assertEqual(len(v['rejected']),8)
            self.assertEqual({r['mode'] for r in v['checks']},set(MODES))
            for r in v['checks']:
                self.assertTrue(r['payload_exact'] and r['scale_exact'] and r['semantic_tolerance_passed'])
            if name in ('memcheck','synccheck','racecheck'):
                log=(self.root/f'reports/o378_roof_v62_{name}.log').read_text()
                self.assertIn('0 errors',log)
                if name=='racecheck':self.assertIn('0 warnings',log)


if __name__=='__main__':unittest.main()
