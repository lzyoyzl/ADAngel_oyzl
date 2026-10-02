from pathlib import Path
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


if __name__=='__main__':unittest.main()
