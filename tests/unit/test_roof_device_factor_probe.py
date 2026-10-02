from pathlib import Path
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


if __name__=='__main__':unittest.main()
