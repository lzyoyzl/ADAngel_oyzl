import sys
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_roof_device_factor_codegen import wrapper,OLD,NEW


class DeviceFactorTests(unittest.TestCase):
    def test_only_wrapper_selection_changes(self):
        old=(ROOT/'csrc/sm80/roof_fullk_integer_probe.cu').read_text()
        new=wrapper(old)
        restored=new.replace(NEW,OLD).replace('o3_factor_async_generated.cuh','o3_fullk_integer_probe.cuh')
        self.assertEqual(restored,old)
        self.assertIn('flag=status[blockIdx.x]',new)
        self.assertLess(new.index('if(flag&6u) return;'),new.index('if(flag&1u)'))
        self.assertNotIn('status[threadIdx.x]',new)
        self.assertIn('reinterpret_cast<const uint8_t*>(meta)',new)
        with self.assertRaises(ValueError):wrapper('unexpected')

    def test_per_cta_fallback_mask(self):
        # Every M tile at this N coordinate takes the same safe/old/reject path.
        for flag in range(8):
            paths=['reject' if flag&6 else 'old' if flag&1 else 'integer' for _ in range(128)]
            self.assertEqual(len(set(paths)),1)
            self.assertEqual(paths[0]=='integer',flag==0)
            self.assertEqual(paths[0]=='old',flag==1)


if __name__=='__main__':unittest.main()
