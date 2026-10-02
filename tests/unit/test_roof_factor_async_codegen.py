from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
import probe_roof_factor_async_codegen as codegen


class FactorAsyncTests(unittest.TestCase):
    def test_only_factor_supply_and_anchor_representation_change(self):
        source=(ROOT/'csrc/sm80/o3_fullk_integer_probe.cuh').read_text()
        changed=codegen.generated_header(source)
        restored=changed.replace(codegen.PREFETCH_NEW,codegen.PREFETCH_OLD).replace(codegen.ANCHOR_NEW,codegen.ANCHOR_OLD)
        self.assertEqual(restored,source)
        self.assertEqual(changed.count('cp.async.commit_group;'),1)
        with self.assertRaises(ValueError): codegen.generated_header('unexpected')

    def test_panel_is_aligned_and_completely_covered(self):
        covered=[]
        for lane in range(32):
            self.assertEqual(lane*4*4%16,0)
            covered.extend(range(lane*4,lane*4+4))
        self.assertEqual(covered,list(range(128)))
        self.assertEqual((3*(64*64+64*64+128*64))%128,0)


if __name__=='__main__': unittest.main()
