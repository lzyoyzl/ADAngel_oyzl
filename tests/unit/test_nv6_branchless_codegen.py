import math
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_nv6_branchless_codegen import OLD,NEW,transform


class Nv6BranchlessTests(unittest.TestCase):
    def test_all_64_codes_match_e2m3_rne(self):
        for c in range(64):
            x=c&31;e=x>>3;m=x&7
            reference=round(m/2 if e==0 else math.ldexp(8+m,e-2))
            if c&32:reference=-reference
            small=(x>>1)+int((x&3)==3);large=(8+(x&7))<<((x>>3)&1)
            mag=large if x&16 else small;sign=-((c>>5)&1)
            result=(mag^sign)-sign
            self.assertEqual(result,reference)
            self.assertGreaterEqual(result,-30);self.assertLessEqual(result,30)

    def test_only_decoder_changes(self):
        header=(ROOT/'csrc/sm80/roof_vector_conversion_impl.cuh').read_text()
        changed=transform(header)
        self.assertEqual(changed.replace(NEW,OLD),header)
        self.assertIn('selp.u32',changed)
        self.assertIn('effective[dst_group]=value;',changed)
        with self.assertRaises(ValueError):transform(changed)
