from pathlib import Path
import runpy
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
CHECK = runpy.run_path(str(ROOT / 'scripts/probe_roof_l2_codegen.py'))['static_entries']


class L2PrefetchTests(unittest.TestCase):
    def test_probe_is_not_production(self):
        source = (ROOT / 'csrc/sm80/roof_l2_prefetch_probe.cu').read_text()
        self.assertNotIn('roof_l2_prefetch_probe.cu', (ROOT / 'setup.py').read_text())
        for hint in ('cp.async.cg.shared.global [', 'cp.async.cg.shared.global.L2::128B',
                     'cp.async.cg.shared.global.L2::256B'):
            self.assertIn(hint, source)
        self.assertNotIn('cp.async.ca.', source)
        for body in ('o3_row_scale_epilogue_candidate.cuh', 'o78_unsigned_payload_candidate.cuh'):
            self.assertIn(f'#include "{body}"', source)

    def test_same_entry_checks_fail_closed(self):
        sass = '\n'.join(f'''Function : adangel_roof_l2_o{name}
/*0000*/ LDGSTS.E.BYPASS.128 [R1], [R2.64];
/*0010*/ IMMA.16832.U4.S4 R4, R5, R6, R7;
/*0020*/ IMMA.16832.S4.S4 R4, R5, R6, R7;''' for name in ('3', '78'))
        self.assertTrue(all(x['all_copies_bypass_l1'] for x in CHECK(sass).values()))
        self.assertFalse(all(x['all_copies_bypass_l1'] for x in CHECK(sass.replace('.BYPASS', '')).values()))
        self.assertFalse(all(x['native_u4_s4'] for x in CHECK(sass.replace('.U4.', '.U8.')).values()))
        with self.assertRaises(ValueError):
            CHECK(sass.replace('adangel_roof_l2_o78', 'unrelated_kernel'))


if __name__ == '__main__':
    unittest.main()
