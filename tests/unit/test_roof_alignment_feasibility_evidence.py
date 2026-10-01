"""Recompute archived CPU-only scale range evidence, not a kernel acceptance."""
import json
from pathlib import Path
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[2]


class AlignmentEvidenceTests(unittest.TestCase):
    def test_recompute_all_windows(self):
        script=runpy.run_path(str(ROOT/'scripts/inspect_o3_integer_alignment_feasibility.py'))
        d=json.loads((ROOT/'docs/evidence/o3_integer_alignment_feasibility_v1/reports/o3_integer_alignment_feasibility_v1/feasibility.json').read_text())
        self.assertEqual(len(d['samples']),24)
        self.assertEqual(len({s['sample_id'] for s in d['samples']}),24)
        self.assertEqual(d['partial_bound'],131072)
        self.assertEqual(d['summary'],[script['aggregate'](d['samples'],w) for w in script['WINDOWS']])
        self.assertEqual([s['windows'] for s in d['summary']],[1572864,786432,393216,98304])
        self.assertEqual([s['maximum_absolute_integer_bound'] for s in d['summary']],
                         [4325376,9830400,19791872,81657856])
        self.assertTrue(all(s['int32_safe_fraction']==1.0 for s in d['summary']))
        self.assertEqual(d['summary'][0]['exponent_spread_histogram'],
                         {'0':1175389,'1':374168,'2':21469,'3':1756,'4':76,'5':6})
        self.assertIn('no_new_kernel_or_MSE_or_timing',d['scope'])


if __name__=='__main__': unittest.main()
