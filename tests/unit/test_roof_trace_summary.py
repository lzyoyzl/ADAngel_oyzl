"""CPU checks of sample-paired aggregation; no CUDA/imported model required."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import benchmark_a100_roof_trace as runner

spec=importlib.util.spec_from_file_location('roof_metrics',ROOT/'python/adangel/benchmark/metrics.py')
metrics=importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class RoofTraceSummaryTest(unittest.TestCase):
    def test_two_case_order_balanced_for_each_mode_and_variant(self):
        for variant in range(3):
            for mode in range(4):
                for round_index in range(3):
                    orders=[runner.measurement_order([-1,6],si,variant,round_index,mode) for si in range(24)]
                    self.assertEqual(orders.count([-1,6]),12)
                    self.assertEqual(orders.count([6,-1]),12)
                    self.assertNotEqual(orders[0],orders[1])
        self.assertNotEqual(runner.measurement_order([-1,6],0,0,0),
                            runner.measurement_order([-1,6],0,0,1))

    def rows(self):
        rows=[]
        for sid in ('a','b'):
            for r in range(2):
                for tune in (-1,1):
                    value=(1+r)*(.5 if sid=='a' else 1)
                    rows.append(dict(sample_id=sid,variant='o7',round=r,tune=tune,
                        summary=dict(median_ms=value*(2 if tune==-1 else 1),cv_percent=5 if r else 1),
                        mse_vs_o0=.1,mse_vs_paired_fp16=.02))
        return rows

    def call(self,rows):
        with patch.dict(sys.modules,{'adangel':types.ModuleType('adangel'),
                'adangel.benchmark':types.ModuleType('adangel.benchmark'),
                'adangel.benchmark.metrics':metrics}):
            return runner.summarize(rows)

    def test_sample_pairing_and_no_cv_filtering(self):
        out=self.call(self.rows())
        candidate=next(r for r in out if r['tune']==1)
        self.assertEqual(candidate['samples'],2)
        self.assertEqual(candidate['records'],4)
        self.assertEqual(candidate['cv_failed_records'],2)
        self.assertEqual(candidate['paired_speedup_median'],2)
        self.assertEqual(candidate['paired_speedup_ci95'],[2,2])
        self.assertEqual(candidate['median_ms'],1.125)
        self.assertEqual(candidate['median_mse_vs_paired_fp16'],.02)

    def test_missing_control_duplicate_and_changed_mse(self):
        rows=self.rows()
        with self.assertRaises(ValueError): self.call(rows+[rows[0]])
        with self.assertRaises(KeyError): self.call([r for r in rows if r['tune']!=-1])
        rows[-1]['mse_vs_paired_fp16']=.03
        with self.assertRaises(ValueError): self.call(rows)

    def test_modes_are_separate(self):
        rows=self.rows()
        both=[dict(r,mode=mode) for mode in ('compute_only','cold') for r in rows]
        out=self.call(both)
        self.assertEqual(len(out),4)
        self.assertEqual({r['mode'] for r in out},{'compute_only','cold'})


if __name__=='__main__': unittest.main()
