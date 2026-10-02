import math
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_nv6_branchless_codegen import OLD,NEW,transform,audit_payloads


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

    def test_archived_sass_and_reaudit(self):
        base=ROOT/'docs/evidence/a100_o378_roof_v66/reports/o378_roof_v66_build'
        receipt=json.loads((base/'audit_rechecked.json').read_text())
        original=(base/'audit.json').read_bytes()
        self.assertEqual(hashlib.sha256(original).hexdigest(),receipt['original_audit_sha256'])
        self.assertFalse(json.loads(original)['passed'])
        result=audit_payloads([(base/f'policy_{i}/conversion.sass').read_text() for i in (0,1)],
                             [(base/f'policy_{i}/conversion.resources.txt').read_text() for i in (0,1)])
        for key,value in result.items():self.assertEqual(receipt[key],value)
        self.assertTrue(result['passed']);self.assertEqual(len(result['preserved_entries']),20)
        self.assertTrue(all(result['preserved_entries'].values()))
        for row in result['targets']:
            if 'ELi16E' in row['symbol']:
                self.assertEqual(row['resources']['REG'],19 if row['policy']==0 else 31)
                self.assertEqual(row['static_BRA'],33 if row['policy']==0 else 1)
        from compare_a100_codegen import compare
        before=ROOT/'docs/evidence/a100_o378_roof_v53/reports/o378_roof_v53_build2/conversion.sass'
        self.assertTrue(compare(before.read_text(),(base/'policy_0/conversion.sass').read_text(),
                                r'adangel_sm80_vector_fixed_conversion')['passed'])

    def test_archived_measurements_and_mse(self):
        base=ROOT/'docs/evidence/a100_o378_roof_v66/reports'
        from benchmark_vector_conversion_probe import summarize
        from benchmark_a100_o1 import stats
        rows=[json.loads(s) for s in (base/'o378_roof_v66_screen/results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),24)
        for row in rows:
            self.assertEqual(len(row['timings_ms']),200)
            self.assertEqual(row['inner'],100)
            self.assertTrue(row['bitwise_payload_and_scale'])
            calculated=stats(row['timings_ms'])
            self.assertEqual(set(calculated),set(row['summary']))
            for key,value in calculated.items():
                # NumPy builds differ in the last bit of std/CV reduction;
                # native CUDA Event samples themselves remain exact.
                self.assertAlmostEqual(value,row['summary'][key],places=12)
        result=summarize(rows)
        self.assertEqual(result,json.loads((base/'o378_roof_v66_screen/summary.json').read_text()))
        self.assertLess(result[1]['paired_speedup_ci95'][1],1)
        self.assertTrue(all(r['cv_failed_records']==0 for r in result))
        mse=[json.loads(s) for s in (base/'o378_roof_v66_screen/output_mse.jsonl').read_text().splitlines()]
        self.assertEqual(len(mse),8)
        for item in mse:
            self.assertEqual(item['mse_vs_current_best'],0)
            self.assertTrue(item['output_bitwise_current_best'])
            self.assertEqual(item['paired_reference'],'o6')
        validation=json.loads((base/'o378_roof_v66_validation/validation.json').read_text())
        self.assertTrue(validation['passed'])
        self.assertEqual(len(validation['checks']),393)
        self.assertEqual(len(validation['rejected']),18)
