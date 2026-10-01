"""Recompute the negative cache experiment from immutable raw evidence."""
import csv
import hashlib
import io
import json
from pathlib import Path
import runpy
import re
import unittest

ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'docs/evidence/a100_o378_roof_v37'
P=E/'reports/o378_roof_v37'


def read(path): return json.loads(path.read_text())


class CacheEvidenceTests(unittest.TestCase):
    def test_full_trace_pairs_recompute_and_keep_failures(self):
        mod=runpy.run_path(str(ROOT/'scripts/compare_roof_trace_candidates.py'))
        for suffix,reference,candidate,variants,reassociated,count in (
                ('o3',54,61,['o3'],True,216),('o78',59,62,['o7','o8'],False,432)):
            directory=E/f'runs/o378_roof_v37_trace24_{suffix}'
            rows=[json.loads(s) for s in (directory/'results.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),count)
            self.assertEqual(len({r['sample_id'] for r in rows}),24)
            self.assertTrue(read(directory/'summary.json')['correctness_passed'])
            for row in rows:
                self.assertEqual(len(row['raw_ms']),200)
                self.assertTrue(row.get('bitwise_equal_production') or row.get('mse_regression_passed'))
                if row['tune']==61:
                    self.assertTrue(row['bitwise_equal_row_scale_baseline'])
                    self.assertFalse(row['kernel']['gemm_math_changed_vs_reference'])
            result=mod['compare'](rows,reference,candidate,24,3,variants,['compute_only'],reassociated)
            saved=read(P/f'trace24_{suffix}_paired.json')['rows']
            self.assertEqual(result,saved)
            for row in result:
                self.assertLess(row['paired_speedup_ci95'][1],1)
                self.assertGreater(row['cv'][str(candidate)]['selected_stage_failed'],0)
                self.assertEqual(row['median_mse_vs_paired_fp16'],row['median_reference_mse_vs_paired_fp16'])

    def test_safety_and_codegen_scope(self):
        for suffix,count in (('o3',84),('o78',168)):
            self.assertEqual(len(read(E/f'runs/o378_roof_v37_preflight_{suffix}/validation.json')['checks']),count)
            for prefix,checks in (('mem',count//2),('sync',count//2)):
                self.assertIn('ERROR SUMMARY: 0 errors',(P/f'{prefix}_{suffix}.log').read_text())
                self.assertEqual(len(read(E/f'runs/o378_roof_v37_{prefix}_{suffix}/validation.json')['checks']),checks)
        self.assertEqual(len(read(P/'async_guards.json')['checks']),13)
        self.assertTrue(read(E/'runs/o378_roof_v37_mixed_regression/validation.json')['passed'])
        for name,count in (('production',12),('candidate',148)):
            item=read(P/f'{name}_codegen.json')
            self.assertTrue(item['passed']);self.assertEqual(item['old_symbols'],count)
        allcode=read(P/'all_sm80_codegen.json')
        self.assertFalse(allcode['passed']);self.assertEqual(len(allcode['changed']),2)
        self.assertTrue(all('mixed_binary' in s for s in allcode['changed']))
        audit=read(P/'audit/audit.json')
        self.assertTrue(audit['passed']);self.assertEqual(len(audit['functions']),150)
        self.assertEqual(audit['binary_sha256'],'fe9c2b18d89787231bf988b704a29d2041edcfdad558c98acee3f5b2195b366f')
        for tune in (61,62):
            f=next(f for f in audit['functions'] if f'ELi{tune}EE' in f['symbol'])
            self.assertTrue(f['checks']['cache_ca_ptx'] and f['checks']['cache_no_bypass_sass'])
            self.assertTrue(f['checks']['sass_u4s4'] and f['checks']['sass_s4s4'])
        self.assertIn('Ran 238 tests',(P/'cpu_tests_final.log').read_text())
        self.assertIn('OK',(P/'cpu_tests_final.log').read_text())

    def test_ncu_recompute_and_exact_math_work(self):
        mod=runpy.run_path(str(ROOT/'scripts/analyze_roof_scale_ncu.py'))
        for variant,tunes in (('o3',(54,61)),('o7',(59,62))):
            saved=read(P/f'ncu_{variant}_analysis.json')
            for source in saved['sources']:
                self.assertEqual(hashlib.sha256((E/source['file']).read_bytes()).hexdigest(),source['sha256'])
            recomputed=[]; instructions=[]
            for tune in tunes:
                raw=(P/f'ncu_{variant}_t{tune}_raw.csv').read_text()
                sass=(P/f'ncu_{variant}_t{tune}_source_sass.csv').read_text()
                recomputed.append(mod['analyze'](raw,sass,tune,variant,True,True))
                stream=io.StringIO(sass);next(csv.reader(stream))
                source=list(csv.DictReader(stream))
                base=int(source[0]['Address'],16)
                def normalize(row):
                    text=row['Source'].replace('.BYPASS','')
                    # NCU relocates code independently in each process. Normalize
                    # absolute branch/call targets, not arithmetic immediates.
                    if re.search(r'\b(?:BRA|CALL)\b',text):
                        text=re.sub(r'0x[0-9a-f]{10,}',lambda m:hex(int(m[0],16)-base),text)
                    return text
                instructions.append([normalize(r) for r in source])
            self.assertEqual(recomputed,saved['rows'])
            self.assertEqual(instructions[0],instructions[1])
            a,b=recomputed
            self.assertEqual(a['opcodes'],b['opcodes'])
            self.assertGreater(b['l1_data_pipe_peak_percent'],a['l1_data_pipe_peak_percent'])
            self.assertGreater(b['long_scoreboard_stall_per_issue'],a['long_scoreboard_stall_per_issue'])
            self.assertLess(b['eligible_warps'],a['eligible_warps'])
            self.assertGreater(b['optimistic_fixed_work_lower_bound_ms'],a['optimistic_fixed_work_lower_bound_ms'])


if __name__=='__main__': unittest.main()
