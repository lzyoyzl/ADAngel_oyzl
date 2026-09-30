"""Reconcile v24 claims with archived same-binary records, never filtered timings."""
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]
EVIDENCE=ROOT/'docs/evidence/a100_o378_roof_v24'
BINARY='07c8631383290138cb6094e01ea79827df86b586e01c030ac028604e92368f93'


def read(relative):
    return json.loads((EVIDENCE/relative).read_text())


@unittest.skipUnless((EVIDENCE/'runs/o378_roof_v24_trace24/summary.json').exists(),
                     'v24 archive has not yet been imported')
class V24EvidenceTest(unittest.TestCase):
    def test_compute_pairing_and_bitwise_claims(self):
        path=EVIDENCE/'runs/o378_roof_v24_trace24'
        env=json.loads((path/'environment.json').read_text())
        self.assertEqual(env['binary_sha256'],BINARY)
        self.assertEqual(env['args']['rounds'],5)
        summary=json.loads((path/'summary.json').read_text())
        self.assertTrue(summary['all_24_samples'] and summary['no_filtering'])
        self.assertEqual(summary['numerical_policy'],'bitwise')
        rows=[json.loads(line) for line in (path/'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),1800)
        self.assertEqual(len({r['sample_id'] for r in rows}),24)
        for r in rows:
            self.assertTrue(r['bitwise_equal_production'])
            self.assertEqual(r['mse_vs_production'],0)
            if r['tune'] in (41,42):
                self.assertTrue(r['payload_layout_bitwise_verified'])
        for variant,file in (('o3','trace24_t41_vs22'),('o7','trace24_t42_vs23'),('o8','trace24_t42_vs23')):
            d=read(f'reports/o378_roof_v24/{file}.json')
            self.assertEqual(d['source_binary_sha256'],BINARY)
            row=next(r for r in d['rows'] if r['variant']==variant)
            self.assertGreater(row['paired_speedup_ci95'][0],1)
            self.assertEqual(row['median_candidate_mse_vs_production'],0)

    def test_four_modes_keep_reorder_cost_and_all_records(self):
        for suffix,count in (('o3',288),('o78',576)):
            path=EVIDENCE/f'runs/o378_roof_v24_{suffix}_four24'
            env=json.loads((path/'environment.json').read_text())
            self.assertEqual(env['binary_sha256'],BINARY)
            self.assertEqual(env['args']['rounds'],1)
            self.assertEqual((env['args']['warmup'],env['args']['repeats'],env['args']['inner']),
                             (50,200,100))
            summary=json.loads((path/'summary.json').read_text())
            self.assertTrue(summary['all_four_modes_completed'] and summary['no_filtering'])
            rows=[json.loads(line) for line in (path/'results.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),count)
            self.assertEqual({r['mode'] for r in rows},
                             {'conversion_only','compute_only','cold','steady_state'})
            for row in rows:
                self.assertTrue(row['bitwise_equal_production'])
                if row['tune'] not in (41,42): continue
                self.assertTrue(row['payload_layout_bitwise_verified'])
                meta=row['kernel']
                self.assertTrue(meta['payload_reorder_in_conversion'])
                self.assertFalse(meta['payload_reorder_fused'])
                self.assertEqual(meta['activation_payload_reorder_traffic_bytes'],2*4096**2)
                self.assertEqual(meta['weight_payload_reorder_traffic_bytes'],4096**2)

    def test_native_audit_and_previous_codegen(self):
        audit=read('reports/o378_roof_v24/audit/audit.json')
        self.assertTrue(audit['passed'])
        self.assertEqual(audit['binary_sha256'],BINARY)
        self.assertEqual(len(audit['functions']),120)
        for file,count in (('production_codegen',12),('candidate_codegen',114)):
            result=read(f'reports/o378_roof_v24/{file}.json')
            self.assertTrue(result['passed'])
            self.assertEqual(result['old_symbols'],count)
            self.assertEqual(len(result['unchanged']),count)
        broad=read('reports/o378_roof_v24/all_sm80_codegen.json')
        self.assertFalse(broad['passed'])  # Never relabel a raw machine-code difference as equality.
        self.assertEqual(broad['old_symbols'],239)
        self.assertEqual(len(broad['changed']),4)
        self.assertFalse(broad['missing'])
        self.assertTrue(all('adangel_sm80_mixed_binary' in s for s in broad['changed']))
        details=read('reports/o378_roof_v24/all_sm80_delta_details.json')
        self.assertTrue(all(r['normalized_text_equal'] and r['old_count']==r['new_count']
                            for r in details['rows']))
        regression=read('runs/o378_roof_v24_mixed_regression/validation.json')
        self.assertTrue(regression['passed'])
        self.assertEqual(regression['binary_sha256'],BINARY)
        self.assertEqual(len(regression['binary_gemm_checks']),480)

    def test_ncu_work_reduction_is_specific_not_mma_removal(self):
        d=read('reports/o378_roof_v24/ncu_analysis.json')
        rows={r['tune']:r for r in d['rows']}
        self.assertEqual(set(rows),{22,23,41,42})
        for tune,row in rows.items():
            copy=row['source_memory_work_by_opcode']['LDGSTS']
            self.assertEqual(copy['L1 Wavefronts Shared Excessive'],0 if tune in (41,42) else 8388608)
            self.assertEqual(copy['L1 Wavefronts Shared'],8388608 if tune in (41,42) else 16777216)
            self.assertEqual(row['source_memory_work']['L1 Wavefronts Shared'],
                             30801920 if tune in (41,42) else 39190528)
            for opcode in ('IMMA','I2F','FMUL','FFMA'):
                self.assertEqual(row['opcodes'][opcode],16777216)
            self.assertEqual(row['opcodes']['LDSM'],4194304)
            self.assertEqual(row['registers_per_thread'],168)
            self.assertEqual(row['max_ctas_per_sm_from_launch_limits'],3)

    def test_o3_ncu_does_not_hide_remaining_scale_traffic(self):
        rows={r['tune']:r for r in read('reports/o378_roof_v24/ncu_o3_analysis.json')['rows']}
        self.assertEqual(set(rows),{22,41})
        for tune,row in rows.items():
            self.assertEqual(row['source_memory_work']['L1 Wavefronts Shared Excessive'],
                             8388608 if tune==22 else 0)
            self.assertEqual(row['source_memory_work']['L2 Theoretical Sectors Global Excessive'],8126464)
            for opcode in ('IMMA','I2F','FFMA'):
                self.assertEqual(row['opcodes'][opcode],16777216)
            self.assertEqual(row['opcodes'].get('FMUL',0),0)  # Guarded O3 exponent path.


if __name__=='__main__': unittest.main()
