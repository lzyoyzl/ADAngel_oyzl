from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_roof_interleaved_merge_codegen import generated_header, HEADERS, START, END


class InterleavedMergeTests(unittest.TestCase):
    def test_control_unchanged(self):
        for name in HEADERS.values():
            text = (ROOT / "csrc/sm80" / name).read_text()
            self.assertEqual(generated_header(text, 0), text)

    def test_only_partial_path_changes(self):
        for name in HEADERS.values():
            text = (ROOT / "csrc/sm80" / name).read_text()
            candidate = generated_header(text, 1)
            branch = candidate[candidate.index(START):candidate.index(END)]
            self.assertNotIn("phs", branch)
            self.assertEqual(branch.count("cute::gemm(HA{}"), 2)
            self.assertEqual(branch.count("cute::gemm(LA{}"), 2)
            self.assertIn("pls(vi,ni)*=16", branch)
            self.assertIn("const int partial=pl(vi);", candidate)
            for marker in ("cp.async.commit_group", "__syncthreads", "__fmaf_rn", "__fmul_rn"):
                self.assertEqual(candidate.count(marker), text.count(marker))

    def test_reject_drift(self):
        with self.assertRaises(ValueError):
            generated_header("missing branch", 1)
        with self.assertRaises(ValueError):
            generated_header("anything", 2)

    def test_all_int8_int4_pairs_and_intermediate_bounds(self):
        for a in range(-128, 128):
            lo = a & 15
            hi = a // 16
            for w in range(-8, 8):
                self.assertEqual((hi*w)*16 + lo*w, a*w)
                # Conservative bounds cover every 128-element dot and prefix.
                self.assertLessEqual(abs(hi*w)*128*16 + abs(lo*w)*128, 253952)
        self.assertLess(253952, 2**31)

    def test_both_policies_use_m64_launcher(self):
        import benchmark_roof_interleaved_merge_probe as probe
        class FakeDriver:
            def __init__(self, library, cubins, variant, smem):
                self.path=cubins[0]
                self.resources={0:dict(m128=0,cta_tile=[64,128,128])}
                self.closed=False
            def run(self, index, *args):
                self_index=index
                return self.path,self_index,args
            def close(self): self.closed=True
        with patch.object(probe,'M128Driver',FakeDriver):
            driver=probe.Driver('library',{0:'control',1:'candidate'},'o7',34304)
            self.assertEqual(driver.run(1,'inputs'),('candidate',0,('inputs',)))
            self.assertEqual(driver.resources[1]['cta_tile'],[64,128,128])
            self.assertEqual(driver.resources[1]['partial_registers_per_four_n_atoms'],16)
            driver.close()
            self.assertEqual(driver.drivers,{})


class InterleavedMergeEvidenceTests(unittest.TestCase):
    root=ROOT/'docs/evidence/a100_o378_roof_v63'

    def test_same_control_and_native_isa(self):
        from compare_a100_codegen import compare
        from probe_roof_fullk_integer_codegen import static_entries
        directory=self.root/'reports/o378_roof_v63'
        result=json.loads((directory/'codegen.json').read_text())
        previous=ROOT/'docs/evidence/a100_o378_roof_v57/reports/o378_roof_v57/m128_64.sass'
        self.assertTrue(compare(previous.read_text(),(directory/'interleaved_merge_0.sass').read_text(),
                                r'^adangel_roof_m128_(?:o3|o78)$')['passed'])
        for policy in (0,1):
            text=(directory/f'interleaved_merge_{policy}.sass').read_text()
            entries=static_entries(text,r'^adangel_roof_m128_(?:o3|o78)$',
                                   {'adangel_roof_m128_o3','adangel_roof_m128_o78'})
            self.assertEqual(entries,result['variants'][str(policy)]['entries'])
            for info in entries.values():
                self.assertTrue(info['native_u4_s4'] and info['native_s4_s4'] and info['all_copies_bypass_l1'])
                self.assertFalse(info['int8_mma'])
                self.assertEqual(info['opcode_counts']['IMMA'],64)
                self.assertEqual(info['opcode_counts']['LDSM'],16)
            for family,name in HEADERS.items():
                source=(ROOT/'csrc/sm80'/name).read_text()
                emitted=(directory/f'policy_{policy}/{family}_m128_generated.cuh').read_text()
                self.assertEqual(emitted,generated_header(source,policy))

    def test_raw_pairs_mse_and_cv(self):
        from benchmark_roof_interleaved_merge_probe import summary
        from benchmark_a100_o1 import stats
        directory=self.root/'runs/o378_roof_v63_screen'
        rows=[json.loads(s) for s in (directory/'results.jsonl').read_text().splitlines()]
        saved=json.loads((directory/'summary.json').read_text())
        self.assertEqual(len(rows),72)
        self.assertEqual(summary(rows),saved['records'])
        self.assertFalse(saved['production_default_changed'])
        self.assertEqual({r['sample_id'] for r in rows},{f'layer_00_{p}_proj' for p in ('q','k','v','o')})
        for row in rows:
            self.assertEqual(len(row['raw_ms']),200)
            for key,value in stats(row['raw_ms']).items():
                # Python/NumPy builds differ at ~1e-16 in the CPU CV reduction.
                self.assertAlmostEqual(value,row['summary'][key],places=12)
            self.assertTrue(row['bitwise_equal_current_best'])
            self.assertEqual(row['mse_vs_current_best'],0)
            self.assertEqual(row['probe_resources']['cta_tile'],[64,128,128])
            self.assertEqual(row['probe_resources']['registers_per_thread'],168)
            self.assertEqual(row['probe_resources']['active_blocks_per_sm'],3)
        for bad in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(ValueError): summary(bad)

    def test_scoped_safety_and_binary_provenance(self):
        directory=self.root/'reports/o378_roof_v63'
        codegen=json.loads((directory/'codegen.json').read_text())
        for name in ('preflight','memcheck','synccheck','racecheck','screen'):
            run=self.root/f'runs/o378_roof_v63_{name}'
            check=json.loads((run/'validation.json').read_text())
            self.assertTrue(check['passed']);self.assertEqual(len(check['checks']),96)
            for row in check['checks']:
                self.assertTrue(row['bitwise_equal_best'] and row['finite_fp32'])
            env=json.loads((run/'environment.json').read_text())
            self.assertEqual(env['extension_sha256'],'94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462')
            for policy in (0,1):
                self.assertEqual(env['cubins'][str(policy)],codegen['variants'][str(policy)]['cubin_sha256'])
            if name in ('memcheck','synccheck','racecheck'):
                text=(directory/f'{name}.log').read_text()
                self.assertIn('0 errors',text)
                if name=='racecheck':self.assertIn('0 warnings',text)


if __name__ == "__main__": unittest.main()
