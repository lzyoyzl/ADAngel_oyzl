"""N64 candidate source/metadata contracts, not GPU performance acceptance."""
from pathlib import Path
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[2]


class NarrowPayloadTests(unittest.TestCase):
    def test_reuses_same_body_without_modifying_old_instantiations(self):
        new=(ROOT/'csrc/sm80/roof_narrow_payload.cu').read_text()
        old=(ROOT/'csrc/sm80/roof_grouped_payload.cu').read_text()
        self.assertIn('__launch_bounds__(128,4)',new)
        self.assertIn('__launch_bounds__(128,3)',old)
        self.assertIn('#include "o3_grouped_payload_candidate.cuh"',new)
        key='o3_grouped_payload_experiment::o3_body<'
        call=lambda text:text[text.index(key):text.index(';',text.index(key))]
        self.assertEqual(call(new).replace('<64,64,128,','<64,128,128,'),call(old))
        self.assertNotIn('mma.sync',new)
        self.assertNotIn('__global__',(ROOT/'csrc/sm80/roof_narrow_payload_api.h').read_text())

    def test_explicit_host_config_and_select_before_old_payload_route(self):
        text=(ROOT/'csrc/sm80/roof_candidates.cuh').read_text()
        self.assertIn('case 45:case 46:return {64,128,128,4,6,64,tune==45?2:3,',text)
        self.assertIn('narrow_payload_shared_bytes(dual,tune)',text)
        route=text[text.index('auto select_roof_kernel'):]
        self.assertLess(route.index('select_narrow_payload_kernel'),route.index('select_grouped_payload_kernel'))
        self.assertNotIn('ROOF_PICK(45)',text)
        self.assertIn('bool roof_fused_payload(int tune) {return tune==43 || tune==44;}',text)

    def test_register_occupancy_hypothesis_is_not_an_acceptance_claim(self):
        self.assertEqual(64*64//128,32)
        for stages in (2,3):
            for dual in (False,True):
                smem=stages*(64*64*3+64*4+(64*4 if dual else 0))
                self.assertLess(4*(smem+1024),164*1024)
        self.assertEqual(128*128*4,65536)
        setup=(ROOT/'setup.py').read_text()
        self.assertGreater(setup.index('"csrc/sm80/roof_narrow_payload.cu"'),setup.index('if target == "sm80":'))

    def test_three_stage_native_audit_includes46(self):
        check=runpy.run_path(str(ROOT/'scripts/audit_a100_o1.py'))['roof_pipeline_checks']
        checks=check('adangel_sm80_roof_candidateILb1ELb0ELi46EE',
            'cp.async.wait_group 1; cp.async.wait_group 0;',
            'DEPBAR.LE SB0, 0x1; DEPBAR.LE SB0, 0x0;')
        self.assertTrue(checks and all(checks.values()))
        for name in ('benchmark_a100_roof_candidates.py','benchmark_a100_roof_trace.py'):
            self.assertIn('41,42,', (ROOT/'scripts'/name).read_text())
            self.assertIn('45,46,47,48,49,50)', (ROOT/'scripts'/name).read_text())
        comparison=(ROOT/'scripts/compare_roof_trace_candidates.py').read_text()
        self.assertIn("in (41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56) and not r.get('payload_layout_bitwise_verified')",comparison)


if __name__=='__main__': unittest.main()
