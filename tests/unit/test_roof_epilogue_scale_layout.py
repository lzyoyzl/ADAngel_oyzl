"""Layout-only extension of the guarded row-scale epilogue candidates."""
from pathlib import Path
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[2]


class EpilogueScaleLayoutTests(unittest.TestCase):
    def test_group_major_address_mapping_including_single_and_odd_groups(self):
        for n in (128,256):
            for groups in (1,2,3,5,32):
                natural=[(row*7+group*11)%255 for row in range(n) for group in range(groups)]
                reordered=[natural[(i%n)*groups+i//n] for i in range(n*groups)]
                for row in range(n):
                    for group in range(groups):
                        self.assertEqual(reordered[group*n+row],natural[row*groups+group])

    def test_device_reuses_exact_body_changing_only_scale_layout(self):
        old=(ROOT/'csrc/sm80/roof_row_scale_epilogue.cu').read_text()
        new=(ROOT/'csrc/sm80/roof_epilogue_scale_layout.cu').read_text()
        for s in (old,new):
            self.assertIn('#include "o3_row_scale_epilogue_candidate.cuh"',s)
            self.assertIn('__launch_bounds__(128,3)',s)
            self.assertIn('o3_body<64,128,128,false,false,false,2,false,true,false,true,true,true,false,',s)
        self.assertIn('false,false,6,false,false,false,false,Stages>',old)
        self.assertIn('false,true,6,false,false,false,false,Stages>',new)
        self.assertIn('!DualScale && !Fast && (Tune==53 || Tune==54)',new)
        self.assertIn('constexpr int Stages=Tune==53?2:3',new)

    def test_both_host_paths_pack_scales_and_publish_correct_costs(self):
        host=(ROOT/'csrc/sm80/o1_o3.cu').read_text()
        core=(ROOT/'csrc/sm80/roof_candidates.cuh').read_text()
        self.assertIn('return tune==13 || tune==53 || tune==54',core)
        self.assertIn('case 53:return roof_config_for<41>(dual);',core)
        self.assertIn('case 54:return roof_config_for<42>(dual);',core)
        self.assertIn('roof_ws=roof_group_major_w_scale(roof_tune) ? at::empty',host)
        self.assertIn('if(roof_group_major_w_scale(roof_tune)) roof_reorder_o3_scale',host)
        self.assertIn('if(roof_group_major_w_scale(tune)) roof_reorder_o3_scale',core)
        self.assertIn('meta["weight_scale_reorder_bytes"]=roof_group_major_w_scale(roof_tune) ? int64_t(2)*n*(k/128)',host)
        self.assertIn('meta["weight_scale_reorder_bytes"]=roof_group_major_w_scale(tune) ? int64_t(2)*n*g',core)
        self.assertIn('meta["conversion_kernels_per_operand"]=py::none()',core)
        self.assertIn('meta["weight_conversion_kernels"]=3',core)
        self.assertIn('bool roof_row_scale_epilogue(int tune) {return (tune>=51 && tune<=54) || tune==61;}',core)

    def test_bitwise_reference_is_explicit_and_outside_timing(self):
        module=runpy.run_path(str(ROOT/'scripts/roof_reduction_validation.py'))
        calls=[]
        class Native:
            def _benchmark_roof_candidate(self,*args):
                calls.append(args)
                return {'output':'reference'}
        f=module['row_scale_layout_reference']
        self.assertIsNone(f(Native(),'o3',51,('a','as','w','ws')))
        self.assertEqual(f(Native(),'o3',53,('a','as','w','ws')),'reference')
        self.assertEqual(f(Native(),'o3',54,('a','as','w','ws')),'reference')
        self.assertEqual(calls,[('o3',51,'a','as','w','ws',0,1),('o3',52,'a','as','w','ws',0,1)])
        for name in ('benchmark_a100_roof_candidates.py','benchmark_a100_roof_trace.py'):
            source=(ROOT/'scripts'/name).read_text()
            self.assertIn('row_scale_baseline=row_scale_layout_reference(native,variant,tune,values)',source)
        payload=(ROOT/'scripts/roof_payload_validation.py').read_text()
        self.assertIn('torch.equal(actual_ws,natural_ws.T.contiguous())',payload)
        compare=(ROOT/'scripts/compare_roof_trace_candidates.py').read_text()
        self.assertIn("not r.get('bitwise_equal_row_scale_baseline')",compare)

    def test_pipeline_and_ncu_work_contract(self):
        audit=runpy.run_path(str(ROOT/'scripts/audit_a100_o1.py'))
        checks=audit['roof_pipeline_checks']('adangel_sm80_roof_candidateILb0ELb0ELi54EE',
            'cp.async.wait_group 1; cp.async.wait_group 0;',
            'DEPBAR.LE SB0, 0x1; DEPBAR.LE SB0, 0x0;')
        self.assertTrue(checks and all(checks.values()))
        parser=runpy.run_path(str(ROOT/'scripts/analyze_roof_scale_ncu.py'))
        for tune in (53,54):
            parser['validate_arithmetic_work']({'IMMA':16777216,'I2F':16777216,
                'FFMA':16777216,'FMUL':524288},tune,False)


if __name__=='__main__': unittest.main()
