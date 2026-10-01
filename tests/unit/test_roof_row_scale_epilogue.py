"""Factoring a row scale is an explicit numerical candidate, not integer reduction."""
from fractions import Fraction
from pathlib import Path
import random
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[2]


class RowScaleEpilogueTests(unittest.TestCase):
    def test_distributivity_in_exact_arithmetic_not_bitwise_fp32_claim(self):
        rng=random.Random(20261001)
        for groups in (1,2,3,5,32):
            for _ in range(50):
                a=Fraction(rng.randrange(0,1024),1024)
                pairs=[(rng.randrange(-131072,131073),Fraction(2)**rng.randrange(-20,5))
                       for _ in range(groups)]
                self.assertEqual(sum(p*a*w for p,w in pairs),a*sum(p*w for p,w in pairs))

    def test_gpu_body_keeps_fp32_fma_groups_and_only_scales_rows_at_end(self):
        s=(ROOT/'csrc/sm80/o3_row_scale_epilogue_candidate.cuh').read_text()
        self.assertIn('!DualScale && !Fast && !Magic && RoofTune==6 && Stream && VectorStore',s)
        self.assertIn('auto acc=cute::make_fragment_like<float>(low)',s)
        self.assertNotIn('rows(i)=as[',s)
        start=s.index('auto finish=[&]')
        end=s.index('if constexpr(RoofTune&2)',start)
        body=s[start:end]
        self.assertIn('__fmaf_rn(float(partial),column,acc(vi,mi,full_ni))',body)
        self.assertNotIn('__float_as_uint',body)
        self.assertNotIn('as[',body)
        epi=s.index('// FP32 across-group accumulator remains FP32.')
        self.assertGreater(epi,s.index('for(int stage=0;stage<k/K;++stage)'))
        self.assertIn('__fmul_rn(acc(i),as[blockIdx.y*M+cute::get<0>(coords(i))])',s[epi:])

    def test_guard_is_in_both_host_paths_and_dual_formats_are_rejected(self):
        s=(ROOT/'csrc/sm80/roof_candidates.cuh').read_text()
        self.assertIn('check_roof_row_scale_guard(tune,as,ws,k)',s)
        self.assertIn('check_roof_row_scale_guard(roof_tune,as,ws,k)',(ROOT/'csrc/sm80/o1_o3.cu').read_text())
        self.assertIn('ws.min().item<int>()>=1',s)
        self.assertIn('bound<=std::numeric_limits<float>::max()',s)
        self.assertIn('bound*as.max().item<double>()<=std::numeric_limits<float>::max()',s)
        self.assertIn('!roof_row_scale_epilogue(tune) || !dual',s)
        self.assertIn('case 51:return roof_config_for<41>(dual);',s)
        self.assertIn('case 52:return roof_config_for<42>(dual);',s)
        self.assertLess(s.index('select_row_scale_epilogue_kernel'),s.index('if(roof_grouped_payload(tune)) return'))

    def test_numerical_opt_in_payload_and_three_stage_safety_are_preserved(self):
        for name in ('benchmark_a100_roof_candidates.py','benchmark_a100_roof_trace.py'):
            s=(ROOT/'scripts'/name).read_text()
            self.assertIn('39,40,51,52,53,54,61) for t in args.tunes) and not args.allow_reassociation',s)
            self.assertIn('row-scale epilogue candidates51/52/53/54 are O3 only',s)
        audit=runpy.run_path(str(ROOT/'scripts/audit_a100_o1.py'))
        checks=audit['roof_pipeline_checks']('adangel_sm80_roof_candidateILb0ELb0ELi52EE',
            'cp.async.wait_group 1; cp.async.wait_group 0;',
            'DEPBAR.LE SB0, 0x1; DEPBAR.LE SB0, 0x0;')
        self.assertTrue(checks and all(checks.values()))
        parser=runpy.run_path(str(ROOT/'scripts/analyze_roof_scale_ncu.py'))
        expected={'IMMA':16777216,'I2F':16777216,'FFMA':16777216,'FMUL':524288}
        parser['validate_arithmetic_work'](expected,51,False)
        with self.assertRaises(ValueError):
            parser['validate_arithmetic_work']({**expected,'FMUL':16777216},51,False)


if __name__=='__main__': unittest.main()
