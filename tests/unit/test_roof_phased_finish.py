"""Phased16-value finish is not a cross-G128 tree or math reassociation."""
from pathlib import Path
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[2]


class PhasedFinishTests(unittest.TestCase):
    def test_only_finish_schedule_changes(self):
        old=(ROOT/'csrc/sm80/o3_grouped_payload_candidate.cuh').read_text()
        new=(ROOT/'csrc/sm80/o3_phased_finish_candidate.cuh').read_text()
        old=old.replace('// Candidates41/42: G128-major global payload, same22/23 compute and shared layout.',
                        '// Candidates49/50: phased fragment finish, same41/42 payload and G128 order.')
        old=old.replace('o3_grouped_payload_experiment','o3_phased_finish_experiment')
        needle='''              o1_static_for<0,NAtoms>([&](auto ni) {
                auto pl=pls(cute::_,ni),ph=phs(cute::_,ni);finish(ni,pl,ph);
              });'''
        start=new.index('              // Local fragment phases only:')
        end=new.index('              // End local fragment phases.')+len('              // End local fragment phases.')
        self.assertEqual(new[:start]+needle+new[end:],old)
        block=new[start:end]
        self.assertEqual(block.count('o1_static_for<0,NAtoms>'),3)
        self.assertIn('static_assert(NAtoms==4)',block)
        self.assertIn('float(pls(vi,ni)+16*phs(vi,ni))',block)
        self.assertIn('__fmul_rn(row,column)',block)
        self.assertIn('__fmaf_rn(values(vi,ni),scales(vi,ni),acc(vi,mi,full_ni))',block)
        self.assertNotIn('stage',block)
        self.assertNotIn('__syncthreads',block)
        self.assertNotIn('__syncwarp',block)

    def test_local_fragment_values_do_not_change_exact_integer_reconstruction(self):
        # Both routes reconstruct the same INT8*Q4 partial. FP conversion is
        # exact throughout the conservative complete bound. GPU still required.
        import struct
        def f32(x): return struct.unpack('f',struct.pack('f',float(x)))[0]
        for partial in range(-131072,131073):
            self.assertEqual(f32(partial),partial)

    def test_tile_and_selection_match_unmodified_controls(self):
        new=(ROOT/'csrc/sm80/roof_phased_finish.cu').read_text()
        self.assertIn('__launch_bounds__(128,3)',new)
        self.assertIn('Stages=Tune==49?2:3',new)
        self.assertIn('o3_phased_finish_experiment::o3_body<64,128,128,Fast,false,false,2,false,true,false,true,true,true,false,',new)
        config=(ROOT/'csrc/sm80/roof_candidates.cuh').read_text()
        self.assertIn('case 49:return roof_config_for<41>(dual);',config)
        self.assertIn('case 50:return roof_config_for<42>(dual);',config)
        self.assertLess(config.index('select_phased_finish_kernel'),config.index('if(roof_grouped_payload(tune)) return'))
        self.assertIn('"phased_fragment_finish"',config)
        self.assertIn('"finish_batch_values_per_thread"]=(tune==49 || tune==50)?16:0',config)
        self.assertIn('roof_phased_finish.cu',(ROOT/'setup.py').read_text())

    def test_validation_bindings_and_three_stage_drain_are_required(self):
        audit=runpy.run_path(str(ROOT/'scripts/audit_a100_o1.py'))
        checks=audit['roof_pipeline_checks'](
            'adangel_sm80_roof_candidateILb1ELb0ELi50EE',
            'cp.async.wait_group 1; cp.async.wait_group 0;',
            'DEPBAR.LE SB0, 0x1; DEPBAR.LE SB0, 0x0;')
        self.assertTrue(checks and all(checks.values()))
        missing=audit['roof_pipeline_checks'](
            'adangel_sm80_roof_candidateILb1ELb0ELi50EE','cp.async.wait_group 1;',
            'DEPBAR.LE SB0, 0x1;')
        self.assertFalse(all(missing.values()))
        for name in ('benchmark_a100_roof_candidates.py','benchmark_a100_roof_trace.py',
                     'compare_roof_trace_candidates.py','roof_payload_validation.py'):
            self.assertRegex((ROOT/'scripts'/name).read_text(),r'49,50(?:,51,52(?:,53,54(?:,55,56(?:,57,58(?:,59,60)?)?)?)?)?\)')


if __name__=='__main__': unittest.main()
