"""Static slot scheduling contracts; GPU audit and numerical tests remain required."""
from pathlib import Path
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[2]


class StaticRingTests(unittest.TestCase):
    def test_arithmetic_payload_and_barriers_are_preserved(self):
        old=(ROOT/'csrc/sm80/o3_grouped_payload_candidate.cuh').read_text()
        new=(ROOT/'csrc/sm80/o3_static_ring_candidate.cuh').read_text()
        old=old.replace('// Candidates41/42: G128-major global payload, same22/23 compute and shared layout.',
                        '// Candidates47/48: static ring slots, same41/42 group arithmetic and payload.')
        old=old.replace('o3_grouped_payload_experiment','o3_static_ring_experiment')
        old=old.replace('(s,(stage+2)%Stages,stage+2,a,w,ws,m,k,as,n);',
                        '(s,(decltype(slot)::value+2)%Stages,stage+2,a,w,ws,m,k,as,n);')
        old_start=old.index('  if constexpr(PhasePair) {',old.index('auto process_stage='))
        new_start=new.index('  // Static physical ring phases')
        self.assertEqual(old[:old_start],new[:new_start])
        old_tail=old.index('  if constexpr(VectorStore) {',old_start)
        new_tail=new.index('  if constexpr(VectorStore) {',new_start)
        self.assertEqual(old[old_tail:],new[new_tail:])
        loop=new[new_start:new_tail]
        self.assertIn('first+=Stages',loop)
        self.assertIn('o1_static_for<0,Stages>',loop)
        self.assertIn('if(stage<k/K) process_stage(stage,phase);',loop)
        self.assertNotIn('stage%',loop)

    def test_ring_trace_all_short_and_odd_lengths(self):
        for stages in (2,3):
            for groups in range(1,66):
                owner={i:i for i in range(min(stages-1,groups))}
                copies=list(owner.values());consumed=[]
                for first in range(0,groups,stages):
                    for phase in range(stages):
                        group=first+phase
                        if group>=groups:continue
                        self.assertEqual(owner[phase],group)
                        consumed.append(group)
                        future=group+stages-1
                        if future<groups:
                            target=(phase+stages-1)%stages
                            self.assertEqual(target,future%stages)
                            if target in owner:self.assertIn(owner[target],consumed)
                            owner[target]=future;copies.append(future)
                self.assertEqual(consumed,list(range(groups)))
                self.assertEqual(copies,list(range(groups)))

    def test_tile_and_bound_match_existing_control(self):
        old=(ROOT/'csrc/sm80/roof_grouped_payload.cu').read_text()
        new=(ROOT/'csrc/sm80/roof_static_ring.cu').read_text()
        key='::o3_body<'
        call=lambda text:text[text.index(key):text.index(';',text.index(key))]
        self.assertEqual(call(old),call(new))
        self.assertIn('__launch_bounds__(128,3)',new)
        cfg=(ROOT/'csrc/sm80/roof_candidates.cuh').read_text()
        self.assertIn('case 47:return roof_config_for<41>(dual);',cfg)
        self.assertIn('case 48:return roof_config_for<42>(dual);',cfg)
        route=cfg[cfg.index('auto select_roof_kernel'):]
        self.assertLess(route.index('select_static_ring_kernel'),route.index('select_grouped_payload_kernel'))
        self.assertIn('roof_tune<=56',(ROOT/'csrc/sm80/mixed_benchmark.cuh').read_text())

    def test_validation_and_three_stage_drain(self):
        checks=runpy.run_path(str(ROOT/'scripts/audit_a100_o1.py'))['roof_pipeline_checks'](
            'adangel_sm80_roof_candidateILb1ELb0ELi48EE',
            'cp.async.wait_group 1; cp.async.wait_group 0;',
            'DEPBAR.LE SB0, 0x1; DEPBAR.LE SB0, 0x0;')
        self.assertTrue(checks and all(checks.values()))
        for name in ('benchmark_a100_roof_candidates.py','benchmark_a100_roof_trace.py',
                     'compare_roof_trace_candidates.py','roof_payload_validation.py'):
            self.assertRegex((ROOT/'scripts'/name).read_text(),r'47,48,49,50(?:,51,52(?:,53,54(?:,55,56(?:,57,58(?:,59,60)?)?)?)?)?\)')
        self.assertIn('"csrc/sm80/roof_static_ring.cu"',(ROOT/'setup.py').read_text())


if __name__=='__main__': unittest.main()
