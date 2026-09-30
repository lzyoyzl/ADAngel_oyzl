"""CPU order/source checks; does not prove compiler overlap or GPU performance."""
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]


class FinishPipelineContract(unittest.TestCase):
    def test_only_independent_atom_finish_order_changes(self):
        old=(ROOT/'csrc/sm80/o3_reuse_pipeline_candidate.cuh').read_text()
        new=(ROOT/'csrc/sm80/o3_finish_pipeline_candidate.cuh').read_text()
        new='\n'.join(new.splitlines()[2:])+'\n'
        new=new.replace('o3_finish_pipeline_experiment','o3_reuse_pipeline_experiment')
        old_start=old.index('              o1_static_for<0,NAtoms>([&](auto ni) {\n'
                            '                auto pl=pls(cute::_,ni),ph=phs(cute::_,ni);\n'
                            '                cute::gemm(LA{},pl,ra1')
        new_start=new.index('              // Complete this atom')
        end='            } else {\n              o1_static_for<0,NAtoms>'
        self.assertEqual(old[:old_start],new[:new_start])
        self.assertEqual(old[old.index(end,old_start):].strip(),new[new.index(end,new_start):].strip())
        changed=new[new_start:new.index(end,new_start)]
        self.assertIn('if constexpr(decltype(ni)::value>0)',changed)
        self.assertIn('finish(previous,previous_low,previous_high)',changed)
        self.assertIn('finish(last,last_low,last_high)',changed)
        self.assertNotIn('make_tensor',changed)  # no extra partial/accumulator bank

    def test_atom_completion_and_per_output_group_order(self):
        for atoms in (1,2,4,8):
            for groups in (1,2,3,5,32):
                actual={n:[] for n in range(atoms)}
                for g in range(groups):
                    ready={n:{(0,'low'),(0,'high')} for n in range(atoms)}
                    finished=[]
                    def finish(n):
                        self.assertEqual(ready[n],{(h,path) for h in (0,1) for path in ('low','high')})
                        self.assertNotIn(n,finished)
                        finished.append(n);actual[n].append(g)
                    for n in range(atoms):
                        ready[n].update(((1,'low'),(1,'high')))
                        if n: finish(n-1)
                    finish(atoms-1)
                    self.assertEqual(finished,list(range(atoms)))
                self.assertEqual(actual,{n:list(range(groups)) for n in range(atoms)})

    def test_isolated_launch_and_exact_numerical_gate(self):
        separate=(ROOT/'csrc/sm80/roof_finish_pipeline.cu').read_text()
        host=(ROOT/'csrc/sm80/roof_candidates.cuh').read_text()
        self.assertIn('__launch_bounds__(128,3)',separate)
        self.assertIn('Stages=Tune==30?2:3',separate)
        self.assertIn('select_finish_pipeline_kernel(dual,fast,tune)',host)
        self.assertNotIn('#include "o3_finish_pipeline_candidate.cuh"',host)
        self.assertNotIn('ROOF_PICK(30)',host)
        self.assertNotIn('ROOF_PICK(31)',host)
        self.assertIn('meta["fp32_reassociated"]=tune>=24 && tune<=27;',host)
        self.assertIn('"csrc/sm80/roof_finish_pipeline.cu"',(ROOT/'setup.py').read_text())
        for name in ('o1_o3.cu','mixed_benchmark.cuh'):
            source=(ROOT/'csrc/sm80'/name).read_text()
            self.assertIn('meta["interleaved_mma_finish"]=roof_tune==30 || roof_tune==31',source)
            self.assertIn('meta["group_accumulation"]=(roof_tune>=24 && roof_tune<=27)',source)


if __name__=='__main__': unittest.main()
