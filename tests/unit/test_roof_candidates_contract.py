"""Source gates only; GPU numerical/ISA/performance acceptance is separate."""
from pathlib import Path
import hashlib
import unittest

ROOT = Path(__file__).resolve().parents[2]


class RoofCandidatesContractTest(unittest.TestCase):
    def test_three_stage_cannot_edit_the_preserved_two_stage_source(self):
        # Audited v8/c971aff source. A future production change must deliberately
        # update this gate AND pass same-binary numerical/ISA/performance tests.
        body=(ROOT/'csrc/sm80/o3_optimized.cuh').read_text()
        self.assertEqual(hashlib.sha256(body.encode()).hexdigest(),
                         'e3d8e0c0f4be4d128cf3aee41fe6ef774e5b279618b50d13b7c070bc1f56706a')
        experiment=(ROOT/'csrc/sm80/o3_pipeline_candidate.cuh').read_text()
        self.assertIn('namespace o3_pipeline_experiment',experiment)
        self.assertNotIn('__global__',experiment)
        self.assertIn('static_assert(Stages==3 && K==128',experiment)
        self.assertIn('static_assert(RoofTune==2 || RoofTune==6)',experiment)
        self.assertIn('__fmaf_rn(float(partial),scale,acc(vi,mi,full_ni))',experiment)
        host=(ROOT/'csrc/sm80/o1_o3.cu').read_text()
        roof=(ROOT/'csrc/sm80/roof_candidates.cuh').read_text()
        separate=(ROOT/'csrc/sm80/roof_pipeline.cu').read_text()
        self.assertNotIn('#include "o3_pipeline_candidate.cuh"',host+roof)
        self.assertIn('#include "o3_pipeline_candidate.cuh"',separate)
        self.assertIn('"csrc/sm80/roof_pipeline.cu"',(ROOT/'setup.py').read_text())
        self.assertIn('select_three_stage_kernel(dual,fast,tune)',roof)
        self.assertNotIn('ROOF_PICK(16)',roof)
        self.assertNotIn('ROOF_PICK(17)',roof)
        self.assertNotIn('ROOF_PICK(18)',roof)
        self.assertNotIn('ROOF_PICK(19)',roof)
        self.assertIn('__launch_bounds__(256,(Tune>=18?2:3))',separate)
        self.assertIn('CoreTune=(Tune==16 || Tune==18)?2:6',separate)

    def test_production_default_unchanged(self):
        host = (ROOT / "csrc/sm80/o1_o3.cu").read_text()
        self.assertIn('implementation="o3_swizzle_64x128_k256_exp_static_stream_bound2_store2";', host)
        self.assertIn('int RoofTune=0,bool ActivationPower2=false,bool PrebiasActivationScale=false,bool AsyncScale=false,bool CombinedScalePanels=false>', (ROOT / "csrc/sm80/o3_optimized.cuh").read_text())

    def test_same_group_math_no_magic(self):
        body = (ROOT / "csrc/sm80/o3_optimized.cuh").read_text()
        candidate = body.split("if constexpr(RoofTune!=0) {", 1)[1].split("} else {\n          o1_static_for<0,NAtoms>", 1)[0]
        self.assertIn("const int partial=pl(vi)+16*ph(vi);", candidate)
        self.assertIn("__fmul_rn(row,column)", candidate)
        self.assertIn("__fmaf_rn(float(partial),scale,acc(vi,mi,full_ni))", candidate)
        self.assertNotIn("__int_as_float", candidate)
        self.assertIn("SM80_16x8x64_S32U4S4S32_TN", candidate)
        self.assertIn("SM80_16x8x64_S32S4S4S32_TN", candidate)

    def test_exact_old_entry_and_guarded_shapes(self):
        text = (ROOT / "csrc/sm80/roof_candidates.cuh").read_text()
        self.assertIn("if(tune==-1 && !dual)", text)
        self.assertIn("const bool existing_dual=tune==-1 && dual;", text)
        self.assertIn("adangel_sm80_split_grouped_major<128,256>", text)
        self.assertIn("m64*k64<=2147483647LL", text)
        self.assertIn("as.stride(0)==1", text)
        self.assertIn("ws.ne(255)", text)

    def test_four_mode_opt_in_preserves_defaults(self):
        host=(ROOT/'csrc/sm80/o1_o3.cu').read_text()
        mixed=(ROOT/'csrc/sm80/mixed_benchmark.cuh').read_text()
        self.assertEqual(host.count('py::arg("roof_tune")=-1'),2)
        self.assertIn('candidate_not_production',host)
        self.assertIn('candidate_not_production',mixed)
        self.assertIn('no silent fallback',host)
        self.assertIn('roof candidate requires O7/O8 group-major 64x128x256',mixed)

    def test_targeted_occupancy_candidates_use_configured_launch(self):
        text=(ROOT/'csrc/sm80/roof_candidates.cuh').read_text()
        self.assertIn('WN=Tune==8?4:2',text)
        self.assertIn('MinBlocks=(Tune==9 || Tune==10 || Tune==16 || Tune==17 || (Tune>=22 && Tune<=23))?3:2',text)
        self.assertIn('CoreTune=(Tune==16 || Tune==18 || Tune==20)?2:(Tune>=11?6:(Tune>=8?2:Tune))',text)
        self.assertIn('K=(Tune==10 || (Tune>=16 && Tune<=19) || (Tune>=22 && Tune<=23))?128:256',text)
        self.assertIn('Stages=((Tune>=16 && Tune<=19) || Tune==23)?3:2',text)
        self.assertIn('dim3(n/cfg.n,m/64),cfg.threads,smem',text)
        self.assertIn('cudaOccupancyMaxActiveBlocksPerMultiprocessor',text)
        for name in ('o1_o3.cu','mixed_benchmark.cuh'):
            self.assertIn('dim3(n/roof_cfg.n,m/64),roof_cfg.threads',
                          (ROOT/'csrc/sm80'/name).read_text())

    def test_two_by_two_warps_reuse_b_without_modifying_controls(self):
        separate=(ROOT/'csrc/sm80/roof_warp_reuse.cu').read_text()
        experiment=(ROOT/'csrc/sm80/o3_warp_reuse_candidate.cuh').read_text()
        host=(ROOT/'csrc/sm80/roof_candidates.cuh').read_text()
        self.assertIn('__launch_bounds__(128,2)',separate)
        self.assertIn('static constexpr int WM=2;',experiment)
        self.assertIn('static_assert(M==64 && WN==2 && K==256)',experiment)
        self.assertNotIn('__global__',experiment)
        self.assertIn('Threads=(Tune>=20 && Tune<=23)?128:128*WN',host)
        self.assertIn('select_warp_reuse_kernel(dual,fast,tune)',host)
        self.assertNotIn('#include "o3_warp_reuse_candidate.cuh"',host)
        self.assertIn('"csrc/sm80/roof_warp_reuse.cu"',(ROOT/'setup.py').read_text())
        # The experiment is copied from the preserved body, with exactly one
        # intentional config change. Enforce all arithmetic/pipeline code stays
        # identical here; source equivalence does not replace GPU acceptance.
        original=(ROOT/'csrc/sm80/o3_optimized.cuh').read_text()
        start='template<int N,bool Cached> struct O3ScaleCodeScratch'
        end='template<int M,int N,int K,bool Fast,bool Cached=false,bool Magic=Fast,int WN=2,bool Merge=false,bool StaticCopy=false,bool PhasePair=false,bool Stream=false>\n__global__'
        original=original[original.index(start):original.index(end)].strip()
        actual=experiment[experiment.index(start):experiment.index('} // namespace o3_warp_reuse_experiment')].strip()
        changed='// Two M warps each reuse B across two M atoms; no cross-warp exchange.\n  static constexpr int WM=2;\n  static_assert(M==64 && WN==2 && K==256);'
        self.assertEqual(actual.replace(changed,'static constexpr int WM=M==32?2:4;'),original)
        self.assertEqual(64*128//128,64)  # FP32 output accumulators/thread.
        # Logical B elements supplied per group: WM duplicate readers. This
        # is a request-count model, not a measured wavefront or latency claim.
        self.assertEqual((2*128*128)/(4*128*128),0.5)

    def test_reuse_k128_candidates_are_isolated_and_preserve_group_math(self):
        experiment=(ROOT/'csrc/sm80/o3_reuse_pipeline_candidate.cuh').read_text()
        separate=(ROOT/'csrc/sm80/roof_reuse_pipeline.cu').read_text()
        roof=(ROOT/'csrc/sm80/roof_candidates.cuh').read_text()
        self.assertIn('__launch_bounds__(128,3)',separate)
        self.assertIn('Stages=Tune==22?2:3',separate)
        self.assertIn('select_reuse_pipeline_kernel(dual,fast,tune)',roof)
        self.assertNotIn('#include "o3_reuse_pipeline_candidate.cuh"',roof)
        self.assertNotIn('ROOF_PICK(22)',roof)
        self.assertNotIn('ROOF_PICK(23)',roof)
        self.assertIn('"csrc/sm80/roof_reuse_pipeline.cu"',(ROOT/'setup.py').read_text())
        old=(ROOT/'csrc/sm80/o3_pipeline_candidate.cuh').read_text()
        old=old[old.index('#pragma once'):].replace('o3_pipeline_experiment','o3_reuse_pipeline_experiment')
        old=old.replace('static constexpr int WM=M==32?2:4;',
                        'static constexpr int WM=2;\n  static_assert(M==64 && WN==2 && K==128);')
        old=old.replace('static_assert(Stages==3 && K==128',
                        'static_assert((Stages==2 || Stages==3) && K==128')
        self.assertEqual(experiment[experiment.index('#pragma once'):].strip(),old.strip())
        # Same formulas for host buffer allocation; driver adds reserved bytes.
        for stages in (2,3):
            for dual in (False,True):
                size=stages*(64*128+128*128//2+128*4+(64*4 if dual else 0))
                self.assertLess(size+1024,164*1024//3)
        self.assertEqual(128//32*3,12)  # Potential, not yet measured residency.

    def test_power2_scale_guard_preserves_i2f_and_fma(self):
        text=(ROOT/'csrc/sm80/roof_candidates.cuh').read_text()
        self.assertIn('at::bitwise_and(abits,0x007fffff).ne(0)',text)
        self.assertIn('amin+wmin-127>=1 && amax+wmax-127<=254',text)
        self.assertIn('activation_power2_guard_fallback',text)
        body=(ROOT/'csrc/sm80/o3_optimized.cuh').read_text()
        self.assertIn('__float_as_uint(column)+__float_as_uint(row)-0x3f800000u',body)
        self.assertIn('__fmaf_rn(float(partial),scale,acc(vi,mi,full_ni))',body)
        self.assertIn('static_assert(!PrebiasActivationScale || ActivationPower2)',body)
        self.assertIn('row_scale=__uint_as_float(__float_as_uint(row_scale)-0x3f800000u)',body)
        self.assertIn('o3_prefetch<M,N,K,Fast,Cached,WN,StaticCopy,VectorScale,DualScale,GroupMajorScale,PrebiasActivationScale,AsyncScale,CombinedScalePanels>',body)

    def test_three_stage_prologue_wait_drain_and_slot_reuse(self):
        body=(ROOT/'csrc/sm80/o3_pipeline_candidate.cuh').read_text()
        self.assertIn('(s,1,1,a,w,ws,m,k,as,n)',body)
        self.assertIn('if(stage+2<k/K) asm volatile("cp.async.wait_group 1;',body)
        self.assertIn('(s,(stage+2)%Stages,stage+2,a,w,ws,m,k,as,n)',body)
        self.assertIn('process_stage(stage,stage%Stages)',body)
        self.assertIn('static_assert(Stages==3 && K==128',body)
        roof=(ROOT/'csrc/sm80/roof_candidates.cuh').read_text()
        self.assertIn('o3_pipeline_experiment::o3_body',(ROOT/'csrc/sm80/roof_pipeline.cu').read_text())
        self.assertIn('if constexpr(R::Stages==3)',roof)
        for total in (1,2,3,4,5,6,7,32):
            slots={};pending=[];completed=set();consumed=[]
            def issue(g):
                slot=g%3
                self.assertTrue(slot not in slots or slots[slot] in consumed)
                slots[slot]=g;pending.append(g)
            for g in range(min(total,2)):
                issue(g)
            for g in range(total):
                leave=1 if g+2<total else 0
                while len(pending)>leave:
                    completed.add(pending.pop(0))
                self.assertIn(g,completed)
                self.assertEqual(slots[g%3],g)
                if g+2<total:
                    issue(g+2)
                consumed.append(g)
            self.assertEqual(consumed,list(range(total)))
            self.assertFalse(pending)
        # Allocated payload+FP32 scale bytes; not an occupancy claim.
        for dual,expected in ((False,50688),(True,51456)):
            self.assertEqual(3*(64*128+128*128//2)+3*128*4+(3*64*4 if dual else 0),expected)

    def test_async_scale_uses_existing_payload_wait_and_group_major_alignment(self):
        body=(ROOT/'csrc/sm80/o3_optimized.cuh').read_text()
        block=body.split('if constexpr(AsyncScale) {',1)[1].split('} else if constexpr(DualScale)',1)[0]
        self.assertIn('copy16(s.activation_scales+',block)
        self.assertIn('copy16(s.scales+',block)
        self.assertNotIn('__fmul',block)
        self.assertIn('!AsyncScale || (DualScale && GroupMajorScale && !PrebiasActivationScale)',body)
        self.assertIn('alignas(16) float activation_scales',body)
        stage=body.split('auto process_stage=',1)[1]
        self.assertLess(stage.index('cp.async.wait_group 0'),stage.index('__syncthreads()'))
        self.assertLess(stage.index('__syncthreads()'),stage.index('auto process_group='))
        for rows in (64,128):
            touched=[]
            for slot in (0,1):
                for group in (0,1):
                    for lane in range(rows//4):
                        first=(slot*2+group)*rows+lane*4
                        self.assertEqual(first*4%16,0)
                        touched.extend(range(first,first+4))
            self.assertEqual(touched,list(range(4*rows)))
        host=(ROOT/'csrc/sm80/roof_candidates.cuh').read_text()
        self.assertIn('return adangel_sm80_roof_candidate<true,false,14>',host)
        self.assertIn('if(tune==11 || tune==12) fast=roof_power2_activation_guard',host)

    def test_combined_scale_panel_copy_preserves_two_distinct_groups(self):
        body=(ROOT/'csrc/sm80/o3_optimized.cuh').read_text()
        self.assertIn('unsigned group=off/M,row=off%M;',body)
        self.assertIn('unsigned group=off/N,col=off%N;',body)
        self.assertIn('(M*C::Groups)%128==0 && (N*C::Groups)%128==0',body)
        for rows in (64,128):
            for slot in (0,1):
                for stage in (0,1,2,15):
                    srcdst=[]
                    # One or two completely active warps, each lane copies4 floats.
                    self.assertEqual((rows*2//4)%32,0)
                    for lane in range(rows*2//4):
                        off=lane*4;group,row=divmod(off,rows)
                        for j in range(4):
                            srcdst.append((slot*2*rows+off+j,(stage*2+group)*4096+rows*3+row+j))
                    expected=[((slot*2+group)*rows+row,(stage*2+group)*4096+rows*3+row)
                              for group in (0,1) for row in range(rows)]
                    self.assertEqual(srcdst,expected)

    def test_o3_group_major_scale_is_timed_in_weight_conversion(self):
        host=(ROOT/'csrc/sm80/o1_o3.cu').read_text()
        cvw=host.split('auto cvw=[&](){',1)[1].split('auto cva=',1)[0]
        self.assertIn('roof_reorder_o3_scale(ws,roof_ws,n,k/128,stream)',cvw)
        self.assertIn('auto roof_ws=roof_tune==13 ? at::empty({k/128,n},ws.options()) : ws',host)
        self.assertIn('weight_scale_reorder_in_conversion',host)
        body=(ROOT/'csrc/sm80/o3_optimized.cuh').read_text()
        self.assertIn('(stage*C::Groups+group)*total_n+blockIdx.x*N+threadIdx.x',body)
        self.assertIn('weight_scale_reorder_bytes',(ROOT/'scripts/benchmark_a100_roof_trace.py').read_text())

    def test_o3_scale_permutation_indexing(self):
        for n,groups in ((128,2),(256,4),(128,6),(4096,32)):
            source=[(row*17+group*31)%255 for row in range(n) for group in range(groups)]
            actual=[source[(i%n)*groups+i//n] for i in range(n*groups)]
            expected=[source[row*groups+group] for group in range(groups) for row in range(n)]
            self.assertEqual(actual,expected)


if __name__ == "__main__":
    unittest.main()
