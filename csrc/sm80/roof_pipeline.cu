// Independently compiled three-stage candidates. No production default lives
// here; keeping the experimental device IR out of o1_o3.cu is intentional.
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cutlass/integer_subbyte.h>
#include <stdexcept>
#include "roof_pipeline_api.h"

namespace {
__device__ void copy16(void* dst, const void* src) {
  const uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
      "r"(address), "l"(src) : "memory");
}
template<int I,int End,class F>
__device__ __forceinline__ void o1_static_for(F const& f) {
  if constexpr(I<End) {f(cute::Int<I>{});o1_static_for<I+1,End>(f);}
}
#include "o3_pipeline_candidate.cuh"

template<bool DualScale,bool Fast,int Tune>
__global__ __launch_bounds__(256,(Tune>=18?2:3))
void adangel_sm80_roof_candidate(
    const uint8_t* a,const uint8_t* w,const float* as,const uint8_t* ws,
    float* y,int m,int n,int k) {
  static_assert(Tune>=16 && Tune<=19);
  static_assert(!DualScale || !Fast);
  // 18/19 change only the launch-bound register budget versus 16/17.
  // Actual registers/residency still require cudaFuncGetAttributes + NCU.
  constexpr int CoreTune=(Tune==16 || Tune==18)?2:6;
  o3_pipeline_experiment::o3_body<64,128,128,Fast,false,false,2,false,true,false,true,true,true,false,
          DualScale,DualScale,CoreTune,false,false,false,false,3>(a,w,as,ws,y,m,n,k);
}
} // namespace

namespace adangel_sm80_experiment {
Kernel select_three_stage_kernel(bool dual,bool fast,int tune) {
  if(dual && fast) throw std::invalid_argument("three-stage dual scales use exact FP32 multiplication");
  if(tune==16) return dual ? adangel_sm80_roof_candidate<true,false,16> :
      (fast ? adangel_sm80_roof_candidate<false,true,16> : adangel_sm80_roof_candidate<false,false,16>);
  if(tune==17) return dual ? adangel_sm80_roof_candidate<true,false,17> :
      (fast ? adangel_sm80_roof_candidate<false,true,17> : adangel_sm80_roof_candidate<false,false,17>);
  if(tune==18) return dual ? adangel_sm80_roof_candidate<true,false,18> :
      (fast ? adangel_sm80_roof_candidate<false,true,18> : adangel_sm80_roof_candidate<false,false,18>);
  if(tune==19) return dual ? adangel_sm80_roof_candidate<true,false,19> :
      (fast ? adangel_sm80_roof_candidate<false,true,19> : adangel_sm80_roof_candidate<false,false,19>);
  throw std::invalid_argument("expected three-stage candidate16..19");
}
size_t three_stage_shared_bytes(bool dual) {
  using namespace o3_pipeline_experiment;
  return dual ? sizeof(typename O3AmpereConfig<64,128,128,false,2,true,3>::Storage)
              : sizeof(typename O3AmpereConfig<64,128,128,false,2,false,3>::Storage);
}
} // namespace adangel_sm80_experiment
