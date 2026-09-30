// Isolated K128 / 2x2-warp experiment, not a production default.
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
__device__ void copy16(void* dst,const void* src) {
  const uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
      "r"(address), "l"(src) : "memory");
}
template<int I,int End,class F>
__device__ __forceinline__ void o1_static_for(F const& f) {
  if constexpr(I<End) {f(cute::Int<I>{});o1_static_for<I+1,End>(f);}
}
#include "o3_reuse_pipeline_candidate.cuh"

template<bool DualScale,bool Fast,int Tune>
__global__ __launch_bounds__(128,3)
void adangel_sm80_roof_candidate(
    const uint8_t* a,const uint8_t* w,const float* as,const uint8_t* ws,
    float* y,int m,int n,int k) {
  static_assert(Tune==22 || Tune==23);
  static_assert(!DualScale || !Fast);
  constexpr int Stages=Tune==22?2:3;
  // Each warp owns M32 (64 FP32 accumulators/thread). K128 reduces shared
  // allocation; bound3 limits registers for potential 12 warps/SM, compared
  // with candidate21's 8 warps. Spills and actual residency must be measured.
  // N64 stream and ascending G128 arithmetic remain identical.
  o3_reuse_pipeline_experiment::o3_body<64,128,128,Fast,false,false,2,false,true,false,true,true,true,false,
      DualScale,DualScale,6,false,false,false,false,Stages>(a,w,as,ws,y,m,n,k);
}
}

namespace adangel_sm80_experiment {
Kernel select_reuse_pipeline_kernel(bool dual,bool fast,int tune) {
  if(dual && fast) throw std::invalid_argument("reuse pipeline uses exact FP32 dual scale multiplication");
  if(tune==22) return dual ? adangel_sm80_roof_candidate<true,false,22> :
      (fast ? adangel_sm80_roof_candidate<false,true,22> : adangel_sm80_roof_candidate<false,false,22>);
  if(tune==23) return dual ? adangel_sm80_roof_candidate<true,false,23> :
      (fast ? adangel_sm80_roof_candidate<false,true,23> : adangel_sm80_roof_candidate<false,false,23>);
  throw std::invalid_argument("expected reuse-pipeline candidate22/23");
}
}
