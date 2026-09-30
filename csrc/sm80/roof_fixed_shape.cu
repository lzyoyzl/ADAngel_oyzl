// Candidates32/33: fixed4096 dimensions, otherwise exactly22/23's device body.
// Host entry points reject other shapes. No production default is changed.
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
  static_assert(Tune==32 || Tune==33);
  static_assert(!DualScale || !Fast);
  constexpr int Stages=Tune==32?2:3;
  // Constant dimensions permit strength reduction of address calculations.
  // Do not assume less SASS or faster execution until measured and audited.
  o3_reuse_pipeline_experiment::o3_body<64,128,128,Fast,false,false,2,false,true,false,true,true,true,false,
      DualScale,DualScale,6,false,false,false,false,Stages>(a,w,as,ws,y,4096,4096,4096);
}
}

namespace adangel_sm80_experiment {
Kernel select_fixed_shape_kernel(bool dual,bool fast,int tune) {
  if(dual && fast) throw std::invalid_argument("fixed shape uses exact dual scale multiplication");
  if(tune==32) return dual ? adangel_sm80_roof_candidate<true,false,32> :
      (fast ? adangel_sm80_roof_candidate<false,true,32> : adangel_sm80_roof_candidate<false,false,32>);
  if(tune==33) return dual ? adangel_sm80_roof_candidate<true,false,33> :
      (fast ? adangel_sm80_roof_candidate<false,true,33> : adangel_sm80_roof_candidate<false,false,33>);
  throw std::invalid_argument("expected fixed4096 candidate32/33");
}
}
