// Candidate29: coalesced pair copies, three physical G128 slots, exact math.
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
#include "o3_paired_pipeline_candidate.cuh"

template<bool DualScale,bool Fast,int Tune>
__global__ __launch_bounds__(128,3)
void adangel_sm80_roof_candidate(
    const uint8_t* a,const uint8_t* w,const float* as,const uint8_t* ws,
    float* y,int m,int n,int k) {
  static_assert(Tune==29 && (!DualScale || !Fast));
  o3_paired_pipeline_experiment::o3_body<64,128,128,Fast,false,false,2,false,true,false,true,true,true,false,
      DualScale,DualScale,6,false,false,false,false,3>(a,w,as,ws,y,m,n,k);
}
}

namespace adangel_sm80_experiment {
Kernel select_paired_pipeline_kernel(bool dual,bool fast,int tune) {
  if(tune!=29 || (dual && fast))
    throw std::invalid_argument("expected paired-pipeline candidate29 with exact dual scale multiplication");
  return dual ? adangel_sm80_roof_candidate<true,false,29> :
      (fast ? adangel_sm80_roof_candidate<false,true,29> : adangel_sm80_roof_candidate<false,false,29>);
}
size_t paired_pipeline_shared_bytes(bool dual) {
  return dual ? sizeof(o3_paired_pipeline_experiment::O3AmpereConfig<64,128,128,false,2,true,3>::Storage) :
      sizeof(o3_paired_pipeline_experiment::O3AmpereConfig<64,128,128,false,2,false,3>::Storage);
}
}
