// 61/62: cache-policy-only counterparts of O3/54 and O7/O8/59.
// Reuse the exact device bodies; preserve every arithmetic and pipeline step.
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cutlass/integer_subbyte.h>
#include <stdexcept>
#include "roof_cache_policy_api.h"

namespace {
__device__ void copy16(void* dst,const void* src) {
  const uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::
      "r"(address),"l"(src):"memory");
}
template<int I,int End,class F>
__device__ __forceinline__ void o1_static_for(F const& f) {
  if constexpr(I<End) {f(cute::Int<I>{});o1_static_for<I+1,End>(f);}
}
#include "o3_row_scale_epilogue_candidate.cuh"
#include "o78_unsigned_payload_candidate.cuh"

template<bool DualScale,bool Fast,int Tune>
__global__ __launch_bounds__(128,3)
void adangel_sm80_roof_candidate(const uint8_t* a,const uint8_t* w,const float* as,
    const uint8_t* ws,float* y,int m,int n,int k) {
  static_assert(!Fast && ((!DualScale && Tune==61) || (DualScale && Tune==62)));
  if constexpr(Tune==61) {
    o3_row_scale_epilogue_experiment::o3_body<64,128,128,false,false,false,2,false,true,false,true,true,true,false,
        false,true,6,false,false,false,false,3>(a,w,as,ws,y,m,n,k);
  } else {
    o78_unsigned_payload_experiment::o3_body<64,128,128,false,false,false,2,false,true,false,true,true,true,false,
        true,true,6,false,false,true,false,2>(a,w,as,ws,y,m,n,k);
  }
}
}
namespace adangel_sm80_experiment {
Kernel select_cache_policy_kernel(bool dual,int tune) {
  if(!dual && tune==61) return adangel_sm80_roof_candidate<false,false,61>;
  if(dual && tune==62) return adangel_sm80_roof_candidate<true,false,62>;
  throw std::invalid_argument("cache candidate61 requires O3;62 requires O7/O8");
}
}
