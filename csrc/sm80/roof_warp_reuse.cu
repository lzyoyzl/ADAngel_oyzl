// Isolated 2x2-warp fragment-reuse experiment; not a production default.
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
#include "o3_warp_reuse_candidate.cuh"

template<bool DualScale,bool Fast,int Tune>
__global__ __launch_bounds__(128,2)
void adangel_sm80_roof_candidate(
    const uint8_t* a,const uint8_t* w,const float* as,const uint8_t* ws,
    float* y,int m,int n,int k) {
  static_assert(Tune==20 || Tune==21);
  static_assert(!DualScale || !Fast);
  constexpr int CoreTune=Tune==20?2:6;
  // Same output tile, scales and two K256 buffers as candidate6. Each warp
  // owns M32 instead of M16, reusing each B register fragment across M atoms.
  // FP32 accumulator count is64/thread; higher register pressure is measured,
  // not assumed harmless. Each output still visits G128 in ascending order.
  o3_warp_reuse_experiment::o3_body<64,128,256,Fast,false,false,2,false,true,false,true,true,true,false,
      DualScale,DualScale,CoreTune>(a,w,as,ws,y,m,n,k);
}
}

namespace adangel_sm80_experiment {
Kernel select_warp_reuse_kernel(bool dual,bool fast,int tune) {
  if(dual && fast) throw std::invalid_argument("warp reuse uses exact FP32 dual scale multiplication");
  if(tune==20) return dual ? adangel_sm80_roof_candidate<true,false,20> :
      (fast ? adangel_sm80_roof_candidate<false,true,20> : adangel_sm80_roof_candidate<false,false,20>);
  if(tune==21) return dual ? adangel_sm80_roof_candidate<true,false,21> :
      (fast ? adangel_sm80_roof_candidate<false,true,21> : adangel_sm80_roof_candidate<false,false,21>);
  throw std::invalid_argument("expected warp-reuse candidate20/21");
}
}
