// Candidates45/46: retain G128-major payload and B-fragment reuse, halve N.
// Unlike old9/10, use four warps (WM2/WN2) and 32 FP32 outputs per thread.
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cutlass/integer_subbyte.h>
#include <stdexcept>
#include "roof_narrow_payload_api.h"

namespace {
__device__ void copy16(void* dst,const void* src) {
  const uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
      "r"(address),"l"(src):"memory");
}
template<int I,int End,class F>
__device__ __forceinline__ void o1_static_for(F const& f) {
  if constexpr(I<End) {f(cute::Int<I>{});o1_static_for<I+1,End>(f);}
}
// Reuse the EXACT group arithmetic and pipeline code of41/42; no copied fork.
#include "o3_grouped_payload_candidate.cuh"

template<bool DualScale,bool Fast,int Tune>
__global__ __launch_bounds__(128,4)
void adangel_sm80_roof_candidate(const uint8_t* a,const uint8_t* w,const float* as,
    const uint8_t* ws,float* y,int m,int n,int k) {
  static_assert(Tune==45 || Tune==46);
  static_assert(!DualScale || !Fast);
  constexpr int Stages=Tune==45?2:3;
  o3_grouped_payload_experiment::o3_body<64,64,128,Fast,false,false,2,false,true,false,true,true,true,false,
      DualScale,DualScale,6,false,false,false,false,Stages>(a,w,as,ws,y,m,n,k);
}
template<bool Dual,int Stages> constexpr size_t shared_bytes() {
  return sizeof(typename o3_grouped_payload_experiment::O3AmpereConfig<64,64,128,false,2,Dual,Stages>::Storage);
}
}

namespace adangel_sm80_experiment {
Kernel select_narrow_payload_kernel(bool dual,bool fast,int tune) {
  if((tune!=45 && tune!=46) || (dual && fast)) throw std::invalid_argument("expected N64 payload candidate45/46");
  if(tune==45) return dual?adangel_sm80_roof_candidate<true,false,45>:
      (fast?adangel_sm80_roof_candidate<false,true,45>:adangel_sm80_roof_candidate<false,false,45>);
  return dual?adangel_sm80_roof_candidate<true,false,46>:
      (fast?adangel_sm80_roof_candidate<false,true,46>:adangel_sm80_roof_candidate<false,false,46>);
}
size_t narrow_payload_shared_bytes(bool dual,int tune) {
  if(tune==45) return dual?shared_bytes<true,2>():shared_bytes<false,2>();
  if(tune==46) return dual?shared_bytes<true,3>():shared_bytes<false,3>();
  throw std::invalid_argument("expected N64 payload candidate45/46");
}
}
