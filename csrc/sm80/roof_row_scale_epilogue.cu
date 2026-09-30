// Candidate only: keep the FP32 group reduction, apply invariant A_scale last.
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cutlass/integer_subbyte.h>
#include <stdexcept>
#include "roof_row_scale_epilogue_api.h"

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
#include "o3_row_scale_epilogue_candidate.cuh"

template<bool DualScale,bool Fast,int Tune>
__global__ __launch_bounds__(128,3)
void adangel_sm80_roof_candidate(const uint8_t* a,const uint8_t* w,const float* as,
    const uint8_t* ws,float* y,int m,int n,int k) {
  static_assert(!DualScale && !Fast && (Tune==51 || Tune==52));
  constexpr int Stages=Tune==51?2:3;
  o3_row_scale_epilogue_experiment::o3_body<64,128,128,false,false,false,2,false,true,false,true,true,true,false,
      false,false,6,false,false,false,false,Stages>(a,w,as,ws,y,m,n,k);
}
}
namespace adangel_sm80_experiment {
Kernel select_row_scale_epilogue_kernel(int tune) {
  if(tune==51) return adangel_sm80_roof_candidate<false,false,51>;
  if(tune==52) return adangel_sm80_roof_candidate<false,false,52>;
  throw std::invalid_argument("expected O3 row-scale epilogue candidate51/52");
}
}
