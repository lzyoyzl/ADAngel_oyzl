// v39 isolated integer partial recomposition, not a production adapter.
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cutlass/integer_subbyte.h>

#ifndef ADANGEL_RECOMPOSE_POLICY
#error "Select recomposition policy 0, 1 or 2"
#endif
static_assert(ADANGEL_RECOMPOSE_POLICY>=0 && ADANGEL_RECOMPOSE_POLICY<=2);
namespace {
__device__ __forceinline__ int roof_recompose(int low,int high) {
#if ADANGEL_RECOMPOSE_POLICY == 0
  return low+16*high;
#elif ADANGEL_RECOMPOSE_POLICY == 1
  int result;
  asm volatile("{ .reg .b32 shifted; shl.b32 shifted, %2, 4; add.s32 %0, %1, shifted; }"
      : "=r"(result) : "r"(low),"r"(high));
  return result;
#else
  int shifted,result;
  asm volatile("shl.b32 %0, %1, 4;" : "=r"(shifted) : "r"(high));
  asm volatile("add.s32 %0, %1, %2;" : "=r"(result) : "r"(low),"r"(shifted));
  return result;
#endif
}
__device__ void copy16(void* dst,const void* src) {
  const uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(address),"l"(src):"memory");
}
template<int I,int End,class F>
__device__ __forceinline__ void o1_static_for(F const& f) {
  if constexpr(I<End) {f(cute::Int<I>{});o1_static_for<I+1,End>(f);}
}
#include "o3_partial_recompose_probe.cuh"
#include "o78_partial_recompose_probe.cuh"
}
extern "C" __global__ __launch_bounds__(128,3)
void adangel_roof_recompose_o3(const uint8_t* a,const uint8_t* w,const float* as,
    const uint8_t* ws,float* y,int m,int n,int k) {
  o3_row_scale_epilogue_experiment_v39::o3_body<64,128,128,false,false,false,2,false,true,false,true,true,true,false,
      false,true,6,false,false,false,false,3>(a,w,as,ws,y,m,n,k);
}
extern "C" __global__ __launch_bounds__(128,3)
void adangel_roof_recompose_o78(const uint8_t* a,const uint8_t* w,const float* as,
    const uint8_t* ws,float* y,int m,int n,int k) {
  o78_unsigned_payload_experiment_v39::o3_body<64,128,128,false,false,false,2,false,true,false,true,true,true,false,
      true,true,6,false,false,true,false,2>(a,w,as,ws,y,m,n,k);
}
