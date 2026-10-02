// v55 isolated full-K probe, no public extension binding or default changes.
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cutlass/integer_subbyte.h>
#ifndef ADANGEL_FULLK_INTEGER
#error "Select control0 or full-K candidate1"
#endif
static_assert(ADANGEL_FULLK_INTEGER>=0 && ADANGEL_FULLK_INTEGER<=1);
namespace {
__device__ void copy16(void* dst,const void* src) {
  const uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(address),"l"(src):"memory");
}
template<int I,int End,class F>
__device__ __forceinline__ void o1_static_for(F const& f) {
  if constexpr(I<End) {f(cute::Int<I>{});o1_static_for<I+1,End>(f);}
}
#include "o3_row_scale_epilogue_candidate.cuh"
#include "o78_unsigned_payload_candidate.cuh"
namespace O3=o3_row_scale_epilogue_experiment;
namespace O78=o78_unsigned_payload_experiment;
#if ADANGEL_FULLK_INTEGER>0
#include "o3_factor_async_generated.cuh"
#endif
}
extern "C" __global__ __launch_bounds__(128,3)
void adangel_roof_fullk_integer_o3(const uint8_t* a,const uint8_t* w,const float* as,
    const uint8_t* ws,float* y,int m,int n,int k) {
#if ADANGEL_FULLK_INTEGER==0
  O3::o3_body<64,128,128,false,false,false,2,false,true,false,true,true,true,false,
      false,true,6,false,false,false,false,3>(a,w,as,ws,y,m,n,k);
#else
  o3_fullk_integer_experiment::body(a,w,as,ws,y,m,n,k);
#endif
}
extern "C" __global__ __launch_bounds__(128,3)
void adangel_roof_fullk_integer_o78(const uint8_t* a,const uint8_t* w,const float* as,
    const uint8_t* ws,float* y,int m,int n,int k) {
  O78::o3_body<64,128,128,false,false,false,2,false,true,false,true,true,true,false,
      true,true,6,false,false,true,false,2>(a,w,as,ws,y,m,n,k);
}
