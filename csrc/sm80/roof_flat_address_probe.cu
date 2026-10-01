// Independent v44 cubins: 0=original, 1=flat payload indices, 2=also flat FP32 scale indices.
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cutlass/integer_subbyte.h>
#ifndef ADANGEL_FLAT_ADDRESS
#error "Select flat address policy 0, 1 or 2"
#endif
static_assert(ADANGEL_FLAT_ADDRESS>=0 && ADANGEL_FLAT_ADDRESS<=2);
constexpr int ProbeThreads=128;
constexpr int ProbeMinBlocks=3;
namespace {
__device__ void copy16(void* dst,const void* src) {
  const uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(address),"l"(src):"memory");
}
template<int I,int End,class F>
__device__ __forceinline__ void o1_static_for(F const& f) {
  if constexpr(I<End) {f(cute::Int<I>{});o1_static_for<I+1,End>(f);}
}
#if ADANGEL_FLAT_ADDRESS==0
#include "o3_row_scale_epilogue_candidate.cuh"
#include "o78_unsigned_payload_candidate.cuh"
namespace O3=o3_row_scale_epilogue_experiment;
namespace O78=o78_unsigned_payload_experiment;
#else
#include "o3_flat_address_probe.cuh"
#include "o78_flat_address_probe.cuh"
namespace O3=o3_row_scale_epilogue_experiment_v44;
namespace O78=o78_unsigned_payload_experiment_v44;
#endif
}
extern "C" __global__ __launch_bounds__(ProbeThreads,ProbeMinBlocks)
void adangel_roof_flat_address_o3(const uint8_t* a,const uint8_t* w,const float* as,
    const uint8_t* ws,float* y,int m,int n,int k) {
  O3::o3_body<64,128,128,false,false,false,2,false,true,false,true,true,true,false,
      false,true,6,false,false,false,false,3>(a,w,as,ws,y,m,n,k);
}
extern "C" __global__ __launch_bounds__(ProbeThreads,ProbeMinBlocks)
void adangel_roof_flat_address_o78(const uint8_t* a,const uint8_t* w,const float* as,
    const uint8_t* ws,float* y,int m,int n,int k) {
  O78::o3_body<64,128,128,false,false,false,2,false,true,false,true,true,true,false,
      true,true,6,false,false,true,false,2>(a,w,as,ws,y,m,n,k);
}



