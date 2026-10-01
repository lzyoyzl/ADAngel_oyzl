// v38 codegen preflight only; NOT linked into the production extension.
// Compile this SAME unit with ADANGEL_L2_PREFETCH_BYTES=0/128/256.
// Establish whether an explicit prefetch hint changes actual SM80 instructions
// before adding more runtime candidates or claiming a performance effect.
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cutlass/integer_subbyte.h>

#ifndef ADANGEL_L2_PREFETCH_BYTES
#error "Explicitly select 0, 128 or 256 byte L2 hint"
#endif
static_assert(ADANGEL_L2_PREFETCH_BYTES==0 || ADANGEL_L2_PREFETCH_BYTES==128 ||
              ADANGEL_L2_PREFETCH_BYTES==256);

namespace {
__device__ void copy16(void* dst,const void* src) {
  const uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
#if ADANGEL_L2_PREFETCH_BYTES == 128
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;" ::
      "r"(address),"l"(src):"memory");
#elif ADANGEL_L2_PREFETCH_BYTES == 256
  asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;" ::
      "r"(address),"l"(src):"memory");
#else
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
      "r"(address),"l"(src):"memory");
#endif
}
template<int I,int End,class F>
__device__ __forceinline__ void o1_static_for(F const& f) {
  if constexpr(I<End) {f(cute::Int<I>{});o1_static_for<I+1,End>(f);}
}
#include "o3_row_scale_epilogue_candidate.cuh"
#include "o78_unsigned_payload_candidate.cuh"
}

extern "C" __global__ __launch_bounds__(128,3)
void adangel_roof_l2_o3(const uint8_t* a,const uint8_t* w,const float* as,
    const uint8_t* ws,float* y,int m,int n,int k) {
  o3_row_scale_epilogue_experiment::o3_body<64,128,128,false,false,false,2,false,true,false,true,true,true,false,
      false,true,6,false,false,false,false,3>(a,w,as,ws,y,m,n,k);
}
extern "C" __global__ __launch_bounds__(128,3)
void adangel_roof_l2_o78(const uint8_t* a,const uint8_t* w,const float* as,
    const uint8_t* ws,float* y,int m,int n,int k) {
  o78_unsigned_payload_experiment::o3_body<64,128,128,false,false,false,2,false,true,false,true,true,true,false,
      true,true,6,false,false,true,false,2>(a,w,as,ws,y,m,n,k);
}
