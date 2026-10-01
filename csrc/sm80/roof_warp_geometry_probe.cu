// v40 isolated 2x2 versus 2x4 warp geometry; production defaults untouched.
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cutlass/integer_subbyte.h>

#ifndef ADANGEL_WARP_GEOMETRY
#error "Select geometry 0 (2x2) or 1 (2x4)"
#endif
static_assert(ADANGEL_WARP_GEOMETRY==0 || ADANGEL_WARP_GEOMETRY==1);
constexpr int ProbeWN=ADANGEL_WARP_GEOMETRY?4:2;
constexpr int ProbeThreads=64*ProbeWN;
constexpr int ProbeMinBlocks=ADANGEL_WARP_GEOMETRY?2:3;
namespace {
__device__ void copy16(void* dst,const void* src) {
  const uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(address),"l"(src):"memory");
}
template<int I,int End,class F>
__device__ __forceinline__ void o1_static_for(F const& f) {
  if constexpr(I<End) {f(cute::Int<I>{});o1_static_for<I+1,End>(f);}
}
#include "o3_warp_geometry_probe.cuh"
#include "o78_warp_geometry_probe.cuh"
// Thread geometry must not change the allocation or cp.async ring layout.
static_assert(sizeof(o3_row_scale_epilogue_experiment_v40::O3AmpereConfig<64,128,128,false,2,false,3>::Storage)==
              sizeof(o3_row_scale_epilogue_experiment_v40::O3AmpereConfig<64,128,128,false,ProbeWN,false,3>::Storage));
static_assert(sizeof(o78_unsigned_payload_experiment_v40::O3AmpereConfig<64,128,128,false,2,true,2>::Storage)==
              sizeof(o78_unsigned_payload_experiment_v40::O3AmpereConfig<64,128,128,false,ProbeWN,true,2>::Storage));
}
extern "C" __global__ __launch_bounds__(ProbeThreads,ProbeMinBlocks)
void adangel_roof_warp_o3(const uint8_t* a,const uint8_t* w,const float* as,
    const uint8_t* ws,float* y,int m,int n,int k) {
  o3_row_scale_epilogue_experiment_v40::o3_body<64,128,128,false,false,false,ProbeWN,false,true,false,true,true,true,false,
      false,true,6,false,false,false,false,3>(a,w,as,ws,y,m,n,k);
}
extern "C" __global__ __launch_bounds__(ProbeThreads,ProbeMinBlocks)
void adangel_roof_warp_o78(const uint8_t* a,const uint8_t* w,const float* as,
    const uint8_t* ws,float* y,int m,int n,int k) {
  o78_unsigned_payload_experiment_v40::o3_body<64,128,128,false,false,false,ProbeWN,false,true,false,true,true,true,false,
      true,true,6,false,false,true,false,2>(a,w,as,ws,y,m,n,k);
}
