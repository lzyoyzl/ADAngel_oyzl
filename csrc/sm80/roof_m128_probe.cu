// Isolated v57: four warps reuse B across four M atoms instead of two.
// No source-format, G128, INT4 route, scale, or FP32 accumulation-order change.
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cutlass/integer_subbyte.h>
#ifndef ADANGEL_PROBE_M
#error "ADANGEL_PROBE_M must be 64 or 128"
#endif
static_assert(ADANGEL_PROBE_M==64 || ADANGEL_PROBE_M==128);
constexpr int ProbeM=ADANGEL_PROBE_M;
constexpr int MinBlocks=ProbeM==64?3:2;
namespace {
__device__ void copy16(void* dst,const void* src) {
  const uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(address),"l"(src):"memory");
}
template<int I,int End,class F>
__device__ __forceinline__ void o1_static_for(F const& f) {
  if constexpr(I<End) {f(cute::Int<I>{});o1_static_for<I+1,End>(f);}
}
// Generated copies differ ONLY in the explicit M==64 config assertion.
// Their hashes and exact mechanical substitution are recorded by the builder.
#include "o3_m128_generated.cuh"
#include "o78_m128_generated.cuh"
namespace O3=o3_row_scale_epilogue_experiment;
namespace O78=o78_unsigned_payload_experiment;
using C3=O3::O3AmpereConfig<ProbeM,128,128,false,2,false,3>;
using C78=O78::O3AmpereConfig<ProbeM,128,128,false,2,true,2>;
static_assert(C3::Threads==128 && C78::Threads==128);
static_assert(sizeof(C3::Storage)==(ProbeM==64?50688:75264));
static_assert(sizeof(C78::Storage)==(ProbeM==64?34304:51200));
}
extern "C" __global__ __launch_bounds__(128,MinBlocks)
void adangel_roof_m128_o3(const uint8_t* a,const uint8_t* w,const float* as,
    const uint8_t* ws,float* y,int m,int n,int k) {
  O3::o3_body<ProbeM,128,128,false,false,false,2,false,true,false,true,true,true,false,
      false,true,6,false,false,false,false,3>(a,w,as,ws,y,m,n,k);
}
extern "C" __global__ __launch_bounds__(128,MinBlocks)
void adangel_roof_m128_o78(const uint8_t* a,const uint8_t* w,const float* as,
    const uint8_t* ws,float* y,int m,int n,int k) {
  O78::o3_body<ProbeM,128,128,false,false,false,2,false,true,false,true,true,true,false,
      true,true,6,false,false,true,false,2>(a,w,as,ws,y,m,n,k);
}
