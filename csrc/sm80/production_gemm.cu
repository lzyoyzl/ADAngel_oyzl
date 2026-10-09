// Accepted kernels only. No benchmark-report files or JIT compilation at runtime.
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cutlass/integer_subbyte.h>
#include "production_api.h"

namespace {
__device__ void copy16(void* dst,const void* src) {
  const uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(address),"l"(src):"memory");
}
template<int I,int End,class F>
__device__ __forceinline__ void o1_static_for(F const& f) {
  if constexpr(I<End) {f(cute::Int<I>{});o1_static_for<I+1,End>(f);}
}
#include "roof_grouped_cta.cuh"
#include "o3_row_scale_epilogue_candidate.cuh"
namespace O3=o3_row_scale_epilogue_experiment;
#include "production_generated/o3_fallback.cuh"
#include "production_generated/o3.cuh"
#include "o78_unsigned_payload_candidate.cuh"
namespace O78=o78_unsigned_payload_experiment;
#include "production_generated/o78.cuh"
}

extern "C" __global__ __launch_bounds__(128,3)
void adangel_sm80_o3_fullk_grouped(const uint8_t* a,const uint8_t* w,
    const float* as,const uint8_t* ws,const int32_t* meta,
    const uint32_t* status,float* y,int m,int n,int k) {
  const uint32_t flag=status[roof_grouped_cta::tile().x];
  if(flag&6u)return; // Host rejects invalid input before exposing output.
  if(flag&1u)
    o3_grouped_fallback::o3_body<64,128,128,false,false,false,2,false,true,false,true,true,true,false,
        false,true,6,false,false,false,false,3>(a,w,as,ws,y,m,n,k);
  else o3_grouped_cta_experiment::body(a,w,as,reinterpret_cast<const uint8_t*>(meta),y,m,n,k);
}

extern "C" __global__ __launch_bounds__(128,3)
void adangel_sm80_o78_fullk_streaming(const uint8_t* a,const uint8_t* w,
    const float* as,const float* ws,const int32_t* af,const int32_t* wf,
    const float* base_a,const float* base_w,const uint32_t* status,
    float* y,int m,int n,int k) {
  const uint32_t flag=status[blockIdx.y*(n/128)+blockIdx.x];
  if(flag>1u)return;
  if(flag==1u)
    O78::o3_body<64,128,128,false,false,false,2,false,true,false,true,true,true,false,
        true,true,6,false,false,true,false,2>(a,w,as,reinterpret_cast<const uint8_t*>(ws),y,m,n,k);
  else o78_output_streaming_experiment::body(a,w,af,wf,base_a,base_w,y,m,n,k);
}

namespace adangel_sm80_production {
cudaError_t resources(bool mixed,cudaFuncAttributes* attributes,int* active_blocks) {
  const void* kernel=mixed?reinterpret_cast<const void*>(adangel_sm80_o78_fullk_streaming):
                           reinterpret_cast<const void*>(adangel_sm80_o3_fullk_grouped);
  const int smem=mixed?34304:50688;
  auto rc=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
  if(rc!=cudaSuccess)return rc;
  rc=cudaFuncGetAttributes(attributes,kernel);
  if(rc!=cudaSuccess)return rc;
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(active_blocks,kernel,128,smem);
}
void o3_gemm(const uint8_t* a,const uint8_t* w,const float* as,const uint8_t* ws,
    const int32_t* metadata,const uint32_t* status,float* y,int m,int n,cudaStream_t stream) {
  adangel_sm80_o3_fullk_grouped<<<dim3(n/128,m/64),128,50688,stream>>>(a,w,as,ws,metadata,status,y,m,n,4096);
}
void mixed_gemm(const uint64_t* v,int m,int n,cudaStream_t stream) {
  #define P(T,I) reinterpret_cast<T*>(v[I])
  adangel_sm80_o78_fullk_streaming<<<dim3(n/128,m/64),128,34304,stream>>>(
    P(const uint8_t,0),P(const uint8_t,1),P(const float,2),P(const float,3),
    P(const int32_t,4),P(const int32_t,5),P(const float,6),P(const float,7),
    P(const uint32_t,14),P(float,15),m,n,4096);
  #undef P
}
}
