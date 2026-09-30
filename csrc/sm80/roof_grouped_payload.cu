// Candidates41/42: make each G128 panel contiguous in global memory.
// No quantization or arithmetic change; packing is measured by host adapters.
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
      "r"(address),"l"(src):"memory");
}
template<int I,int End,class F>
__device__ __forceinline__ void o1_static_for(F const& f) {
  if constexpr(I<End) {f(cute::Int<I>{});o1_static_for<I+1,End>(f);}
}
#include "o3_grouped_payload_candidate.cuh"

template<bool DualScale,bool Fast,int Tune>
__global__ __launch_bounds__(128,3)
void adangel_sm80_roof_candidate(const uint8_t* a,const uint8_t* w,const float* as,
    const uint8_t* ws,float* y,int m,int n,int k) {
  static_assert(Tune==41 || Tune==42);
  static_assert(!DualScale || !Fast);
  constexpr int Stages=Tune==41?2:3;
  o3_grouped_payload_experiment::o3_body<64,128,128,Fast,false,false,2,false,true,false,true,true,true,false,
      DualScale,DualScale,6,false,false,false,false,Stages>(a,w,as,ws,y,m,n,k);
}

// Each thread moves one16B vector from [plane,row,group,byte64]
// to [plane,group,row,byte64]. Payload bits are not decoded or modified.
__global__ void adangel_sm80_pack_g128_payload(const uint4* src,uint4* dst,
    int rows,int groups,int vectors) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=vectors) return;
  const int plane_span=rows*groups*4;
  const int plane=i/plane_span,local=i%plane_span;
  const int group=local/(rows*4),row=(local/4)%rows,col=local%4;
  dst[i]=src[(plane*rows+row)*groups*4+group*4+col];
}
}

namespace adangel_sm80_experiment {
Kernel select_grouped_payload_kernel(bool dual,bool fast,int tune) {
  if((tune!=41 && tune!=42) || (dual && fast)) throw std::invalid_argument("expected G128 payload candidate41/42");
  if(tune==41) return dual?adangel_sm80_roof_candidate<true,false,41>:
      (fast?adangel_sm80_roof_candidate<false,true,41>:adangel_sm80_roof_candidate<false,false,41>);
  return dual?adangel_sm80_roof_candidate<true,false,42>:
      (fast?adangel_sm80_roof_candidate<false,true,42>:adangel_sm80_roof_candidate<false,false,42>);
}
void pack_g128_payload(const uint8_t* src,uint8_t* dst,int planes,int rows,int k,cudaStream_t stream) {
  const int vectors=planes*rows*(k/32);
  adangel_sm80_pack_g128_payload<<<(vectors+255)/256,256,0,stream>>>(
      reinterpret_cast<const uint4*>(src),reinterpret_cast<uint4*>(dst),rows,k/128,vectors);
}
}
