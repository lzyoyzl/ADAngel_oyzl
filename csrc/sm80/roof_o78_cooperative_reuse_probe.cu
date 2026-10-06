// v111 compiler prototype: proportional CTA cooperation, not a default.
#include "roof_o78_eight_chain_probe.cu"
namespace {
#include "o78_cooperative_reuse_generated.cuh"
}
extern "C" __global__ __launch_bounds__(384,1)
void adangel_roof_o78_cooperative_reuse_candidate(const uint8_t* a,const uint8_t* w,
    const float* as,const float* ws,const int32_t* af,const int32_t* wf,
    const float* base_a,const float* base_w,const uint32_t* status,
    float* y,int m,int n,int k) {
  // The old per-output bound remains sufficient despite the larger tile.
  const unsigned old_n=n/128,first=(blockIdx.x*192)/128;
  unsigned flag=0;
  #pragma unroll
  for(unsigned dy=0;dy<2;++dy) {
    flag|=status[(blockIdx.y*2+dy)*old_n+first];
    if(first+1<old_n)flag|=status[(blockIdx.y*2+dy)*old_n+first+1];
  }
  // Compiler-only prototype. Nonzero tiles MUST NOT be benchmarked/exposed;
  // runtime integration must supply unchanged fallback before full acceptance.
  if(flag!=0u)return;
  o78_cooperative_reuse_experiment::body(a,w,af,wf,base_a,base_w,y,m,n,k);
}
