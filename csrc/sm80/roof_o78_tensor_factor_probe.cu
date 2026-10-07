// v129: integer Tensor Core computes scale outer products, never payload GEMM.
// Original payload remains two native INT4 routes. Independent probe only.
#include "roof_o78_eight_chain_probe.cu"
namespace {
#include "o78_tensor_factor_generated.cuh"
}
extern "C" __global__ __launch_bounds__(128,3)
void adangel_roof_o78_tensor_factor_candidate(const uint8_t* a,const uint8_t* w,
    const float* as,const float* ws,const int32_t* af,const int32_t* wf,
    const float* base_a,const float* base_w,const uint32_t* status,
    float* y,int m,int n,int k) {
  const uint32_t flag=status[blockIdx.y*(n/128)+blockIdx.x];
  if(flag>1u) return;
  if(flag==1u) {
    O78::o3_body<64,128,128,false,false,false,2,false,true,false,true,true,true,false,
        true,true,6,false,false,true,false,2>(a,w,as,reinterpret_cast<const uint8_t*>(ws),y,m,n,k);
    return;
  }
  // Preserve original full-K coefficient/product/prefix guard. This extra
  // exact U8 check is online GEMM work, not free/offline preparation.
  bool too_wide=false;
  #pragma unroll 1
  for(int g=0;g<32;++g) {
    if(threadIdx.x<64)
      too_wide|=static_cast<uint32_t>(af[g*m+blockIdx.y*64+threadIdx.x])>255u;
    too_wide|=static_cast<uint32_t>(wf[g*n+blockIdx.x*128+threadIdx.x])>255u;
  }
  if(__syncthreads_or(too_wide))
    o78_eight_chain_experiment::body(a,w,af,wf,base_a,base_w,y,m,n,k);
  else
    o78_tensor_factor_experiment::body(a,w,af,wf,base_a,base_w,y,m,n,k);
}
