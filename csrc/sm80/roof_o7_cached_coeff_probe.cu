// v132: materialize W_factor*2^d once in WEIGHT conversion, not in each CTA.
// Independent candidate; keeps v78 as control and both original fallbacks.
#include "roof_o78_eight_chain_probe.cu"
namespace {
#include "o7_cached_coeff_generated.cuh"
}

// Allocated output: (32*N) original factors + (32*N*10) cached coefficients.
// Input/output must not alias. Original status guards remain authoritative.
extern "C" __global__ void adangel_roof_o7_build_cached_coeff(
    const int32_t* factors,uint32_t* extended,int n) {
  const unsigned col=blockIdx.x*128+threadIdx.x,group=blockIdx.y;
  const uint32_t f=static_cast<uint32_t>(factors[group*n+col]);
  extended[group*n+col]=f;
  #pragma unroll
  for(unsigned d=0;d<10;++d) {
    // N8-contiguous layout preserves adjacent coefficient pairs.
    const unsigned local=(threadIdx.x/8)*80+d*8+threadIdx.x%8;
    extended[32*n+(group*n+blockIdx.x*128)*10+local]=f<<d;
  }
}

extern "C" __global__ __launch_bounds__(128,3)
void adangel_roof_o7_cached_coeff_candidate(const uint8_t* a,const uint8_t* w,
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
  // Exact on-GPU range guard remains part of GEMM, not free host metadata.
  bool bad=false;
  #pragma unroll
  for(int item=threadIdx.x;item<32*64;item+=128) {
    const uint32_t f=static_cast<uint32_t>(af[(item/64)*m+blockIdx.y*64+item%64]);
    bad|=f==0u || (f&(f-1u))!=0u || f>=1024u;
  }
  if(__syncthreads_or(bad))
    o78_eight_chain_experiment::body(a,w,af,wf,base_a,base_w,y,m,n,k);
  else
    o7_cached_coeff_experiment::body(a,w,af,wf,base_a,base_w,y,m,n,k);
}
