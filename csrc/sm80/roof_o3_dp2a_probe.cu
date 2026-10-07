// v115: isolated fixed O3 dot-fused candidate; defaults/5090 unchanged.
#include "roof_o3_grouped_cta_probe.cu"
namespace {
#include "o3_dp2a_generated.cuh"
}

// Original33*N metadata + packed32*N + one eligibility flag per N128.
// This extra preparation must count in Cold, and is cacheable in steady.
extern "C" __global__ void adangel_roof_o3_dp2a_pack_metadata(
    const int32_t* original,int32_t* joined,int n) {
  const int column=blockIdx.x*128+threadIdx.x;
  bool safe=true;
  for(int group=0;group<32;++group) {
    const int value=original[group*n+column];
    joined[group*n+column]=value;
    const bool fits=value>=1 && value<=15;
    joined[(33+group)*n+column]=fits?value*4097:0;
    safe=safe && fits;
  }
  joined[32*n+column]=original[32*n+column];
  const int failed=__syncthreads_count(!safe);
  if(threadIdx.x==0)joined[65*n+blockIdx.x]=(failed==0);
}

extern "C" __global__ __launch_bounds__(128,3)
void adangel_roof_o3_dp2a_candidate(const uint8_t* a,const uint8_t* w,
    const float* as,const uint8_t* ws,const int32_t* joined,
    const uint32_t* status,float* y,int m,int n,int k) {
  const uint32_t column_tile=roof_grouped_cta::tile().x;
  const uint32_t flag=status[column_tile];
  if(flag&6u)return;
  if(flag&1u)
    o3_grouped_fallback::o3_body<64,128,128,false,false,false,2,false,true,false,true,true,true,false,
        false,true,6,false,false,false,false,3>(a,w,as,ws,y,m,n,k);
  else if(joined[65*n+column_tile])
    o3_dp2a_experiment::body(a,w,as,reinterpret_cast<const uint8_t*>(joined),y,m,n,k);
  else
    o3_grouped_cta_experiment::body(a,w,as,reinterpret_cast<const uint8_t*>(joined),y,m,n,k);
}
