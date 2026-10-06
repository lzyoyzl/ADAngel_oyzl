// v112: isolated cache-fed register fragments; no production binding.
#include "roof_o78_eight_chain_probe.cu"
#include "o78_register_layout_mapping.cuh"
namespace {
#include "o78_direct_fragment_generated.cuh"
}
extern "C" __global__ __launch_bounds__(128,4)
void adangel_roof_o78_direct_fragment_candidate(const uint8_t* a,const uint8_t* w,
    const float* as,const float* ws,const int32_t* af,const int32_t* wf,
    const float* base_a,const float* base_w,const uint32_t* status,
    float* y,int m,int n,int k) {
  const uint32_t flag=status[blockIdx.y*(n/128)+blockIdx.x];
  if(flag>1u)return; // Internal host must reject invalid source metadata.
  const auto* pa=a+size_t(m)*k;
  const auto* pw=w+size_t(n)*(k/2);
  if(flag==1u)
    o78_direct_fragment_experiment::body<false>(pa,pw,as,ws,base_a,base_w,y,m,n,k);
  else
    o78_direct_fragment_experiment::body<true>(pa,pw,af,wf,base_a,base_w,y,m,n,k);
}
