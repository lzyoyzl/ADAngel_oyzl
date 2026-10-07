// v119: independent register-only MMA compiler-dependency experiment.
// Keep the original complete entry for encoded-SASS regression. No defaults.
#include "roof_o78_eight_chain_probe.cu"
#include "register_mma_generated.cuh"
namespace {
#include "o78_register_mma_generated.cuh"
}
extern "C" __global__ __launch_bounds__(128,3)
void adangel_roof_o78_register_mma_candidate(const uint8_t* a,const uint8_t* w,
    const float* as,const float* ws,const int32_t* af,const int32_t* wf,
    const float* base_a,const float* base_w,const uint32_t* status,
    float* y,int m,int n,int k) {
  const uint32_t flag=status[blockIdx.y*(n/128)+blockIdx.x];
  if(flag>1u) return;
  if(flag==1u)
    O78::o3_body<64,128,128,false,false,false,2,false,true,false,true,true,true,false,
        true,true,6,false,false,true,false,2>(a,w,as,reinterpret_cast<const uint8_t*>(ws),y,m,n,k);
  else
    o78_register_mma_experiment::body(a,w,af,wf,base_a,base_w,y,m,n,k);
}
