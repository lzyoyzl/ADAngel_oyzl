//v98 fixed8-warp geometry; unchanged O7/O8 quantization, guard and two stages.
#include "roof_o78_eight_chain_probe.cu"
namespace {
#include "o78_eight_warp_fullk_fallback_generated.cuh"
#include "o78_eight_warp_fullk_generated.cuh"
}
extern "C" __global__ __launch_bounds__(256,2)
void adangel_roof_o78_eight_warp_fullk_candidate(const uint8_t* a,const uint8_t* w,
    const float* as,const float* ws,const int32_t* af,const int32_t* wf,
    const float* base_a,const float* base_w,const uint32_t* status,
    float* y,int m,int n,int k) {
  const uint32_t flag=status[blockIdx.y*(n/128)+blockIdx.x];
  if(flag>1u) return;
  if(flag==1u)
    o78_eight_warp_fallback::o3_body<64,128,128,false,false,false,4,false,true,false,true,true,true,false,
        true,true,6,false,false,true,false,2>(a,w,as,reinterpret_cast<const uint8_t*>(ws),y,m,n,k);
  else
    o78_eight_warp_fullk_experiment::body(a,w,af,wf,base_a,base_w,y,m,n,k);
}
