// v122 isolated candidate. Status3 means fullK-safe but wide G128 partials.
// Preserve both validated eight-warp fallback paths and the v78 control.
#include "roof_o78_eight_warp_fullk_probe.cu"
namespace {
#include "o78_partial_handoff_probe.cuh"
}
extern "C" __global__ __launch_bounds__(256,2)
void adangel_roof_o78_partial_handoff_candidate(const uint8_t* a,const uint8_t* w,
    const float* as,const float* ws,const int32_t* af,const int32_t* wf,
    const float* base_a,const float* base_w,const uint32_t* status,
    float* y,int m,int n,int k) {
  const uint32_t flag=status[blockIdx.y*(n/128)+blockIdx.x];
  if(flag==2u || flag>3u) return; // invalid input is also rejected by host
  if(flag==1u)
    o78_eight_warp_fallback::o3_body<64,128,128,false,false,false,4,false,true,false,true,true,true,false,
        true,true,6,false,false,true,false,2>(a,w,as,reinterpret_cast<const uint8_t*>(ws),y,m,n,k);
  else if(flag==3u)
    o78_eight_warp_fullk_experiment::body(a,w,af,wf,base_a,base_w,y,m,n,k);
  else if(threadIdx.x<128)
    o78_partial_handoff::producer(a,w,af,wf,base_a,base_w,y,m,n,k);
  else
    o78_partial_handoff::consumer(base_a,base_w,y,n);
}
