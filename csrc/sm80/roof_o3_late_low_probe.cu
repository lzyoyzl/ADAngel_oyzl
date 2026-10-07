// v131: same-G128 low-A loads after the first signed-high dot.
// No new buffer, synchronization, scale representation or default switch.
#include "roof_o3_grouped_cta_probe.cu"
namespace {
#include "o3_late_low_generated.cuh"
}
extern "C" __global__ __launch_bounds__(128,3)
void adangel_roof_o3_late_low_candidate(const uint8_t* a,const uint8_t* w,
    const float* as,const uint8_t* ws,const int32_t* meta,
    const uint32_t* status,float* y,int m,int n,int k) {
  const uint32_t flag=status[roof_grouped_cta::tile().x];
  if(flag&6u) return;
  if(flag&1u)
    o3_grouped_fallback::o3_body<64,128,128,false,false,false,2,false,true,false,true,true,true,false,
        false,true,6,false,false,false,false,3>(a,w,as,ws,y,m,n,k);
  else
    o3_late_low_experiment::body(a,w,as,reinterpret_cast<const uint8_t*>(meta),y,m,n,k);
}
