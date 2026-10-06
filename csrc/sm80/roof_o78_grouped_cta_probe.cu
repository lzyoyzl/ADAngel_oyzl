// v89: same v78 math, pipeline and preparation; grouped CTA coordinates only.
#include "roof_o78_eight_chain_probe.cu"
namespace {
#include "roof_grouped_cta.cuh"
#include "o78_grouped_fallback_generated.cuh"
#include "o78_grouped_cta_generated.cuh"
}
extern "C" __global__ __launch_bounds__(128,3)
void adangel_roof_o78_grouped_cta_candidate(const uint8_t* a,const uint8_t* w,
    const float* as,const float* ws,const int32_t* af,const int32_t* wf,
    const float* base_a,const float* base_w,const uint32_t* status,
    float* y,int m,int n,int k) {
  const auto tile=roof_grouped_cta::tile();
  const uint32_t flag=status[tile.y*(n/128)+tile.x];
  if(flag>1u) return;
  if(flag==1u)
    o78_grouped_fallback::o3_body<64,128,128,false,false,false,2,false,true,false,true,true,true,false,
        true,true,6,false,false,true,false,2>(a,w,as,reinterpret_cast<const uint8_t*>(ws),y,m,n,k);
  else
    o78_grouped_cta_experiment::body(a,w,af,wf,base_a,base_w,y,m,n,k);
}
