// Fixed8-warp two-route experiment. No public binding/default changes.
#include "roof_o3_grouped_cta_probe.cu"
namespace {
#include "o3_route_cohort_candidate.cuh"
}
extern "C" __global__ __launch_bounds__(256,2)
void adangel_roof_o3_route_cohort_candidate(const uint8_t* a,const uint8_t* w,
    const float* as,const uint8_t* ws,const int32_t* meta,
    const uint32_t* status,float* y,int m,int n,int k) {
  const uint32_t flag=status[roof_grouped_cta::tile().x];
  if(flag&6u) return;
  if(flag&1u) {
    // Entire upper warps retire before the original128-thread fallback starts.
    // Runtime acceptance must include synccheck and mixed safe/fallback grids.
    if(threadIdx.x>=128) return;
    o3_grouped_fallback::o3_body<64,128,128,false,false,false,2,false,true,false,true,true,true,false,
        false,true,6,false,false,false,false,3>(a,w,as,ws,y,m,n,k);
  } else {
    o3_route_cohort_experiment::body(a,w,as,reinterpret_cast<const uint8_t*>(meta),y,m,n,k);
  }
}
