// Independent v85: lane-vector payload layout, same v78 integer math.
// Natural payload is retained in the first half for the unchanged fallback.
#include "roof_o78_eight_chain_probe.cu"
#include "o78_register_layout_mapping.cuh"
namespace {
#include "o78_register_layout_generated.cuh"
}
extern "C" __global__ __launch_bounds__(128,3)
void adangel_roof_o78_register_layout_candidate(const uint8_t* a,const uint8_t* w,
    const float* as,const float* ws,const int32_t* af,const int32_t* wf,
    const float* base_a,const float* base_w,const uint32_t* status,
    float* y,int m,int n,int k) {
  const uint32_t flag=status[blockIdx.y*(n/128)+blockIdx.x];
  if(flag>1u) return;
  if(flag==1u)
    O78::o3_body<64,128,128,false,false,false,2,false,true,false,true,true,true,false,
        true,true,6,false,false,true,false,2>(a,w,as,reinterpret_cast<const uint8_t*>(ws),y,m,n,k);
  else
    o78_register_layout_experiment::body(a+size_t(m)*k,w+size_t(n)*(k/2),
        af,wf,base_a,base_w,y,m,n,k);
}
