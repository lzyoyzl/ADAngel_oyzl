// v79 isolated O3 scheduling candidate; existing guard and fallback unchanged.
#include "o3_device_control_generated.cu"
namespace {
#include "o3_eight_chain_generated.cuh"
}
extern "C" __global__ __launch_bounds__(128,3)
void adangel_roof_o3_eight_chain_candidate(const uint8_t* a,const uint8_t* w,
    const float* as,const uint8_t* ws,const int32_t* meta,
    const uint32_t* status,float* y,int m,int n,int k) {
  const uint32_t flag=status[blockIdx.x];
  if(flag&6u) return;
  if(flag&1u)
    O3::o3_body<64,128,128,false,false,false,2,false,true,false,true,true,true,false,
        false,true,6,false,false,false,false,3>(a,w,as,ws,y,m,n,k);
  else
    o3_eight_chain_experiment::body(a,w,as,reinterpret_cast<const uint8_t*>(meta),y,m,n,k);
}
