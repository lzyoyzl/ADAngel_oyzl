// One larger-N reuse candidate, derived from v78; production stays unchanged.
#include "roof_o78_eight_chain_probe.cu"
namespace {
#include "o78_n256_generated.cuh"
}
extern "C" __global__ __launch_bounds__(128,2)
void adangel_roof_o78_n256_candidate(const uint8_t* a,const uint8_t* w,
    const float* as,const float* ws,const int32_t* af,const int32_t* wf,
    const float* base_a,const float* base_w,const uint32_t* status,
    float* y,int m,int n,int k) {
  // Existing guard describes N128 tiles; both halves must be integer-safe.
  // Uniformly fall back for both halves if either half needs FP32. No unsafe
  // integer path is allowed by merging two independently checked regions.
  const unsigned index=blockIdx.y*(n/128)+2*blockIdx.x;
  const uint32_t flag=status[index]|status[index+1];
  if(flag>1u) return; // host rejects invalid/no-store status before launch
  if(flag==1u)
    O78::o3_body<64,256,128,false,false,false,2,false,true,false,true,true,true,false,
        true,true,6,false,false,true,false,2>(a,w,as,reinterpret_cast<const uint8_t*>(ws),y,m,n,k);
  else
    o78_n256_experiment::body(a,w,af,wf,base_a,base_w,y,m,n,k);
}
