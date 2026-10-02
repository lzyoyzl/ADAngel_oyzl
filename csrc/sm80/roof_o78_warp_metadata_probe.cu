// v81: same v78 GEMM, full-warp factor copies. Independent experiment only.
#include "roof_o78_eight_chain_probe.cu"
namespace {
__device__ __forceinline__ void copy_factor8(void* dst,const void* src) {
  const uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  // PTX .cg supports only16B. .ca8B lets all32 lanes cover64 A factors once.
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(address),"l"(src):"memory");
}
#include "o78_warp_metadata_generated.cuh"
}
extern "C" __global__ __launch_bounds__(128,3)
void adangel_roof_o78_warp_metadata_candidate(const uint8_t* a,const uint8_t* w,
    const float* as,const float* ws,const int32_t* af,const int32_t* wf,
    const float* base_a,const float* base_w,const uint32_t* status,
    float* y,int m,int n,int k) {
  const uint32_t flag=status[blockIdx.y*(n/128)+blockIdx.x];
  if(flag>1u) return;
  if(flag==1u)
    O78::o3_body<64,128,128,false,false,false,2,false,true,false,true,true,true,false,
        true,true,6,false,false,true,false,2>(a,w,as,reinterpret_cast<const uint8_t*>(ws),y,m,n,k);
  else
    o78_warp_metadata_experiment::body(a,w,af,wf,base_a,base_w,y,m,n,k);
}
