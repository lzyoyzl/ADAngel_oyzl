// v76: isolated shared coefficient-table hypothesis. No production dispatch.
// Include the unmodified v67 entries as controls in this translation unit.
#include "roof_o78_fullk_probe.cu"
namespace {
#include "o7_factor_table_generated.cuh"
}

extern "C" __global__ __launch_bounds__(128,3)
void adangel_roof_o7_factor_table_candidate(const uint8_t* a,const uint8_t* w,
    const float* as,const float* ws,const int32_t* af,const int32_t* wf,
    const float* base_a,const float* base_w,const uint32_t* status,
    float* y,int m,int n,int k) {
  const uint32_t flag=status[blockIdx.y*(n/128)+blockIdx.x];
  if(flag>1u) return;  // host rejects invalid sources, unchanged v67 contract
  if(flag==1u) {
    O78::o3_body<64,128,128,false,false,false,2,false,true,false,true,true,true,false,
        true,true,6,false,false,true,false,2>(a,w,as,reinterpret_cast<const uint8_t*>(ws),y,m,n,k);
    return;
  }
  // CTA-uniform exact guard; included in GEMM timing. No host decision or new
  // global metadata. All 32 groups x64 rows must be powers of two <=512.
  bool bad=false;
  #pragma unroll
  for(int item=threadIdx.x;item<32*64;item+=128) {
    const uint32_t f=static_cast<uint32_t>(af[(item/64)*m+blockIdx.y*64+item%64]);
    bad|=f==0u || (f&(f-1u))!=0u || f>=1024u;
  }
  if(__syncthreads_or(bad))
    o78_fullk_integer_experiment::body(a,w,af,wf,base_a,base_w,y,m,n,k);
  else
    o7_factor_table_experiment::body(a,w,af,wf,base_a,base_w,y,m,n,k);
}
