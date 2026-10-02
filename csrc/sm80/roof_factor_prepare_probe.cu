// Isolated v60 preparation only. The v59 GEMM cubins are reused unchanged.
#include <stdint.h>
#include <cuda_runtime.h>

extern "C" __global__ __launch_bounds__(128)
void adangel_roof_factor_prepare(const uint8_t* ws, int32_t* meta,
                                  uint32_t* block_status, int n) {
  const int col=blockIdx.x*128+threadIdx.x;
  unsigned flag=0;
  if(col<n) {
    unsigned code[32],anchor=255,sum=0;
    #pragma unroll
    for(int g=0;g<32;++g) {
      code[g]=ws[g*n+col];
      anchor=min(anchor,code[g]);
      if(code[g]==255) flag|=2;
      if(code[g]==0) flag|=4;
    }
    #pragma unroll
    for(int g=0;g<32;++g) {
      // Any delta>=14 is already unsafe: 131072*16384 > INT32_MAX.
      // Clamping keeps even invalid/unsafe inputs free of shift/overflow UB.
      unsigned factor=1u<<min(code[g]-anchor,14u);
      meta[g*n+col]=int32_t(factor);
      sum+=factor; // <=32*16384, exact uint32.
    }
    meta[32*n+col]=int32_t(anchor);
    if(sum>16383) flag|=1;
  }
  #pragma unroll
  for(int d=16;d>0;d/=2) flag|=__shfl_xor_sync(0xffffffff,flag,d);
  __shared__ unsigned warp_flags[4];
  if((threadIdx.x&31)==0) warp_flags[threadIdx.x/32]=flag;
  __syncthreads();
  if(threadIdx.x==0)
    block_status[blockIdx.x]=warp_flags[0]|warp_flags[1]|warp_flags[2]|warp_flags[3];
}
