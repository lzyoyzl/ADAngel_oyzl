// v115 ISA cost gate only. No Tensor Core kernel, launch or timing claim.
// Packed weights are [f,16*f,0,0], 0 <= f <= 15. Both dots are exact S16.
#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__ void adangel_dp2a_recompose_cost(
    const int* low, const int* high, const uint32_t* packed_factor,
    const int* accumulator, int* output) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  uint32_t dots;
  asm("prmt.b32 %0, %1, %2, 0x5410;"
      : "=r"(dots) : "r"(low[i]), "r"(high[i]));
  int result;
  asm("dp2a.lo.s32.u32 %0, %1, %2, %3;"
      : "=r"(result) : "r"(dots), "r"(packed_factor[i]), "r"(accumulator[i]));
  output[i]=result;
}

extern "C" __global__ void adangel_scalar_recompose_cost(
    const int* low, const int* high, const uint32_t* factor,
    const int* accumulator, int* output) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  output[i]=accumulator[i]+(low[i]+16*high[i])*static_cast<int>(factor[i]);
}
