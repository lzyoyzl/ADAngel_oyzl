#include <cuda_fp16.h>
#include <stdint.h>

#include "adangel/data_types.cuh"

// These kernels are deliberately allocation-free. The production binding supplies preallocated
// output tensors so CUDA Event regions contain only the conversion work.
extern "C" __global__ void adangel_mxfp4_to_fp16(
    const uint8_t* packed, const uint8_t* scales, half* output, int rows, int k) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int count = rows * k;
  if (index >= count) return;
  int row = index / k;
  int col = index - row * k;
  uint8_t byte = packed[row * (k / 2) + col / 2];
  uint8_t code = (col & 1) ? byte >> 4 : byte & 0xF;
  float scale = adangel::decode_ue8m0(scales[row * (k / 32) + col / 32]);
  output[index] = __float2half_rn(adangel::decode_e2m1(code) * scale);
}

extern "C" __global__ void adangel_int8_to_fp16(
    const int8_t* input, const float* scales, half* output, int rows, int k) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * k) return;
  int row = index / k;
  output[index] = __float2half_rn(static_cast<float>(input[index]) * scales[row]);
}

extern "C" __global__ void adangel_mxfp4_to_int8(
    const uint8_t* packed, int8_t* output, int rows, int k) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * k) return;
  int row = index / k;
  int col = index - row * k;
  uint8_t byte = packed[row * (k / 2) + col / 2];
  output[index] = adangel::e2m1_to_int8_base((col & 1) ? byte >> 4 : byte & 0xF);
}
