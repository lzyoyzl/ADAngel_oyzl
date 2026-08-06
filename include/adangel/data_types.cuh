#pragma once

#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

namespace adangel {

constexpr int kMxBlock = 32;
constexpr int kUe8m0Bias = 127;

__host__ __device__ inline float decode_e2m1(uint8_t code) {
  constexpr float table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  float value = table[code & 7];
  return (code & 8) ? -value : value;
}

__host__ __device__ inline int8_t e2m1_to_int8_base(uint8_t code) {
  constexpr int8_t table[8] = {0, 1, 2, 3, 4, 6, 8, 12};
  int8_t value = table[code & 7];
  return (code & 8) ? -value : value;
}

__device__ inline float decode_ue8m0(uint8_t code) {
  return ldexpf(1.0f, static_cast<int>(code) - kUe8m0Bias);
}

}  // namespace adangel
