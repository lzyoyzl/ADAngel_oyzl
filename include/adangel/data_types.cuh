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

// Q=4 fixed-point mapping used by O3/O4.  F=Q-4=0, so this is RNE of
// the private E2M1 value followed by two's-complement Q4 packing.
__host__ __device__ inline int8_t e2m1_to_q4(uint8_t code) {
  constexpr int8_t table[8] = {0, 0, 1, 2, 2, 3, 4, 6};
  int8_t value = table[code & 7];
  return (code & 8) ? -value : value;
}

// OCP E2M1 round-to-nearest, ties-to-even encoder. The code parity is the
// significand parity at every positive midpoint.
__host__ __device__ inline uint8_t encode_e2m1_rne(float value) {
  const bool negative = signbit(value) && value != 0.0f;
  const float magnitude = fabsf(value);
  uint8_t code = 0;
  if (magnitude <= 0.25f) code = 0;
  else if (magnitude < 0.75f) code = 1;
  else if (magnitude <= 1.25f) code = 2;
  else if (magnitude < 1.75f) code = 3;
  else if (magnitude <= 2.5f) code = 4;
  else if (magnitude < 3.5f) code = 5;
  else if (magnitude <= 5.0f) code = 6;
  else code = 7;
  return static_cast<uint8_t>(code | (negative ? 8 : 0));
}

__device__ inline float decode_ue8m0(uint8_t code) {
  return ldexpf(1.0f, static_cast<int>(code) - kUe8m0Bias);
}

}  // namespace adangel
