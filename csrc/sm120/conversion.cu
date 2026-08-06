#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <torch/extension.h>

#include "adangel/data_types.cuh"

// These kernels are deliberately allocation-free. The production binding supplies preallocated
// output tensors so CUDA Event regions contain only the conversion work.
extern "C" __global__ void adangel_mxfp4_to_fp16(
    const uint8_t* packed, const uint8_t* scales, half* output, int rows, int k) {
  const int row = static_cast<int>(blockIdx.y);
  const int pair = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int pairs_per_row = k / 2;
  if (row >= rows || pair >= pairs_per_row) return;
  const uint8_t byte = packed[row * pairs_per_row + pair];
  const float scale =
      adangel::decode_ue8m0(scales[row * (k / 32) + pair / 16]);
  const half low = __float2half_rn(adangel::decode_e2m1(byte & 0xF) * scale);
  const half high = __float2half_rn(adangel::decode_e2m1(byte >> 4) * scale);
  reinterpret_cast<half2*>(output)[row * pairs_per_row + pair] =
      __halves2half2(low, high);
}

extern "C" __global__ void adangel_int8_to_fp16(
    const int8_t* input, const float* scales, half* output, int rows, int k) {
  const int row = static_cast<int>(blockIdx.y);
  const int pair = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int pairs_per_row = k / 2;
  if (row >= rows || pair >= pairs_per_row) return;
  const char2 values = reinterpret_cast<const char2*>(input)[row * pairs_per_row + pair];
  const float scale = scales[row];
  const half low = __float2half_rn(static_cast<float>(values.x) * scale);
  const half high = __float2half_rn(static_cast<float>(values.y) * scale);
  reinterpret_cast<half2*>(output)[row * pairs_per_row + pair] =
      __halves2half2(low, high);
}

extern "C" __global__ void adangel_mxfp4_to_int8(
    const uint8_t* packed, int8_t* output, int rows, int k) {
  const int row = static_cast<int>(blockIdx.y);
  const int pair = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int pairs_per_row = k / 2;
  if (row >= rows || pair >= pairs_per_row) return;
  const uint8_t byte = packed[row * pairs_per_row + pair];
  char2 values;
  values.x = adangel::e2m1_to_int8_base(byte & 0xF);
  values.y = adangel::e2m1_to_int8_base(byte >> 4);
  reinterpret_cast<char2*>(output)[row * pairs_per_row + pair] = values;
}

namespace {

void check_launch(const char* operation) {
  cudaError_t status = cudaGetLastError();
  TORCH_CHECK(status == cudaSuccess, operation, " launch failed: ", cudaGetErrorString(status));
}

}  // namespace

void adangel_launch_mxfp4_to_fp16(
    const at::Tensor& packed,
    const at::Tensor& scales,
    at::Tensor& output,
    cudaStream_t stream) {
  const int rows = static_cast<int>(output.size(0));
  const int k = static_cast<int>(output.size(1));
  const int pairs_per_row = k / 2;
  constexpr int threads = 256;
  dim3 blocks((pairs_per_row + threads - 1) / threads, rows);
  adangel_mxfp4_to_fp16<<<blocks, threads, 0, stream>>>(
      packed.data_ptr<uint8_t>(),
      scales.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      rows,
      k);
  check_launch("MXFP4-to-FP16");
}

void adangel_launch_int8_to_fp16(
    const at::Tensor& input,
    const at::Tensor& scales,
    at::Tensor& output,
    cudaStream_t stream) {
  const int rows = static_cast<int>(input.size(0));
  const int k = static_cast<int>(input.size(1));
  const int pairs_per_row = k / 2;
  constexpr int threads = 256;
  dim3 blocks((pairs_per_row + threads - 1) / threads, rows);
  adangel_int8_to_fp16<<<blocks, threads, 0, stream>>>(
      input.data_ptr<int8_t>(),
      scales.data_ptr<float>(),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      rows,
      k);
  check_launch("INT8-to-FP16");
}

void adangel_launch_mxfp4_to_int8(
    const at::Tensor& packed,
    at::Tensor& output,
    cudaStream_t stream) {
  const int rows = static_cast<int>(output.size(0));
  const int k = static_cast<int>(output.size(1));
  const int pairs_per_row = k / 2;
  constexpr int threads = 256;
  dim3 blocks((pairs_per_row + threads - 1) / threads, rows);
  adangel_mxfp4_to_int8<<<blocks, threads, 0, stream>>>(
      packed.data_ptr<uint8_t>(),
      output.data_ptr<int8_t>(),
      rows,
      k);
  check_launch("MXFP4-to-INT8");
}
