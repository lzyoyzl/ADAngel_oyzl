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

extern "C" __global__ void adangel_mxfp4_to_q4(
    const uint8_t* packed, uint8_t* output, int rows, int pairs_per_row) {
  const int pair = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int row = static_cast<int>(blockIdx.y);
  if (row >= rows || pair >= pairs_per_row) return;
  const uint8_t byte = packed[row * pairs_per_row + pair];
  const uint8_t low = static_cast<uint8_t>(
      adangel::e2m1_to_q4(byte & 0xF)) & 0xF;
  const uint8_t high = static_cast<uint8_t>(
      adangel::e2m1_to_q4(byte >> 4)) & 0xF;
  output[row * pairs_per_row + pair] = low | (high << 4);
}

// Paper Split representation: the first M rows hold unsigned low nibbles;
// the next M rows hold signed high nibbles as raw two's-complement bits.
extern "C" __global__ void adangel_split_int8_to_int4(
    const int8_t* input, uint8_t* output, int rows, int pairs_per_row) {
  const int pair = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int row = static_cast<int>(blockIdx.y);
  if (row >= rows || pair >= pairs_per_row) return;
  const uchar2 raw = reinterpret_cast<const uchar2*>(input)[
      row * pairs_per_row + pair];
  output[row * pairs_per_row + pair] =
      (raw.x & 0xF) | ((raw.y & 0xF) << 4);
  output[(rows + row) * pairs_per_row + pair] =
      (raw.x >> 4) | ((raw.y >> 4) << 4);
}

extern "C" __global__ void adangel_int8_bitplanes(
    const int8_t* input,
    uint32_t* output,
    int rows,
    int words_per_row) {
  const int word = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int row = static_cast<int>(blockIdx.y);
  const int plane = static_cast<int>(blockIdx.z);
  if (row >= rows || word >= words_per_row) return;
  uint32_t packed = 0;
#pragma unroll
  for (int bit = 0; bit < 32; ++bit) {
    const uint8_t raw = static_cast<uint8_t>(
        input[row * words_per_row * 32 + word * 32 + bit]);
    packed |= static_cast<uint32_t>((raw >> plane) & 1u) << bit;
  }
  output[(plane * rows + row) * words_per_row + word] = packed;
}

extern "C" __global__ void adangel_q4_bitplanes(
    const uint8_t* input,
    uint32_t* output,
    int rows,
    int words_per_row) {
  const int word = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int row = static_cast<int>(blockIdx.y);
  const int plane = static_cast<int>(blockIdx.z);
  if (row >= rows || word >= words_per_row) return;
  uint32_t packed = 0;
#pragma unroll
  for (int bit = 0; bit < 32; ++bit) {
    const int element = word * 32 + bit;
    const uint8_t byte = input[
        row * words_per_row * 16 + (element >> 1)];
    const uint8_t nibble = (element & 1) ? (byte >> 4) : (byte & 0xF);
    packed |= static_cast<uint32_t>((nibble >> plane) & 1u) << bit;
  }
  output[(plane * rows + row) * words_per_row + word] = packed;
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

void adangel_launch_mxfp4_to_q4(
    const at::Tensor& packed,
    at::Tensor& output,
    cudaStream_t stream) {
  const int rows = static_cast<int>(packed.size(0));
  const int pairs_per_row = static_cast<int>(packed.size(1));
  constexpr int threads = 256;
  dim3 blocks((pairs_per_row + threads - 1) / threads, rows);
  adangel_mxfp4_to_q4<<<blocks, threads, 0, stream>>>(
      packed.data_ptr<uint8_t>(),
      output.data_ptr<uint8_t>(),
      rows,
      pairs_per_row);
  check_launch("MXFP4-to-Q4");
}

void adangel_launch_split_int8_to_int4(
    const at::Tensor& input,
    at::Tensor& output,
    cudaStream_t stream) {
  const int rows = static_cast<int>(input.size(0));
  const int pairs_per_row = static_cast<int>(input.size(1) / 2);
  constexpr int threads = 256;
  dim3 blocks((pairs_per_row + threads - 1) / threads, rows);
  adangel_split_int8_to_int4<<<blocks, threads, 0, stream>>>(
      input.data_ptr<int8_t>(), output.data_ptr<uint8_t>(), rows, pairs_per_row);
  check_launch("INT8 Split-to-INT4");
}

void adangel_launch_int8_bitplanes(
    const at::Tensor& input,
    at::Tensor& output,
    cudaStream_t stream) {
  const int rows = static_cast<int>(input.size(0));
  const int words_per_row = static_cast<int>(input.size(1) / 32);
  constexpr int threads = 128;
  dim3 blocks((words_per_row + threads - 1) / threads, rows, 8);
  adangel_int8_bitplanes<<<blocks, threads, 0, stream>>>(
      input.data_ptr<int8_t>(),
      reinterpret_cast<uint32_t*>(output.data_ptr<int32_t>()),
      rows,
      words_per_row);
  check_launch("INT8-to-B1-planes");
}

void adangel_launch_q4_bitplanes(
    const at::Tensor& packed,
    at::Tensor& output,
    cudaStream_t stream) {
  const int rows = static_cast<int>(packed.size(0));
  const int words_per_row = static_cast<int>(packed.size(1) * 2 / 32);
  constexpr int threads = 128;
  dim3 blocks((words_per_row + threads - 1) / threads, rows, 4);
  adangel_q4_bitplanes<<<blocks, threads, 0, stream>>>(
      packed.data_ptr<uint8_t>(),
      reinterpret_cast<uint32_t*>(output.data_ptr<int32_t>()),
      rows,
      words_per_row);
  check_launch("Q4-to-B1-planes");
}
