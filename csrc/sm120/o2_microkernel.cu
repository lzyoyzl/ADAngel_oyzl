#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdint.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Exact PTX ISA probe for the target O2 instruction. A/B fragment registers and the per-lane
// scale words are supplied by the layout test. This is not the 4096^3 performance kernel.
extern "C" __global__ void adangel_o2_mxfp4_layout_probe(
    const uint32_t* a, const uint32_t* b, const uint32_t* sfa, const uint32_t* sfb, float* d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  int lane = threadIdx.x & 31;
  float x0 = 0, x1 = 0, x2 = 0, x3 = 0;
  uint32_t a0 = a[lane * 4 + 0], a1 = a[lane * 4 + 1];
  uint32_t a2 = a[lane * 4 + 2], a3 = a[lane * 4 + 3];
  uint32_t b0 = b[lane * 2 + 0], b1 = b[lane * 2 + 1];
  uint32_t scale_a = sfa[lane], scale_b = sfb[lane];
  asm volatile(
      "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X."
      "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
      "%10, {0,0}, %11, {0,0};"
      : "+f"(x0), "+f"(x1), "+f"(x2), "+f"(x3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "r"(scale_a), "r"(scale_b));
  d[lane * 4 + 0] = x0;
  d[lane * 4 + 1] = x1;
  d[lane * 4 + 2] = x2;
  d[lane * 4 + 3] = x3;
#endif
}

namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

float ue8m0(uint8_t code) { return std::ldexp(1.0f, static_cast<int>(code) - 127); }

}  // namespace

std::pair<bool, float> adangel_verify_o2_layout_cuda() {
  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
  if (properties.major != 12 || properties.minor != 0) {
    throw std::runtime_error("O2 layout verification requires compute capability 12.0");
  }

  std::vector<uint32_t> a(32 * 4, 0x11111111u);  // Every E2M1 nibble is +0.5.
  std::vector<uint32_t> b(32 * 2, 0x11111111u);
  std::vector<uint32_t> scale_a(32, 0), scale_b(32, 0);
  uint8_t logical_a[16][2]{};
  uint8_t logical_b[2][8]{};
  for (int row = 0; row < 16; ++row) {
    logical_a[row][0] = static_cast<uint8_t>(125 + row % 4);
    logical_a[row][1] = static_cast<uint8_t>(126 + row % 3);
  }
  for (int col = 0; col < 8; ++col) {
    logical_b[0][col] = static_cast<uint8_t>(125 + col % 3);
    logical_b[1][col] = static_cast<uint8_t>(126 + col % 2);
  }
  for (int q = 0; q < 8; ++q) {
    scale_a[4 * q] = static_cast<uint32_t>(logical_a[q][0]) |
                     (static_cast<uint32_t>(logical_a[q][1]) << 8);
    scale_a[4 * q + 1] = static_cast<uint32_t>(logical_a[q + 8][0]) |
                         (static_cast<uint32_t>(logical_a[q + 8][1]) << 8);
    scale_b[4 * q] = static_cast<uint32_t>(logical_b[0][q]) |
                     (static_cast<uint32_t>(logical_b[1][q]) << 8);
  }

  uint32_t *device_a = nullptr, *device_b = nullptr, *device_scale_a = nullptr, *device_scale_b = nullptr;
  float* device_d = nullptr;
  check_cuda(cudaMalloc(&device_a, a.size() * sizeof(uint32_t)), "cudaMalloc A");
  check_cuda(cudaMalloc(&device_b, b.size() * sizeof(uint32_t)), "cudaMalloc B");
  check_cuda(cudaMalloc(&device_scale_a, scale_a.size() * sizeof(uint32_t)), "cudaMalloc SFA");
  check_cuda(cudaMalloc(&device_scale_b, scale_b.size() * sizeof(uint32_t)), "cudaMalloc SFB");
  check_cuda(cudaMalloc(&device_d, 32 * 4 * sizeof(float)), "cudaMalloc D");
  check_cuda(cudaMemcpy(device_a, a.data(), a.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), "copy A");
  check_cuda(cudaMemcpy(device_b, b.data(), b.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), "copy B");
  check_cuda(cudaMemcpy(device_scale_a, scale_a.data(), scale_a.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), "copy SFA");
  check_cuda(cudaMemcpy(device_scale_b, scale_b.data(), scale_b.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), "copy SFB");

  adangel_o2_mxfp4_layout_probe<<<1, 32>>>(device_a, device_b, device_scale_a, device_scale_b, device_d);
  check_cuda(cudaGetLastError(), "launch O2 layout probe");
  std::vector<float> output(32 * 4);
  check_cuda(cudaMemcpy(output.data(), device_d, output.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy D");
  cudaFree(device_a); cudaFree(device_b); cudaFree(device_scale_a); cudaFree(device_scale_b); cudaFree(device_d);

  float max_abs_error = 0.0f;
  for (int lane = 0; lane < 32; ++lane) {
    int row0 = lane >> 2;
    int row1 = row0 + 8;
    int col0 = (lane & 3) * 2;
    int rows[4] = {row0, row0, row1, row1};
    int cols[4] = {col0, col0 + 1, col0, col0 + 1};
    for (int value = 0; value < 4; ++value) {
      int row = rows[value], col = cols[value];
      // 32 products per group and 0.5*0.5 per product gives the leading factor 8.
      float expected = 8.0f * (
          ue8m0(logical_a[row][0]) * ue8m0(logical_b[0][col]) +
          ue8m0(logical_a[row][1]) * ue8m0(logical_b[1][col]));
      max_abs_error = std::max(max_abs_error, std::abs(output[lane * 4 + value] - expected));
    }
  }
  return {max_abs_error <= 1.0e-3f, max_abs_error};
}
