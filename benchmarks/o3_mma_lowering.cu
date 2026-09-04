#include <cuda_runtime.h>

#include <cute/arch/mma_sm80.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kWarpSize = 32;

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + cudaGetErrorString(status));
  }
}

class Event {
 public:
  Event() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
  ~Event() { cudaEventDestroy(event_); }
  Event(Event const&) = delete;
  Event& operator=(Event const&) = delete;
  cudaEvent_t get() const { return event_; }

 private:
  cudaEvent_t event_{};
};

template <class Mma, int Chains>
__device__ __forceinline__ uint32_t run_mma_loop(
    uint32_t seed, int iterations) {
  static_assert(Chains == 1 || Chains == 4, "microbenchmark supports one or four chains");

  // These values are loaded from global memory before the loop, so ptxas
  // cannot constant-fold the MMA sequence.  Every nibble/byte is deliberately
  // small enough to keep the non-saturating S32 accumulators in range.
  uint32_t a0 = seed;
  uint32_t a1 = seed ^ 0x00010001u;
  uint32_t a2 = seed ^ 0x00100010u;
  uint32_t a3 = seed ^ 0x01000100u;
  uint32_t b0 = seed ^ 0x00000101u;
  uint32_t b1 = seed ^ 0x00010000u;
  uint32_t accumulators[Chains][4];
#pragma unroll
  for (int chain = 0; chain < Chains; ++chain) {
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      // Different C fragments are essential: identical zero-initialized
      // chains let ptxas merge the chains or serialize them through a common
      // temporary, which defeats the throughput experiment.
      accumulators[chain][item] =
          seed ^ static_cast<uint32_t>(0x01020408u * (chain + 1) + item);
    }
  }

  // Keep the loop rolled: this measures steady-state instruction issue while
  // avoiding an enormous cubin.  Four independent accumulator chains expose
  // throughput; one chain exposes dependency latency.
#pragma unroll 1
  for (int iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll
    for (int chain = 0; chain < Chains; ++chain) {
      uint32_t d0, d1, d2, d3;
      Mma::fma(
          d0, d1, d2, d3,
          a0, a1, a2, a3, b0, b1,
          accumulators[chain][0], accumulators[chain][1],
          accumulators[chain][2], accumulators[chain][3]);
      accumulators[chain][0] = d0;
      accumulators[chain][1] = d1;
      accumulators[chain][2] = d2;
      accumulators[chain][3] = d3;
    }
  }

  uint32_t checksum = 2166136261u;
#pragma unroll
  for (int chain = 0; chain < Chains; ++chain) {
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      checksum = (checksum ^ accumulators[chain][item]) * 16777619u;
    }
  }
  return checksum;
}

template <class Mma, int Chains>
__device__ __forceinline__ uint32_t run_mma_loop_m8n8(
    uint32_t seed, int iterations) {
  static_assert(Chains == 1 || Chains == 4, "microbenchmark supports one or four chains");
  uint32_t a0 = seed;
  uint32_t b0 = seed ^ 0x00000101u;
  uint32_t accumulators[Chains][2];
#pragma unroll
  for (int chain = 0; chain < Chains; ++chain) {
#pragma unroll
    for (int item = 0; item < 2; ++item) {
      accumulators[chain][item] =
          seed ^ static_cast<uint32_t>(0x01020408u * (chain + 1) + item);
    }
  }
#pragma unroll 1
  for (int iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll
    for (int chain = 0; chain < Chains; ++chain) {
      uint32_t d0, d1;
      Mma::fma(
          d0, d1, a0, b0,
          accumulators[chain][0], accumulators[chain][1]);
      accumulators[chain][0] = d0;
      accumulators[chain][1] = d1;
    }
  }
  uint32_t checksum = 2166136261u;
#pragma unroll
  for (int chain = 0; chain < Chains; ++chain) {
#pragma unroll
    for (int item = 0; item < 2; ++item) {
      checksum = (checksum ^ accumulators[chain][item]) * 16777619u;
    }
  }
  return checksum;
}

template <class Mma, int Chains>
__device__ __forceinline__ uint32_t run_mma_loop_m16n8k32(
    uint32_t seed, int iterations) {
  static_assert(Chains == 1 || Chains == 4, "microbenchmark supports one or four chains");
  uint32_t a0 = seed;
  uint32_t a1 = seed ^ 0x00010001u;
  uint32_t b0 = seed ^ 0x00000101u;
  uint32_t accumulators[Chains][4];
#pragma unroll
  for (int chain = 0; chain < Chains; ++chain) {
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      accumulators[chain][item] =
          seed ^ static_cast<uint32_t>(0x01020408u * (chain + 1) + item);
    }
  }
#pragma unroll 1
  for (int iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll
    for (int chain = 0; chain < Chains; ++chain) {
      uint32_t d0, d1, d2, d3;
      Mma::fma(
          d0, d1, d2, d3, a0, a1, b0,
          accumulators[chain][0], accumulators[chain][1],
          accumulators[chain][2], accumulators[chain][3]);
      accumulators[chain][0] = d0;
      accumulators[chain][1] = d1;
      accumulators[chain][2] = d2;
      accumulators[chain][3] = d3;
    }
  }
  uint32_t checksum = 2166136261u;
#pragma unroll
  for (int chain = 0; chain < Chains; ++chain) {
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      checksum = (checksum ^ accumulators[chain][item]) * 16777619u;
    }
  }
  return checksum;
}

template <class Mma, int Chains>
__device__ __forceinline__ void run_kernel_body(
    const uint32_t* input, uint32_t* output, int iterations) {
  const int thread = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint32_t seed = input[thread];
  output[thread] = run_mma_loop<Mma, Chains>(seed, iterations);
}

extern "C" __global__ __launch_bounds__(256)
void adangel_o3_mma_micro_u4s4_c1(
    const uint32_t* input, uint32_t* output, int iterations) {
  run_kernel_body<cute::SM80_16x8x64_S32U4S4S32_TN, 1>(input, output, iterations);
}

extern "C" __global__ __launch_bounds__(256)
void adangel_o3_mma_micro_u4s4_c4(
    const uint32_t* input, uint32_t* output, int iterations) {
  run_kernel_body<cute::SM80_16x8x64_S32U4S4S32_TN, 4>(input, output, iterations);
}

extern "C" __global__ __launch_bounds__(256)
void adangel_o3_mma_micro_s4s4_c1(
    const uint32_t* input, uint32_t* output, int iterations) {
  run_kernel_body<cute::SM80_16x8x64_S32S4S4S32_TN, 1>(input, output, iterations);
}

extern "C" __global__ __launch_bounds__(256)
void adangel_o3_mma_micro_s4s4_c4(
    const uint32_t* input, uint32_t* output, int iterations) {
  run_kernel_body<cute::SM80_16x8x64_S32S4S4S32_TN, 4>(input, output, iterations);
}

extern "C" __global__ __launch_bounds__(256)
void adangel_o3_mma_micro_u4s4_sat_c1(
    const uint32_t* input, uint32_t* output, int iterations) {
  run_kernel_body<cute::SM80_16x8x64_S32U4S4S32_TN_SATURATE, 1>(
      input, output, iterations);
}

extern "C" __global__ __launch_bounds__(256)
void adangel_o3_mma_micro_u4s4_sat_c4(
    const uint32_t* input, uint32_t* output, int iterations) {
  run_kernel_body<cute::SM80_16x8x64_S32U4S4S32_TN_SATURATE, 4>(
      input, output, iterations);
}

extern "C" __global__ __launch_bounds__(256)
void adangel_o3_mma_micro_s4s4_sat_c1(
    const uint32_t* input, uint32_t* output, int iterations) {
  run_kernel_body<cute::SM80_16x8x64_S32S4S4S32_TN_SATURATE, 1>(
      input, output, iterations);
}

extern "C" __global__ __launch_bounds__(256)
void adangel_o3_mma_micro_s4s4_sat_c4(
    const uint32_t* input, uint32_t* output, int iterations) {
  run_kernel_body<cute::SM80_16x8x64_S32S4S4S32_TN_SATURATE, 4>(
      input, output, iterations);
}

extern "C" __global__ __launch_bounds__(256)
void adangel_o3_mma_micro_s8s8_c1(
    const uint32_t* input, uint32_t* output, int iterations) {
  run_kernel_body<cute::SM80_16x8x32_S32S8S8S32_TN, 1>(input, output, iterations);
}

extern "C" __global__ __launch_bounds__(256)
void adangel_o3_mma_micro_s8s8_c4(
    const uint32_t* input, uint32_t* output, int iterations) {
  run_kernel_body<cute::SM80_16x8x32_S32S8S8S32_TN, 4>(input, output, iterations);
}

extern "C" __global__ __launch_bounds__(256)
void adangel_o3_mma_micro_s4u4_c1(
    const uint32_t* input, uint32_t* output, int iterations) {
  run_kernel_body<cute::SM80_16x8x64_S32S4U4S32_TN, 1>(input, output, iterations);
}

extern "C" __global__ __launch_bounds__(256)
void adangel_o3_mma_micro_s4u4_c4(
    const uint32_t* input, uint32_t* output, int iterations) {
  run_kernel_body<cute::SM80_16x8x64_S32S4U4S32_TN, 4>(input, output, iterations);
}

template <int Chains>
__device__ __forceinline__ uint32_t run_split_pair_loop(
    uint32_t seed, int iterations) {
  uint32_t low_a0 = seed;
  uint32_t low_a1 = seed ^ 0x00010001u;
  uint32_t low_a2 = seed ^ 0x00100010u;
  uint32_t low_a3 = seed ^ 0x01000100u;
  uint32_t high_a0 = seed ^ 0x11111111u;
  uint32_t high_a1 = seed ^ 0x22222222u;
  uint32_t high_a2 = seed ^ 0x44444444u;
  uint32_t high_a3 = seed ^ 0x33333333u;
  uint32_t b0 = seed ^ 0x00000101u;
  uint32_t b1 = seed ^ 0x00010000u;
  uint32_t low[Chains][4];
  uint32_t high[Chains][4];
#pragma unroll
  for (int chain = 0; chain < Chains; ++chain) {
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      low[chain][item] =
          seed ^ static_cast<uint32_t>(0x01020408u * (chain + 1) + item);
      high[chain][item] =
          seed ^ static_cast<uint32_t>(0x10204080u * (chain + 1) + item);
    }
  }

#pragma unroll 1
  for (int iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll
    for (int chain = 0; chain < Chains; ++chain) {
      uint32_t l0, l1, l2, l3, h0, h1, h2, h3;
      // Keep the two formal O3 PTX operations in one asm block and share the
      // exact same signed-Q4 B fragment.  This tests whether ptxas can reuse
      // operand expansion across the low/high pair without changing O3 math.
      asm volatile(
          "mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32 "
          "{%0,%1,%2,%3},{%8,%9,%10,%11},{%16,%17},{%18,%19,%20,%21};\n"
          "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
          "{%4,%5,%6,%7},{%12,%13,%14,%15},{%16,%17},{%22,%23,%24,%25};\n"
          : "=&r"(l0), "=&r"(l1), "=&r"(l2), "=&r"(l3),
            "=&r"(h0), "=&r"(h1), "=&r"(h2), "=&r"(h3)
          : "r"(low_a0), "r"(low_a1), "r"(low_a2), "r"(low_a3),
            "r"(high_a0), "r"(high_a1), "r"(high_a2), "r"(high_a3),
            "r"(b0), "r"(b1),
            "r"(low[chain][0]), "r"(low[chain][1]),
            "r"(low[chain][2]), "r"(low[chain][3]),
            "r"(high[chain][0]), "r"(high[chain][1]),
            "r"(high[chain][2]), "r"(high[chain][3]));
      low[chain][0] = l0;
      low[chain][1] = l1;
      low[chain][2] = l2;
      low[chain][3] = l3;
      high[chain][0] = h0;
      high[chain][1] = h1;
      high[chain][2] = h2;
      high[chain][3] = h3;
    }
  }

  uint32_t checksum = 2166136261u;
#pragma unroll
  for (int chain = 0; chain < Chains; ++chain) {
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      checksum = (checksum ^ low[chain][item]) * 16777619u;
      checksum = (checksum ^ high[chain][item]) * 16777619u;
    }
  }
  return checksum;
}

template <int Chains>
__device__ __forceinline__ void run_split_pair_kernel_body(
    const uint32_t* input, uint32_t* output, int iterations) {
  const int thread = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  output[thread] = run_split_pair_loop<Chains>(input[thread], iterations);
}

extern "C" __global__ __launch_bounds__(256)
void adangel_o3_mma_micro_split_pair_c1(
    const uint32_t* input, uint32_t* output, int iterations) {
  run_split_pair_kernel_body<1>(input, output, iterations);
}

extern "C" __global__ __launch_bounds__(256)
void adangel_o3_mma_micro_split_pair_c4(
    const uint32_t* input, uint32_t* output, int iterations) {
  run_split_pair_kernel_body<4>(input, output, iterations);
}

template <class Mma, int Chains>
__device__ __forceinline__ void run_m8n8_kernel_body(
    const uint32_t* input, uint32_t* output, int iterations) {
  const int thread = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  output[thread] = run_mma_loop_m8n8<Mma, Chains>(input[thread], iterations);
}

extern "C" __global__ __launch_bounds__(256)
void adangel_o3_mma_micro_m8n8_u4s4_c1(
    const uint32_t* input, uint32_t* output, int iterations) {
  run_m8n8_kernel_body<cute::SM80_8x8x32_S32U4S4S32_TN, 1>(input, output, iterations);
}

extern "C" __global__ __launch_bounds__(256)
void adangel_o3_mma_micro_m8n8_u4s4_c4(
    const uint32_t* input, uint32_t* output, int iterations) {
  run_m8n8_kernel_body<cute::SM80_8x8x32_S32U4S4S32_TN, 4>(input, output, iterations);
}

extern "C" __global__ __launch_bounds__(256)
void adangel_o3_mma_micro_m8n8_s4s4_c1(
    const uint32_t* input, uint32_t* output, int iterations) {
  run_m8n8_kernel_body<cute::SM80_8x8x32_S32S4S4S32_TN, 1>(input, output, iterations);
}

extern "C" __global__ __launch_bounds__(256)
void adangel_o3_mma_micro_m8n8_s4s4_c4(
    const uint32_t* input, uint32_t* output, int iterations) {
  run_m8n8_kernel_body<cute::SM80_8x8x32_S32S4S4S32_TN, 4>(input, output, iterations);
}

extern "C" __global__ __launch_bounds__(256)
void adangel_o3_mma_micro_m8n8_s8s8_c1(
    const uint32_t* input, uint32_t* output, int iterations) {
  run_m8n8_kernel_body<cute::SM80_8x8x16_S32S8S8S32_TN, 1>(input, output, iterations);
}

extern "C" __global__ __launch_bounds__(256)
void adangel_o3_mma_micro_m8n8_s8s8_c4(
    const uint32_t* input, uint32_t* output, int iterations) {
  run_m8n8_kernel_body<cute::SM80_8x8x16_S32S8S8S32_TN, 4>(input, output, iterations);
}

template <class Mma, int Chains>
__device__ __forceinline__ void run_m16n8k32_kernel_body(
    const uint32_t* input, uint32_t* output, int iterations) {
  const int thread = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  output[thread] = run_mma_loop_m16n8k32<Mma, Chains>(input[thread], iterations);
}

extern "C" __global__ __launch_bounds__(256)
void adangel_o3_mma_micro_m16n8k32_u4s4_c1(
    const uint32_t* input, uint32_t* output, int iterations) {
  run_m16n8k32_kernel_body<cute::SM80_16x8x32_S32U4S4S32_TN, 1>(
      input, output, iterations);
}

extern "C" __global__ __launch_bounds__(256)
void adangel_o3_mma_micro_m16n8k32_u4s4_c4(
    const uint32_t* input, uint32_t* output, int iterations) {
  run_m16n8k32_kernel_body<cute::SM80_16x8x32_S32U4S4S32_TN, 4>(
      input, output, iterations);
}

extern "C" __global__ __launch_bounds__(256)
void adangel_o3_mma_micro_m16n8k32_s4s4_c1(
    const uint32_t* input, uint32_t* output, int iterations) {
  run_m16n8k32_kernel_body<cute::SM80_16x8x32_S32S4S4S32_TN, 1>(
      input, output, iterations);
}

extern "C" __global__ __launch_bounds__(256)
void adangel_o3_mma_micro_m16n8k32_s4s4_c4(
    const uint32_t* input, uint32_t* output, int iterations) {
  run_m16n8k32_kernel_body<cute::SM80_16x8x32_S32S4S4S32_TN, 4>(
      input, output, iterations);
}

extern "C" __global__ __launch_bounds__(256)
void adangel_o3_mma_micro_m16n8k32_s8s8_c1(
    const uint32_t* input, uint32_t* output, int iterations) {
  run_m16n8k32_kernel_body<cute::SM80_16x8x16_S32S8S8S32_TN, 1>(
      input, output, iterations);
}

extern "C" __global__ __launch_bounds__(256)
void adangel_o3_mma_micro_m16n8k32_s8s8_c4(
    const uint32_t* input, uint32_t* output, int iterations) {
  run_m16n8k32_kernel_body<cute::SM80_16x8x16_S32S8S8S32_TN, 4>(
      input, output, iterations);
}

using Kernel = void (*)(const uint32_t*, uint32_t*, int);

struct Options {
  std::string kind = "u4s4";
  std::string shape = "m16n8k64";
  int chains = 4;
  int blocks = 512;
  int warps = 8;
  int iterations = 512;
  int warmup = 10;
  int repeats = 50;
};

int parse_positive(const char* value, const char* option, bool allow_zero = false) {
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed < (allow_zero ? 0 : 1) || parsed > 1000000) {
    throw std::runtime_error(std::string("invalid value for ") + option + ": " + value);
  }
  return static_cast<int>(parsed);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto require_value = [&]() -> const char* {
      if (++index >= argc) throw std::runtime_error("missing value after " + argument);
      return argv[index];
    };
    if (argument == "--kind") options.kind = require_value();
    else if (argument == "--shape") options.shape = require_value();
    else if (argument == "--chains") options.chains = parse_positive(require_value(), "--chains");
    else if (argument == "--blocks") options.blocks = parse_positive(require_value(), "--blocks");
    else if (argument == "--warps") options.warps = parse_positive(require_value(), "--warps");
    else if (argument == "--iterations") {
      options.iterations = parse_positive(require_value(), "--iterations");
    } else if (argument == "--warmup") {
      options.warmup = parse_positive(require_value(), "--warmup", true);
    } else if (argument == "--repeats") {
      options.repeats = parse_positive(require_value(), "--repeats");
    } else if (argument == "--help") {
      std::cout
          << "Usage: o3_mma_lowering [--kind u4s4|s4s4|u4s4_sat|s4s4_sat|s4u4|s8s8|split_pair] "
             "[--shape m16n8k64|m16n8k32|m8n8k32] [--chains 1|4] "
             "[--blocks N] [--warps N] [--iterations N] [--warmup N] [--repeats N]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  if (options.kind != "u4s4" && options.kind != "s4s4" &&
      options.kind != "u4s4_sat" && options.kind != "s4s4_sat" &&
      options.kind != "s4u4" && options.kind != "s8s8" &&
      options.kind != "split_pair") {
    throw std::runtime_error(
        "--kind must be u4s4, s4s4, u4s4_sat, s4s4_sat, s4u4, s8s8, or split_pair");
  }
  if (options.shape != "m16n8k64" && options.shape != "m16n8k32" &&
      options.shape != "m8n8k32") {
    throw std::runtime_error(
        "--shape must be m16n8k64, m16n8k32, or m8n8k32");
  }
  if (options.chains != 1 && options.chains != 4) {
    throw std::runtime_error("--chains must be 1 or 4");
  }
  if (options.warps > 8) {
    throw std::runtime_error("--warps must not exceed 8 (__launch_bounds__(256))");
  }
  if ((options.kind == "s4u4" || options.kind == "split_pair" ||
       options.kind == "u4s4_sat" || options.kind == "s4s4_sat") &&
      options.shape != "m16n8k64") {
    throw std::runtime_error(
        "s4u4, split_pair, and satfinite variants require --shape m16n8k64");
  }
  return options;
}

Kernel select_kernel(
    const Options& options, const char** symbol, int* mma_m, int* mma_n, int* mma_k) {
  *mma_m = options.shape == "m8n8k32" ? 8 : 16;
  *mma_n = 8;
  const bool short_k = options.shape != "m16n8k64";
  *mma_k = options.kind == "s8s8" ? (short_k ? 16 : 32)
                                      : (short_k ? 32 : 64);
  if (options.kind == "split_pair") {
    if (options.chains == 1) {
      *symbol = "adangel_o3_mma_micro_split_pair_c1";
      return adangel_o3_mma_micro_split_pair_c1;
    }
    *symbol = "adangel_o3_mma_micro_split_pair_c4";
    return adangel_o3_mma_micro_split_pair_c4;
  }
  if (options.kind == "s4u4") {
    if (options.chains == 1) {
      *symbol = "adangel_o3_mma_micro_s4u4_c1";
      return adangel_o3_mma_micro_s4u4_c1;
    }
    *symbol = "adangel_o3_mma_micro_s4u4_c4";
    return adangel_o3_mma_micro_s4u4_c4;
  }
  if (options.kind == "u4s4_sat") {
    if (options.chains == 1) {
      *symbol = "adangel_o3_mma_micro_u4s4_sat_c1";
      return adangel_o3_mma_micro_u4s4_sat_c1;
    }
    *symbol = "adangel_o3_mma_micro_u4s4_sat_c4";
    return adangel_o3_mma_micro_u4s4_sat_c4;
  }
  if (options.kind == "s4s4_sat") {
    if (options.chains == 1) {
      *symbol = "adangel_o3_mma_micro_s4s4_sat_c1";
      return adangel_o3_mma_micro_s4s4_sat_c1;
    }
    *symbol = "adangel_o3_mma_micro_s4s4_sat_c4";
    return adangel_o3_mma_micro_s4s4_sat_c4;
  }
  if (options.shape == "m8n8k32") {
    if (options.kind == "u4s4" && options.chains == 1) {
      *symbol = "adangel_o3_mma_micro_m8n8_u4s4_c1";
      return adangel_o3_mma_micro_m8n8_u4s4_c1;
    }
    if (options.kind == "u4s4" && options.chains == 4) {
      *symbol = "adangel_o3_mma_micro_m8n8_u4s4_c4";
      return adangel_o3_mma_micro_m8n8_u4s4_c4;
    }
    if (options.kind == "s4s4" && options.chains == 1) {
      *symbol = "adangel_o3_mma_micro_m8n8_s4s4_c1";
      return adangel_o3_mma_micro_m8n8_s4s4_c1;
    }
    if (options.kind == "s4s4" && options.chains == 4) {
      *symbol = "adangel_o3_mma_micro_m8n8_s4s4_c4";
      return adangel_o3_mma_micro_m8n8_s4s4_c4;
    }
    if (options.kind == "s8s8" && options.chains == 1) {
      *symbol = "adangel_o3_mma_micro_m8n8_s8s8_c1";
      return adangel_o3_mma_micro_m8n8_s8s8_c1;
    }
    *symbol = "adangel_o3_mma_micro_m8n8_s8s8_c4";
    return adangel_o3_mma_micro_m8n8_s8s8_c4;
  }
  if (options.shape == "m16n8k32") {
    if (options.kind == "u4s4" && options.chains == 1) {
      *symbol = "adangel_o3_mma_micro_m16n8k32_u4s4_c1";
      return adangel_o3_mma_micro_m16n8k32_u4s4_c1;
    }
    if (options.kind == "u4s4" && options.chains == 4) {
      *symbol = "adangel_o3_mma_micro_m16n8k32_u4s4_c4";
      return adangel_o3_mma_micro_m16n8k32_u4s4_c4;
    }
    if (options.kind == "s4s4" && options.chains == 1) {
      *symbol = "adangel_o3_mma_micro_m16n8k32_s4s4_c1";
      return adangel_o3_mma_micro_m16n8k32_s4s4_c1;
    }
    if (options.kind == "s4s4" && options.chains == 4) {
      *symbol = "adangel_o3_mma_micro_m16n8k32_s4s4_c4";
      return adangel_o3_mma_micro_m16n8k32_s4s4_c4;
    }
    if (options.kind == "s8s8" && options.chains == 1) {
      *symbol = "adangel_o3_mma_micro_m16n8k32_s8s8_c1";
      return adangel_o3_mma_micro_m16n8k32_s8s8_c1;
    }
    *symbol = "adangel_o3_mma_micro_m16n8k32_s8s8_c4";
    return adangel_o3_mma_micro_m16n8k32_s8s8_c4;
  }
  if (options.kind == "u4s4" && options.chains == 1) {
    *symbol = "adangel_o3_mma_micro_u4s4_c1";
    return adangel_o3_mma_micro_u4s4_c1;
  }
  if (options.kind == "u4s4" && options.chains == 4) {
    *symbol = "adangel_o3_mma_micro_u4s4_c4";
    return adangel_o3_mma_micro_u4s4_c4;
  }
  if (options.kind == "s4s4" && options.chains == 1) {
    *symbol = "adangel_o3_mma_micro_s4s4_c1";
    return adangel_o3_mma_micro_s4s4_c1;
  }
  if (options.kind == "s4s4" && options.chains == 4) {
    *symbol = "adangel_o3_mma_micro_s4s4_c4";
    return adangel_o3_mma_micro_s4s4_c4;
  }
  if (options.kind == "s8s8" && options.chains == 1) {
    *symbol = "adangel_o3_mma_micro_s8s8_c1";
    return adangel_o3_mma_micro_s8s8_c1;
  }
  *symbol = "adangel_o3_mma_micro_s8s8_c4";
  return adangel_o3_mma_micro_s8s8_c4;
}

double percentile(std::vector<float> values, double fraction) {
  std::sort(values.begin(), values.end());
  const double position = fraction * static_cast<double>(values.size() - 1);
  const auto lower = static_cast<std::size_t>(position);
  const auto upper = std::min(lower + 1, values.size() - 1);
  const double alpha = position - static_cast<double>(lower);
  return values[lower] * (1.0 - alpha) + values[upper] * alpha;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const char* kernel_symbol = nullptr;
    int mma_m = 0;
    int mma_n = 0;
    int mma_k = 0;
    const Kernel kernel = select_kernel(
        options, &kernel_symbol, &mma_m, &mma_n, &mma_k);

    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
    if (properties.major != 12 || properties.minor != 0) {
      throw std::runtime_error("this microbenchmark requires an SM120 GPU");
    }
    const int threads = options.warps * kWarpSize;
    const std::size_t elements = static_cast<std::size_t>(options.blocks) * threads;
    std::vector<uint32_t> host_input(elements, 0x01010101u);
    std::vector<uint32_t> host_output(elements);
    uint32_t* device_input = nullptr;
    uint32_t* device_output = nullptr;
    check_cuda(cudaMalloc(&device_input, elements * sizeof(uint32_t)), "cudaMalloc input");
    check_cuda(cudaMalloc(&device_output, elements * sizeof(uint32_t)), "cudaMalloc output");
    check_cuda(
        cudaMemcpy(
            device_input, host_input.data(), elements * sizeof(uint32_t),
            cudaMemcpyHostToDevice),
        "cudaMemcpy input");

    for (int iteration = 0; iteration < options.warmup; ++iteration) {
      kernel<<<options.blocks, threads>>>(device_input, device_output, options.iterations);
    }
    check_cuda(cudaGetLastError(), "warmup launch");
    check_cuda(cudaDeviceSynchronize(), "warmup synchronization");

    std::vector<float> milliseconds;
    milliseconds.reserve(options.repeats);
    Event begin;
    Event end;
    for (int repeat = 0; repeat < options.repeats; ++repeat) {
      check_cuda(cudaEventRecord(begin.get()), "cudaEventRecord begin");
      kernel<<<options.blocks, threads>>>(device_input, device_output, options.iterations);
      check_cuda(cudaEventRecord(end.get()), "cudaEventRecord end");
      check_cuda(cudaEventSynchronize(end.get()), "cudaEventSynchronize");
      float elapsed_ms = 0.0f;
      check_cuda(
          cudaEventElapsedTime(&elapsed_ms, begin.get(), end.get()),
          "cudaEventElapsedTime");
      milliseconds.push_back(elapsed_ms);
    }
    check_cuda(cudaGetLastError(), "timed launch");
    check_cuda(
        cudaMemcpy(
            host_output.data(), device_output, elements * sizeof(uint32_t),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy output");

    cudaFuncAttributes attributes{};
    check_cuda(cudaFuncGetAttributes(&attributes, kernel), "cudaFuncGetAttributes");
    const double mean_ms = std::accumulate(milliseconds.begin(), milliseconds.end(), 0.0) /
        static_cast<double>(milliseconds.size());
    const double median_ms = percentile(milliseconds, 0.5);
    const double p5_ms = percentile(milliseconds, 0.05);
    const double p95_ms = percentile(milliseconds, 0.95);
    const int mma_per_chain = options.kind == "split_pair" ? 2 : 1;
    const double mma_per_launch = static_cast<double>(options.blocks) * options.warps *
        options.iterations * options.chains * mma_per_chain;
    const double logical_operations =
        mma_per_launch * 2.0 * mma_m * mma_n * mma_k;
    const double logical_tops = logical_operations / (median_ms * 1.0e9);
    uint64_t checksum = 0;
    for (uint32_t value : host_output) checksum += value;

    std::cout << std::fixed << std::setprecision(9)
              << "{\n"
              << "  \"gpu\": \"" << properties.name << "\",\n"
              << "  \"compute_capability\": \"" << properties.major << "."
              << properties.minor << "\",\n"
              << "  \"kind\": \"" << options.kind << "\",\n"
              << "  \"shape\": \"" << options.shape << "\",\n"
              << "  \"chains\": " << options.chains << ",\n"
              << "  \"kernel_symbol\": \"" << kernel_symbol << "\",\n"
              << "  \"mma_shape\": \"m" << mma_m << "n" << mma_n << "k" << mma_k
              << "\",\n"
              << "  \"blocks\": " << options.blocks << ",\n"
              << "  \"warps_per_block\": " << options.warps << ",\n"
              << "  \"iterations\": " << options.iterations << ",\n"
              << "  \"mma_per_launch\": " << std::setprecision(0) << mma_per_launch << ",\n"
              << std::setprecision(9)
              << "  \"median_ms\": " << median_ms << ",\n"
              << "  \"mean_ms\": " << mean_ms << ",\n"
              << "  \"p5_ms\": " << p5_ms << ",\n"
              << "  \"p95_ms\": " << p95_ms << ",\n"
              << "  \"logical_tops\": " << logical_tops << ",\n"
              << "  \"registers_per_thread\": " << attributes.numRegs << ",\n"
              << "  \"local_bytes\": " << attributes.localSizeBytes << ",\n"
              << "  \"static_shared_bytes\": " << attributes.sharedSizeBytes << ",\n"
              << "  \"checksum\": " << checksum << "\n"
              << "}\n";

    cudaFree(device_output);
    cudaFree(device_input);
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "o3_mma_lowering: " << error.what() << '\n';
    return 1;
  }
}
