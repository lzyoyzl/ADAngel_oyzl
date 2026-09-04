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
  uint32_t accumulators[Chains][4] = {};

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

  uint32_t checksum = 0;
#pragma unroll
  for (int chain = 0; chain < Chains; ++chain) {
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      checksum ^= accumulators[chain][item] + static_cast<uint32_t>(17 * chain + item);
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
  uint32_t accumulators[Chains][2] = {};
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
  uint32_t checksum = 0;
#pragma unroll
  for (int chain = 0; chain < Chains; ++chain) {
#pragma unroll
    for (int item = 0; item < 2; ++item) {
      checksum ^= accumulators[chain][item] + static_cast<uint32_t>(17 * chain + item);
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
void adangel_o3_mma_micro_s8s8_c1(
    const uint32_t* input, uint32_t* output, int iterations) {
  run_kernel_body<cute::SM80_16x8x32_S32S8S8S32_TN, 1>(input, output, iterations);
}

extern "C" __global__ __launch_bounds__(256)
void adangel_o3_mma_micro_s8s8_c4(
    const uint32_t* input, uint32_t* output, int iterations) {
  run_kernel_body<cute::SM80_16x8x32_S32S8S8S32_TN, 4>(input, output, iterations);
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

using Kernel = void (*)(const uint32_t*, uint32_t*, int);

struct Options {
  std::string kind = "u4s4";
  std::string shape = "m16n8";
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
          << "Usage: o3_mma_lowering [--kind u4s4|s4s4|s8s8] "
             "[--shape m16n8|m8n8] [--chains 1|4] "
             "[--blocks N] [--warps N] [--iterations N] [--warmup N] [--repeats N]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  if (options.kind != "u4s4" && options.kind != "s4s4" && options.kind != "s8s8") {
    throw std::runtime_error("--kind must be u4s4, s4s4, or s8s8");
  }
  if (options.shape != "m16n8" && options.shape != "m8n8") {
    throw std::runtime_error("--shape must be m16n8 or m8n8");
  }
  if (options.chains != 1 && options.chains != 4) {
    throw std::runtime_error("--chains must be 1 or 4");
  }
  if (options.warps > 8) {
    throw std::runtime_error("--warps must not exceed 8 (__launch_bounds__(256))");
  }
  return options;
}

Kernel select_kernel(
    const Options& options, const char** symbol, int* mma_m, int* mma_n, int* mma_k) {
  *mma_m = options.shape == "m16n8" ? 16 : 8;
  *mma_n = 8;
  *mma_k = options.kind == "s8s8" ? (*mma_m == 16 ? 32 : 16)
                                      : (*mma_m == 16 ? 64 : 32);
  if (options.shape == "m8n8") {
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
    const double mma_per_launch = static_cast<double>(options.blocks) * options.warps *
        options.iterations * options.chains;
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
