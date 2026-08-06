#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <mma.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <cstdint>
#include <string>
#include <vector>

#include "adangel/data_types.cuh"
#include "adangel/kernel_api.h"

namespace py = pybind11;

namespace {

constexpr int kGroupSize = 32;
constexpr int kWmmaK = 16;
constexpr int kTileM = 64;
constexpr int kTileN = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreadsPerBlock = kWarpsPerBlock * 32;
constexpr int kOutputsPerThread = (kTileM * kTileN) / kThreadsPerBlock;

static_assert(kOutputsPerThread == 8);

void check_cuda(cudaError_t status, const char* operation) {
  TORCH_CHECK(status == cudaSuccess, operation, " failed: ", cudaGetErrorString(status));
}

enum class TimingMode { kConversionOnly, kComputeOnly, kCold, kSteadyState };

TimingMode parse_mode(const std::string& mode) {
  if (mode == "conversion_only") return TimingMode::kConversionOnly;
  if (mode == "compute_only") return TimingMode::kComputeOnly;
  if (mode == "cold") return TimingMode::kCold;
  if (mode == "steady_state") return TimingMode::kSteadyState;
  TORCH_CHECK(false, "unknown O1 timing mode: ", mode);
  return TimingMode::kCold;
}

class CudaEvent {
 public:
  CudaEvent() { check_cuda(cudaEventCreateWithFlags(&event_, cudaEventDefault), "cudaEventCreate"); }
  ~CudaEvent() {
    if (event_ != nullptr) cudaEventDestroy(event_);
  }
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;
  CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) { other.event_ = nullptr; }
  CudaEvent& operator=(CudaEvent&& other) noexcept {
    if (this != &other) {
      if (event_ != nullptr) cudaEventDestroy(event_);
      event_ = other.event_;
      other.event_ = nullptr;
    }
    return *this;
  }
  cudaEvent_t get() const { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

struct EventSet {
  CudaEvent e0;
  CudaEvent e1;
  CudaEvent e2;
};

float elapsed_ms(const CudaEvent& begin, const CudaEvent& end) {
  float value = 0.0f;
  check_cuda(cudaEventElapsedTime(&value, begin.get(), end.get()), "cudaEventElapsedTime");
  return value;
}

void record(cudaEvent_t event, cudaStream_t stream) {
  check_cuda(cudaEventRecord(event, stream), "cudaEventRecord");
}

}  // namespace

// One CTA owns a 64x32 output tile. Eight warps compute 16x16 subtiles. For each K32 group,
// two signed-int8 WMMA operations create one exact INT32 partial per output element. The partial
// is staged only in shared memory, immediately rescaled, and accumulated in per-thread FP32
// registers. Each final FP32 output is written to global memory exactly once.
extern "C" __global__ void adangel_o1_fused_tiled(
    const int8_t* a_int8,
    const float* a_scale,
    const int8_t* w_int8,
    const uint8_t* w_scale,
    float* output,
    int m,
    int n,
    int k,
    int groups) {
  __shared__ __align__(32) int8_t shared_a_lo[kTileM * kWmmaK];
  __shared__ __align__(32) int8_t shared_a_hi[kTileM * kWmmaK];
  // B is W^T. Each half is stored column-major [K16,N32] for WMMA matrix_b.
  __shared__ __align__(32) int8_t shared_b_lo[kTileN * kWmmaK];
  __shared__ __align__(32) int8_t shared_b_hi[kTileN * kWmmaK];
  __shared__ __align__(32) int32_t shared_partial[kWarpsPerBlock * 16 * 16];

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread >> 5;
  const int warp_row = warp >> 1;
  const int warp_column = warp & 1;
  const int tile_row = static_cast<int>(blockIdx.y) * kTileM;
  const int tile_column = static_cast<int>(blockIdx.x) * kTileN;
  const int local_column = thread & (kTileN - 1);
  const int first_local_row = thread >> 5;
  const int global_column = tile_column + local_column;

  float row_scales[kOutputsPerThread];
  float accumulators[kOutputsPerThread];
#pragma unroll
  for (int item = 0; item < kOutputsPerThread; ++item) {
    const int local_row = first_local_row + item * 8;
    const int global_row = tile_row + local_row;
    row_scales[item] = global_row < m ? a_scale[global_row] : 0.0f;
    accumulators[item] = 0.0f;
  }

  for (int group = 0; group < groups; ++group) {
    const int group_k = group * kGroupSize;
    for (int linear = thread; linear < kTileM * kGroupSize;
         linear += kThreadsPerBlock) {
      const int local_row = linear / kGroupSize;
      const int local_k = linear - local_row * kGroupSize;
      const int global_row = tile_row + local_row;
      const int8_t value = global_row < m
          ? a_int8[static_cast<int64_t>(global_row) * k + group_k + local_k]
          : int8_t{0};
      if (local_k < kWmmaK) {
        shared_a_lo[local_row * kWmmaK + local_k] = value;
      } else {
        shared_a_hi[local_row * kWmmaK + local_k - kWmmaK] = value;
      }
    }
    for (int linear = thread; linear < kTileN * kGroupSize;
         linear += kThreadsPerBlock) {
      const int local_col = linear / kGroupSize;
      const int local_k = linear - local_col * kGroupSize;
      const int global_col = tile_column + local_col;
      const int8_t value = global_col < n
          ? w_int8[static_cast<int64_t>(global_col) * k + group_k + local_k]
          : int8_t{0};
      if (local_k < kWmmaK) {
        shared_b_lo[local_col * kWmmaK + local_k] = value;
      } else {
        shared_b_hi[local_col * kWmmaK + local_k - kWmmaK] = value;
      }
    }
    __syncthreads();

    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a_fragment;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b_fragment;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> partial_fragment;
    wmma::fill_fragment(partial_fragment, 0);

    const int a_offset = warp_row * 16 * kWmmaK;
    const int b_offset = warp_column * 16 * kWmmaK;
    wmma::load_matrix_sync(a_fragment, shared_a_lo + a_offset, kWmmaK);
    wmma::load_matrix_sync(b_fragment, shared_b_lo + b_offset, kWmmaK);
    wmma::mma_sync(partial_fragment, a_fragment, b_fragment, partial_fragment);
    wmma::load_matrix_sync(a_fragment, shared_a_hi + a_offset, kWmmaK);
    wmma::load_matrix_sync(b_fragment, shared_b_hi + b_offset, kWmmaK);
    wmma::mma_sync(partial_fragment, a_fragment, b_fragment, partial_fragment);
    wmma::store_matrix_sync(
        shared_partial + warp * 16 * 16,
        partial_fragment,
        16,
        wmma::mem_row_major);
    __syncthreads();

    const float column_scale = global_column < n
        ? adangel::decode_ue8m0(w_scale[global_column * groups + group])
        : 0.0f;
#pragma unroll
    for (int item = 0; item < kOutputsPerThread; ++item) {
      const int local_row = first_local_row + item * 8;
      const int owner_warp = (local_row / 16) * 2 + local_column / 16;
      const int owner_index =
          (local_row % 16) * 16 + (local_column % 16);
      const int32_t partial = shared_partial[owner_warp * 16 * 16 + owner_index];
      float scale = __fmul_rn(row_scales[item], column_scale);
      scale = __fmul_rn(scale, 0.5f);
      const float contribution = __fmul_rn(static_cast<float>(partial), scale);
      accumulators[item] = group == 0
          ? contribution
          : __fadd_rn(accumulators[item], contribution);
    }
    // The next iteration may overwrite A/B shared storage immediately. Its load barrier also
    // prevents any warp from overwriting shared_partial before every thread consumed this group.
  }

  if (global_column < n) {
#pragma unroll
    for (int item = 0; item < kOutputsPerThread; ++item) {
      const int global_row = tile_row + first_local_row + item * 8;
      if (global_row < m) {
        output[static_cast<int64_t>(global_row) * n + global_column] = accumulators[item];
      }
    }
  }
}

namespace {

void launch_fused_o1(
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_int8,
    const at::Tensor& w_scale,
    at::Tensor& output,
    cudaStream_t stream) {
  const int m = static_cast<int>(a_int8.size(0));
  const int k = static_cast<int>(a_int8.size(1));
  const int n = static_cast<int>(w_int8.size(0));
  const int groups = k / kGroupSize;
  dim3 grid((n + kTileN - 1) / kTileN, (m + kTileM - 1) / kTileM);
  adangel_o1_fused_tiled<<<grid, kThreadsPerBlock, 0, stream>>>(
      a_int8.data_ptr<int8_t>(),
      a_scale.data_ptr<float>(),
      w_int8.data_ptr<int8_t>(),
      w_scale.data_ptr<uint8_t>(),
      output.data_ptr<float>(),
      m,
      n,
      k,
      groups);
  check_cuda(cudaGetLastError(), "O1 fused tiled WMMA launch");
}

py::dict fused_metadata(int groups) {
  py::dict result;
  result["library"] = "CUDA WMMA";
  result["algorithm_id"] = -1;
  result["workspace_bytes"] = 0;
  result["numerical_impl_flags"] = 0;
  result["tensor_core"] = true;
  result["mma_family"] = "IMMA";
  result["mma_api"] = "nvcuda::wmma";
  result["mma_shape"] = "m16n16k16";
  result["implementation"] = "fused_tiled";
  result["kernel_symbol"] = "adangel_o1_fused_tiled";
  result["cta_tile"] = py::make_tuple(kTileM, kTileN, kGroupSize);
  result["split_k"] = 1;
  result["compute_type"] = "S8xS8_TO_S32";
  result["input_dtype"] = "int8";
  result["partial_dtype"] = "int32";
  result["accumulation_dtype"] = "fp32";
  result["output_dtype"] = "fp32";
  result["group_size"] = kGroupSize;
  result["group_count"] = groups;
  result["scale_formula"] = "A_scale*decode_ue8m0(W_scale)/2";
  result["global_partial_buffer"] = false;
  result["output_stores_per_element"] = 1;
  return result;
}

}  // namespace

bool adangel_o1_is_implemented() { return true; }

py::dict adangel_benchmark_o1(
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4,
    const at::Tensor& w_scale,
    const std::string& mode_name,
    int warmup,
    int repeats) {
  TORCH_CHECK(warmup >= 0, "warmup must be non-negative");
  TORCH_CHECK(repeats > 0, "repeats must be positive");
  adangel_validate_cuda_inputs(a_int8, a_scale, w_mxfp4, w_scale);
  const TimingMode mode = parse_mode(mode_name);
  c10::cuda::CUDAGuard device_guard(a_int8.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(a_int8.get_device()).stream();

  const int64_t m = a_int8.size(0);
  const int64_t k = a_int8.size(1);
  const int64_t n = w_mxfp4.size(0);
  const int groups = static_cast<int>(k / kGroupSize);
  auto w_int8 = at::empty({n, k}, a_int8.options().dtype(at::kChar));
  auto output = at::empty({m, n}, a_int8.options().dtype(at::kFloat));

  auto convert_weight = [&]() { adangel_launch_mxfp4_to_int8(w_mxfp4, w_int8, stream); };
  auto gemm = [&]() { launch_fused_o1(a_int8, a_scale, w_int8, w_scale, output, stream); };

  if (mode == TimingMode::kComputeOnly || mode == TimingMode::kSteadyState) convert_weight();
  for (int iteration = 0; iteration < warmup; ++iteration) {
    if (mode == TimingMode::kConversionOnly) {
      convert_weight();
    } else if (mode == TimingMode::kCold) {
      convert_weight();
      gemm();
    } else {
      gemm();
    }
  }
  check_cuda(cudaStreamSynchronize(stream), "O1 warmup synchronization");

  std::vector<EventSet> events;
  events.reserve(repeats);
  for (int iteration = 0; iteration < repeats; ++iteration) events.emplace_back();

  for (int iteration = 0; iteration < repeats; ++iteration) {
    auto& marker = events[iteration];
    record(marker.e0.get(), stream);
    if (mode == TimingMode::kConversionOnly) {
      convert_weight();
      record(marker.e1.get(), stream);
    } else if (mode == TimingMode::kCold) {
      convert_weight();
      record(marker.e1.get(), stream);
      gemm();
      record(marker.e2.get(), stream);
    } else {
      gemm();
      record(marker.e1.get(), stream);
    }
  }
  cudaEvent_t final_event = mode == TimingMode::kCold
      ? events.back().e2.get()
      : events.back().e1.get();
  check_cuda(cudaEventSynchronize(final_event), "O1 measurement synchronization");

  std::vector<float> weight_ms;
  std::vector<float> gemm_ms;
  std::vector<float> total_ms;
  weight_ms.reserve(repeats);
  gemm_ms.reserve(repeats);
  total_ms.reserve(repeats);
  for (const auto& marker : events) {
    if (mode == TimingMode::kConversionOnly) {
      const float value = elapsed_ms(marker.e0, marker.e1);
      weight_ms.push_back(value);
      total_ms.push_back(value);
    } else if (mode == TimingMode::kCold) {
      weight_ms.push_back(elapsed_ms(marker.e0, marker.e1));
      gemm_ms.push_back(elapsed_ms(marker.e1, marker.e2));
      total_ms.push_back(elapsed_ms(marker.e0, marker.e2));
    } else {
      const float value = elapsed_ms(marker.e0, marker.e1);
      gemm_ms.push_back(value);
      total_ms.push_back(value);
    }
  }

  // conversion_only returns a valid O1 output, but the fused kernel is outside its event range.
  if (mode == TimingMode::kConversionOnly) gemm();

  py::dict timings;
  if (!weight_ms.empty()) timings["weight_conversion"] = weight_ms;
  if (!gemm_ms.empty()) timings["gemm"] = gemm_ms;
  timings["total"] = total_ms;

  py::dict result;
  result["output"] = output;
  result["converted_weight"] = w_int8;
  result["converted_activation"] = py::none();
  result["timings_ms"] = timings;
  result["kernel"] = fused_metadata(groups);
  return result;
}

py::dict adangel_run_o1(
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4,
    const at::Tensor& w_scale,
    const std::string& mode) {
  py::dict measured =
      adangel_benchmark_o1(a_int8, a_scale, w_mxfp4, w_scale, mode, 0, 1);
  py::dict timings = measured["timings_ms"].cast<py::dict>();
  auto scalar = [&](const char* name) {
    if (!timings.contains(name)) return 0.0f;
    return timings[name].cast<std::vector<float>>().front();
  };
  measured["weight_conversion_ms"] = scalar("weight_conversion");
  measured["activation_conversion_ms"] = 0.0f;
  measured["gemm_ms"] = scalar("gemm");
  measured["total_ms"] = scalar("total");
  return measured;
}

// Retained as an independent direct-PTX smoke probe. The instruction audit must additionally
// associate integer mma instructions with the adangel_o1_fused_tiled production kernel symbol.
extern "C" __global__ void adangel_o1_int8_mma_probe(
    const uint32_t* a, const uint32_t* b, int32_t* d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  int x0 = 0, x1 = 0, x2 = 0, x3 = 0;
  uint32_t a0 = a[threadIdx.x * 4 + 0], a1 = a[threadIdx.x * 4 + 1];
  uint32_t a2 = a[threadIdx.x * 4 + 2], a3 = a[threadIdx.x * 4 + 3];
  uint32_t b0 = b[threadIdx.x * 2 + 0], b1 = b[threadIdx.x * 2 + 1];
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+r"(x0), "+r"(x1), "+r"(x2), "+r"(x3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  d[threadIdx.x * 4 + 0] = x0;
#endif
}
