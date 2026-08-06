#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cublasLt.h>
#include <cuda_runtime_api.h>
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
constexpr int kScaleTileM = 16;
constexpr int kScaleTileN = 256;
constexpr uint64_t kMaxWorkspaceBytes = 64ULL * 1024ULL * 1024ULL;
constexpr int kRequestedAlgorithms = 64;

void check_cuda(cudaError_t status, const char* operation) {
  TORCH_CHECK(status == cudaSuccess, operation, " failed: ", cudaGetErrorString(status));
}

void check_cublas(cublasStatus_t status, const char* operation) {
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      operation,
      " failed with cuBLAS status ",
      static_cast<int>(status));
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

// One block reuses 256 column scales across 16 output rows. Each warp therefore reads/writes
// contiguous output values while the strided W_scale load is paid only once per output tile.
__global__ void adangel_o1_scale_accumulate(
    const int32_t* partial,
    const float* a_scale,
    const uint8_t* w_scale,
    float* output,
    int m,
    int n,
    int groups,
    int group) {
  __shared__ float column_scale[kScaleTileN];
  __shared__ float row_scale[kScaleTileM];
  const int lane = static_cast<int>(threadIdx.x);
  const int column = static_cast<int>(blockIdx.x) * kScaleTileN + lane;
  const int row_begin = static_cast<int>(blockIdx.y) * kScaleTileM;
  if (lane < kScaleTileM) {
    const int row = row_begin + lane;
    row_scale[lane] = row < m ? a_scale[row] : 0.0f;
  }
  column_scale[lane] = column < n
      ? adangel::decode_ue8m0(w_scale[column * groups + group])
      : 0.0f;
  __syncthreads();
  if (column >= n) return;

#pragma unroll
  for (int row_offset = 0; row_offset < kScaleTileM; ++row_offset) {
    const int row = row_begin + row_offset;
    if (row >= m) break;
    const int64_t index = static_cast<int64_t>(row) * n + column;
    float scale = __fmul_rn(row_scale[row_offset], column_scale[lane]);
    scale = __fmul_rn(scale, 0.5f);
    const float contribution = __fmul_rn(static_cast<float>(partial[index]), scale);
    output[index] = group == 0 ? contribution : __fadd_rn(output[index], contribution);
  }
}

class O1LtPlan {
 public:
  O1LtPlan(int64_t m, int64_t n, int64_t full_k, uint64_t max_workspace_bytes) {
    try {
      check_cublas(cublasLtCreate(&handle_), "cublasLtCreate");
      check_cublas(
          cublasLtMatmulDescCreate(&operation_, CUBLAS_COMPUTE_32I, CUDA_R_32I),
          "cublasLtMatmulDescCreate");
      // Row-major [M,K] and [N,K] buffers are memory-equivalent to column-major [K,M] and
      // [K,N]. This TN description satisfies cuBLASLt's regular-order IMMA requirements.
      cublasOperation_t transpose_a = CUBLAS_OP_T;
      cublasOperation_t transpose_b = CUBLAS_OP_N;
      check_cublas(
          cublasLtMatmulDescSetAttribute(
              operation_, CUBLASLT_MATMUL_DESC_TRANSA, &transpose_a, sizeof(transpose_a)),
          "set TRANSA");
      check_cublas(
          cublasLtMatmulDescSetAttribute(
              operation_, CUBLASLT_MATMUL_DESC_TRANSB, &transpose_b, sizeof(transpose_b)),
          "set TRANSB");

      check_cublas(
          cublasLtMatrixLayoutCreate(
              &a_layout_, CUDA_R_8I, kGroupSize, m, full_k),
          "create A K32 layout");
      check_cublas(
          cublasLtMatrixLayoutCreate(
              &b_layout_, CUDA_R_8I, kGroupSize, n, full_k),
          "create W K32 layout");
      check_cublas(
          cublasLtMatrixLayoutCreate(&c_layout_, CUDA_R_32I, m, n, n),
          "create partial C layout");
      check_cublas(
          cublasLtMatrixLayoutCreate(&d_layout_, CUDA_R_32I, m, n, n),
          "create partial D layout");
      cublasLtOrder_t column_order = CUBLASLT_ORDER_COL;
      cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
      check_cublas(
          cublasLtMatrixLayoutSetAttribute(
              a_layout_, CUBLASLT_MATRIX_LAYOUT_ORDER, &column_order, sizeof(column_order)),
          "set A column-major layout");
      check_cublas(
          cublasLtMatrixLayoutSetAttribute(
              b_layout_, CUBLASLT_MATRIX_LAYOUT_ORDER, &column_order, sizeof(column_order)),
          "set W column-major layout");
      check_cublas(
          cublasLtMatrixLayoutSetAttribute(
              c_layout_, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)),
          "set C row-major layout");
      check_cublas(
          cublasLtMatrixLayoutSetAttribute(
              d_layout_, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)),
          "set D row-major layout");

      check_cublas(cublasLtMatmulPreferenceCreate(&preference_), "create matmul preference");
      check_cublas(
          cublasLtMatmulPreferenceSetAttribute(
              preference_,
              CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
              &max_workspace_bytes,
              sizeof(max_workspace_bytes)),
          "set maximum workspace");

      std::vector<cublasLtMatmulHeuristicResult_t> candidates(kRequestedAlgorithms);
      int returned = 0;
      check_cublas(
          cublasLtMatmulAlgoGetHeuristic(
              handle_,
              operation_,
              a_layout_,
              b_layout_,
              c_layout_,
              d_layout_,
              preference_,
              static_cast<int>(candidates.size()),
              candidates.data(),
              &returned),
          "cublasLtMatmulAlgoGetHeuristic");

      bool found = false;
      for (int index = 0; index < returned; ++index) {
        const auto& candidate = candidates[index];
        if (candidate.state != CUBLAS_STATUS_SUCCESS ||
            candidate.workspaceSize > max_workspace_bytes) {
          continue;
        }
        uint64_t flags = 0;
        size_t flag_bytes = 0;
        cublasStatus_t flag_status = cublasLtMatmulAlgoCapGetAttribute(
            &candidate.algo,
            CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
            &flags,
            sizeof(flags),
            &flag_bytes);
        constexpr uint64_t required_flags =
            CUBLASLT_NUMERICAL_IMPL_FLAGS_IMMA |
            CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32I |
            CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8I;
        uint32_t split_k = 0;
        size_t split_k_bytes = 0;
        cublasStatus_t split_k_status = cublasLtMatmulAlgoConfigGetAttribute(
            &candidate.algo,
            CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
            &split_k,
            sizeof(split_k),
            &split_k_bytes);
        if (flag_status != CUBLAS_STATUS_SUCCESS || flag_bytes != sizeof(flags) ||
            (flags & required_flags) != required_flags ||
            split_k_status != CUBLAS_STATUS_SUCCESS ||
            split_k_bytes != sizeof(split_k) ||
            split_k > 1) {
          continue;
        }
        heuristic_ = candidate;
        numerical_flags_ = flags;
        split_k_ = split_k;
        found = true;
        break;
      }
      TORCH_CHECK(
          found,
          "cuBLASLt returned no INT8 Tensor Core (IMMA) algorithm for O1 K32 shape ",
          m,
          "x",
          n,
          " within ",
          max_workspace_bytes,
          " workspace bytes");

      size_t algorithm_bytes = 0;
      check_cublas(
          cublasLtMatmulAlgoConfigGetAttribute(
              &heuristic_.algo,
              CUBLASLT_ALGO_CONFIG_ID,
              &algorithm_id_,
              sizeof(algorithm_id_),
              &algorithm_bytes),
          "read cuBLASLt algorithm ID");
      TORCH_CHECK(
          algorithm_bytes == sizeof(algorithm_id_), "invalid cuBLASLt algorithm ID size");
    } catch (...) {
      release();
      throw;
    }
  }

  ~O1LtPlan() { release(); }
  O1LtPlan(const O1LtPlan&) = delete;
  O1LtPlan& operator=(const O1LtPlan&) = delete;

  void run_group(
      const int8_t* a,
      const int8_t* w,
      at::Tensor& partial,
      at::Tensor& workspace,
      cudaStream_t stream) const {
    void* workspace_pointer = heuristic_.workspaceSize == 0 ? nullptr : workspace.data_ptr();
    if (workspace_pointer != nullptr) {
      TORCH_CHECK(
          (reinterpret_cast<uintptr_t>(workspace_pointer) & 255U) == 0,
          "cuBLASLt workspace must be at least 256-byte aligned");
    }
    const int32_t alpha = 1;
    const int32_t beta = 0;
    check_cublas(
        cublasLtMatmul(
            handle_,
            operation_,
            &alpha,
            a,
            a_layout_,
            w,
            b_layout_,
            &beta,
            partial.data_ptr<int32_t>(),
            c_layout_,
            partial.data_ptr<int32_t>(),
            d_layout_,
            &heuristic_.algo,
            workspace_pointer,
            heuristic_.workspaceSize,
            stream),
        "O1 cublasLtMatmul K32");
  }

  py::dict metadata(int groups) const {
    py::dict result;
    result["library"] = "cublasLt+CUDA";
    result["algorithm_id"] = algorithm_id_;
    result["workspace_bytes"] = static_cast<uint64_t>(heuristic_.workspaceSize);
    result["numerical_impl_flags"] = numerical_flags_;
    result["tensor_core"] = true;
    result["mma_family"] = "IMMA";
    result["split_k"] = split_k_;
    result["compute_type"] = "CUBLAS_COMPUTE_32I";
    result["input_dtype"] = "int8";
    result["partial_dtype"] = "int32";
    result["output_dtype"] = "fp32";
    result["group_size"] = kGroupSize;
    result["group_count"] = groups;
    result["scale_formula"] = "A_scale*decode_ue8m0(W_scale)/2";
    return result;
  }

 private:
  void release() noexcept {
    if (preference_ != nullptr) cublasLtMatmulPreferenceDestroy(preference_);
    if (d_layout_ != nullptr) cublasLtMatrixLayoutDestroy(d_layout_);
    if (c_layout_ != nullptr) cublasLtMatrixLayoutDestroy(c_layout_);
    if (b_layout_ != nullptr) cublasLtMatrixLayoutDestroy(b_layout_);
    if (a_layout_ != nullptr) cublasLtMatrixLayoutDestroy(a_layout_);
    if (operation_ != nullptr) cublasLtMatmulDescDestroy(operation_);
    if (handle_ != nullptr) cublasLtDestroy(handle_);
    preference_ = nullptr;
    d_layout_ = c_layout_ = b_layout_ = a_layout_ = nullptr;
    operation_ = nullptr;
    handle_ = nullptr;
  }

  cublasLtHandle_t handle_ = nullptr;
  cublasLtMatmulDesc_t operation_ = nullptr;
  cublasLtMatrixLayout_t a_layout_ = nullptr;
  cublasLtMatrixLayout_t b_layout_ = nullptr;
  cublasLtMatrixLayout_t c_layout_ = nullptr;
  cublasLtMatrixLayout_t d_layout_ = nullptr;
  cublasLtMatmulPreference_t preference_ = nullptr;
  cublasLtMatmulHeuristicResult_t heuristic_{};
  uint64_t numerical_flags_ = 0;
  uint32_t split_k_ = 0;
  int algorithm_id_ = -1;
};

void launch_scale_accumulate(
    const at::Tensor& partial,
    const at::Tensor& a_scale,
    const at::Tensor& w_scale,
    at::Tensor& output,
    int group,
    cudaStream_t stream) {
  const int m = static_cast<int>(output.size(0));
  const int n = static_cast<int>(output.size(1));
  const int groups = static_cast<int>(w_scale.size(1));
  dim3 blocks(
      (n + kScaleTileN - 1) / kScaleTileN,
      (m + kScaleTileM - 1) / kScaleTileM);
  adangel_o1_scale_accumulate<<<blocks, kScaleTileN, 0, stream>>>(
      partial.data_ptr<int32_t>(),
      a_scale.data_ptr<float>(),
      w_scale.data_ptr<uint8_t>(),
      output.data_ptr<float>(),
      m,
      n,
      groups,
      group);
  check_cuda(cudaGetLastError(), "O1 K32 scale/accumulate launch");
}

void record(cudaEvent_t event, cudaStream_t stream) {
  check_cuda(cudaEventRecord(event, stream), "cudaEventRecord");
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
  TORCH_CHECK(
      m % 4 == 0 && n % 4 == 0,
      "O1 regular-order INT8 Tensor Core path requires M and N divisible by 4; got ",
      m,
      "x",
      n);
  const int groups = static_cast<int>(k / kGroupSize);
  auto w_int8 = at::empty({n, k}, a_int8.options().dtype(at::kChar));
  auto partial = at::empty({m, n}, a_int8.options().dtype(at::kInt));
  auto output = at::empty({m, n}, a_int8.options().dtype(at::kFloat));
  auto workspace = at::empty(
      {static_cast<int64_t>(kMaxWorkspaceBytes)}, a_int8.options().dtype(at::kByte));
  O1LtPlan plan(m, n, k, kMaxWorkspaceBytes);

  auto convert_weight = [&]() { adangel_launch_mxfp4_to_int8(w_mxfp4, w_int8, stream); };
  auto gemm = [&]() {
    const int8_t* a_pointer = a_int8.data_ptr<int8_t>();
    const int8_t* w_pointer = w_int8.data_ptr<int8_t>();
    for (int group = 0; group < groups; ++group) {
      const int offset = group * kGroupSize;
      plan.run_group(a_pointer + offset, w_pointer + offset, partial, workspace, stream);
      launch_scale_accumulate(partial, a_scale, w_scale, output, group, stream);
    }
  };

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

  // conversion_only returns a valid O1 output, but this K32 GEMM sequence is not timed.
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
  result["kernel"] = plan.metadata(groups);
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

// Retained for the one-time PTX/SASS audit. The production path above additionally rejects
// cuBLASLt algorithms whose numerical implementation flags do not contain IMMA.

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
