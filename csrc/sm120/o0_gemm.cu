#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <cstdint>
#include <string>
#include <vector>

#include "adangel/kernel_api.h"

namespace py = pybind11;

namespace {

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
  TORCH_CHECK(false, "unknown O0 timing mode: ", mode);
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
  CudaEvent e3;
};

struct EventPair {
  CudaEvent begin;
  CudaEvent end;
};

float elapsed_ms(const CudaEvent& begin, const CudaEvent& end) {
  float value = 0.0f;
  check_cuda(cudaEventElapsedTime(&value, begin.get(), end.get()), "cudaEventElapsedTime");
  return value;
}

class O0LtPlan {
 public:
  O0LtPlan(int64_t m, int64_t n, int64_t k, uint64_t max_workspace_bytes) {
    try {
      check_cublas(cublasLtCreate(&handle_), "cublasLtCreate");
      check_cublas(
          cublasLtMatmulDescCreate(&operation_, CUBLAS_COMPUTE_32F, CUDA_R_32F),
          "cublasLtMatmulDescCreate");
      cublasOperation_t transpose_a = CUBLAS_OP_N;
      cublasOperation_t transpose_b = CUBLAS_OP_T;
      check_cublas(
          cublasLtMatmulDescSetAttribute(
              operation_, CUBLASLT_MATMUL_DESC_TRANSA, &transpose_a, sizeof(transpose_a)),
          "set TRANSA");
      check_cublas(
          cublasLtMatmulDescSetAttribute(
              operation_, CUBLASLT_MATMUL_DESC_TRANSB, &transpose_b, sizeof(transpose_b)),
          "set TRANSB");

      // Physical row-major operands are A[M,K] and W[N,K]. TRANSB implements A @ W.T.
      check_cublas(
          cublasLtMatrixLayoutCreate(&a_layout_, CUDA_R_16F, m, k, k),
          "create A layout");
      check_cublas(
          cublasLtMatrixLayoutCreate(&b_layout_, CUDA_R_16F, n, k, k),
          "create W layout");
      check_cublas(
          cublasLtMatrixLayoutCreate(&c_layout_, CUDA_R_32F, m, n, n),
          "create C layout");
      check_cublas(
          cublasLtMatrixLayoutCreate(&d_layout_, CUDA_R_32F, m, n, n),
          "create D layout");
      cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
      for (auto layout : {a_layout_, b_layout_, c_layout_, d_layout_}) {
        check_cublas(
            cublasLtMatrixLayoutSetAttribute(
                layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)),
            "set row-major layout");
      }

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
        size_t bytes_written = 0;
        cublasStatus_t cap_status = cublasLtMatmulAlgoCapGetAttribute(
            &candidate.algo,
            CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
            &flags,
            sizeof(flags),
            &bytes_written);
        constexpr uint64_t required_flags =
            CUBLASLT_NUMERICAL_IMPL_FLAGS_HMMA |
            CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F |
            CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16F;
        uint32_t split_k = 0;
        size_t split_k_bytes = 0;
        cublasStatus_t split_k_status = cublasLtMatmulAlgoConfigGetAttribute(
            &candidate.algo,
            CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
            &split_k,
            sizeof(split_k),
            &split_k_bytes);
        if (cap_status != CUBLAS_STATUS_SUCCESS || bytes_written != sizeof(flags) ||
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
          "cuBLASLt returned no FP16 Tensor Core (HMMA) algorithm for O0 shape ",
          m,
          "x",
          n,
          "x",
          k,
          " within ",
          max_workspace_bytes,
          " workspace bytes");

      size_t bytes_written = 0;
      check_cublas(
          cublasLtMatmulAlgoConfigGetAttribute(
              &heuristic_.algo,
              CUBLASLT_ALGO_CONFIG_ID,
              &algorithm_id_,
              sizeof(algorithm_id_),
              &bytes_written),
          "read cuBLASLt algorithm ID");
      TORCH_CHECK(bytes_written == sizeof(algorithm_id_), "invalid cuBLASLt algorithm ID size");
    } catch (...) {
      release();
      throw;
    }
  }

  ~O0LtPlan() { release(); }
  O0LtPlan(const O0LtPlan&) = delete;
  O0LtPlan& operator=(const O0LtPlan&) = delete;

  void run(
      const at::Tensor& a_fp16,
      const at::Tensor& w_fp16,
      at::Tensor& output,
      at::Tensor& workspace,
      cudaStream_t stream) const {
    void* workspace_pointer = heuristic_.workspaceSize == 0 ? nullptr : workspace.data_ptr();
    if (workspace_pointer != nullptr) {
      TORCH_CHECK(
          (reinterpret_cast<uintptr_t>(workspace_pointer) & 255U) == 0,
          "cuBLASLt workspace must be at least 256-byte aligned");
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    check_cublas(
        cublasLtMatmul(
            handle_,
            operation_,
            &alpha,
            a_fp16.data_ptr(),
            a_layout_,
            w_fp16.data_ptr(),
            b_layout_,
            &beta,
            output.data_ptr(),
            c_layout_,
            output.data_ptr(),
            d_layout_,
            &heuristic_.algo,
            workspace_pointer,
            heuristic_.workspaceSize,
            stream),
        "cublasLtMatmul");
  }

  py::dict metadata() const {
    py::dict result;
    result["library"] = "cublasLt";
    result["algorithm_id"] = algorithm_id_;
    result["workspace_bytes"] = static_cast<uint64_t>(heuristic_.workspaceSize);
    result["numerical_impl_flags"] = numerical_flags_;
    result["tensor_core"] = true;
    result["split_k"] = split_k_;
    result["compute_type"] = "CUBLAS_COMPUTE_32F";
    result["input_dtype"] = "fp16";
    result["output_dtype"] = "fp32";
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

void record(cudaEvent_t event, cudaStream_t stream) {
  check_cuda(cudaEventRecord(event, stream), "cudaEventRecord");
}

template <class Operation>
std::vector<float> measure_batched_conversion(
    Operation&& operation,
    int repeats,
    int inner_repeats,
    cudaStream_t stream,
    const char* synchronization_name) {
  std::vector<EventPair> events;
  events.reserve(repeats);
  for (int iteration = 0; iteration < repeats; ++iteration) events.emplace_back();
  for (auto& marker : events) {
    record(marker.begin.get(), stream);
    for (int inner = 0; inner < inner_repeats; ++inner) operation();
    record(marker.end.get(), stream);
  }
  check_cuda(cudaEventSynchronize(events.back().end.get()), synchronization_name);

  std::vector<float> samples;
  samples.reserve(repeats);
  for (const auto& marker : events) {
    samples.push_back(
        elapsed_ms(marker.begin, marker.end) / static_cast<float>(inner_repeats));
  }
  return samples;
}

py::dict timing_metadata(const std::string& mode, int conversion_inner_repeats) {
  py::dict result;
  result["strategy"] = "conversion_amortized_end_to_end_direct";
  result["conversion_stage_timing"] =
      "isolated_batched_cuda_event_average";
  result["conversion_inner_repeats"] = conversion_inner_repeats;
  result["conversion_only_total_timing"] = "batched_cuda_event_average";
  result["end_to_end_total_timing"] = "direct_single_path";
  result["mode_total_timing"] = mode == "conversion_only"
      ? "batched_cuda_event_average"
      : "direct_single_path";
  result["mode_total_inner_repeats"] =
      mode == "conversion_only" ? conversion_inner_repeats : 1;
  result["component_and_total_measured_separately"] = true;
  return result;
}

}  // namespace

bool adangel_o0_is_implemented() { return true; }

py::dict adangel_benchmark_o0(
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4,
    const at::Tensor& w_scale,
    const std::string& mode_name,
    int warmup,
    int repeats,
    int conversion_inner_repeats) {
  TORCH_CHECK(warmup >= 0, "warmup must be non-negative");
  TORCH_CHECK(repeats > 0, "repeats must be positive");
  TORCH_CHECK(
      conversion_inner_repeats > 1,
      "conversion_inner_repeats must exceed one");
  adangel_validate_cuda_inputs(a_int8, a_scale, w_mxfp4, w_scale);
  const TimingMode mode = parse_mode(mode_name);
  c10::cuda::CUDAGuard device_guard(a_int8.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(a_int8.get_device()).stream();

  const int64_t m = a_int8.size(0);
  const int64_t k = a_int8.size(1);
  const int64_t n = w_mxfp4.size(0);
  auto a_fp16 = at::empty({m, k}, a_int8.options().dtype(at::kHalf));
  auto w_fp16 = at::empty({n, k}, a_int8.options().dtype(at::kHalf));
  auto output = at::empty({m, n}, a_int8.options().dtype(at::kFloat));
  auto workspace = at::empty(
      {static_cast<int64_t>(kMaxWorkspaceBytes)}, a_int8.options().dtype(at::kByte));
  O0LtPlan plan(m, n, k, kMaxWorkspaceBytes);

  auto convert_weight = [&]() {
    adangel_launch_mxfp4_to_fp16(w_mxfp4, w_scale, w_fp16, stream);
  };
  auto convert_activation = [&]() {
    adangel_launch_int8_to_fp16(a_int8, a_scale, a_fp16, stream);
  };
  auto gemm = [&]() { plan.run(a_fp16, w_fp16, output, workspace, stream); };

  if (mode == TimingMode::kComputeOnly) {
    convert_weight();
    convert_activation();
  } else if (mode == TimingMode::kSteadyState) {
    convert_weight();
  }
  for (int iteration = 0; iteration < warmup; ++iteration) {
    if (mode == TimingMode::kConversionOnly) {
      convert_weight();
      convert_activation();
    } else if (mode == TimingMode::kComputeOnly) {
      gemm();
    } else if (mode == TimingMode::kCold) {
      convert_weight();
      convert_activation();
      gemm();
    } else {
      convert_activation();
      gemm();
    }
  }
  check_cuda(cudaStreamSynchronize(stream), "O0 warmup synchronization");

  std::vector<EventSet> events;
  events.reserve(repeats);
  for (int iteration = 0; iteration < repeats; ++iteration) events.emplace_back();

  for (int iteration = 0; iteration < repeats; ++iteration) {
    auto& marker = events[iteration];
    record(marker.e0.get(), stream);
    if (mode == TimingMode::kConversionOnly) {
      convert_weight();
      record(marker.e1.get(), stream);
      convert_activation();
      record(marker.e2.get(), stream);
    } else if (mode == TimingMode::kComputeOnly) {
      gemm();
      record(marker.e1.get(), stream);
    } else if (mode == TimingMode::kCold) {
      convert_weight();
      record(marker.e1.get(), stream);
      convert_activation();
      record(marker.e2.get(), stream);
      gemm();
      record(marker.e3.get(), stream);
    } else {
      convert_activation();
      record(marker.e1.get(), stream);
      gemm();
      record(marker.e2.get(), stream);
    }
  }
  cudaEvent_t final_event = mode == TimingMode::kCold
      ? events.back().e3.get()
      : (mode == TimingMode::kComputeOnly ? events.back().e1.get() : events.back().e2.get());
  check_cuda(cudaEventSynchronize(final_event), "O0 measurement synchronization");

  // Conversion components are measured in isolated batched intervals so CUDA Event
  // quantization and rare scheduling outliers do not dominate microsecond-scale kernels.
  // The cold/steady-state main-path events above still contain one real conversion and
  // therefore preserve direct end-to-end total latency.
  std::vector<float> isolated_weight_ms;
  std::vector<float> isolated_activation_ms;
  std::vector<float> conversion_only_total_ms;
  if (mode == TimingMode::kConversionOnly || mode == TimingMode::kCold) {
    isolated_weight_ms = measure_batched_conversion(
        convert_weight,
        repeats,
        conversion_inner_repeats,
        stream,
        "O0 weight-conversion timing synchronization");
  }
  if (mode == TimingMode::kConversionOnly ||
      mode == TimingMode::kCold ||
      mode == TimingMode::kSteadyState) {
    isolated_activation_ms = measure_batched_conversion(
        convert_activation,
        repeats,
        conversion_inner_repeats,
        stream,
        "O0 activation-conversion timing synchronization");
  }
  if (mode == TimingMode::kConversionOnly) {
    auto convert_all = [&]() {
      convert_weight();
      convert_activation();
    };
    conversion_only_total_ms = measure_batched_conversion(
        convert_all,
        repeats,
        conversion_inner_repeats,
        stream,
        "O0 conversion-only total timing synchronization");
  }

  std::vector<float> weight_ms;
  std::vector<float> activation_ms;
  std::vector<float> gemm_ms;
  std::vector<float> total_ms;
  weight_ms.reserve(repeats);
  activation_ms.reserve(repeats);
  gemm_ms.reserve(repeats);
  total_ms.reserve(repeats);
  for (int iteration = 0; iteration < repeats; ++iteration) {
    const auto& marker = events[iteration];
    if (mode == TimingMode::kConversionOnly) {
      weight_ms.push_back(isolated_weight_ms[iteration]);
      activation_ms.push_back(isolated_activation_ms[iteration]);
      total_ms.push_back(conversion_only_total_ms[iteration]);
    } else if (mode == TimingMode::kComputeOnly) {
      float value = elapsed_ms(marker.e0, marker.e1);
      gemm_ms.push_back(value);
      total_ms.push_back(value);
    } else if (mode == TimingMode::kCold) {
      weight_ms.push_back(isolated_weight_ms[iteration]);
      activation_ms.push_back(isolated_activation_ms[iteration]);
      gemm_ms.push_back(elapsed_ms(marker.e2, marker.e3));
      total_ms.push_back(elapsed_ms(marker.e0, marker.e3));
    } else {
      activation_ms.push_back(isolated_activation_ms[iteration]);
      gemm_ms.push_back(elapsed_ms(marker.e1, marker.e2));
      total_ms.push_back(elapsed_ms(marker.e0, marker.e2));
    }
  }

  // conversion_only still returns a semantically valid O0 output, but this GEMM is not timed.
  if (mode == TimingMode::kConversionOnly) gemm();

  py::dict timings;
  if (!weight_ms.empty()) timings["weight_conversion"] = weight_ms;
  if (!activation_ms.empty()) timings["activation_conversion"] = activation_ms;
  if (!gemm_ms.empty()) timings["gemm"] = gemm_ms;
  timings["total"] = total_ms;

  py::dict result;
  result["output"] = output;
  result["converted_weight"] = w_fp16;
  result["converted_activation"] = a_fp16;
  result["timings_ms"] = timings;
  result["timing_method"] =
      timing_metadata(mode_name, conversion_inner_repeats);
  result["kernel"] = plan.metadata();
  return result;
}

py::dict adangel_run_o0(
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4,
    const at::Tensor& w_scale,
    const std::string& mode) {
  py::dict measured =
      adangel_benchmark_o0(
          a_int8,
          a_scale,
          w_mxfp4,
          w_scale,
          mode,
          0,
          1,
          kAdangelDefaultConversionTimingInnerRepeats);
  py::dict timings = measured["timings_ms"].cast<py::dict>();
  auto scalar = [&](const char* name) {
    if (!timings.contains(name)) return 0.0f;
    return timings[name].cast<std::vector<float>>().front();
  };
  measured["weight_conversion_ms"] = scalar("weight_conversion");
  measured["activation_conversion_ms"] = scalar("activation_conversion");
  measured["gemm_ms"] = scalar("gemm");
  measured["total_ms"] = scalar("total");
  return measured;
}

// This probe is retained for the one-time PTX/SASS audit. The production path above additionally
// rejects any cuBLASLt heuristic whose numerical implementation flags do not contain HMMA.
extern "C" __global__ void adangel_o0_fp16_mma_probe(
    const uint32_t* a, const uint32_t* b, float* d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  float x0 = 0, x1 = 0, x2 = 0, x3 = 0;
  uint32_t a0 = a[threadIdx.x * 4 + 0], a1 = a[threadIdx.x * 4 + 1];
  uint32_t a2 = a[threadIdx.x * 4 + 2], a3 = a[threadIdx.x * 4 + 3];
  uint32_t b0 = b[threadIdx.x * 2 + 0], b1 = b[threadIdx.x * 2 + 1];
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(x0), "+f"(x1), "+f"(x2), "+f"(x3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  d[threadIdx.x * 4 + 0] = x0;
#endif
}
