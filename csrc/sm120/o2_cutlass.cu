#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/util/packed_stride.hpp>
#include <cuda_runtime_api.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "adangel/data_types.cuh"
#include "adangel/kernel_api.h"

namespace py = pybind11;

namespace {

constexpr int kGroupSize = 32;
constexpr int kQuantWarpsPerBlock = 8;
constexpr int kQuantThreads = kQuantWarpsPerBlock * 32;
constexpr char kCutlassCommit[] = "db1c288993354c88e551c40c19a8fb93a774a241";

using ElementA = cutlass::float_e2m1_t;
using ElementB = cutlass::float_e2m1_t;
using ElementC = float;
using ElementD = float;
using ElementAccumulator = float;
using ElementCompute = float;
using ElementSF = cutlass::float_ue8m0_t;
using ElementPairA = cutlass::mx_float4_t<ElementA>;
using ElementPairB = cutlass::mx_float4_t<ElementB>;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

constexpr int kAlignmentA = 16 * 8 / cutlass::sizeof_bits<ElementA>::value;
constexpr int kAlignmentB = 16 * 8 / cutlass::sizeof_bits<ElementB>::value;
constexpr int kAlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int kAlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using TileShape = cute::Shape<cute::_128, cute::_128, cute::_256>;
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    cutlass::arch::OpClassTensorOp,
    TileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    LayoutC,
    kAlignmentC,
    ElementD,
    LayoutD,
    kAlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    cutlass::arch::OpClassBlockScaledTensorOp,
    ElementPairA,
    LayoutA,
    kAlignmentA,
    ElementPairB,
    LayoutB,
    kAlignmentB,
    ElementAccumulator,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

using ProblemShape = cute::Shape<int, int, int, int>;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<kGroupSize>;

enum class TimingMode { kConversionOnly, kComputeOnly, kCold, kSteadyState };

TimingMode parse_mode(const std::string& mode) {
  if (mode == "conversion_only") return TimingMode::kConversionOnly;
  if (mode == "compute_only") return TimingMode::kComputeOnly;
  if (mode == "cold") return TimingMode::kCold;
  if (mode == "steady_state") return TimingMode::kSteadyState;
  TORCH_CHECK(false, "unknown O2 timing mode: ", mode);
  return TimingMode::kCold;
}

void check_cuda(cudaError_t status, const char* operation) {
  TORCH_CHECK(status == cudaSuccess, operation, " failed: ", cudaGetErrorString(status));
}

void check_cutlass(cutlass::Status status, const char* operation) {
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      operation,
      " failed: ",
      cutlassGetStatusString(status));
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

void record(cudaEvent_t event, cudaStream_t stream) {
  check_cuda(cudaEventRecord(event, stream), "cudaEventRecord");
}

float elapsed_ms(const CudaEvent& begin, const CudaEvent& end) {
  float value = 0.0f;
  check_cuda(cudaEventElapsedTime(&value, begin.get(), end.get()), "cudaEventElapsedTime");
  return value;
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

__global__ void adangel_o2_quantize_activation(
    const int8_t* input,
    const float* row_scale,
    uint8_t* packed,
    uint8_t* natural_scale,
    int rows,
    int k,
    int groups) {
  const int warp_in_block = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int linear_group =
      static_cast<int>(blockIdx.x) * kQuantWarpsPerBlock + warp_in_block;
  const int total_groups = rows * groups;
  if (linear_group >= total_groups) return;

  const int row = linear_group / groups;
  const int group = linear_group - row * groups;
  const int column = group * kGroupSize + lane;
  const float value =
      static_cast<float>(input[static_cast<int64_t>(row) * k + column]) * row_scale[row];

  float max_abs = fabsf(value);
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, offset));
  }
  max_abs = __shfl_sync(0xffffffffu, max_abs, 0);

  int exponent = 0;
  if (max_abs != 0.0f) {
    exponent = ilogbf(max_abs) - 2;
    exponent = exponent < -127 ? -127 : (exponent > 127 ? 127 : exponent);
  }
  const float scale = ldexpf(1.0f, exponent);
  const uint8_t code = adangel::encode_e2m1_rne(value / scale);

  // The full-warp mask requires every lane to execute this intrinsic.  Only
  // even lanes consume the shuffled value, but odd lanes must still
  // participate so that lane 2*i can safely read the code from lane 2*i+1.
  const int next_lane_code =
      __shfl_down_sync(0xffffffffu, static_cast<int>(code), 1);
  if ((lane & 1) == 0) {
    const uint8_t high = static_cast<uint8_t>(next_lane_code);
    packed[
        static_cast<int64_t>(row) * (k / 2) +
        static_cast<int64_t>(group) * (kGroupSize / 2) +
        lane / 2] = static_cast<uint8_t>(code | (high << 4));
  }
  if (lane == 0) {
    natural_scale[static_cast<int64_t>(row) * groups + group] =
        static_cast<uint8_t>(exponent + 127);
  }
}

template <class ScaleLayout>
__global__ void adangel_o2_repack_scale(
    const uint8_t* natural,
    uint8_t* physical,
    ScaleLayout layout,
    int rows,
    int groups) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= rows * groups) return;
  const int row = index / groups;
  const int group = index - row * groups;
  const auto coordinate = cute::make_coord(row, group * kGroupSize, 0);
  physical[static_cast<int64_t>(layout(coordinate))] = natural[index];
}

void launch_quantize_activation(
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    at::Tensor& a_mxfp4,
    at::Tensor& a_scale_natural,
    cudaStream_t stream) {
  const int rows = static_cast<int>(a_int8.size(0));
  const int k = static_cast<int>(a_int8.size(1));
  const int groups = k / kGroupSize;
  const int total_groups = rows * groups;
  const int blocks = (total_groups + kQuantWarpsPerBlock - 1) / kQuantWarpsPerBlock;
  adangel_o2_quantize_activation<<<blocks, kQuantThreads, 0, stream>>>(
      a_int8.data_ptr<int8_t>(),
      a_scale.data_ptr<float>(),
      a_mxfp4.data_ptr<uint8_t>(),
      a_scale_natural.data_ptr<uint8_t>(),
      rows,
      k,
      groups);
  check_cuda(cudaGetLastError(), "O2 activation quantization launch");
}

template <class ScaleLayout>
void launch_repack_scale(
    const at::Tensor& natural,
    at::Tensor& physical,
    ScaleLayout const& layout,
    int rows,
    int groups,
    cudaStream_t stream) {
  constexpr int threads = 256;
  const int elements = rows * groups;
  const int blocks = (elements + threads - 1) / threads;
  adangel_o2_repack_scale<<<blocks, threads, 0, stream>>>(
      natural.data_ptr<uint8_t>(),
      physical.data_ptr<uint8_t>(),
      layout,
      rows,
      groups);
  check_cuda(cudaGetLastError(), "O2 CUTLASS scale-layout repack launch");
}

py::dict kernel_metadata(
    size_t workspace_bytes, int groups, int conversion_inner_repeats) {
  py::dict result;
  result["library"] = "CUTLASS";
  result["cutlass_commit"] = kCutlassCommit;
  result["implementation"] = "cutlass_sm120_mxf4_tma_warp_specialized";
  result["kernel_symbol"] = "cutlass::device_kernel<GemmUniversal<SM120_MXFP4>>";
  result["data_movement"] = "TMA";
  result["kernel_schedule"] = "cooperative_warp_specialized";
  result["stage_count_policy"] = "StageCountAutoCarveout";
  result["cta_tile"] = py::make_tuple(128, 128, 256);
  result["cluster"] = py::make_tuple(1, 1, 1);
  result["tensor_core"] = true;
  result["mma_family"] = "MXFP4_BLOCK_SCALED";
  result["mma_shape"] = "m16n8k64";
  result["mma_instruction"] =
      "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X."
      "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0";
  result["input_dtype"] = "mxfp4_e2m1";
  result["scale_dtype"] = "ue8m0";
  result["scale_vector_size"] = kGroupSize;
  result["group_count"] = groups;
  result["accumulation_dtype"] = "fp32";
  result["output_dtype"] = "fp32";
  result["workspace_bytes"] = static_cast<uint64_t>(workspace_bytes);
  result["weight_scale_layout_repack"] = true;
  result["weight_scale_repack_timing_method"] = "batched_cuda_event_average";
  result["weight_scale_repack_timing_isolated"] = true;
  result["weight_scale_repack_inner_repeats"] =
      conversion_inner_repeats;
  result["activation_conversion_timing_method"] =
      "batched_cuda_event_average";
  result["activation_conversion_timing_isolated"] = true;
  result["activation_conversion_inner_repeats"] =
      conversion_inner_repeats;
  result["total_timing_semantics"] =
      "conversion_only_amortized_cold_steady_direct";
  result["global_partial_buffer"] = false;
  result["output_stores_per_element"] = 1;
  return result;
}

}  // namespace

bool adangel_o2_cutlass_is_implemented() { return true; }

py::dict adangel_benchmark_o2(
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
  TORCH_CHECK(a_int8.size(1) % 64 == 0, "O2 K must be divisible by 64");
  const TimingMode mode = parse_mode(mode_name);
  c10::cuda::CUDAGuard device_guard(a_int8.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(a_int8.get_device()).stream();

  TORCH_CHECK(at::isfinite(a_scale).all().item<bool>(), "A_scale contains NaN/Inf");
  TORCH_CHECK(
      !at::any(at::eq(w_scale, 255)).item<bool>(),
      "W_scale contains forbidden UE8M0 NaN code 255");

  const int m = static_cast<int>(a_int8.size(0));
  const int k = static_cast<int>(a_int8.size(1));
  const int n = static_cast<int>(w_mxfp4.size(0));
  const int groups = k / kGroupSize;
  TORCH_CHECK(m % 4 == 0 && n % 4 == 0, "O2 requires M and N divisible by 4");

  const ProblemShape problem_shape = cute::make_shape(m, n, k, 1);
  const auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(problem_shape);
  const auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(problem_shape);
  const int64_t sfa_elements = static_cast<int64_t>(cute::cosize(layout_sfa));
  const int64_t sfb_elements = static_cast<int64_t>(cute::cosize(layout_sfb));

  auto a_mxfp4 = at::empty({m, k / 2}, a_int8.options().dtype(at::kByte));
  auto a_scale_natural = at::empty({m, groups}, a_int8.options().dtype(at::kByte));
  auto sfa_storage = at::zeros({sfa_elements}, a_int8.options().dtype(at::kByte));
  auto sfb_storage = at::zeros({sfb_elements}, a_int8.options().dtype(at::kByte));
  auto output = at::zeros({m, n}, a_int8.options().dtype(at::kFloat));

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;
  using ArrayElementA = typename CollectiveMainloop::ArrayElementA;
  using ArrayElementB = typename CollectiveMainloop::ArrayElementB;

  const auto stride_a =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  const auto stride_b =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  const auto stride_c =
      cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
  const auto stride_d =
      cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));

  typename CollectiveMainloop::Arguments mainloop_arguments{
      reinterpret_cast<ArrayElementA*>(a_mxfp4.data_ptr<uint8_t>()),
      stride_a,
      reinterpret_cast<ArrayElementB*>(w_mxfp4.data_ptr<uint8_t>()),
      stride_b,
      reinterpret_cast<ElementSF*>(sfa_storage.data_ptr<uint8_t>()),
      layout_sfa,
      reinterpret_cast<ElementSF*>(sfb_storage.data_ptr<uint8_t>()),
      layout_sfb};
  typename CollectiveEpilogue::Arguments epilogue_arguments{
      {1.0f, 0.0f},
      output.data_ptr<float>(),
      stride_c,
      output.data_ptr<float>(),
      stride_d};

  cutlass::KernelHardwareInfo hardware_info;
  hardware_info.device_id = a_int8.get_device();
  hardware_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hardware_info.device_id);
  typename GemmKernel::TileScheduler::Arguments scheduler_arguments{};
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      mainloop_arguments,
      epilogue_arguments,
      hardware_info,
      scheduler_arguments};

  Gemm gemm_operator;
  check_cutlass(gemm_operator.can_implement(arguments), "O2 CUTLASS can_implement");
  const size_t workspace_bytes = Gemm::get_workspace_size(arguments);
  auto workspace = at::empty(
      {static_cast<int64_t>(std::max<size_t>(workspace_bytes, 1))},
      a_int8.options().dtype(at::kByte));
  void* workspace_pointer = workspace_bytes == 0 ? nullptr : workspace.data_ptr();
  check_cutlass(
      gemm_operator.initialize(arguments, workspace_pointer, stream),
      "O2 CUTLASS initialize");

  auto convert_weight = [&]() {
    launch_repack_scale(w_scale, sfb_storage, layout_sfb, n, groups, stream);
  };
  auto convert_activation = [&]() {
    launch_quantize_activation(a_int8, a_scale, a_mxfp4, a_scale_natural, stream);
    launch_repack_scale(
        a_scale_natural, sfa_storage, layout_sfa, m, groups, stream);
  };
  auto gemm = [&]() {
    check_cutlass(gemm_operator.run(stream), "O2 CUTLASS run");
  };

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
  check_cuda(cudaStreamSynchronize(stream), "O2 warmup synchronization");

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
  check_cuda(cudaEventSynchronize(final_event), "O2 measurement synchronization");

  // Conversion components use isolated batched intervals. The cold/steady-state
  // main-path events above still contain exactly one online conversion sequence,
  // preserving direct end-to-end latency and the static-weight cache semantics.
  std::vector<float> isolated_weight_ms;
  std::vector<float> isolated_activation_ms;
  std::vector<float> conversion_only_total_ms;
  if (mode == TimingMode::kConversionOnly || mode == TimingMode::kCold) {
    isolated_weight_ms = measure_batched_conversion(
        convert_weight,
        repeats,
        conversion_inner_repeats,
        stream,
        "O2 weight-repack timing synchronization");
  }
  if (mode == TimingMode::kConversionOnly ||
      mode == TimingMode::kCold ||
      mode == TimingMode::kSteadyState) {
    isolated_activation_ms = measure_batched_conversion(
        convert_activation,
        repeats,
        conversion_inner_repeats,
        stream,
        "O2 activation-conversion timing synchronization");
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
        "O2 conversion-only total timing synchronization");
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
      const float value = elapsed_ms(marker.e0, marker.e1);
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

  // conversion_only still returns a valid O2 output, but GEMM is outside its event range.
  if (mode == TimingMode::kConversionOnly) gemm();

  py::dict timings;
  if (!weight_ms.empty()) timings["weight_conversion"] = weight_ms;
  if (!activation_ms.empty()) timings["activation_conversion"] = activation_ms;
  if (!gemm_ms.empty()) timings["gemm"] = gemm_ms;
  timings["total"] = total_ms;

  py::dict result;
  result["output"] = output;
  result["converted_weight"] = py::none();
  result["converted_activation"] = py::make_tuple(a_mxfp4, a_scale_natural);
  result["timings_ms"] = timings;
  result["timing_method"] =
      timing_metadata(mode_name, conversion_inner_repeats);
  result["kernel"] =
      kernel_metadata(workspace_bytes, groups, conversion_inner_repeats);
  return result;
}

py::dict adangel_run_o2(
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4,
    const at::Tensor& w_scale,
    const std::string& mode) {
  py::dict measured =
      adangel_benchmark_o2(
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
