#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cutlass.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>
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
constexpr int kPipelineStages = 3;
constexpr int kProducerWarps = 1;
constexpr int kConsumerWarps = 8;
constexpr int kProducerThreads = kProducerWarps * 32;
constexpr int kConsumerThreads = kConsumerWarps * 32;
constexpr int kThreadsPerBlock = kProducerThreads + kConsumerThreads;
constexpr int kOutputsPerThread = (kTileM * kTileN) / kConsumerThreads;
constexpr int kAStageElements = kTileM * kGroupSize;
constexpr int kBStageElements = kTileN * kGroupSize;
using O1Pipeline = cutlass::PipelineTmaAsync<kPipelineStages>;
using SmemLayoutA = decltype(cute::make_layout(
    cute::make_shape(cute::Int<kTileM>{}, cute::Int<kGroupSize>{}, cute::Int<kPipelineStages>{}),
    cute::make_stride(
        cute::Int<kGroupSize>{},
        cute::Int<1>{},
        cute::Int<kAStageElements>{})));
using SmemLayoutB = decltype(cute::make_layout(
    cute::make_shape(cute::Int<kTileN>{}, cute::Int<kGroupSize>{}, cute::Int<kPipelineStages>{}),
    cute::make_stride(
        cute::Int<kGroupSize>{},
        cute::Int<1>{},
        cute::Int<kBStageElements>{})));
struct alignas(128) O1SharedStorage {
  alignas(128) uint8_t a[kPipelineStages * kAStageElements];
  alignas(128) uint8_t b[kPipelineStages * kBStageElements];
  alignas(32) int32_t partial[kConsumerWarps * 16 * 16];
  alignas(16) O1Pipeline::SharedStorage pipeline;
};

static_assert(kOutputsPerThread == 8);
static_assert(kThreadsPerBlock == 288);

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

struct EventPair {
  CudaEvent begin;
  CudaEvent end;
};

float elapsed_ms(const CudaEvent& begin, const CudaEvent& end) {
  float value = 0.0f;
  check_cuda(cudaEventElapsedTime(&value, begin.get(), end.get()), "cudaEventElapsedTime");
  return value;
}

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

// One CTA owns a 64x32 output tile. Warp 0 is a dedicated TMA producer and warps 1-8
// cooperatively consume a three-stage shared-memory pipeline. The numerical path remains exact:
// every K32 group produces an INT32 partial, applies A_scale*W_scale/2, accumulates in FP32
// registers, and writes each final output element exactly once.
template <class TmaA, class TmaB>
__global__ __launch_bounds__(kThreadsPerBlock) void adangel_o1_tma_warp_specialized(
    CUTE_GRID_CONSTANT TmaA const tma_a,
    CUTE_GRID_CONSTANT TmaB const tma_b,
    const float* a_scale,
    const uint8_t* w_scale,
    float* output,
    int m,
    int n,
    int k,
    int groups) {
  // PipelineTmaAsync::SharedStorage contains barrier objects whose default constructor is
  // intentionally unavailable. Back the aggregate with raw dynamic shared memory so CUDA does
  // not try to construct it, then let PipelineTmaAsync initialize its barriers explicitly.
  extern __shared__ __align__(128) uint8_t shared_bytes[];
  auto& shared_storage = *reinterpret_cast<O1SharedStorage*>(shared_bytes);

  auto mA = tma_a.get_tma_tensor(cute::make_shape(m, k));
  auto mB = tma_b.get_tma_tensor(cute::make_shape(n, k));
  auto cta_tiler = cute::make_shape(
      cute::Int<kTileM>{}, cute::Int<kTileN>{}, cute::Int<kGroupSize>{});
  auto cta_coord = cute::make_coord(
      static_cast<int>(blockIdx.y), static_cast<int>(blockIdx.x), cute::_);
  auto gA = cute::local_tile(
      mA, cta_tiler, cta_coord, cute::Step<cute::_1, cute::X, cute::_1>{});
  auto gB = cute::local_tile(
      mB, cta_tiler, cta_coord, cute::Step<cute::X, cute::_1, cute::_1>{});
  auto sA = cute::make_tensor(cute::make_smem_ptr(shared_storage.a), SmemLayoutA{});
  auto sB = cute::make_tensor(cute::make_smem_ptr(shared_storage.b), SmemLayoutB{});
  auto [tAgA, tAsA] = cute::tma_partition(
      tma_a,
      cute::Int<0>{},
      cute::Layout<cute::_1>{},
      cute::group_modes<0, 2>(sA),
      cute::group_modes<0, 2>(gA));
  auto [tBgB, tBsB] = cute::tma_partition(
      tma_b,
      cute::Int<0>{},
      cute::Layout<cute::_1>{},
      cute::group_modes<0, 2>(sB),
      cute::group_modes<0, 2>(gB));

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread >> 5;
  const int lane = thread & 31;
  typename O1Pipeline::Params pipeline_params;
  pipeline_params.num_consumers = kConsumerThreads;
  pipeline_params.transaction_bytes = kAStageElements + kBStageElements;
  pipeline_params.initializing_warp = 0;
  if (warp == 0 && lane == 0) {
    pipeline_params.role = O1Pipeline::ThreadCategory::Producer;
    pipeline_params.is_leader = 1;
  } else if (warp > 0) {
    pipeline_params.role = O1Pipeline::ThreadCategory::Consumer;
  }
  O1Pipeline pipeline(
      shared_storage.pipeline,
      pipeline_params,
      cute::Shape<cute::_1, cute::_1, cute::_1>{});
  cutlass::pipeline_init_wait(1);

  if (warp == 0) {
    if (lane == 0) {
      auto write_state = cutlass::make_producer_start_state<O1Pipeline>();
      for (int group = 0; group < groups; ++group) {
        pipeline.producer_acquire(write_state);
        auto* tma_barrier = pipeline.producer_get_barrier(write_state);
        cute::copy(
            tma_a.with(*tma_barrier),
            tAgA(cute::_, group),
            tAsA(cute::_, write_state.index()));
        cute::copy(
            tma_b.with(*tma_barrier),
            tBgB(cute::_, group),
            tBsB(cute::_, write_state.index()));
        ++write_state;
      }
      pipeline.producer_tail(write_state);
    }
    return;
  }

  const int compute_thread = thread - kProducerThreads;
  const int compute_warp = compute_thread >> 5;
  const int warp_row = compute_warp >> 1;
  const int warp_column = compute_warp & 1;
  const int tile_row = static_cast<int>(blockIdx.y) * kTileM;
  const int tile_column = static_cast<int>(blockIdx.x) * kTileN;
  const int local_column = compute_thread & (kTileN - 1);
  const int first_local_row = compute_thread >> 5;
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

  typename O1Pipeline::PipelineState read_state;
  for (int group = 0; group < groups; ++group) {
    pipeline.consumer_wait(read_state);
    const int read_stage = read_state.index();
    const int8_t* shared_a = reinterpret_cast<const int8_t*>(
        shared_storage.a + read_stage * kAStageElements);
    const int8_t* shared_b = reinterpret_cast<const int8_t*>(
        shared_storage.b + read_stage * kBStageElements);

    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a_fragment;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b_fragment;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> partial_fragment;
    wmma::fill_fragment(partial_fragment, 0);

    const int a_offset = warp_row * 16 * kGroupSize;
    const int b_offset = warp_column * 16 * kGroupSize;
    wmma::load_matrix_sync(a_fragment, shared_a + a_offset, kGroupSize);
    wmma::load_matrix_sync(b_fragment, shared_b + b_offset, kGroupSize);
    wmma::mma_sync(partial_fragment, a_fragment, b_fragment, partial_fragment);
    wmma::load_matrix_sync(a_fragment, shared_a + a_offset + kWmmaK, kGroupSize);
    wmma::load_matrix_sync(b_fragment, shared_b + b_offset + kWmmaK, kGroupSize);
    wmma::mma_sync(partial_fragment, a_fragment, b_fragment, partial_fragment);
    wmma::store_matrix_sync(
        shared_storage.partial + compute_warp * 16 * 16,
        partial_fragment,
        16,
        wmma::mem_row_major);
    cutlass::arch::NamedBarrier::sync(
        kConsumerThreads, cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);

    const float column_scale = global_column < n
        ? adangel::decode_ue8m0(w_scale[global_column * groups + group])
        : 0.0f;
#pragma unroll
    for (int item = 0; item < kOutputsPerThread; ++item) {
      const int local_row = first_local_row + item * 8;
      const int owner_warp = (local_row / 16) * 2 + local_column / 16;
      const int owner_index = (local_row % 16) * 16 + (local_column % 16);
      const int32_t partial =
          shared_storage.partial[owner_warp * 16 * 16 + owner_index];
      float scale = __fmul_rn(row_scales[item], column_scale);
      scale = __fmul_rn(scale, 0.5f);
      const float contribution = __fmul_rn(static_cast<float>(partial), scale);
      accumulators[item] = group == 0
          ? contribution
          : __fadd_rn(accumulators[item], contribution);
    }

    cutlass::arch::NamedBarrier::sync(
        kConsumerThreads, cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
    pipeline.consumer_release(read_state);
    ++read_state;
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

auto make_o1_tma_a(const at::Tensor& a_int8) {
  const int m = static_cast<int>(a_int8.size(0));
  const int k = static_cast<int>(a_int8.size(1));
  auto tensor = cute::make_tensor(
      reinterpret_cast<const uint8_t*>(a_int8.data_ptr<int8_t>()),
      cute::make_shape(m, k),
      cute::make_stride(k, cute::Int<1>{}));
  auto layout = SmemLayoutA{};
  return cute::make_tma_atom(
      cute::SM90_TMA_LOAD{},
      tensor,
      layout(cute::_, cute::_, cute::Int<0>{}),
      cute::make_shape(cute::Int<kTileM>{}, cute::Int<kGroupSize>{}));
}

auto make_o1_tma_b(const at::Tensor& w_int8) {
  const int n = static_cast<int>(w_int8.size(0));
  const int k = static_cast<int>(w_int8.size(1));
  auto tensor = cute::make_tensor(
      reinterpret_cast<const uint8_t*>(w_int8.data_ptr<int8_t>()),
      cute::make_shape(n, k),
      cute::make_stride(k, cute::Int<1>{}));
  auto layout = SmemLayoutB{};
  return cute::make_tma_atom(
      cute::SM90_TMA_LOAD{},
      tensor,
      layout(cute::_, cute::_, cute::Int<0>{}),
      cute::make_shape(cute::Int<kTileN>{}, cute::Int<kGroupSize>{}));
}

template <class TmaA, class TmaB>
void launch_tma_o1(
    const at::Tensor& a_scale,
    const at::Tensor& w_scale,
    at::Tensor& output,
    int m,
    int n,
    int k,
    TmaA const& tma_a,
    TmaB const& tma_b,
    cudaStream_t stream) {
  const int groups = k / kGroupSize;
  dim3 grid((n + kTileN - 1) / kTileN, (m + kTileM - 1) / kTileM);
  adangel_o1_tma_warp_specialized<<<
      grid, kThreadsPerBlock, sizeof(O1SharedStorage), stream>>>(
      tma_a,
      tma_b,
      a_scale.data_ptr<float>(),
      w_scale.data_ptr<uint8_t>(),
      output.data_ptr<float>(),
      m,
      n,
      k,
      groups);
  check_cuda(cudaGetLastError(), "O1 TMA warp-specialized WMMA launch");
}

py::dict tma_metadata(int groups) {
  py::dict result;
  result["library"] = "CUTLASS CuTe + CUDA WMMA";
  result["algorithm_id"] = -1;
  result["workspace_bytes"] = 0;
  result["numerical_impl_flags"] = 0;
  result["tensor_core"] = true;
  result["mma_family"] = "IMMA";
  result["mma_api"] = "nvcuda::wmma";
  result["mma_shape"] = "m16n16k16";
  result["implementation"] = "tma_warp_specialized";
  result["kernel_symbol"] = "adangel_o1_tma_warp_specialized";
  result["data_movement"] = "TMA";
  result["kernel_schedule"] = "cooperative_warp_specialized";
  result["pipeline_stages"] = kPipelineStages;
  result["producer_warps"] = kProducerWarps;
  result["consumer_warps"] = kConsumerWarps;
  result["tma_operands"] = py::make_tuple("A_int8", "W_int8");
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
  const int groups = static_cast<int>(k / kGroupSize);
  auto w_int8 = at::empty({n, k}, a_int8.options().dtype(at::kChar));
  auto output = at::empty({m, n}, a_int8.options().dtype(at::kFloat));

  // Tensor-map descriptors are tied to the stable preallocated buffers and are intentionally
  // encoded before warmup and before every CUDA Event timing interval.
  auto tma_a = make_o1_tma_a(a_int8);
  auto tma_b = make_o1_tma_b(w_int8);

  auto convert_weight = [&]() { adangel_launch_mxfp4_to_int8(w_mxfp4, w_int8, stream); };
  auto gemm = [&]() {
    launch_tma_o1(
        a_scale,
        w_scale,
        output,
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        tma_a,
        tma_b,
        stream);
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

  // O1 has one conversion component. Measure it in an isolated batched interval;
  // cold total above still contains exactly one weight conversion and one GEMM.
  std::vector<float> isolated_weight_ms;
  if (mode == TimingMode::kConversionOnly || mode == TimingMode::kCold) {
    isolated_weight_ms = measure_batched_conversion(
        convert_weight,
        repeats,
        conversion_inner_repeats,
        stream,
        "O1 weight-conversion timing synchronization");
  }

  std::vector<float> weight_ms;
  std::vector<float> gemm_ms;
  std::vector<float> total_ms;
  weight_ms.reserve(repeats);
  gemm_ms.reserve(repeats);
  total_ms.reserve(repeats);
  for (int iteration = 0; iteration < repeats; ++iteration) {
    const auto& marker = events[iteration];
    if (mode == TimingMode::kConversionOnly) {
      const float value = isolated_weight_ms[iteration];
      weight_ms.push_back(value);
      total_ms.push_back(value);
    } else if (mode == TimingMode::kCold) {
      weight_ms.push_back(isolated_weight_ms[iteration]);
      gemm_ms.push_back(elapsed_ms(marker.e1, marker.e2));
      total_ms.push_back(elapsed_ms(marker.e0, marker.e2));
    } else {
      const float value = elapsed_ms(marker.e0, marker.e1);
      gemm_ms.push_back(value);
      total_ms.push_back(value);
    }
  }

  // conversion_only returns a valid O1 output, but the TMA kernel is outside its event range.
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
  result["timing_method"] =
      timing_metadata(mode_name, conversion_inner_repeats);
  result["kernel"] = tma_metadata(groups);
  return result;
}

py::dict adangel_run_o1(
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4,
    const at::Tensor& w_scale,
    const std::string& mode) {
  py::dict measured =
      adangel_benchmark_o1(
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
  measured["activation_conversion_ms"] = 0.0f;
  measured["gemm_ms"] = scalar("gemm");
  measured["total_ms"] = scalar("total");
  return measured;
}

// Retained as an independent direct-PTX smoke probe. The instruction audit must additionally
// associate TMA loads and integer MMA with the adangel_o1_tma_warp_specialized production kernel.
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
