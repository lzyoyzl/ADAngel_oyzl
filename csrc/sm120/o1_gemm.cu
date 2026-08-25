#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cutlass.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cuda_runtime_api.h>
#include <mma.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <cstdint>
#include <string>
#include <type_traits>
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
  alignas(32) int32_t shared_partial[kConsumerWarps * 16 * 16];
  alignas(16) O1Pipeline::SharedStorage pipeline;
};

static_assert(kOutputsPerThread == 8);
static_assert(kThreadsPerBlock == 288);
constexpr char kProductionO1Implementation[] = "register_64x32";

struct O1Register64Config {
  static constexpr int kTileM = 64;
  static constexpr int kTileN = 32;
  static constexpr int kProducerWarps = 1;
  static constexpr int kConsumerWarps = 8;
  static constexpr int kProducerThreads = 32;
  static constexpr int kConsumerThreads = 256;
  static constexpr int kThreadsPerBlock = 288;
  static constexpr int kGroupsPerStage = 1;
  static constexpr int kAStageElements = 64 * kGroupSize;
  static constexpr int kBStageElements = 32 * kGroupSize;
  using Pipeline = cutlass::PipelineTmaAsync<kPipelineStages>;
  using SmemLayoutA = decltype(cute::make_layout(
      cute::make_shape(cute::_64{}, cute::_32{}, cute::_3{}),
      cute::make_stride(cute::_32{}, cute::_1{}, cute::Int<64 * 32>{})));
  using SmemLayoutB = decltype(cute::make_layout(
      cute::make_shape(cute::_32{}, cute::_32{}, cute::_3{}),
      cute::make_stride(cute::_32{}, cute::_1{}, cute::Int<32 * 32>{})));
  using TiledMma = cute::TiledMMA<
      cute::MMA_Atom<cute::SM80_16x8x32_S32S8S8S32_TN>,
      cute::Layout<cute::Shape<cute::_4, cute::_2, cute::_1>>,
      cute::Tile<cute::_64, cute::_32, cute::_32>>;
};

struct O1Register128Config {
  static constexpr int kTileM = 128;
  static constexpr int kTileN = 128;
  static constexpr int kProducerWarps = 1;
  static constexpr int kConsumerWarps = 16;
  static constexpr int kProducerThreads = 32;
  static constexpr int kConsumerThreads = 512;
  static constexpr int kThreadsPerBlock = 544;
  static constexpr int kGroupsPerStage = 1;
  static constexpr int kAStageElements = 128 * kGroupSize;
  static constexpr int kBStageElements = 128 * kGroupSize;
  using Pipeline = cutlass::PipelineTmaAsync<kPipelineStages>;
  using SmemLayoutA = decltype(cute::make_layout(
      cute::make_shape(cute::_128{}, cute::_32{}, cute::_3{}),
      cute::make_stride(cute::_32{}, cute::_1{}, cute::Int<128 * 32>{})));
  using SmemLayoutB = SmemLayoutA;
  using TiledMma = cute::TiledMMA<
      cute::MMA_Atom<cute::SM80_16x8x32_S32S8S8S32_TN>,
      cute::Layout<cute::Shape<cute::_4, cute::_4, cute::_1>>,
      cute::Tile<cute::_128, cute::_128, cute::_32>>;
};

struct O1Register64K64Config {
  static constexpr int kTileM = 64;
  static constexpr int kTileN = 32;
  static constexpr int kProducerWarps = 1;
  static constexpr int kConsumerWarps = 8;
  static constexpr int kProducerThreads = 32;
  static constexpr int kConsumerThreads = 256;
  static constexpr int kThreadsPerBlock = 288;
  static constexpr int kGroupsPerStage = 2;
  static constexpr int kAGroupElements = 64 * kGroupSize;
  static constexpr int kBGroupElements = 32 * kGroupSize;
  static constexpr int kAStageElements = kGroupsPerStage * kAGroupElements;
  static constexpr int kBStageElements = kGroupsPerStage * kBGroupElements;
  using Pipeline = cutlass::PipelineTmaAsync<kPipelineStages>;
  using SmemLayoutA = decltype(cute::make_layout(
      cute::make_shape(cute::_64{}, cute::_32{}, cute::_2{}, cute::_3{}),
      cute::make_stride(
          cute::_32{},
          cute::_1{},
          cute::Int<64 * 32>{},
          cute::Int<2 * 64 * 32>{})));
  using SmemLayoutB = decltype(cute::make_layout(
      cute::make_shape(cute::_32{}, cute::_32{}, cute::_2{}, cute::_3{}),
      cute::make_stride(
          cute::_32{},
          cute::_1{},
          cute::Int<32 * 32>{},
          cute::Int<2 * 32 * 32>{})));
  using TiledMma = cute::TiledMMA<
      cute::MMA_Atom<cute::SM80_16x8x32_S32S8S8S32_TN>,
      cute::Layout<cute::Shape<cute::_4, cute::_2, cute::_1>>,
      cute::Tile<cute::_64, cute::_32, cute::_32>>;
};

// Same K64 pipeline as O1Register64K64Config, but doubles the CTA M extent.
// Sixteen consumer warps keep the per-thread accumulator footprint unchanged
// while each CTA reuses one W/scale tile for twice as many output rows.
struct O1Register128x32K64Config {
  static constexpr int kTileM = 128;
  static constexpr int kTileN = 32;
  static constexpr int kProducerWarps = 1;
  static constexpr int kConsumerWarps = 16;
  static constexpr int kProducerThreads = 32;
  static constexpr int kConsumerThreads = 512;
  static constexpr int kThreadsPerBlock = 544;
  static constexpr int kGroupsPerStage = 2;
  static constexpr int kAGroupElements = 128 * kGroupSize;
  static constexpr int kBGroupElements = 32 * kGroupSize;
  static constexpr int kAStageElements = kGroupsPerStage * kAGroupElements;
  static constexpr int kBStageElements = kGroupsPerStage * kBGroupElements;
  using Pipeline = cutlass::PipelineTmaAsync<kPipelineStages>;
  using SmemLayoutA = decltype(cute::make_layout(
      cute::make_shape(cute::_128{}, cute::_32{}, cute::_2{}, cute::_3{}),
      cute::make_stride(
          cute::_32{},
          cute::_1{},
          cute::Int<128 * 32>{},
          cute::Int<2 * 128 * 32>{})));
  using SmemLayoutB = decltype(cute::make_layout(
      cute::make_shape(cute::_32{}, cute::_32{}, cute::_2{}, cute::_3{}),
      cute::make_stride(
          cute::_32{},
          cute::_1{},
          cute::Int<32 * 32>{},
          cute::Int<2 * 32 * 32>{})));
  using TiledMma = cute::TiledMMA<
      cute::MMA_Atom<cute::SM80_16x8x32_S32S8S8S32_TN>,
      cute::Layout<cute::Shape<cute::_8, cute::_2, cute::_1>>,
      cute::Tile<cute::_128, cute::_32, cute::_32>>;
};

template <class Config>
struct alignas(128) O1RegisterSharedStorage {
  alignas(128) uint8_t a[kPipelineStages * Config::kAStageElements];
  alignas(128) uint8_t b[kPipelineStages * Config::kBStageElements];
  alignas(16) typename Config::Pipeline::SharedStorage pipeline;
};

// Candidate storage for CTA-wide W-scale reuse.  The producer warp decodes each
// (column, K32 group) scale exactly once and publishes the FP32 scale/2 factor
// through the same three-stage lifetime as the TMA operand buffers.
template <class Config>
struct alignas(128) O1RegisterScaleSharedStorage {
  alignas(128) uint8_t a[kPipelineStages * Config::kAStageElements];
  alignas(128) uint8_t b[kPipelineStages * Config::kBStageElements];
  alignas(128) float column_scale_factor[
      kPipelineStages * Config::kGroupsPerStage * Config::kTileN];
  alignas(16) typename Config::Pipeline::SharedStorage pipeline;
};

enum class O1Implementation {
  kSharedPartial,
  kRegister64x32,
  kRegister64x32ScaleShared,
  kRegister64x32K64ScaleShared,
  kRegister128x32K64ScaleShared,
  kRegister128x128,
};

static_assert(O1Register64Config::kThreadsPerBlock == 288);
static_assert(O1Register64K64Config::kThreadsPerBlock == 288);
static_assert(O1Register128x32K64Config::kThreadsPerBlock == 544);
static_assert(O1Register128Config::kThreadsPerBlock == 544);

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
__global__ __launch_bounds__(kThreadsPerBlock) void adangel_o1_shared_partial_baseline(
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
        shared_storage.shared_partial + compute_warp * 16 * 16,
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
          shared_storage.shared_partial[owner_warp * 16 * 16 + owner_index];
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

template <class Config, bool kShareColumnScale, class TmaA, class TmaB, class SharedStorage>
__device__ __forceinline__ void adangel_o1_register_partial_body(
    TmaA const& tma_a,
    TmaB const& tma_b,
    const float* a_scale,
    const uint8_t* w_scale,
    float* output,
    int m,
    int n,
    int k,
    int groups,
    SharedStorage& shared_storage) {
  auto mA = tma_a.get_tma_tensor(cute::make_shape(m, k));
  auto mB = tma_b.get_tma_tensor(cute::make_shape(n, k));
  auto cta_tiler = cute::make_shape(
      cute::Int<Config::kTileM>{},
      cute::Int<Config::kTileN>{},
      cute::Int<kGroupSize>{});
  auto cta_coord = cute::make_coord(
      static_cast<int>(blockIdx.y), static_cast<int>(blockIdx.x), cute::_);
  auto gA = cute::local_tile(
      mA, cta_tiler, cta_coord, cute::Step<cute::_1, cute::X, cute::_1>{});
  auto gB = cute::local_tile(
      mB, cta_tiler, cta_coord, cute::Step<cute::X, cute::_1, cute::_1>{});
  auto sA = cute::make_tensor(
      cute::make_smem_ptr(shared_storage.a), typename Config::SmemLayoutA{});
  auto sB = cute::make_tensor(
      cute::make_smem_ptr(shared_storage.b), typename Config::SmemLayoutB{});
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
  typename Config::Pipeline::Params pipeline_params;
  pipeline_params.num_consumers = Config::kConsumerThreads;
  pipeline_params.transaction_bytes =
      Config::kAStageElements + Config::kBStageElements;
  pipeline_params.initializing_warp = 0;
  if (warp == 0 && lane == 0) {
    pipeline_params.role = Config::Pipeline::ThreadCategory::Producer;
    pipeline_params.is_leader = 1;
  } else if (warp > 0) {
    pipeline_params.role = Config::Pipeline::ThreadCategory::Consumer;
  }
  typename Config::Pipeline pipeline(
      shared_storage.pipeline,
      pipeline_params,
      cute::Shape<cute::_1, cute::_1, cute::_1>{});
  cutlass::pipeline_init_wait(1);

  if (warp == 0) {
    if constexpr (kShareColumnScale) {
      auto write_state =
          cutlass::make_producer_start_state<typename Config::Pipeline>();
      for (int group = 0; group < groups; ++group) {
        if (lane == 0) pipeline.producer_acquire(write_state);
        __syncwarp();
        const int write_stage = write_state.index();
        for (int local_column = lane;
             local_column < Config::kTileN;
             local_column += 32) {
          const int global_column =
              static_cast<int>(blockIdx.x) * Config::kTileN + local_column;
          const float decoded = global_column < n
              ? adangel::decode_ue8m0(
                    w_scale[global_column * groups + group])
              : 0.0f;
          shared_storage.column_scale_factor[
              write_stage * Config::kTileN + local_column] =
              __fmul_rn(decoded, 0.5f);
        }
        __threadfence_block();
        __syncwarp();
        if (lane == 0) {
          auto* tma_barrier = pipeline.producer_get_barrier(write_state);
          cute::copy(
              tma_a.with(*tma_barrier),
              tAgA(cute::_, group),
              tAsA(cute::_, write_stage));
          cute::copy(
              tma_b.with(*tma_barrier),
              tBgB(cute::_, group),
              tBsB(cute::_, write_stage));
        }
        ++write_state;
      }
      if (lane == 0) pipeline.producer_tail(write_state);
    } else if (lane == 0) {
      auto write_state =
          cutlass::make_producer_start_state<typename Config::Pipeline>();
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

  const int compute_thread = thread - Config::kProducerThreads;
  const int tile_row = static_cast<int>(blockIdx.y) * Config::kTileM;
  const int tile_column = static_cast<int>(blockIdx.x) * Config::kTileN;
  typename Config::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_slice(compute_thread);
  auto tCrA = thr_mma.partition_fragment_A(sA(cute::_, cute::_, cute::Int<0>{}));
  auto tCrB = thr_mma.partition_fragment_B(sB(cute::_, cute::_, cute::Int<0>{}));
  auto cC = cute::make_identity_tensor(
      cute::make_shape(
          cute::Int<Config::kTileM>{}, cute::Int<Config::kTileN>{}));
  auto tCcC = thr_mma.partition_C(cC);
  auto tCrPartial = thr_mma.make_fragment_C(tCcC);
  auto tCrAccumulator = cute::make_fragment_like<float>(tCrPartial);
  auto tCrRowScale = cute::make_fragment_like<float>(tCrPartial);
  cute::clear(tCrAccumulator);
  for (int item = 0; item < cute::size(tCrRowScale); ++item) {
    const int local_row = cute::get<0>(tCcC(item));
    const int global_row = tile_row + local_row;
    tCrRowScale(item) = global_row < m ? a_scale[global_row] : 0.0f;
  }

  using SmemCopyAtom =
      cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, uint8_t>;
  SmemCopyAtom smem_copy_atom;
  auto tiled_copy_a = cute::make_tiled_copy_A(smem_copy_atom, tiled_mma);
  auto tiled_copy_b = cute::make_tiled_copy_B(smem_copy_atom, tiled_mma);
  auto thr_copy_a = tiled_copy_a.get_slice(compute_thread);
  auto thr_copy_b = tiled_copy_b.get_slice(compute_thread);
  auto tXsA = thr_copy_a.partition_S(sA);
  auto tXsB = thr_copy_b.partition_S(sB);
  auto tXrA = thr_copy_a.retile_D(tCrA);
  auto tXrB = thr_copy_b.retile_D(tCrB);

  typename Config::Pipeline::PipelineState read_state;
  for (int group = 0; group < groups; ++group) {
    pipeline.consumer_wait(read_state);
    const int read_stage = read_state.index();
    cute::copy(
        smem_copy_atom,
        tXsA(cute::_, cute::_, cute::_, read_stage),
        tXrA);
    cute::copy(
        smem_copy_atom,
        tXsB(cute::_, cute::_, cute::_, read_stage),
        tXrB);
    cute::clear(tCrPartial);
    cute::gemm(tiled_mma, tCrA, tCrB, tCrPartial);

    float warp_column_scales[Config::kTileN / 32];
#pragma unroll
    for (int round = 0; round < Config::kTileN / 32; ++round) {
      const int local_column = round * 32 + lane;
      const int global_column = tile_column + local_column;
      if constexpr (kShareColumnScale) {
        warp_column_scales[round] =
            shared_storage.column_scale_factor[
                read_stage * Config::kTileN + local_column];
      } else {
        warp_column_scales[round] = global_column < n
            ? adangel::decode_ue8m0(
                  w_scale[global_column * groups + group])
            : 0.0f;
      }
    }

    for (int item = 0; item < cute::size(tCrPartial); ++item) {
      const int local_column = cute::get<1>(tCcC(item));
      float column_scale = 0.0f;
#pragma unroll
      for (int round = 0; round < Config::kTileN / 32; ++round) {
        const float candidate = __shfl_sync(
            0xffffffffu, warp_column_scales[round], local_column & 31);
        if ((local_column >> 5) == round) column_scale = candidate;
      }
      if constexpr (kShareColumnScale) {
        const float scale = __fmul_rn(tCrRowScale(item), column_scale);
        tCrAccumulator(item) = __fmaf_rn(
            static_cast<float>(tCrPartial(item)),
            scale,
            tCrAccumulator(item));
      } else {
        float scale = __fmul_rn(tCrRowScale(item), column_scale);
        scale = __fmul_rn(scale, 0.5f);
        const float contribution =
            __fmul_rn(static_cast<float>(tCrPartial(item)), scale);
        tCrAccumulator(item) = group == 0
            ? contribution
            : __fadd_rn(tCrAccumulator(item), contribution);
      }
    }

    pipeline.consumer_release(read_state);
    ++read_state;
  }

  auto mC = cute::make_tensor(
      cute::make_gmem_ptr(output),
      cute::make_shape(m, n),
      cute::make_stride(n, cute::Int<1>{}));
  auto gC = cute::local_tile(
      mC,
      cute::make_shape(
          cute::Int<Config::kTileM>{}, cute::Int<Config::kTileN>{}),
      cute::make_coord(
          static_cast<int>(blockIdx.y), static_cast<int>(blockIdx.x)));
  auto tCgC = thr_mma.partition_C(gC);
  for (int item = 0; item < cute::size(tCrAccumulator); ++item) {
    const int local_row = cute::get<0>(tCcC(item));
    const int local_column = cute::get<1>(tCcC(item));
    if (tile_row + local_row < m && tile_column + local_column < n) {
      tCgC(item) = tCrAccumulator(item);
    }
  }
}

// K64 pipeline candidate: one producer/consumer pipeline state owns two adjacent
// K32 groups.  The two INT8 MMAs and their UE8M0 scale applications remain
// independent, so the public K32 block-scale semantics are unchanged.
template <class Config, class TmaA, class TmaB>
__device__ __forceinline__ void adangel_o1_register_partial_k64_body(
    TmaA const& tma_a,
    TmaB const& tma_b,
    const float* a_scale,
    const uint8_t* w_scale,
    float* output,
    int m,
    int n,
    int k,
    int groups,
    O1RegisterScaleSharedStorage<Config>& shared_storage) {
  auto mA = tma_a.get_tma_tensor(cute::make_shape(m, k));
  auto mB = tma_b.get_tma_tensor(cute::make_shape(n, k));
  auto cta_tiler = cute::make_shape(
      cute::Int<Config::kTileM>{},
      cute::Int<Config::kTileN>{},
      cute::Int<kGroupSize>{});
  auto cta_coord = cute::make_coord(
      static_cast<int>(blockIdx.y), static_cast<int>(blockIdx.x), cute::_);
  auto gA = cute::local_tile(
      mA, cta_tiler, cta_coord, cute::Step<cute::_1, cute::X, cute::_1>{});
  auto gB = cute::local_tile(
      mB, cta_tiler, cta_coord, cute::Step<cute::X, cute::_1, cute::_1>{});
  auto sA = cute::make_tensor(
      cute::make_smem_ptr(shared_storage.a), typename Config::SmemLayoutA{});
  auto sB = cute::make_tensor(
      cute::make_smem_ptr(shared_storage.b), typename Config::SmemLayoutB{});
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
  typename Config::Pipeline::Params pipeline_params;
  pipeline_params.num_consumers = Config::kConsumerThreads;
  pipeline_params.transaction_bytes =
      Config::kAStageElements + Config::kBStageElements;
  pipeline_params.initializing_warp = 0;
  if (warp == 0 && lane == 0) {
    pipeline_params.role = Config::Pipeline::ThreadCategory::Producer;
    pipeline_params.is_leader = 1;
  } else if (warp > 0) {
    pipeline_params.role = Config::Pipeline::ThreadCategory::Consumer;
  }
  typename Config::Pipeline pipeline(
      shared_storage.pipeline,
      pipeline_params,
      cute::Shape<cute::_1, cute::_1, cute::_1>{});
  cutlass::pipeline_init_wait(1);

  const int pipeline_groups =
      (groups + Config::kGroupsPerStage - 1) / Config::kGroupsPerStage;
  if (warp == 0) {
    auto write_state =
        cutlass::make_producer_start_state<typename Config::Pipeline>();
    for (int pipeline_group = 0;
         pipeline_group < pipeline_groups;
         ++pipeline_group) {
      if (lane == 0) pipeline.producer_acquire(write_state);
      __syncwarp();
      const int write_stage = write_state.index();
#pragma unroll
      for (int subgroup = 0; subgroup < Config::kGroupsPerStage; ++subgroup) {
        const int group =
            pipeline_group * Config::kGroupsPerStage + subgroup;
        for (int local_column = lane;
             local_column < Config::kTileN;
             local_column += 32) {
          const int global_column =
              static_cast<int>(blockIdx.x) * Config::kTileN + local_column;
          const float decoded = group < groups && global_column < n
              ? adangel::decode_ue8m0(
                    w_scale[global_column * groups + group])
              : 0.0f;
          shared_storage.column_scale_factor[
              (write_stage * Config::kGroupsPerStage + subgroup) *
                  Config::kTileN +
              local_column] = __fmul_rn(decoded, 0.5f);
        }
      }
      __threadfence_block();
      __syncwarp();
      if (lane == 0) {
        auto* tma_barrier = pipeline.producer_get_barrier(write_state);
#pragma unroll
        for (int subgroup = 0; subgroup < Config::kGroupsPerStage; ++subgroup) {
          const int requested_group =
              pipeline_group * Config::kGroupsPerStage + subgroup;
          const int source_group = requested_group < groups ? requested_group : 0;
          cute::copy(
              tma_a.with(*tma_barrier),
              tAgA(cute::_, source_group),
              tAsA(cute::_, subgroup, write_stage));
          cute::copy(
              tma_b.with(*tma_barrier),
              tBgB(cute::_, source_group),
              tBsB(cute::_, subgroup, write_stage));
        }
      }
      ++write_state;
    }
    if (lane == 0) pipeline.producer_tail(write_state);
    return;
  }

  const int compute_thread = thread - Config::kProducerThreads;
  const int tile_row = static_cast<int>(blockIdx.y) * Config::kTileM;
  const int tile_column = static_cast<int>(blockIdx.x) * Config::kTileN;
  typename Config::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_slice(compute_thread);
  auto tCrA = thr_mma.partition_fragment_A(
      sA(cute::_, cute::_, cute::Int<0>{}, cute::Int<0>{}));
  auto tCrB = thr_mma.partition_fragment_B(
      sB(cute::_, cute::_, cute::Int<0>{}, cute::Int<0>{}));
  auto cC = cute::make_identity_tensor(
      cute::make_shape(
          cute::Int<Config::kTileM>{}, cute::Int<Config::kTileN>{}));
  auto tCcC = thr_mma.partition_C(cC);
  auto tCrPartial = thr_mma.make_fragment_C(tCcC);
  auto tCrAccumulator = cute::make_fragment_like<float>(tCrPartial);
  auto tCrRowScale = cute::make_fragment_like<float>(tCrPartial);
  cute::clear(tCrAccumulator);
  for (int item = 0; item < cute::size(tCrRowScale); ++item) {
    const int local_row = cute::get<0>(tCcC(item));
    const int global_row = tile_row + local_row;
    tCrRowScale(item) = global_row < m ? a_scale[global_row] : 0.0f;
  }

  using SmemCopyAtom =
      cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, uint8_t>;
  SmemCopyAtom smem_copy_atom;
  auto tiled_copy_a = cute::make_tiled_copy_A(smem_copy_atom, tiled_mma);
  auto tiled_copy_b = cute::make_tiled_copy_B(smem_copy_atom, tiled_mma);
  auto thr_copy_a = tiled_copy_a.get_slice(compute_thread);
  auto thr_copy_b = tiled_copy_b.get_slice(compute_thread);
  auto tXrA = thr_copy_a.retile_D(tCrA);
  auto tXrB = thr_copy_b.retile_D(tCrB);

  typename Config::Pipeline::PipelineState read_state;
  for (int pipeline_group = 0;
       pipeline_group < pipeline_groups;
       ++pipeline_group) {
    pipeline.consumer_wait(read_state);
    const int read_stage = read_state.index();
#pragma unroll
    for (int subgroup = 0; subgroup < Config::kGroupsPerStage; ++subgroup) {
      const int group =
          pipeline_group * Config::kGroupsPerStage + subgroup;
      if (group >= groups) continue;
      auto tXsA = thr_copy_a.partition_S(
          sA(cute::_, cute::_, subgroup, read_stage));
      auto tXsB = thr_copy_b.partition_S(
          sB(cute::_, cute::_, subgroup, read_stage));
      cute::copy(smem_copy_atom, tXsA, tXrA);
      cute::copy(smem_copy_atom, tXsB, tXrB);
      cute::clear(tCrPartial);
      cute::gemm(tiled_mma, tCrA, tCrB, tCrPartial);

      float warp_column_scales[Config::kTileN / 32];
#pragma unroll
      for (int round = 0; round < Config::kTileN / 32; ++round) {
        const int local_column = round * 32 + lane;
        warp_column_scales[round] =
            shared_storage.column_scale_factor[
                (read_stage * Config::kGroupsPerStage + subgroup) *
                    Config::kTileN +
                local_column];
      }
      for (int item = 0; item < cute::size(tCrPartial); ++item) {
        const int local_column = cute::get<1>(tCcC(item));
        float column_scale = 0.0f;
#pragma unroll
        for (int round = 0; round < Config::kTileN / 32; ++round) {
          const float candidate = __shfl_sync(
              0xffffffffu, warp_column_scales[round], local_column & 31);
          if ((local_column >> 5) == round) column_scale = candidate;
        }
        const float scale = __fmul_rn(tCrRowScale(item), column_scale);
        tCrAccumulator(item) = __fmaf_rn(
            static_cast<float>(tCrPartial(item)),
            scale,
            tCrAccumulator(item));
      }
    }
    pipeline.consumer_release(read_state);
    ++read_state;
  }

  auto mC = cute::make_tensor(
      cute::make_gmem_ptr(output),
      cute::make_shape(m, n),
      cute::make_stride(n, cute::Int<1>{}));
  auto gC = cute::local_tile(
      mC,
      cute::make_shape(
          cute::Int<Config::kTileM>{}, cute::Int<Config::kTileN>{}),
      cute::make_coord(
          static_cast<int>(blockIdx.y), static_cast<int>(blockIdx.x)));
  auto tCgC = thr_mma.partition_C(gC);
  for (int item = 0; item < cute::size(tCrAccumulator); ++item) {
    const int local_row = cute::get<0>(tCcC(item));
    const int local_column = cute::get<1>(tCcC(item));
    if (tile_row + local_row < m && tile_column + local_column < n) {
      tCgC(item) = tCrAccumulator(item);
    }
  }
}

template <class TmaA, class TmaB>
__global__ __launch_bounds__(O1Register64Config::kThreadsPerBlock)
void adangel_o1_register_partial_64x32(
    CUTE_GRID_CONSTANT TmaA const tma_a,
    CUTE_GRID_CONSTANT TmaB const tma_b,
    const float* a_scale,
    const uint8_t* w_scale,
    float* output,
    int m,
    int n,
    int k,
    int groups) {
  extern __shared__ __align__(128) uint8_t shared_bytes[];
  auto& shared_storage =
      *reinterpret_cast<O1RegisterSharedStorage<O1Register64Config>*>(shared_bytes);
  adangel_o1_register_partial_body<O1Register64Config, false>(
      tma_a, tma_b, a_scale, w_scale, output, m, n, k, groups, shared_storage);
}

template <class TmaA, class TmaB>
__global__ __launch_bounds__(O1Register64Config::kThreadsPerBlock)
void adangel_o1_register_partial_64x32_scale_shared(
    CUTE_GRID_CONSTANT TmaA const tma_a,
    CUTE_GRID_CONSTANT TmaB const tma_b,
    const float* a_scale,
    const uint8_t* w_scale,
    float* output,
    int m,
    int n,
    int k,
    int groups) {
  extern __shared__ __align__(128) uint8_t shared_bytes[];
  auto& shared_storage =
      *reinterpret_cast<O1RegisterScaleSharedStorage<O1Register64Config>*>(
          shared_bytes);
  adangel_o1_register_partial_body<O1Register64Config, true>(
      tma_a, tma_b, a_scale, w_scale, output, m, n, k, groups, shared_storage);
}

template <class TmaA, class TmaB>
__global__ __launch_bounds__(O1Register64K64Config::kThreadsPerBlock)
void adangel_o1_register_partial_64x32_k64_scale_shared(
    CUTE_GRID_CONSTANT TmaA const tma_a,
    CUTE_GRID_CONSTANT TmaB const tma_b,
    const float* a_scale,
    const uint8_t* w_scale,
    float* output,
    int m,
    int n,
    int k,
    int groups) {
  extern __shared__ __align__(128) uint8_t shared_bytes[];
  auto& shared_storage =
      *reinterpret_cast<O1RegisterScaleSharedStorage<
          O1Register64K64Config>*>(shared_bytes);
  adangel_o1_register_partial_k64_body<O1Register64K64Config>(
      tma_a, tma_b, a_scale, w_scale, output, m, n, k, groups, shared_storage);
}

template <class TmaA, class TmaB>
__global__ __launch_bounds__(O1Register128x32K64Config::kThreadsPerBlock)
void adangel_o1_register_partial_128x32_k64_scale_shared(
    CUTE_GRID_CONSTANT TmaA const tma_a,
    CUTE_GRID_CONSTANT TmaB const tma_b,
    const float* a_scale,
    const uint8_t* w_scale,
    float* output,
    int m,
    int n,
    int k,
    int groups) {
  extern __shared__ __align__(128) uint8_t shared_bytes[];
  auto& shared_storage =
      *reinterpret_cast<O1RegisterScaleSharedStorage<
          O1Register128x32K64Config>*>(shared_bytes);
  adangel_o1_register_partial_k64_body<O1Register128x32K64Config>(
      tma_a, tma_b, a_scale, w_scale, output, m, n, k, groups, shared_storage);
}

template <class TmaA, class TmaB>
__global__ __launch_bounds__(O1Register128Config::kThreadsPerBlock)
void adangel_o1_register_partial_128x128(
    CUTE_GRID_CONSTANT TmaA const tma_a,
    CUTE_GRID_CONSTANT TmaB const tma_b,
    const float* a_scale,
    const uint8_t* w_scale,
    float* output,
    int m,
    int n,
    int k,
    int groups) {
  extern __shared__ __align__(128) uint8_t shared_bytes[];
  auto& shared_storage =
      *reinterpret_cast<O1RegisterSharedStorage<O1Register128Config>*>(shared_bytes);
  adangel_o1_register_partial_body<O1Register128Config, false>(
      tma_a, tma_b, a_scale, w_scale, output, m, n, k, groups, shared_storage);
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
  adangel_o1_shared_partial_baseline<<<
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

template <class Config>
auto make_register_tma_a(const at::Tensor& a_int8) {
  const int m = static_cast<int>(a_int8.size(0));
  const int k = static_cast<int>(a_int8.size(1));
  auto tensor = cute::make_tensor(
      reinterpret_cast<const uint8_t*>(a_int8.data_ptr<int8_t>()),
      cute::make_shape(m, k),
      cute::make_stride(k, cute::Int<1>{}));
  auto layout = typename Config::SmemLayoutA{};
  if constexpr (Config::kGroupsPerStage == 1) {
    return cute::make_tma_atom(
        cute::SM90_TMA_LOAD{},
        tensor,
        layout(cute::_, cute::_, cute::Int<0>{}),
        cute::make_shape(
            cute::Int<Config::kTileM>{}, cute::Int<kGroupSize>{}));
  } else {
    return cute::make_tma_atom(
        cute::SM90_TMA_LOAD{},
        tensor,
        layout(cute::_, cute::_, cute::Int<0>{}, cute::Int<0>{}),
        cute::make_shape(
            cute::Int<Config::kTileM>{}, cute::Int<kGroupSize>{}));
  }
}

template <class Config>
auto make_register_tma_b(const at::Tensor& w_int8) {
  const int n = static_cast<int>(w_int8.size(0));
  const int k = static_cast<int>(w_int8.size(1));
  auto tensor = cute::make_tensor(
      reinterpret_cast<const uint8_t*>(w_int8.data_ptr<int8_t>()),
      cute::make_shape(n, k),
      cute::make_stride(k, cute::Int<1>{}));
  auto layout = typename Config::SmemLayoutB{};
  if constexpr (Config::kGroupsPerStage == 1) {
    return cute::make_tma_atom(
        cute::SM90_TMA_LOAD{},
        tensor,
        layout(cute::_, cute::_, cute::Int<0>{}),
        cute::make_shape(
            cute::Int<Config::kTileN>{}, cute::Int<kGroupSize>{}));
  } else {
    return cute::make_tma_atom(
        cute::SM90_TMA_LOAD{},
        tensor,
        layout(cute::_, cute::_, cute::Int<0>{}, cute::Int<0>{}),
        cute::make_shape(
            cute::Int<Config::kTileN>{}, cute::Int<kGroupSize>{}));
  }
}

template <class Config, bool kShareColumnScale = false, class TmaA, class TmaB>
void launch_register_o1(
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
  dim3 grid(
      (n + Config::kTileN - 1) / Config::kTileN,
      (m + Config::kTileM - 1) / Config::kTileM);
  constexpr int shared_bytes = kShareColumnScale
      ? sizeof(O1RegisterScaleSharedStorage<Config>)
      : sizeof(O1RegisterSharedStorage<Config>);
  if constexpr (std::is_same_v<Config, O1Register64K64Config>) {
    adangel_o1_register_partial_64x32_k64_scale_shared<<<
        grid, Config::kThreadsPerBlock, shared_bytes, stream>>>(
        tma_a,
        tma_b,
        a_scale.data_ptr<float>(),
        w_scale.data_ptr<uint8_t>(),
        output.data_ptr<float>(),
        m,
        n,
        k,
        groups);
  } else if constexpr (
      std::is_same_v<Config, O1Register128x32K64Config>) {
    adangel_o1_register_partial_128x32_k64_scale_shared<<<
        grid, Config::kThreadsPerBlock, shared_bytes, stream>>>(
        tma_a,
        tma_b,
        a_scale.data_ptr<float>(),
        w_scale.data_ptr<uint8_t>(),
        output.data_ptr<float>(),
        m,
        n,
        k,
        groups);
  } else if constexpr (std::is_same_v<Config, O1Register64Config>) {
    if constexpr (kShareColumnScale) {
      adangel_o1_register_partial_64x32_scale_shared<<<
          grid, Config::kThreadsPerBlock, shared_bytes, stream>>>(
          tma_a,
          tma_b,
          a_scale.data_ptr<float>(),
          w_scale.data_ptr<uint8_t>(),
          output.data_ptr<float>(),
          m,
          n,
          k,
          groups);
    } else {
      adangel_o1_register_partial_64x32<<<
          grid, Config::kThreadsPerBlock, shared_bytes, stream>>>(
          tma_a,
          tma_b,
          a_scale.data_ptr<float>(),
          w_scale.data_ptr<uint8_t>(),
          output.data_ptr<float>(),
          m,
          n,
          k,
          groups);
    }
  } else {
    adangel_o1_register_partial_128x128<<<
        grid, Config::kThreadsPerBlock, shared_bytes, stream>>>(
        tma_a,
        tma_b,
        a_scale.data_ptr<float>(),
        w_scale.data_ptr<uint8_t>(),
        output.data_ptr<float>(),
        m,
        n,
        k,
        groups);
  }
  check_cuda(cudaGetLastError(), "O1 register-partial launch");
}

O1Implementation parse_o1_implementation(const std::string& implementation) {
  const std::string selected = implementation == "production"
      ? std::string(kProductionO1Implementation)
      : implementation;
  if (selected == "shared_partial") return O1Implementation::kSharedPartial;
  if (selected == "register_64x32") return O1Implementation::kRegister64x32;
  if (selected == "register_64x32_scale_shared") {
    return O1Implementation::kRegister64x32ScaleShared;
  }
  if (selected == "register_64x32_k64_scale_shared") {
    return O1Implementation::kRegister64x32K64ScaleShared;
  }
  if (selected == "register_128x32_k64_scale_shared") {
    return O1Implementation::kRegister128x32K64ScaleShared;
  }
  if (selected == "register_128x128") return O1Implementation::kRegister128x128;
  TORCH_CHECK(false, "unknown O1 implementation: ", implementation);
  return O1Implementation::kSharedPartial;
}

py::dict o1_metadata(O1Implementation implementation, int groups) {
  py::dict result;
  result["algorithm_id"] = -1;
  result["workspace_bytes"] = 0;
  result["numerical_impl_flags"] = 0;
  result["tensor_core"] = true;
  result["mma_family"] = "IMMA";
  result["data_movement"] = "TMA";
  result["kernel_schedule"] = "cooperative_warp_specialized";
  result["pipeline_stages"] = kPipelineStages;
  result["tma_operands"] = py::make_tuple("A_int8", "W_int8");
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
  result["production_selected"] =
      implementation == parse_o1_implementation("production");
  if (implementation == O1Implementation::kSharedPartial) {
    result["library"] = "CUTLASS CuTe + CUDA WMMA";
    result["mma_api"] = "nvcuda::wmma";
    result["mma_atom"] = py::none();
    result["mma_shape"] = "m16n16k16";
    result["implementation"] = "tma_warp_specialized_shared_partial";
    result["implementation_key"] = "shared_partial";
    result["kernel_symbol"] = "adangel_o1_shared_partial_baseline";
    result["producer_warps"] = kProducerWarps;
    result["consumer_warps"] = kConsumerWarps;
    result["cta_tile"] = py::make_tuple(kTileM, kTileN, kGroupSize);
    result["partial_storage"] = "shared_memory";
    result["shared_partial_redistribution"] = true;
  } else {
    const bool is_128 = implementation == O1Implementation::kRegister128x128;
    const bool is_128x32_k64 =
        implementation == O1Implementation::kRegister128x32K64ScaleShared;
    const bool scale_shared =
        implementation == O1Implementation::kRegister64x32ScaleShared ||
        implementation == O1Implementation::kRegister64x32K64ScaleShared ||
        is_128x32_k64;
    const bool pipeline_k64 =
        implementation == O1Implementation::kRegister64x32K64ScaleShared ||
        is_128x32_k64;
    result["library"] = "CUTLASS CuTe + CUDA";
    result["mma_api"] = "cute::MMA_Atom";
    result["mma_atom"] = "SM80_16x8x32_S32S8S8S32_TN";
    result["mma_shape"] = "m16n8k32";
    result["implementation"] =
        scale_shared
        ? "tma_warp_specialized_register_partial_scale_shared"
        : "tma_warp_specialized_register_partial";
    result["implementation_key"] = is_128x32_k64
        ? "register_128x32_k64_scale_shared"
        : (pipeline_k64
        ? "register_64x32_k64_scale_shared"
        : (scale_shared
        ? "register_64x32_scale_shared"
        : (is_128 ? "register_128x128" : "register_64x32")));
    result["kernel_symbol"] = is_128x32_k64
        ? "adangel_o1_register_partial_128x32_k64_scale_shared"
        : (pipeline_k64
        ? "adangel_o1_register_partial_64x32_k64_scale_shared"
        : (scale_shared
        ? "adangel_o1_register_partial_64x32_scale_shared"
        : (is_128
              ? "adangel_o1_register_partial_128x128"
              : "adangel_o1_register_partial_64x32")));
    result["producer_warps"] = 1;
    result["consumer_warps"] = (is_128 || is_128x32_k64) ? 16 : 8;
    result["cta_tile"] = is_128
        ? py::make_tuple(128, 128, kGroupSize)
        : (is_128x32_k64
              ? py::make_tuple(128, 32, 64)
              : py::make_tuple(64, 32, pipeline_k64 ? 64 : kGroupSize));
    result["partial_storage"] = "register";
    result["shared_partial_redistribution"] = false;
    result["column_scale_load_scope"] =
        scale_shared ? "cta_once_per_column_group" : "consumer_warp";
    result["column_scale_storage"] =
        scale_shared ? "stage_local_shared_fp32" : "warp_register";
    result["fp32_accumulation_op"] =
        scale_shared ? "fma_rn" : "mul_then_add_rn";
    result["groups_per_pipeline_stage"] = pipeline_k64 ? 2 : 1;
    result["pipeline_tile_k"] = pipeline_k64 ? 64 : 32;
  }
  return result;
}

}  // namespace

bool adangel_o1_is_implemented() { return true; }

namespace {

template <class Gemm>
py::dict measure_o1_implementation(
    const at::Tensor& w_mxfp4,
    at::Tensor& w_int8,
    at::Tensor& output,
    const std::string& mode_name,
    TimingMode mode,
    O1Implementation implementation,
    int warmup,
    int repeats,
    int conversion_inner_repeats,
    int groups,
    cudaStream_t stream,
  Gemm&& gemm) {
  auto convert_weight = [&]() { adangel_launch_mxfp4_to_int8(w_mxfp4, w_int8, stream); };

  // Allocate every timing event before warmup. Creating hundreds of CUDA
  // events after warmup leaves the device idle long enough for clocks to drop,
  // so the measured repetitions would include a fresh boost transition.
  std::vector<EventSet> events;
  events.reserve(repeats);
  for (int iteration = 0; iteration < repeats; ++iteration) events.emplace_back();

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
  result["kernel"] = o1_metadata(implementation, groups);
  return result;
}

template <class Config, bool kShareColumnScale = false>
py::dict benchmark_register_implementation(
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4,
    const at::Tensor& w_scale,
    const std::string& mode_name,
    TimingMode mode,
    O1Implementation implementation,
    int warmup,
    int repeats,
    int conversion_inner_repeats,
    at::Tensor& w_int8,
    at::Tensor& output,
    cudaStream_t stream) {
  const int m = static_cast<int>(a_int8.size(0));
  const int k = static_cast<int>(a_int8.size(1));
  const int n = static_cast<int>(w_mxfp4.size(0));
  auto tma_a = make_register_tma_a<Config>(a_int8);
  auto tma_b = make_register_tma_b<Config>(w_int8);
  auto gemm = [&]() {
    launch_register_o1<Config, kShareColumnScale>(
        a_scale, w_scale, output, m, n, k, tma_a, tma_b, stream);
  };
  return measure_o1_implementation(
      w_mxfp4,
      w_int8,
      output,
      mode_name,
      mode,
      implementation,
      warmup,
      repeats,
      conversion_inner_repeats,
      k / kGroupSize,
      stream,
      gemm);
}

py::dict benchmark_o1_selected(
    const std::string& implementation_name,
    const std::string& mode_name,
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4,
    const at::Tensor& w_scale,
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
  const O1Implementation implementation =
      parse_o1_implementation(implementation_name);
  c10::cuda::CUDAGuard device_guard(a_int8.device());
  cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(a_int8.get_device()).stream();

  const int m = static_cast<int>(a_int8.size(0));
  const int k = static_cast<int>(a_int8.size(1));
  const int n = static_cast<int>(w_mxfp4.size(0));
  auto w_int8 = at::empty({n, k}, a_int8.options().dtype(at::kChar));
  auto output = at::empty({m, n}, a_int8.options().dtype(at::kFloat));

  if (implementation == O1Implementation::kSharedPartial) {
    auto tma_a = make_o1_tma_a(a_int8);
    auto tma_b = make_o1_tma_b(w_int8);
    auto gemm = [&]() {
      launch_tma_o1(
          a_scale, w_scale, output, m, n, k, tma_a, tma_b, stream);
    };
    return measure_o1_implementation(
        w_mxfp4,
        w_int8,
        output,
        mode_name,
        mode,
        implementation,
        warmup,
        repeats,
        conversion_inner_repeats,
        k / kGroupSize,
        stream,
        gemm);
  }
  if (implementation == O1Implementation::kRegister64x32) {
    return benchmark_register_implementation<O1Register64Config>(
        a_int8,
        a_scale,
        w_mxfp4,
        w_scale,
        mode_name,
        mode,
        implementation,
        warmup,
        repeats,
        conversion_inner_repeats,
        w_int8,
        output,
        stream);
  }
  if (implementation == O1Implementation::kRegister64x32ScaleShared) {
    return benchmark_register_implementation<O1Register64Config, true>(
        a_int8,
        a_scale,
        w_mxfp4,
        w_scale,
        mode_name,
        mode,
        implementation,
        warmup,
        repeats,
        conversion_inner_repeats,
        w_int8,
        output,
        stream);
  }
  if (implementation == O1Implementation::kRegister64x32K64ScaleShared) {
    return benchmark_register_implementation<O1Register64K64Config, true>(
        a_int8,
        a_scale,
        w_mxfp4,
        w_scale,
        mode_name,
        mode,
        implementation,
        warmup,
        repeats,
        conversion_inner_repeats,
        w_int8,
        output,
        stream);
  }
  if (implementation == O1Implementation::kRegister128x32K64ScaleShared) {
    return benchmark_register_implementation<O1Register128x32K64Config, true>(
        a_int8,
        a_scale,
        w_mxfp4,
        w_scale,
        mode_name,
        mode,
        implementation,
        warmup,
        repeats,
        conversion_inner_repeats,
        w_int8,
        output,
        stream);
  }
  return benchmark_register_implementation<O1Register128Config>(
      a_int8,
      a_scale,
      w_mxfp4,
      w_scale,
      mode_name,
      mode,
      implementation,
      warmup,
      repeats,
      conversion_inner_repeats,
      w_int8,
      output,
      stream);
}

}  // namespace

py::dict adangel_benchmark_o1_impl(
    const std::string& implementation,
    const std::string& mode,
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4,
    const at::Tensor& w_scale,
    int warmup,
    int repeats,
    int conversion_inner_repeats) {
  return benchmark_o1_selected(
      implementation,
      mode,
      a_int8,
      a_scale,
      w_mxfp4,
      w_scale,
      warmup,
      repeats,
      conversion_inner_repeats);
}

py::dict adangel_benchmark_o1(
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4,
    const at::Tensor& w_scale,
    const std::string& mode,
    int warmup,
    int repeats,
    int conversion_inner_repeats) {
  return benchmark_o1_selected(
      "production",
      mode,
      a_int8,
      a_scale,
      w_mxfp4,
      w_scale,
      warmup,
      repeats,
      conversion_inner_repeats);
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
// associate TMA loads and integer MMA with each exact production/candidate kernel symbol.
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
