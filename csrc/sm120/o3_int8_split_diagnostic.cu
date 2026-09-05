#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>
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

// Diagnostic counterfactual for O3. It preserves the paper Split identity and
// G128/Q4 scaling, but expands both 4-bit activation branches and the Q4
// weight to int8 before the timed GEMM. The GEMM then uses two native S8xS8
// paths instead of the legacy U4/S4 PTX interface. This is deliberately not a
// production O3 implementation because it does not execute INT4 MMA.
constexpr int kGroupSize = 128;
constexpr int kMmaK = 32;
constexpr int kKSubgroups = kGroupSize / kMmaK;
constexpr int kTileM = 128;
constexpr int kTileN = 64;
constexpr int kStages = 2;
constexpr int kProducerThreads = 32;
constexpr int kConsumerWarps = 16;
constexpr int kConsumerThreads = 32 * kConsumerWarps;
constexpr int kThreads = kProducerThreads + kConsumerThreads;
constexpr int kAStageElements = kTileM * kGroupSize;
constexpr int kBStageElements = kTileN * kGroupSize;

using Pipeline = cutlass::PipelineTmaAsync<kStages>;
using FullLayoutA = decltype(cute::make_layout(
    cute::make_shape(cute::Int<kTileM>{}, cute::Int<kGroupSize>{}, cute::Int<kStages>{}),
    cute::make_stride(
        cute::Int<kGroupSize>{}, cute::Int<1>{}, cute::Int<kAStageElements>{})));
using FullLayoutB = decltype(cute::make_layout(
    cute::make_shape(cute::Int<kTileN>{}, cute::Int<kGroupSize>{}, cute::Int<kStages>{}),
    cute::make_stride(
        cute::Int<kGroupSize>{}, cute::Int<1>{}, cute::Int<kBStageElements>{})));
using SubLayoutA = decltype(cute::make_layout(
    cute::make_shape(cute::Int<kTileM>{}, cute::Int<kMmaK>{}),
    cute::make_stride(cute::Int<kGroupSize>{}, cute::Int<1>{})));
using SubLayoutB = decltype(cute::make_layout(
    cute::make_shape(cute::Int<kTileN>{}, cute::Int<kMmaK>{}),
    cute::make_stride(cute::Int<kGroupSize>{}, cute::Int<1>{})));
using TiledMma = cute::TiledMMA<
    cute::MMA_Atom<cute::SM80_16x8x32_S32S8S8S32_TN>,
    cute::Layout<cute::Shape<cute::_8, cute::_2, cute::_1>>,
    cute::Tile<cute::Int<kTileM>, cute::Int<kTileN>, cute::Int<kMmaK>>>;

struct alignas(128) SharedStorage {
  alignas(128) uint8_t a_low[kStages * kAStageElements];
  alignas(128) uint8_t a_high[kStages * kAStageElements];
  alignas(128) uint8_t b[kStages * kBStageElements];
  alignas(128) float column_scale[kStages * kTileN];
  alignas(16) Pipeline::SharedStorage pipeline;
};

void check_cuda(cudaError_t status, const char* operation) {
  TORCH_CHECK(status == cudaSuccess, operation, " failed: ", cudaGetErrorString(status));
}

extern "C" __global__ void adangel_o3_expand_activation_int8x2(
    const int8_t* input, int8_t* low, int8_t* high, int64_t elements) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const uint8_t raw = static_cast<uint8_t>(input[index]);
  low[index] = static_cast<int8_t>(raw & 0x0fu);
  int high_value = static_cast<int>(raw >> 4);
  if (high_value >= 8) high_value -= 16;
  high[index] = static_cast<int8_t>(high_value);
}

extern "C" __global__ void adangel_o3_expand_weight_q4_int8(
    const uint8_t* packed, int8_t* output, int64_t pairs) {
  const int64_t pair =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair >= pairs) return;
  const uint8_t byte = packed[pair];
  char2 values;
  values.x = adangel::e2m1_to_q4(byte & 0x0fu);
  values.y = adangel::e2m1_to_q4(byte >> 4);
  reinterpret_cast<char2*>(output)[pair] = values;
}

void expand_activation(
    const at::Tensor& input,
    at::Tensor& low,
    at::Tensor& high,
    cudaStream_t stream) {
  constexpr int threads = 256;
  const int64_t elements = input.numel();
  const int blocks = static_cast<int>((elements + threads - 1) / threads);
  adangel_o3_expand_activation_int8x2<<<blocks, threads, 0, stream>>>(
      input.data_ptr<int8_t>(),
      low.data_ptr<int8_t>(),
      high.data_ptr<int8_t>(),
      elements);
  check_cuda(cudaGetLastError(), "O3 INT8x2 activation expansion");
}

void expand_weight(
    const at::Tensor& packed,
    at::Tensor& output,
    cudaStream_t stream) {
  constexpr int threads = 256;
  const int64_t pairs = packed.numel();
  const int blocks = static_cast<int>((pairs + threads - 1) / threads);
  adangel_o3_expand_weight_q4_int8<<<blocks, threads, 0, stream>>>(
      packed.data_ptr<uint8_t>(), output.data_ptr<int8_t>(), pairs);
  check_cuda(cudaGetLastError(), "O3 INT8x2 weight expansion");
}

template <class TmaLow, class TmaHigh, class TmaB>
__global__ __launch_bounds__(kThreads) void adangel_o3_split_int8x2_tma_ws(
    CUTE_GRID_CONSTANT TmaLow const tma_low,
    CUTE_GRID_CONSTANT TmaHigh const tma_high,
    CUTE_GRID_CONSTANT TmaB const tma_b,
    const float* a_scale,
    const uint8_t* w_scale,
    float* output,
    int m,
    int n,
    int k,
    int groups) {
  extern __shared__ __align__(128) uint8_t shared_bytes[];
  auto& storage = *reinterpret_cast<SharedStorage*>(shared_bytes);

  auto mLow = tma_low.get_tma_tensor(cute::make_shape(m, k));
  auto mHigh = tma_high.get_tma_tensor(cute::make_shape(m, k));
  auto mB = tma_b.get_tma_tensor(cute::make_shape(n, k));
  auto a_tiler = cute::make_shape(cute::Int<kTileM>{}, cute::Int<kGroupSize>{});
  auto b_tiler = cute::make_shape(cute::Int<kTileN>{}, cute::Int<kGroupSize>{});
  auto gLow = cute::local_tile(
      mLow, a_tiler, cute::make_coord(static_cast<int>(blockIdx.y), cute::_));
  auto gHigh = cute::local_tile(
      mHigh, a_tiler, cute::make_coord(static_cast<int>(blockIdx.y), cute::_));
  auto gB = cute::local_tile(
      mB, b_tiler, cute::make_coord(static_cast<int>(blockIdx.x), cute::_));
  auto sLow = cute::make_tensor(cute::make_smem_ptr(storage.a_low), FullLayoutA{});
  auto sHigh = cute::make_tensor(cute::make_smem_ptr(storage.a_high), FullLayoutA{});
  auto sB = cute::make_tensor(cute::make_smem_ptr(storage.b), FullLayoutB{});
  auto [tLgL, tLsL] = cute::tma_partition(
      tma_low, cute::Int<0>{}, cute::Layout<cute::_1>{},
      cute::group_modes<0, 2>(sLow), cute::group_modes<0, 2>(gLow));
  auto [tHgH, tHsH] = cute::tma_partition(
      tma_high, cute::Int<0>{}, cute::Layout<cute::_1>{},
      cute::group_modes<0, 2>(sHigh), cute::group_modes<0, 2>(gHigh));
  auto [tBgB, tBsB] = cute::tma_partition(
      tma_b, cute::Int<0>{}, cute::Layout<cute::_1>{},
      cute::group_modes<0, 2>(sB), cute::group_modes<0, 2>(gB));

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread >> 5;
  const int lane = thread & 31;
  Pipeline::Params params;
  params.num_consumers = kConsumerThreads;
  params.transaction_bytes = 2 * kAStageElements + kBStageElements;
  params.initializing_warp = 0;
  if (warp == 0 && lane == 0) {
    params.role = Pipeline::ThreadCategory::Producer;
    params.is_leader = 1;
  } else if (warp > 0) {
    params.role = Pipeline::ThreadCategory::Consumer;
  }
  Pipeline pipeline(
      storage.pipeline, params, cute::Shape<cute::_1, cute::_1, cute::_1>{});
  cutlass::pipeline_init_wait(1);

  if (warp == 0) {
    auto write_state = cutlass::make_producer_start_state<Pipeline>();
    for (int group = 0; group < groups; ++group) {
      if (lane == 0) pipeline.producer_acquire(write_state);
      __syncwarp();
      const int stage = write_state.index();
      for (int column = lane; column < kTileN; column += 32) {
        const int global_column = static_cast<int>(blockIdx.x) * kTileN + column;
        storage.column_scale[stage * kTileN + column] =
            global_column < n
            ? adangel::decode_ue8m0(w_scale[global_column * groups + group])
            : 0.0f;
      }
      __threadfence_block();
      __syncwarp();
      if (lane == 0) {
        auto* barrier = pipeline.producer_get_barrier(write_state);
        cute::copy(tma_low.with(*barrier), tLgL(cute::_, group), tLsL(cute::_, stage));
        cute::copy(tma_high.with(*barrier), tHgH(cute::_, group), tHsH(cute::_, stage));
        cute::copy(tma_b.with(*barrier), tBgB(cute::_, group), tBsB(cute::_, stage));
      }
      ++write_state;
    }
    if (lane == 0) pipeline.producer_tail(write_state);
    return;
  }

  const int compute_thread = thread - kProducerThreads;
  const int tile_row = static_cast<int>(blockIdx.y) * kTileM;
  const int tile_column = static_cast<int>(blockIdx.x) * kTileN;
  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_slice(compute_thread);
  auto sLowTemplate = cute::make_tensor(
      cute::make_smem_ptr(storage.a_low), SubLayoutA{});
  auto sHighTemplate = cute::make_tensor(
      cute::make_smem_ptr(storage.a_high), SubLayoutA{});
  auto sBTemplate = cute::make_tensor(
      cute::make_smem_ptr(storage.b), SubLayoutB{});
  auto tCrLow = thr_mma.partition_fragment_A(sLowTemplate);
  auto tCrHigh = thr_mma.partition_fragment_A(sHighTemplate);
  auto tCrB = thr_mma.partition_fragment_B(sBTemplate);
  auto cC = cute::make_identity_tensor(
      cute::make_shape(cute::Int<kTileM>{}, cute::Int<kTileN>{}));
  auto tCcC = thr_mma.partition_C(cC);
  auto tCrPartial = thr_mma.make_fragment_C(tCcC);
  auto tCrGroup = thr_mma.make_fragment_C(tCcC);
  auto tCrAccumulator = cute::make_fragment_like<float>(tCrPartial);
  cute::clear(tCrAccumulator);

  float row_scales[2] = {0.0f, 0.0f};
#pragma unroll
  for (int row_slot = 0; row_slot < 2; ++row_slot) {
    const int item = row_slot * 2;
    const int global_row = tile_row + cute::get<0>(tCcC(item));
    row_scales[row_slot] = global_row < m ? a_scale[global_row] : 0.0f;
  }

  using SmemCopyAtom = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, uint8_t>;
  SmemCopyAtom smem_copy_atom;
  auto tiled_copy_a = cute::make_tiled_copy_A(smem_copy_atom, tiled_mma);
  auto tiled_copy_b = cute::make_tiled_copy_B(smem_copy_atom, tiled_mma);
  auto thr_copy_a = tiled_copy_a.get_slice(compute_thread);
  auto thr_copy_b = tiled_copy_b.get_slice(compute_thread);
  auto tXrLow = thr_copy_a.retile_D(tCrLow);
  auto tXrHigh = thr_copy_a.retile_D(tCrHigh);
  auto tXrB = thr_copy_b.retile_D(tCrB);

  Pipeline::PipelineState read_state;
  for (int group = 0; group < groups; ++group) {
    pipeline.consumer_wait(read_state);
    const int stage = read_state.index();
    cute::clear(tCrGroup);
#pragma unroll
    for (int subgroup = 0; subgroup < kKSubgroups; ++subgroup) {
      auto sLowSub = cute::make_tensor(
          cute::make_smem_ptr(
              storage.a_low + stage * kAStageElements + subgroup * kMmaK),
          SubLayoutA{});
      auto sHighSub = cute::make_tensor(
          cute::make_smem_ptr(
              storage.a_high + stage * kAStageElements + subgroup * kMmaK),
          SubLayoutA{});
      auto sBSub = cute::make_tensor(
          cute::make_smem_ptr(
              storage.b + stage * kBStageElements + subgroup * kMmaK),
          SubLayoutB{});
      auto tXsLow = thr_copy_a.partition_S(sLowSub);
      auto tXsHigh = thr_copy_a.partition_S(sHighSub);
      auto tXsB = thr_copy_b.partition_S(sBSub);
      cute::copy(smem_copy_atom, tXsLow, tXrLow);
      cute::copy(smem_copy_atom, tXsHigh, tXrHigh);
      cute::copy(smem_copy_atom, tXsB, tXrB);

      cute::clear(tCrPartial);
      cute::gemm(tiled_mma, tCrLow, tCrB, tCrPartial);
#pragma unroll
      for (int item = 0; item < cute::size(tCrPartial); ++item) {
        tCrGroup(item) += tCrPartial(item);
      }
      cute::clear(tCrPartial);
      cute::gemm(tiled_mma, tCrHigh, tCrB, tCrPartial);
#pragma unroll
      for (int item = 0; item < cute::size(tCrPartial); ++item) {
        tCrGroup(item) += 16 * tCrPartial(item);
      }
    }

    float warp_column_scales[kTileN / 32];
#pragma unroll
    for (int round = 0; round < kTileN / 32; ++round) {
      warp_column_scales[round] =
          storage.column_scale[stage * kTileN + round * 32 + lane];
    }
#pragma unroll
    for (int item = 0; item < cute::size(tCrAccumulator); ++item) {
      const int local_column = cute::get<1>(tCcC(item));
      float column_scale = 0.0f;
#pragma unroll
      for (int round = 0; round < kTileN / 32; ++round) {
        const float candidate = __shfl_sync(
            0xffffffffu, warp_column_scales[round], local_column & 31);
        if ((local_column >> 5) == round) column_scale = candidate;
      }
      const float scale =
          __fmul_rn(row_scales[(item >> 1) & 1], column_scale);
      tCrAccumulator(item) = __fmaf_rn(
          static_cast<float>(tCrGroup(item)), scale, tCrAccumulator(item));
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
      cute::make_shape(cute::Int<kTileM>{}, cute::Int<kTileN>{}),
      cute::make_coord(static_cast<int>(blockIdx.y), static_cast<int>(blockIdx.x)));
  auto tCgC = thr_mma.partition_C(gC);
#pragma unroll
  for (int item = 0; item < cute::size(tCrAccumulator); ++item) {
    const int global_row = tile_row + cute::get<0>(tCcC(item));
    const int global_column = tile_column + cute::get<1>(tCcC(item));
    if (global_row < m && global_column < n) tCgC(item) = tCrAccumulator(item);
  }
}

template <class Tensor>
auto make_tma_a(const Tensor& input) {
  const int m = static_cast<int>(input.size(0));
  const int k = static_cast<int>(input.size(1));
  auto tensor = cute::make_tensor(
      reinterpret_cast<const uint8_t*>(input.template data_ptr<int8_t>()),
      cute::make_shape(m, k), cute::make_stride(k, cute::Int<1>{}));
  auto layout = FullLayoutA{};
  return cute::make_tma_atom(
      cute::SM90_TMA_LOAD{}, tensor, layout(cute::_, cute::_, cute::Int<0>{}),
      cute::make_shape(cute::Int<kTileM>{}, cute::Int<kGroupSize>{}));
}

template <class Tensor>
auto make_tma_b(const Tensor& input) {
  const int n = static_cast<int>(input.size(0));
  const int k = static_cast<int>(input.size(1));
  auto tensor = cute::make_tensor(
      reinterpret_cast<const uint8_t*>(input.template data_ptr<int8_t>()),
      cute::make_shape(n, k), cute::make_stride(k, cute::Int<1>{}));
  auto layout = FullLayoutB{};
  return cute::make_tma_atom(
      cute::SM90_TMA_LOAD{}, tensor, layout(cute::_, cute::_, cute::Int<0>{}),
      cute::make_shape(cute::Int<kTileN>{}, cute::Int<kGroupSize>{}));
}

template <class TmaLow, class TmaHigh, class TmaB>
void launch_gemm(
    TmaLow const& tma_low,
    TmaHigh const& tma_high,
    TmaB const& tma_b,
    const at::Tensor& a_scale,
    const at::Tensor& w_scale,
    at::Tensor& output,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  dim3 grid((n + kTileN - 1) / kTileN, (m + kTileM - 1) / kTileM);
  check_cuda(
      cudaFuncSetAttribute(
          adangel_o3_split_int8x2_tma_ws<TmaLow, TmaHigh, TmaB>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(sizeof(SharedStorage))),
      "O3 INT8x2 dynamic shared-memory attribute");
  adangel_o3_split_int8x2_tma_ws<<<grid, kThreads, sizeof(SharedStorage), stream>>>(
      tma_low,
      tma_high,
      tma_b,
      a_scale.data_ptr<float>(),
      w_scale.data_ptr<uint8_t>(),
      output.data_ptr<float>(),
      m,
      n,
      k,
      k / kGroupSize);
  check_cuda(cudaGetLastError(), "O3 INT8x2 TMA GEMM launch");
}

enum class TimingMode { kConversionOnly, kComputeOnly, kCold, kSteadyState };

TimingMode parse_mode(const std::string& mode) {
  if (mode == "conversion_only") return TimingMode::kConversionOnly;
  if (mode == "compute_only") return TimingMode::kComputeOnly;
  if (mode == "cold") return TimingMode::kCold;
  if (mode == "steady_state") return TimingMode::kSteadyState;
  TORCH_CHECK(false, "unknown O3 INT8x2 timing mode: ", mode);
  return TimingMode::kCold;
}

class CudaEvent {
 public:
  CudaEvent() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
  ~CudaEvent() { if (event_) cudaEventDestroy(event_); }
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;
  CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
    other.event_ = nullptr;
  }
  cudaEvent_t get() const { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

struct EventSet { CudaEvent e0, e1, e2, e3; };
struct EventPair { CudaEvent begin, end; };

void record(cudaEvent_t event, cudaStream_t stream) {
  check_cuda(cudaEventRecord(event, stream), "cudaEventRecord");
}

float elapsed(const CudaEvent& begin, const CudaEvent& end) {
  float value = 0.0f;
  check_cuda(
      cudaEventElapsedTime(&value, begin.get(), end.get()),
      "cudaEventElapsedTime");
  return value;
}

template <class Operation>
std::vector<float> measure_batched(
    Operation&& operation,
    int repeats,
    int inner,
    cudaStream_t stream,
    const char* name) {
  std::vector<EventPair> events;
  events.reserve(repeats);
  for (int i = 0; i < repeats; ++i) events.emplace_back();
  for (auto& marker : events) {
    record(marker.begin.get(), stream);
    for (int j = 0; j < inner; ++j) operation();
    record(marker.end.get(), stream);
  }
  check_cuda(cudaEventSynchronize(events.back().end.get()), name);
  std::vector<float> samples;
  samples.reserve(repeats);
  for (auto const& marker : events) {
    samples.push_back(elapsed(marker.begin, marker.end) / inner);
  }
  return samples;
}

py::dict timing_metadata(const std::string& mode, int inner) {
  py::dict result;
  result["strategy"] = "conversion_amortized_end_to_end_direct";
  result["conversion_stage_timing"] = "isolated_batched_cuda_event_average";
  result["conversion_inner_repeats"] = inner;
  result["conversion_only_total_timing"] = "sum_of_isolated_batched_stages";
  result["end_to_end_total_timing"] = "direct_single_path";
  result["mode_total_timing"] = mode == "conversion_only"
      ? "batched_cuda_event_average" : "direct_single_path";
  result["mode_total_inner_repeats"] = mode == "conversion_only" ? inner : 1;
  result["component_and_total_measured_separately"] = true;
  return result;
}

void validate_inputs(
    const at::Tensor& a,
    const at::Tensor& a_scale,
    const at::Tensor& w,
    const at::Tensor& w_scale) {
  TORCH_CHECK(
      a.is_cuda() && a_scale.is_cuda() && w.is_cuda() && w_scale.is_cuda(),
      "O3 INT8x2 inputs must be CUDA tensors");
  TORCH_CHECK(
      a.scalar_type() == at::kChar && a.dim() == 2 && a.is_contiguous(),
      "O3 INT8x2 A must be contiguous int8 [M,K]");
  TORCH_CHECK(
      a_scale.scalar_type() == at::kFloat && a_scale.dim() == 1 &&
          a_scale.is_contiguous(),
      "O3 INT8x2 A_scale must be contiguous fp32 [M]");
  TORCH_CHECK(
      w.scalar_type() == at::kByte && w.dim() == 2 && w.is_contiguous(),
      "O3 INT8x2 W_mxfp4_g128 must be contiguous uint8 [N,K/2]");
  TORCH_CHECK(
      w_scale.scalar_type() == at::kByte && w_scale.dim() == 2 &&
          w_scale.is_contiguous(),
      "O3 INT8x2 W_scale_g128 must be contiguous uint8 [N,K/128]");
  const int64_t m = a.size(0);
  const int64_t k = a.size(1);
  const int64_t n = w.size(0);
  TORCH_CHECK(
      m % kTileM == 0 && n % kTileN == 0 && k % kGroupSize == 0,
      "O3 INT8x2 diagnostic requires M%128=0, N%64=0, K%128=0");
  TORCH_CHECK(a_scale.sizes() == at::IntArrayRef({m}), "invalid A_scale shape");
  TORCH_CHECK(w.sizes() == at::IntArrayRef({n, k / 2}), "invalid packed W shape");
  TORCH_CHECK(
      w_scale.sizes() == at::IntArrayRef({n, k / kGroupSize}),
      "invalid W_scale shape");
}

py::dict kernel_metadata(int groups) {
  py::dict result;
  result["implementation"] = "diagnostic_o3_split_two_native_int8_paths";
  result["implementation_key"] = "split_int8x2_128x64x128";
  result["kernel_symbol"] = "adangel_o3_split_int8x2_tma_ws";
  result["diagnostic_only"] = true;
  result["production_selected"] = false;
  result["o3_requirement_compliant"] = false;
  result["requirement_exception"] =
      "replaces required U4xS4/S4xS4 PTX MMA with two expanded S8xS8 paths";
  result["math_semantics"] = "exact_split_low_plus_16_high_g128_q4";
  result["tensor_core"] = true;
  result["mma_family"] = "IMMA_INT8_X2";
  result["ptx_mma_semantics"] = "S8xS8_low_and_S8xS8_high";
  result["mma_api"] = "cute::MMA_Atom";
  result["mma_atom"] = "SM80_16x8x32_S32S8S8S32_TN";
  result["mma_shape"] = "m16n8k32";
  result["logical_int8_paths_per_group"] = 2;
  result["mma_per_path_per_group"] = kKSubgroups;
  result["data_movement"] = "TMA";
  result["kernel_schedule"] = "cooperative_warp_specialized";
  result["cta_tile"] = py::make_tuple(kTileM, kTileN, kGroupSize);
  result["pipeline_stages"] = kStages;
  result["groups_per_pipeline_stage"] = 1;
  result["producer_warps"] = 1;
  result["consumer_warps"] = kConsumerWarps;
  result["partial_storage"] = "register";
  result["global_partial_buffer"] = false;
  result["output_stores_per_element"] = 1;
  result["partial_dtype"] = "int32";
  result["accumulation_dtype"] = "fp32";
  result["output_dtype"] = "fp32";
  result["group_size"] = kGroupSize;
  result["group_count"] = groups;
  result["expanded_activation_dtype"] = "int8_low_and_int8_high";
  result["expanded_weight_dtype"] = "int8_q4_values";
  result["scale_formula"] = "A_scale*decode_ue8m0(W_scale_g128)";
  result["dynamic_shared_memory_bytes"] = sizeof(SharedStorage);
  return result;
}

}  // namespace

py::dict adangel_benchmark_o3_split_int8x2_diagnostic(
    const std::string& mode_name,
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4_g128,
    const at::Tensor& w_scale_g128,
    int warmup,
    int repeats,
    int conversion_inner_repeats) {
  validate_inputs(a_int8, a_scale, w_mxfp4_g128, w_scale_g128);
  TORCH_CHECK(
      warmup >= 0 && repeats > 0 && conversion_inner_repeats > 0,
      "invalid O3 INT8x2 timing repetition count");
  c10::cuda::CUDAGuard guard(a_int8.device());
  cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(a_int8.get_device()).stream();
  const int m = static_cast<int>(a_int8.size(0));
  const int k = static_cast<int>(a_int8.size(1));
  const int n = static_cast<int>(w_mxfp4_g128.size(0));

  auto char_options = a_int8.options().dtype(at::kChar);
  auto a_low = at::empty({m, k}, char_options);
  auto a_high = at::empty({m, k}, char_options);
  auto w_int8 = at::empty({n, k}, char_options);
  auto output = at::empty({m, n}, a_scale.options().dtype(at::kFloat));
  auto convert_w = [&]() { expand_weight(w_mxfp4_g128, w_int8, stream); };
  auto convert_a = [&]() {
    expand_activation(a_int8, a_low, a_high, stream);
  };
  convert_w();
  convert_a();
  auto tma_low = make_tma_a(a_low);
  auto tma_high = make_tma_a(a_high);
  auto tma_b = make_tma_b(w_int8);
  auto gemm = [&]() {
    launch_gemm(
        tma_low,
        tma_high,
        tma_b,
        a_scale,
        w_scale_g128,
        output,
        m,
        n,
        k,
        stream);
  };

  const TimingMode mode = parse_mode(mode_name);
  if (mode == TimingMode::kComputeOnly || mode == TimingMode::kSteadyState) {
    convert_w();
  }
  for (int i = 0; i < warmup; ++i) {
    if (mode == TimingMode::kConversionOnly) {
      convert_w();
      convert_a();
    } else if (mode == TimingMode::kCold) {
      convert_w();
      convert_a();
      gemm();
    } else if (mode == TimingMode::kSteadyState) {
      convert_a();
      gemm();
    } else {
      gemm();
    }
  }
  check_cuda(cudaStreamSynchronize(stream), "O3 INT8x2 warmup synchronization");

  std::vector<EventSet> events;
  events.reserve(repeats);
  for (int i = 0; i < repeats; ++i) events.emplace_back();
  for (auto& marker : events) {
    record(marker.e0.get(), stream);
    if (mode == TimingMode::kConversionOnly) {
      convert_w();
      convert_a();
      record(marker.e1.get(), stream);
    } else if (mode == TimingMode::kCold) {
      convert_w();
      record(marker.e1.get(), stream);
      convert_a();
      record(marker.e2.get(), stream);
      gemm();
      record(marker.e3.get(), stream);
    } else if (mode == TimingMode::kSteadyState) {
      convert_a();
      record(marker.e1.get(), stream);
      gemm();
      record(marker.e2.get(), stream);
    } else {
      gemm();
      record(marker.e1.get(), stream);
    }
  }
  cudaEvent_t final_event = mode == TimingMode::kCold
      ? events.back().e3.get()
      : (mode == TimingMode::kSteadyState
            ? events.back().e2.get()
            : events.back().e1.get());
  check_cuda(
      cudaEventSynchronize(final_event), "O3 INT8x2 timing synchronization");

  std::vector<float> isolated_w;
  std::vector<float> isolated_a;
  if (mode == TimingMode::kConversionOnly || mode == TimingMode::kCold) {
    isolated_w = measure_batched(
        convert_w,
        repeats,
        conversion_inner_repeats,
        stream,
        "O3 INT8x2 weight conversion synchronization");
  }
  if (mode != TimingMode::kComputeOnly) {
    isolated_a = measure_batched(
        convert_a,
        repeats,
        conversion_inner_repeats,
        stream,
        "O3 INT8x2 activation conversion synchronization");
  }

  std::vector<float> weight_ms;
  std::vector<float> activation_ms;
  std::vector<float> gemm_ms;
  std::vector<float> total_ms;
  for (int i = 0; i < repeats; ++i) {
    if (mode == TimingMode::kConversionOnly) {
      weight_ms.push_back(isolated_w[i]);
      activation_ms.push_back(isolated_a[i]);
      total_ms.push_back(isolated_w[i] + isolated_a[i]);
    } else if (mode == TimingMode::kCold) {
      weight_ms.push_back(isolated_w[i]);
      activation_ms.push_back(isolated_a[i]);
      gemm_ms.push_back(elapsed(events[i].e2, events[i].e3));
      total_ms.push_back(elapsed(events[i].e0, events[i].e3));
    } else if (mode == TimingMode::kSteadyState) {
      activation_ms.push_back(isolated_a[i]);
      gemm_ms.push_back(elapsed(events[i].e1, events[i].e2));
      total_ms.push_back(elapsed(events[i].e0, events[i].e2));
    } else {
      gemm_ms.push_back(elapsed(events[i].e0, events[i].e1));
      total_ms.push_back(gemm_ms.back());
    }
  }
  if (mode == TimingMode::kConversionOnly) gemm();

  py::dict timings;
  if (!weight_ms.empty()) timings["weight_conversion"] = weight_ms;
  if (!activation_ms.empty()) timings["activation_conversion"] = activation_ms;
  if (!gemm_ms.empty()) timings["gemm"] = gemm_ms;
  timings["total"] = total_ms;
  py::dict result;
  result["output"] = output;
  result["converted_weight"] = w_int8;
  result["converted_activation"] = py::make_tuple(a_low, a_high);
  result["timings_ms"] = timings;
  result["timing_method"] =
      timing_metadata(mode_name, conversion_inner_repeats);
  result["kernel"] = kernel_metadata(k / kGroupSize);
  return result;
}
