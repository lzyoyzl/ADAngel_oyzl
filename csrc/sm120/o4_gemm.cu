#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/integer_subbyte.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cuda_runtime_api.h>
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

constexpr int kGroupSize = 128;
constexpr int kWordsPerGroup = 4;
constexpr int kAPlanes = 8;
constexpr int kWPlanes = 4;
constexpr int kProducerThreads = 32;
constexpr int kConsumerThreads = 512;
constexpr int kThreads = kProducerThreads + kConsumerThreads;
using BinaryMma = cute::SM80_16x8x128_S32U1U1S32_TN_ANDPOPC;

template <
    int TileM,
    int TileN,
    int GroupsPerStage,
    int WarpM,
    int AccumulatorChains,
    bool CacheBFragments>
struct O4Config {
  static constexpr int kTileM = TileM;
  static constexpr int kTileN = TileN;
  static constexpr int kGroupsPerStage = GroupsPerStage;
  static constexpr int kStages = 2;
  static constexpr int kWarpM = WarpM;
  static constexpr int kWarpN = (kConsumerThreads / 32) / kWarpM;
  static constexpr int kNReplicas = kTileN / (kWarpN * 8);
  static constexpr int kAccumulatorChains = AccumulatorChains;
  static constexpr bool kCacheBFragments = CacheBFragments;
  static constexpr int kPipelineWords = kGroupsPerStage * kWordsPerGroup;
  static constexpr int kPipelineK = kGroupsPerStage * kGroupSize;
  static constexpr int kAStageWords = kAPlanes * kTileM * kPipelineWords;
  static constexpr int kBStageWords = kWPlanes * kTileN * kPipelineWords;
  using Pipeline = cutlass::PipelineTmaAsync<kStages>;
  using WordLayoutA = decltype(cute::make_layout(
      cute::make_shape(
          cute::Int<kAPlanes>{}, cute::Int<kTileM>{},
          cute::Int<kPipelineWords>{}, cute::Int<kStages>{}),
      cute::make_stride(
          cute::Int<kTileM * kPipelineWords>{}, cute::Int<kPipelineWords>{},
          cute::_1{}, cute::Int<kAStageWords>{})));
  using WordLayoutB = decltype(cute::make_layout(
      cute::make_shape(
          cute::Int<kWPlanes>{}, cute::Int<kTileN>{},
          cute::Int<kPipelineWords>{}, cute::Int<kStages>{}),
      cute::make_stride(
          cute::Int<kTileN * kPipelineWords>{}, cute::Int<kPipelineWords>{},
          cute::_1{}, cute::Int<kBStageWords>{})));

  struct alignas(128) SharedStorage {
    alignas(128) uint32_t a[kStages * kAStageWords];
    alignas(128) uint32_t b[kStages * kBStageWords];
    alignas(128) float column_scale[kStages * kGroupsPerStage * kTileN];
    alignas(16) typename Pipeline::SharedStorage pipeline;
  };

  static_assert(kTileM == kWarpM * 16);
  static_assert(kTileN == kWarpN * kNReplicas * 8);
  static_assert(kNReplicas > 0);
  static_assert(kAccumulatorChains == 1 || kAccumulatorChains == 2);
};

using O4N64K256Config = O4Config<128, 64, 2, 8, 1, false>;
using O4N64K256Split2Config = O4Config<128, 64, 2, 8, 2, false>;
using O4N64K256CacheBConfig = O4Config<128, 64, 2, 8, 1, true>;
using O4N64K256Split2CacheBConfig = O4Config<128, 64, 2, 8, 2, true>;
using O4M64N64K512Config = O4Config<64, 64, 4, 4, 1, false>;
using O4M64N64K512OptimizedConfig = O4Config<64, 64, 4, 4, 2, true>;

constexpr const char* kProductionO4Implementation = "m64_n64_k512_optimized";

enum class O4Implementation {
  kN64K256,
  kN64K256Split2,
  kN64K256CacheB,
  kN64K256Split2CacheB,
  kM64N64K512,
  kM64N64K512Optimized,
};

O4Implementation parse_o4_implementation(const std::string& implementation) {
  const std::string selected = implementation == "production"
      ? kProductionO4Implementation : implementation;
  if (selected == "n64_k256") return O4Implementation::kN64K256;
  if (selected == "n64_k256_split2") return O4Implementation::kN64K256Split2;
  if (selected == "n64_k256_cache_b") return O4Implementation::kN64K256CacheB;
  if (selected == "n64_k256_split2_cache_b")
    return O4Implementation::kN64K256Split2CacheB;
  if (selected == "m64_n64_k512") return O4Implementation::kM64N64K512;
  if (selected == "m64_n64_k512_optimized")
    return O4Implementation::kM64N64K512Optimized;
  TORCH_CHECK(false, "unknown O4 implementation: ", implementation);
  return O4Implementation::kN64K256;
}

void check_cuda(cudaError_t status, const char* operation) {
  TORCH_CHECK(status == cudaSuccess, operation, " failed: ", cudaGetErrorString(status));
}

enum class TimingMode { kConversionOnly, kComputeOnly, kCold, kSteadyState };

TimingMode parse_mode(const std::string& mode) {
  if (mode == "conversion_only") return TimingMode::kConversionOnly;
  if (mode == "compute_only") return TimingMode::kComputeOnly;
  if (mode == "cold") return TimingMode::kCold;
  if (mode == "steady_state") return TimingMode::kSteadyState;
  TORCH_CHECK(false, "unknown O4 timing mode: ", mode);
  return TimingMode::kCold;
}

class CudaEvent {
 public:
  CudaEvent() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
  ~CudaEvent() { if (event_) cudaEventDestroy(event_); }
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;
  CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) { other.event_ = nullptr; }
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
  check_cuda(cudaEventElapsedTime(&value, begin.get(), end.get()), "cudaEventElapsedTime");
  return value;
}

template <class Operation>
std::vector<float> measure_batched(
    Operation&& operation, int repeats, int inner, cudaStream_t stream, const char* name) {
  std::vector<EventPair> events;
  events.reserve(repeats);
  for (int i = 0; i < repeats; ++i) events.emplace_back();
  for (auto& marker : events) {
    record(marker.begin.get(), stream);
    for (int j = 0; j < inner; ++j) operation();
    record(marker.end.get(), stream);
  }
  check_cuda(cudaEventSynchronize(events.back().end.get()), name);
  std::vector<float> result;
  result.reserve(repeats);
  for (auto const& marker : events) result.push_back(elapsed(marker.begin, marker.end) / inner);
  return result;
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
    const at::Tensor& a, const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4, const at::Tensor& w_scale) {
  TORCH_CHECK(a.is_cuda() && a_scale.is_cuda() && w_mxfp4.is_cuda() && w_scale.is_cuda(),
              "O4 inputs must be CUDA tensors");
  TORCH_CHECK(a.scalar_type() == at::kChar && a.dim() == 2 && a.is_contiguous(),
              "O4 A must be contiguous int8 [M,K]");
  TORCH_CHECK(a_scale.scalar_type() == at::kFloat && a_scale.dim() == 1 && a_scale.is_contiguous(),
              "O4 A_scale must be contiguous fp32 [M]");
  TORCH_CHECK(w_mxfp4.scalar_type() == at::kByte && w_mxfp4.dim() == 2 && w_mxfp4.is_contiguous(),
              "O4 W_mxfp4_g128 must be contiguous uint8 [N,K/2]");
  TORCH_CHECK(w_scale.scalar_type() == at::kByte && w_scale.dim() == 2 && w_scale.is_contiguous(),
              "O4 W_scale_g128 must be contiguous uint8 [N,K/128]");
  const int64_t m = a.size(0), k = a.size(1), n = w_mxfp4.size(0);
  TORCH_CHECK(k % kGroupSize == 0, "O4 K must be divisible by 128");
  TORCH_CHECK(a_scale.sizes() == at::IntArrayRef({m}), "invalid O4 A_scale shape");
  TORCH_CHECK(w_mxfp4.sizes() == at::IntArrayRef({n, k / 2}), "invalid O4 packed W shape");
  TORCH_CHECK(w_scale.sizes() == at::IntArrayRef({n, k / 128}), "invalid O4 W scale shape");
}

template <class Config, class TmaA, class TmaB>
__global__ __launch_bounds__(kThreads) void adangel_o4_bitwise_tma_ws(
    CUTE_GRID_CONSTANT TmaA const tma_a,
    CUTE_GRID_CONSTANT TmaB const tma_b,
    const float* a_scale,
    const uint8_t* w_scale,
    float* output,
    int m,
    int n,
    int k,
    int groups) {
  using Pipeline = typename Config::Pipeline;
  using SharedStorage = typename Config::SharedStorage;
  using WordLayoutA = typename Config::WordLayoutA;
  using WordLayoutB = typename Config::WordLayoutB;
  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kNReplicas = Config::kNReplicas;
  constexpr int kGroupsPerStage = Config::kGroupsPerStage;
  constexpr int kPipelineWords = Config::kPipelineWords;
  constexpr int kAStageWords = Config::kAStageWords;
  constexpr int kBStageWords = Config::kBStageWords;
  extern __shared__ __align__(128) uint8_t shared_bytes[];
  auto& storage = *reinterpret_cast<SharedStorage*>(shared_bytes);
  // Keep the plane extents in the CuTe type system. Passing the namespace
  // constexpr variables here makes nvcc materialize host symbols in device
  // code instead of treating the extents as compile-time shape constants.
  auto mA = tma_a.get_tma_tensor(cute::make_shape(cute::_8{}, m, k / 32));
  auto mB = tma_b.get_tma_tensor(cute::make_shape(cute::_4{}, n, k / 32));
  auto gA = cute::local_tile(
      mA, cute::make_shape(
          cute::Int<kAPlanes>{}, cute::Int<kTileM>{},
          cute::Int<kPipelineWords>{}),
      cute::make_coord(cute::Int<0>{}, static_cast<int>(blockIdx.y), cute::_));
  auto gB = cute::local_tile(
      mB, cute::make_shape(
          cute::Int<kWPlanes>{}, cute::Int<kTileN>{},
          cute::Int<kPipelineWords>{}),
      cute::make_coord(cute::Int<0>{}, static_cast<int>(blockIdx.x), cute::_));
  auto sAWord = cute::make_tensor(cute::make_smem_ptr(storage.a), WordLayoutA{});
  auto sBWord = cute::make_tensor(cute::make_smem_ptr(storage.b), WordLayoutB{});
  auto [tAgA, tAsA] = cute::tma_partition(
      tma_a, cute::Int<0>{}, cute::Layout<cute::_1>{},
      cute::group_modes<0, 3>(sAWord), cute::group_modes<0, 3>(gA));
  auto [tBgB, tBsB] = cute::tma_partition(
      tma_b, cute::Int<0>{}, cute::Layout<cute::_1>{},
      cute::group_modes<0, 3>(sBWord), cute::group_modes<0, 3>(gB));

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread >> 5;
  const int lane = thread & 31;
  typename Pipeline::Params params;
  params.num_consumers = kConsumerThreads;
  params.transaction_bytes = 4 * (kAStageWords + kBStageWords);
  params.initializing_warp = 0;
  if (warp == 0 && lane == 0) {
    params.role = Pipeline::ThreadCategory::Producer;
    params.is_leader = 1;
  } else if (warp > 0) {
    params.role = Pipeline::ThreadCategory::Consumer;
  }
  Pipeline pipeline(storage.pipeline, params, cute::Shape<cute::_1, cute::_1, cute::_1>{});
  cutlass::pipeline_init_wait(1);

  if (warp == 0) {
    auto write_state = cutlass::make_producer_start_state<Pipeline>();
    const int pipeline_groups = (groups + kGroupsPerStage - 1) / kGroupsPerStage;
    for (int pipeline_group = 0; pipeline_group < pipeline_groups; ++pipeline_group) {
      if (lane == 0) pipeline.producer_acquire(write_state);
      __syncwarp();
      const int stage = write_state.index();
#pragma unroll
      for (int inner_group = 0; inner_group < kGroupsPerStage; ++inner_group) {
        const int group = pipeline_group * kGroupsPerStage + inner_group;
        for (int column = lane; column < kTileN; column += 32) {
          const int global_column = static_cast<int>(blockIdx.x) * kTileN + column;
          storage.column_scale[
              (stage * kGroupsPerStage + inner_group) * kTileN + column] =
              global_column < n && group < groups
              ? adangel::decode_ue8m0(w_scale[global_column * groups + group])
              : 0.0f;
        }
      }
      __threadfence_block();
      __syncwarp();
      if (lane == 0) {
        auto* barrier = pipeline.producer_get_barrier(write_state);
        cute::copy(
            tma_a.with(*barrier), tAgA(cute::_, pipeline_group), tAsA(cute::_, stage));
        cute::copy(
            tma_b.with(*barrier), tBgB(cute::_, pipeline_group), tBsB(cute::_, stage));
      }
      ++write_state;
    }
    if (lane == 0) pipeline.producer_tail(write_state);
    return;
  }

  const int compute_thread = thread - kProducerThreads;
  const int consumer_warp = compute_thread >> 5;
  const int lane_i = lane & 3;
  const int lane_j = lane >> 2;
  const int warp_m = consumer_warp % Config::kWarpM;
  const int warp_n = consumer_warp / Config::kWarpM;
  const int tile_row = static_cast<int>(blockIdx.y) * kTileM;
  const int tile_column = static_cast<int>(blockIdx.x) * kTileN;
  const int local_row0 = warp_m * 16 + lane_j;
  const int local_row1 = local_row0 + 8;
  const int global_row0 = tile_row + local_row0;
  const int global_row1 = tile_row + local_row1;
  const float row_scale0 = global_row0 < m ? a_scale[global_row0] : 0.0f;
  const float row_scale1 = global_row1 < m ? a_scale[global_row1] : 0.0f;
  float accumulators[kNReplicas][4] = {};

  typename Pipeline::PipelineState read_state;
  const int pipeline_groups = (groups + kGroupsPerStage - 1) / kGroupsPerStage;
  for (int pipeline_group = 0; pipeline_group < pipeline_groups; ++pipeline_group) {
    pipeline.consumer_wait(read_state);
    const int stage = read_state.index();
#pragma unroll
    for (int inner_group = 0; inner_group < kGroupsPerStage; ++inner_group) {
      const int group = pipeline_group * kGroupsPerStage + inner_group;
      if (group >= groups) break;
      int32_t group_accumulators[Config::kAccumulatorChains][kNReplicas][4] = {};
      uint32_t cached_b[kWPlanes][kNReplicas] = {};
      if constexpr (Config::kCacheBFragments) {
#pragma unroll
        for (int w_plane = 0; w_plane < kWPlanes; ++w_plane) {
#pragma unroll
          for (int replica = 0; replica < kNReplicas; ++replica) {
            const int atom_n = warp_n * kNReplicas + replica;
            cached_b[w_plane][replica] = storage.b[
                stage * kBStageWords + w_plane * kTileN * kPipelineWords +
                (atom_n * 8 + lane_j) * kPipelineWords +
                inner_group * kWordsPerGroup + lane_i];
          }
        }
      }
#pragma unroll
      for (int a_plane = 0; a_plane < 8; ++a_plane) {
        const uint32_t a0 = storage.a[
            stage * kAStageWords + a_plane * kTileM * kPipelineWords +
            local_row0 * kPipelineWords + inner_group * kWordsPerGroup + lane_i];
        const uint32_t a1 = storage.a[
            stage * kAStageWords + a_plane * kTileM * kPipelineWords +
            local_row1 * kPipelineWords + inner_group * kWordsPerGroup + lane_i];
#pragma unroll
        for (int w_plane = 0; w_plane < 4; ++w_plane) {
        // This is the CuTe SM80 B1 trait mapping written explicitly.  For
        // lane=(i,j), A registers are rows j/j+8 at K word i, B is column j at
        // K word i, and C owns (j,j+8)x(2i,2i+1).  Loading complete b32 words
        // preserves the coalesced bitplane layout and avoids any guessed
        // per-bit lane mapping.
        // Two's-complement bit-plane coefficients. Keeping this expression
        // local allows the unrolled loop to constant-fold every coefficient
        // without referencing host-only lookup tables from device code.
          const int a_weight = a_plane == 7 ? -128 : (1 << a_plane);
          const int w_weight = w_plane == 3 ? -8 : (1 << w_plane);
          const int coefficient = a_weight * w_weight;
#pragma unroll
          for (int replica = 0; replica < kNReplicas; ++replica) {
            const int atom_n = warp_n * kNReplicas + replica;
            const uint32_t b0 = Config::kCacheBFragments
                ? cached_b[w_plane][replica]
                : storage.b[
                    stage * kBStageWords + w_plane * kTileN * kPipelineWords +
                    (atom_n * 8 + lane_j) * kPipelineWords +
                    inner_group * kWordsPerGroup + lane_i];
            uint32_t d0, d1, d2, d3;
            BinaryMma::fma(d0, d1, d2, d3, a0, a1, b0, 0u, 0u, 0u, 0u);
            const int chain = a_plane % Config::kAccumulatorChains;
            group_accumulators[chain][replica][0] +=
                coefficient * static_cast<int32_t>(d0);
            group_accumulators[chain][replica][1] +=
                coefficient * static_cast<int32_t>(d1);
            group_accumulators[chain][replica][2] +=
                coefficient * static_cast<int32_t>(d2);
            group_accumulators[chain][replica][3] +=
                coefficient * static_cast<int32_t>(d3);
          }
        }
      }

#pragma unroll
      for (int replica = 0; replica < kNReplicas; ++replica) {
        const int atom_n = warp_n * kNReplicas + replica;
        const int local_column0 = atom_n * 8 + 2 * lane_i;
        const int local_column1 = local_column0 + 1;
        const int scale_base = (stage * kGroupsPerStage + inner_group) * kTileN;
        const float column_scale0 = storage.column_scale[scale_base + local_column0];
        const float column_scale1 = storage.column_scale[scale_base + local_column1];
        int32_t reconstructed[4] = {};
#pragma unroll
        for (int chain = 0; chain < Config::kAccumulatorChains; ++chain) {
#pragma unroll
          for (int item = 0; item < 4; ++item) {
            reconstructed[item] += group_accumulators[chain][replica][item];
          }
        }
        accumulators[replica][0] = __fmaf_rn(
            static_cast<float>(reconstructed[0]),
            __fmul_rn(row_scale0, column_scale0), accumulators[replica][0]);
        accumulators[replica][1] = __fmaf_rn(
            static_cast<float>(reconstructed[1]),
            __fmul_rn(row_scale0, column_scale1), accumulators[replica][1]);
        accumulators[replica][2] = __fmaf_rn(
            static_cast<float>(reconstructed[2]),
            __fmul_rn(row_scale1, column_scale0), accumulators[replica][2]);
        accumulators[replica][3] = __fmaf_rn(
            static_cast<float>(reconstructed[3]),
            __fmul_rn(row_scale1, column_scale1), accumulators[replica][3]);
      }
    }
    pipeline.consumer_release(read_state);
    ++read_state;
  }

#pragma unroll
  for (int replica = 0; replica < kNReplicas; ++replica) {
    const int atom_n = warp_n * kNReplicas + replica;
    const int global_column0 = tile_column + atom_n * 8 + 2 * lane_i;
    const int global_column1 = global_column0 + 1;
    if (global_row0 < m && global_column0 < n)
      output[static_cast<int64_t>(global_row0) * n + global_column0] = accumulators[replica][0];
    if (global_row0 < m && global_column1 < n)
      output[static_cast<int64_t>(global_row0) * n + global_column1] = accumulators[replica][1];
    if (global_row1 < m && global_column0 < n)
      output[static_cast<int64_t>(global_row1) * n + global_column0] = accumulators[replica][2];
    if (global_row1 < m && global_column1 < n)
      output[static_cast<int64_t>(global_row1) * n + global_column1] = accumulators[replica][3];
  }
}

template <class Config>
auto make_tma_a(const at::Tensor& planes) {
  const int m = static_cast<int>(planes.size(1));
  const int words = static_cast<int>(planes.size(2));
  auto tensor = cute::make_tensor(
      reinterpret_cast<const uint32_t*>(planes.data_ptr<int32_t>()),
      cute::make_shape(kAPlanes, m, words),
      cute::make_stride(m * words, words, cute::_1{}));
  auto layout = typename Config::WordLayoutA{};
  return cute::make_tma_atom(
      cute::SM90_TMA_LOAD{}, tensor,
      layout(cute::_, cute::_, cute::_, cute::Int<0>{}),
      cute::make_shape(
          cute::Int<kAPlanes>{}, cute::Int<Config::kTileM>{},
          cute::Int<Config::kPipelineWords>{}));
}

template <class Config>
auto make_tma_b(const at::Tensor& planes) {
  const int n = static_cast<int>(planes.size(1));
  const int words = static_cast<int>(planes.size(2));
  auto tensor = cute::make_tensor(
      reinterpret_cast<const uint32_t*>(planes.data_ptr<int32_t>()),
      cute::make_shape(kWPlanes, n, words),
      cute::make_stride(n * words, words, cute::_1{}));
  auto layout = typename Config::WordLayoutB{};
  return cute::make_tma_atom(
      cute::SM90_TMA_LOAD{}, tensor,
      layout(cute::_, cute::_, cute::_, cute::Int<0>{}),
      cute::make_shape(
          cute::Int<kWPlanes>{}, cute::Int<Config::kTileN>{},
          cute::Int<Config::kPipelineWords>{}));
}

template <class Config, class TmaA, class TmaB>
void launch_gemm(
    const at::Tensor& a_scale, const at::Tensor& w_scale, at::Tensor& output,
    int m, int n, int k, TmaA const& tma_a, TmaB const& tma_b, cudaStream_t stream) {
  using SharedStorage = typename Config::SharedStorage;
  dim3 grid(
      (n + Config::kTileN - 1) / Config::kTileN,
      (m + Config::kTileM - 1) / Config::kTileM);
  // The staged bit-plane tiles also exceed the default dynamic shared-memory
  // allowance. Request the device-supported opt-in limit for this kernel.
  check_cuda(
      cudaFuncSetAttribute(
          adangel_o4_bitwise_tma_ws<Config, TmaA, TmaB>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(sizeof(SharedStorage))),
      "O4 set dynamic shared-memory limit");
  adangel_o4_bitwise_tma_ws<Config><<<grid, kThreads, sizeof(SharedStorage), stream>>>(
      tma_a, tma_b, a_scale.data_ptr<float>(), w_scale.data_ptr<uint8_t>(),
      output.data_ptr<float>(), m, n, k, k / kGroupSize);
  check_cuda(cudaGetLastError(), "O4 Bitwise TMA warp-specialized launch");
}

template <class Config>
py::dict kernel_metadata(
    int groups, const char* implementation_key, const char* kernel_symbol) {
  py::dict result;
  result["library"] = "CUTLASS CuTe + CUDA";
  result["implementation"] =
      "paper_bitwise_twos_complement_g128_tma_warp_specialized_candidate_matrix";
  result["implementation_key"] = implementation_key;
  result["production_selected"] =
      std::string(implementation_key) == kProductionO4Implementation;
  result["kernel_symbol"] = kernel_symbol;
  result["tensor_core"] = true;
  result["mma_family"] = "BMMA";
  result["mma_api"] = "cute::arch MMA wrapper with trait-derived lane mapping";
  result["mma_atom"] = "SM80_16x8x128_S32U1U1S32_TN_ANDPOPC";
  result["mma_shape"] = "m16n8k128";
  result["bit_operator"] = "and.popc";
  result["logical_mma_per_group"] = 32;
  result["activation_bit_weights"] = py::make_tuple(1, 2, 4, 8, 16, 32, 64, -128);
  result["weight_bit_weights"] = py::make_tuple(1, 2, 4, -8);
  result["data_movement"] = "TMA";
  result["kernel_schedule"] = "cooperative_warp_specialized";
  result["cta_tile"] = py::make_tuple(
      Config::kTileM, Config::kTileN, Config::kPipelineK);
  result["pipeline_stages"] = Config::kStages;
  result["groups_per_pipeline_stage"] = Config::kGroupsPerStage;
  result["dynamic_shared_memory_bytes"] = sizeof(typename Config::SharedStorage);
  result["bmma_accumulator_chains"] = Config::kAccumulatorChains;
  result["b_fragment_cached"] = Config::kCacheBFragments;
  result["warp_layout"] = py::make_tuple(Config::kWarpM, Config::kWarpN);
  result["producer_warps"] = 1;
  result["consumer_warps"] = 16;
  result["group_size"] = kGroupSize;
  result["group_count"] = groups;
  result["scale_formula"] = "A_scale*decode_ue8m0(W_scale_g128)";
  result["partial_storage"] = "register";
  result["global_partial_buffer"] = false;
  result["output_stores_per_element"] = 1;
  result["accumulation_dtype"] = "fp32";
  result["output_dtype"] = "fp32";
  result["activation_decomposition"] = "standalone_batched_conversion";
  result["paper_alignment"] = "Bitwise 8x4 planes; selective fusion is not claimed";
  return result;
}

}  // namespace

bool adangel_o4_is_implemented() { return true; }

template <class Config>
py::dict benchmark_o4_config(
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4_g128,
    const at::Tensor& w_scale_g128,
    const char* implementation_key,
    const char* kernel_symbol,
    const std::string& mode_name,
    int warmup,
    int repeats,
    int conversion_inner_repeats) {
  validate_inputs(a_int8, a_scale, w_mxfp4_g128, w_scale_g128);
  TORCH_CHECK(warmup >= 0 && repeats > 0 && conversion_inner_repeats > 0,
              "invalid O4 timing repetition count");
  c10::cuda::CUDAGuard guard(a_int8.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(a_int8.get_device()).stream();
  const int m = static_cast<int>(a_int8.size(0));
  const int k = static_cast<int>(a_int8.size(1));
  const int n = static_cast<int>(w_mxfp4_g128.size(0));
  auto byte_options = a_int8.options().dtype(at::kByte);
  auto int_options = a_int8.options().dtype(at::kInt);
  auto q4 = at::empty({n, k / 2}, byte_options);
  auto a_planes = at::empty({kAPlanes, m, k / 32}, int_options);
  auto w_planes = at::empty({kWPlanes, n, k / 32}, int_options);
  auto output = at::empty({m, n}, a_scale.options().dtype(at::kFloat));
  auto convert_w = [&]() {
    adangel_launch_mxfp4_to_q4(w_mxfp4_g128, q4, stream);
    adangel_launch_q4_bitplanes(q4, w_planes, stream);
  };
  auto convert_a = [&]() { adangel_launch_int8_bitplanes(a_int8, a_planes, stream); };
  convert_w(); convert_a();
  auto tma_a = make_tma_a<Config>(a_planes);
  auto tma_b = make_tma_b<Config>(w_planes);
  auto gemm = [&]() {
    launch_gemm<Config>(
        a_scale, w_scale_g128, output, m, n, k, tma_a, tma_b, stream);
  };
  const TimingMode mode = parse_mode(mode_name);
  for (int i = 0; i < warmup; ++i) {
    if (mode == TimingMode::kConversionOnly) { convert_w(); convert_a(); }
    else if (mode == TimingMode::kCold) { convert_w(); convert_a(); gemm(); }
    else if (mode == TimingMode::kSteadyState) { convert_a(); gemm(); }
    else gemm();
  }
  check_cuda(cudaStreamSynchronize(stream), "O4 warmup synchronization");

  std::vector<EventSet> events;
  events.reserve(repeats);
  for (int i = 0; i < repeats; ++i) events.emplace_back();
  for (auto& marker : events) {
    record(marker.e0.get(), stream);
    if (mode == TimingMode::kConversionOnly) {
      convert_w(); convert_a(); record(marker.e1.get(), stream);
    } else if (mode == TimingMode::kCold) {
      convert_w(); record(marker.e1.get(), stream);
      convert_a(); record(marker.e2.get(), stream);
      gemm(); record(marker.e3.get(), stream);
    } else if (mode == TimingMode::kSteadyState) {
      convert_a(); record(marker.e1.get(), stream);
      gemm(); record(marker.e2.get(), stream);
    } else {
      gemm(); record(marker.e1.get(), stream);
    }
  }
  cudaEvent_t final_event = mode == TimingMode::kCold ? events.back().e3.get()
      : (mode == TimingMode::kSteadyState ? events.back().e2.get() : events.back().e1.get());
  check_cuda(cudaEventSynchronize(final_event), "O4 timing synchronization");

  std::vector<float> isolated_w, isolated_a;
  if (mode == TimingMode::kConversionOnly || mode == TimingMode::kCold) {
    isolated_w = measure_batched(convert_w, repeats, conversion_inner_repeats, stream,
                                 "O4 weight conversion synchronization");
  }
  if (mode != TimingMode::kComputeOnly) {
    isolated_a = measure_batched(convert_a, repeats, conversion_inner_repeats, stream,
                                 "O4 activation conversion synchronization");
  }
  std::vector<float> weight_ms, activation_ms, gemm_ms, total_ms;
  for (int i = 0; i < repeats; ++i) {
    if (mode == TimingMode::kConversionOnly) {
      weight_ms.push_back(isolated_w[i]); activation_ms.push_back(isolated_a[i]);
      total_ms.push_back(isolated_w[i] + isolated_a[i]);
    } else if (mode == TimingMode::kCold) {
      weight_ms.push_back(isolated_w[i]); activation_ms.push_back(isolated_a[i]);
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
  result["converted_weight"] = py::make_tuple(q4, w_planes);
  result["converted_activation"] = a_planes;
  result["timings_ms"] = timings;
  result["timing_method"] = timing_metadata(mode_name, conversion_inner_repeats);
  result["kernel"] =
      kernel_metadata<Config>(k / kGroupSize, implementation_key, kernel_symbol);
  return result;
}

py::dict adangel_benchmark_o4_impl(
    const std::string& implementation,
    const std::string& mode_name,
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4_g128,
    const at::Tensor& w_scale_g128,
    int warmup,
    int repeats,
    int conversion_inner_repeats) {
  const O4Implementation selected = parse_o4_implementation(implementation);
  if (selected == O4Implementation::kN64K256Split2) {
    return benchmark_o4_config<O4N64K256Split2Config>(
        a_int8, a_scale, w_mxfp4_g128, w_scale_g128,
        "n64_k256_split2", "adangel_o4_bitwise_tma_ws<O4Config<128,64,2,8,2,false>>",
        mode_name, warmup, repeats, conversion_inner_repeats);
  }
  if (selected == O4Implementation::kN64K256CacheB) {
    return benchmark_o4_config<O4N64K256CacheBConfig>(
        a_int8, a_scale, w_mxfp4_g128, w_scale_g128,
        "n64_k256_cache_b", "adangel_o4_bitwise_tma_ws<O4Config<128,64,2,8,1,true>>",
        mode_name, warmup, repeats, conversion_inner_repeats);
  }
  if (selected == O4Implementation::kN64K256Split2CacheB) {
    return benchmark_o4_config<O4N64K256Split2CacheBConfig>(
        a_int8, a_scale, w_mxfp4_g128, w_scale_g128,
        "n64_k256_split2_cache_b",
        "adangel_o4_bitwise_tma_ws<O4Config<128,64,2,8,2,true>>",
        mode_name, warmup, repeats, conversion_inner_repeats);
  }
  if (selected == O4Implementation::kM64N64K512) {
    return benchmark_o4_config<O4M64N64K512Config>(
        a_int8, a_scale, w_mxfp4_g128, w_scale_g128,
        "m64_n64_k512", "adangel_o4_bitwise_tma_ws<O4Config<64,64,4,4,1,false>>",
        mode_name, warmup, repeats, conversion_inner_repeats);
  }
  if (selected == O4Implementation::kM64N64K512Optimized) {
    return benchmark_o4_config<O4M64N64K512OptimizedConfig>(
        a_int8, a_scale, w_mxfp4_g128, w_scale_g128,
        "m64_n64_k512_optimized",
        "adangel_o4_bitwise_tma_ws<O4Config<64,64,4,4,2,true>>",
        mode_name, warmup, repeats, conversion_inner_repeats);
  }
  return benchmark_o4_config<O4N64K256Config>(
      a_int8, a_scale, w_mxfp4_g128, w_scale_g128,
      "n64_k256", "adangel_o4_bitwise_tma_ws<O4Config<128,64,2,8,1,false>>",
      mode_name, warmup, repeats, conversion_inner_repeats);
}

py::dict adangel_benchmark_o4(
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4_g128,
    const at::Tensor& w_scale_g128,
    const std::string& mode_name,
    int warmup,
    int repeats,
    int conversion_inner_repeats) {
  return adangel_benchmark_o4_impl(
      kProductionO4Implementation, mode_name,
      a_int8, a_scale, w_mxfp4_g128, w_scale_g128,
      warmup, repeats, conversion_inner_repeats);
}

py::dict adangel_run_o4(
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4_g128,
    const at::Tensor& w_scale_g128,
    const std::string& mode) {
  py::dict measured = adangel_benchmark_o4(
      a_int8, a_scale, w_mxfp4_g128, w_scale_g128, mode, 1, 1,
      kAdangelDefaultConversionTimingInnerRepeats);
  py::dict timings = measured["timings_ms"].cast<py::dict>();
  for (auto name : {"weight_conversion", "activation_conversion", "gemm", "total"}) {
    if (timings.contains(name)) {
      auto samples = timings[name].cast<std::vector<float>>();
      const std::string key = std::string(name) + "_ms";
      measured[key.c_str()] = samples.front();
    } else {
      const std::string key = std::string(name) + "_ms";
      measured[key.c_str()] = 0.0f;
    }
  }
  return measured;
}
