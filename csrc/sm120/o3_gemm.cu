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
constexpr int kMmaK = 64;
constexpr int kKSubgroups = 2;
constexpr int kProducerThreads = 32;
constexpr int kMaxThreads = 544;
constexpr int kPackedK = kGroupSize / 2;
using LowMma = cute::SM80_16x8x64_S32U4S4S32_TN;
using HighMma = cute::SM80_16x8x64_S32S4S4S32_TN;

template <
    int TileN,
    int GroupsPerStage,
    bool DualK64Chains,
    int TileM = 128,
    bool UseCuteLdsm = false>
struct O3Config {
  static constexpr int kTileM = TileM;
  static constexpr int kTileN = TileN;
  static constexpr int kNReplicas = TileN / 16;
  static constexpr int kMWarps = TileM / 16;
  static constexpr int kNWarpGroups = 2;
  static constexpr int kConsumerWarps = kMWarps * kNWarpGroups;
  static constexpr int kConsumerThreads = 32 * kConsumerWarps;
  static constexpr int kThreads = kProducerThreads + kConsumerThreads;
  static constexpr int kGroupsPerStage = GroupsPerStage;
  static constexpr int kStages = 2;
  static constexpr int kPipelinePackedK = GroupsPerStage * kPackedK;
  static constexpr int kPipelineK = GroupsPerStage * kGroupSize;
  static constexpr int kAStageRowBytes = kPipelinePackedK;
  static constexpr int kBStageRowBytes = kPipelinePackedK;
  static constexpr int kATransactionBytes = kTileM * kPipelinePackedK;
  static constexpr int kBTransactionBytes = kTileN * kPipelinePackedK;
  static constexpr int kAStageBytes = kTileM * kAStageRowBytes;
  static constexpr int kBStageBytes = kTileN * kBStageRowBytes;
  static constexpr bool kDualK64Chains = DualK64Chains;
  static constexpr bool kSwizzledSharedRows = false;
  static constexpr bool kUseCuteLdsm = UseCuteLdsm;
  using Pipeline = cutlass::PipelineTmaAsync<kStages>;
  using ByteLayoutA = decltype(cute::make_layout(
      cute::make_shape(
          cute::Int<kTileM>{}, cute::Int<kPipelinePackedK>{}, cute::Int<kStages>{}),
      cute::make_stride(
          cute::Int<kAStageRowBytes>{}, cute::_1{}, cute::Int<kAStageBytes>{})));
  using ByteLayoutB = decltype(cute::make_layout(
      cute::make_shape(
          cute::Int<kTileN>{}, cute::Int<kPipelinePackedK>{}, cute::Int<kStages>{}),
      cute::make_stride(
          cute::Int<kBStageRowBytes>{}, cute::_1{}, cute::Int<kBStageBytes>{})));
  using NibbleLayoutA = decltype(cute::make_layout(
      cute::make_shape(
          cute::Int<kTileM>{}, cute::Int<kPipelineK>{}, cute::Int<kStages>{}),
      cute::make_stride(
          cute::Int<kPipelineK>{}, cute::_1{}, cute::Int<kTileM * kPipelineK>{})));
  using NibbleLayoutB = decltype(cute::make_layout(
      cute::make_shape(
          cute::Int<kTileN>{}, cute::Int<kPipelineK>{}, cute::Int<kStages>{}),
      cute::make_stride(
          cute::Int<kPipelineK>{}, cute::_1{}, cute::Int<kTileN * kPipelineK>{})));
  using LowTiledMma = cute::TiledMMA<
      cute::MMA_Atom<LowMma>,
      cute::Layout<cute::Shape<
          cute::Int<kMWarps>, cute::Int<kNWarpGroups>, cute::_1>>,
      cute::Tile<cute::Int<kTileM>, cute::Int<kTileN>, cute::Int<kMmaK>>>;
  using HighTiledMma = cute::TiledMMA<
      cute::MMA_Atom<HighMma>,
      cute::Layout<cute::Shape<
          cute::Int<kMWarps>, cute::Int<kNWarpGroups>, cute::_1>>,
      cute::Tile<cute::Int<kTileM>, cute::Int<kTileN>, cute::Int<kMmaK>>>;

  struct alignas(128) SharedStorage {
    alignas(128) uint8_t a_low[kStages * kAStageBytes];
    alignas(128) uint8_t a_high[kStages * kAStageBytes];
    alignas(128) uint8_t b[kStages * kBStageBytes];
    alignas(128) float column_scale[kStages * kGroupsPerStage * kTileN];
    alignas(16) typename Pipeline::SharedStorage pipeline;
  };
};

// NCU reports a four-way conflict for every packed A/B shared-memory load in
// the production row-major layout.  Use CUTLASS' native 64-byte TMA-compatible
// swizzle atom over each 8x64-byte packed tile.  Unlike arbitrary row padding,
// this layout is encoded in the TMA descriptor and preserves the compact stage
// allocation while rotating the K-byte address with the logical row bits.
template <int TileN, bool UseLdsm = false>
struct O3SwizzledConfig {
  static constexpr int kTileM = 128;
  static constexpr int kTileN = TileN;
  static constexpr int kNReplicas = TileN / 16;
  static constexpr int kMWarps = kTileM / 16;
  static constexpr int kNWarpGroups = 2;
  static constexpr int kConsumerWarps = kMWarps * kNWarpGroups;
  static constexpr int kConsumerThreads = 32 * kConsumerWarps;
  static constexpr int kThreads = kProducerThreads + kConsumerThreads;
  static constexpr int kGroupsPerStage = 1;
  static constexpr int kStages = 2;
  static constexpr int kPipelinePackedK = kPackedK;
  static constexpr int kPipelineK = kGroupSize;
  static constexpr int kAStageRowBytes = kPipelinePackedK;
  static constexpr int kBStageRowBytes = kPipelinePackedK;
  static constexpr int kATransactionBytes = kTileM * kPipelinePackedK;
  static constexpr int kBTransactionBytes = kTileN * kPipelinePackedK;
  static constexpr int kAStageBytes = kTileM * kAStageRowBytes;
  static constexpr int kBStageBytes = kTileN * kBStageRowBytes;
  static constexpr bool kDualK64Chains = false;
  static constexpr bool kSwizzledSharedRows = true;
  static constexpr bool kUseCuteLdsm = UseLdsm;
  using Pipeline = cutlass::PipelineTmaAsync<kStages>;
  using SwizzleAtom = decltype(cute::composition(
      cute::Swizzle<2, 4, 3>{},
      cute::Layout<
          cute::Shape<cute::_8, cute::_64>,
          cute::Stride<cute::_64, cute::_1>>{}));
  using ByteLayoutA = decltype(cute::tile_to_shape(
      SwizzleAtom{}, cute::make_shape(
          cute::Int<kTileM>{}, cute::Int<kPipelinePackedK>{}, cute::Int<kStages>{})));
  using ByteLayoutB = decltype(cute::tile_to_shape(
      SwizzleAtom{}, cute::make_shape(
          cute::Int<kTileN>{}, cute::Int<kPipelinePackedK>{}, cute::Int<kStages>{})));

  struct alignas(128) SharedStorage {
    alignas(128) uint8_t a_low[kStages * kAStageBytes];
    alignas(128) uint8_t a_high[kStages * kAStageBytes];
    alignas(128) uint8_t b[kStages * kBStageBytes];
    alignas(128) float column_scale[kStages * kGroupsPerStage * kTileN];
    alignas(16) typename Pipeline::SharedStorage pipeline;
  };
};

using O3N16K128Config = O3Config<16, 1, false>;
using O3N16K256Config = O3Config<16, 2, false>;
using O3N32K128Config = O3Config<32, 1, false>;
using O3N16K128DualConfig = O3Config<16, 1, true>;
using O3N32K256DualConfig = O3Config<32, 2, true>;
using O3N16K128SwizzleConfig = O3SwizzledConfig<16>;
using O3N32K128SwizzleConfig = O3SwizzledConfig<32>;
using O3M64N16K128Config = O3Config<16, 1, false, 64>;
using O3M64N32K128Config = O3Config<32, 1, false, 64>;
using O3M64N16K128CuteLdsmConfig = O3Config<16, 1, false, 64, true>;
using O3N16K128CuteLdsmConfig = O3Config<16, 1, false, 128, true>;
using O3N32K128CuteLdsmConfig = O3Config<32, 1, false, 128, true>;
using O3N16K128LdsmSwizzleConfig = O3SwizzledConfig<16, true>;

constexpr const char* kProductionO3Implementation = "n16_k128";

enum class O3Implementation {
  kN16K128,
  kN16K256,
  kN32K128,
  kN16K128Dual,
  kN32K256Dual,
  kN16K128Swizzle,
  kN32K128Swizzle,
  kM64N16K128,
  kM64N32K128,
  kM64N16K128CuteLdsm,
  kN16K128CuteLdsm,
  kN32K128CuteLdsm,
  kN16K128LdsmSwizzle,
};

O3Implementation parse_o3_implementation(const std::string& implementation) {
  const std::string selected = implementation == "production"
      ? kProductionO3Implementation : implementation;
  if (selected == "n16_k128") return O3Implementation::kN16K128;
  if (selected == "n16_k256") return O3Implementation::kN16K256;
  if (selected == "n32_k128") return O3Implementation::kN32K128;
  if (selected == "n16_k128_dual") return O3Implementation::kN16K128Dual;
  if (selected == "n32_k256_dual") return O3Implementation::kN32K256Dual;
  if (selected == "n16_k128_swizzle") return O3Implementation::kN16K128Swizzle;
  if (selected == "n32_k128_swizzle") return O3Implementation::kN32K128Swizzle;
  if (selected == "m64_n16_k128") return O3Implementation::kM64N16K128;
  if (selected == "m64_n32_k128") return O3Implementation::kM64N32K128;
  if (selected == "m64_n16_k128_cute_ldsm") {
    return O3Implementation::kM64N16K128CuteLdsm;
  }
  if (selected == "n16_k128_cute_ldsm") return O3Implementation::kN16K128CuteLdsm;
  if (selected == "n32_k128_cute_ldsm") return O3Implementation::kN32K128CuteLdsm;
  if (selected == "n16_k128_ldsm_swizzle") {
    return O3Implementation::kN16K128LdsmSwizzle;
  }
  TORCH_CHECK(false, "unknown O3 implementation: ", implementation);
  return O3Implementation::kN16K128;
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
  TORCH_CHECK(false, "unknown O3 timing mode: ", mode);
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
  std::vector<float> samples;
  samples.reserve(repeats);
  for (auto const& marker : events) samples.push_back(elapsed(marker.begin, marker.end) / inner);
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
    const at::Tensor& a, const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4, const at::Tensor& w_scale) {
  TORCH_CHECK(a.is_cuda() && a_scale.is_cuda() && w_mxfp4.is_cuda() && w_scale.is_cuda(),
              "O3 inputs must be CUDA tensors");
  TORCH_CHECK(a.scalar_type() == at::kChar && a.dim() == 2 && a.is_contiguous(),
              "O3 A must be contiguous int8 [M,K]");
  TORCH_CHECK(a_scale.scalar_type() == at::kFloat && a_scale.dim() == 1 && a_scale.is_contiguous(),
              "O3 A_scale must be contiguous fp32 [M]");
  TORCH_CHECK(w_mxfp4.scalar_type() == at::kByte && w_mxfp4.dim() == 2 && w_mxfp4.is_contiguous(),
              "O3 W_mxfp4_g128 must be contiguous uint8 [N,K/2]");
  TORCH_CHECK(w_scale.scalar_type() == at::kByte && w_scale.dim() == 2 && w_scale.is_contiguous(),
              "O3 W_scale_g128 must be contiguous uint8 [N,K/128]");
  const int64_t m = a.size(0), k = a.size(1), n = w_mxfp4.size(0);
  TORCH_CHECK(k % kGroupSize == 0, "O3 K must be divisible by 128");
  TORCH_CHECK(a_scale.sizes() == at::IntArrayRef({m}), "invalid O3 A_scale shape");
  TORCH_CHECK(w_mxfp4.sizes() == at::IntArrayRef({n, k / 2}), "invalid O3 packed W shape");
  TORCH_CHECK(w_scale.sizes() == at::IntArrayRef({n, k / 128}), "invalid O3 W scale shape");
}

template <class Config, class TmaLow, class TmaHigh, class TmaB>
__global__ __launch_bounds__(kMaxThreads) void adangel_o3_split_tma_ws(
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
  using Pipeline = typename Config::Pipeline;
  using SharedStorage = typename Config::SharedStorage;
  using ByteLayoutA = typename Config::ByteLayoutA;
  using ByteLayoutB = typename Config::ByteLayoutB;
  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kNReplicas = Config::kNReplicas;
  constexpr int kMWarps = Config::kMWarps;
  constexpr int kGroupsPerStage = Config::kGroupsPerStage;
  constexpr int kPipelinePackedK = Config::kPipelinePackedK;
  constexpr int kPipelineK = Config::kPipelineK;
  constexpr int kAStageBytes = Config::kAStageBytes;
  constexpr int kBStageBytes = Config::kBStageBytes;
  // TMA B128 swizzle addresses are relative to a 1024-byte aligned shared
  // base.  The stronger alignment is harmless for compact candidates and is
  // required for O3SwizzledConfig to agree with the descriptor mapping.
  extern __shared__ __align__(1024) uint8_t shared_bytes[];
  auto& storage = *reinterpret_cast<SharedStorage*>(shared_bytes);

  auto mLow = tma_low.get_tma_tensor(cute::make_shape(m, k / 2));
  auto mHigh = tma_high.get_tma_tensor(cute::make_shape(m, k / 2));
  auto mB = tma_b.get_tma_tensor(cute::make_shape(n, k / 2));
  auto byte_tiler = cute::make_shape(
      cute::Int<kTileM>{}, cute::Int<kPipelinePackedK>{});
  auto a_coord = cute::make_coord(static_cast<int>(blockIdx.y), cute::_);
  auto b_coord = cute::make_coord(static_cast<int>(blockIdx.x), cute::_);
  auto gLow = cute::local_tile(mLow, byte_tiler, a_coord);
  auto gHigh = cute::local_tile(mHigh, byte_tiler, a_coord);
  auto gB = cute::local_tile(
      mB, cute::make_shape(cute::Int<kTileN>{}, cute::Int<kPipelinePackedK>{}), b_coord);
  auto sLowBytes = cute::make_tensor(cute::make_smem_ptr(storage.a_low), ByteLayoutA{});
  auto sHighBytes = cute::make_tensor(cute::make_smem_ptr(storage.a_high), ByteLayoutA{});
  auto sBBytes = cute::make_tensor(cute::make_smem_ptr(storage.b), ByteLayoutB{});
  auto [tLgL, tLsL] = cute::tma_partition(
      tma_low, cute::Int<0>{}, cute::Layout<cute::_1>{},
      cute::group_modes<0, 2>(sLowBytes), cute::group_modes<0, 2>(gLow));
  auto [tHgH, tHsH] = cute::tma_partition(
      tma_high, cute::Int<0>{}, cute::Layout<cute::_1>{},
      cute::group_modes<0, 2>(sHighBytes), cute::group_modes<0, 2>(gHigh));
  auto [tBgB, tBsB] = cute::tma_partition(
      tma_b, cute::Int<0>{}, cute::Layout<cute::_1>{},
      cute::group_modes<0, 2>(sBBytes), cute::group_modes<0, 2>(gB));

  const int thread = static_cast<int>(threadIdx.x);
  const int warp = thread >> 5;
  const int lane = thread & 31;
  typename Pipeline::Params params;
  params.num_consumers = Config::kConsumerThreads;
  params.transaction_bytes =
      2 * Config::kATransactionBytes + Config::kBTransactionBytes;
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
            tma_low.with(*barrier), tLgL(cute::_, pipeline_group), tLsL(cute::_, stage));
        cute::copy(
            tma_high.with(*barrier), tHgH(cute::_, pipeline_group), tHsH(cute::_, stage));
        cute::copy(tma_b.with(*barrier), tBgB(cute::_, pipeline_group), tBsB(cute::_, stage));
      }
      ++write_state;
    }
    if (lane == 0) pipeline.producer_tail(write_state);
    return;
  }

  // Kept as compile-time documentation of the first CuTe TiledCopy attempt.
  // The high-level sub-byte partition did not match the compact TMA layout;
  // the active candidate uses explicit LDSM fragment addresses below.
  if constexpr (false && Config::kUseCuteLdsm) {
    using NibbleLayoutA = typename Config::NibbleLayoutA;
    using NibbleLayoutB = typename Config::NibbleLayoutB;
    using LowTiledMma = typename Config::LowTiledMma;
    using HighTiledMma = typename Config::HighTiledMma;
    const int compute_thread = thread - kProducerThreads;
    const int tile_row = static_cast<int>(blockIdx.y) * kTileM;
    const int tile_column = static_cast<int>(blockIdx.x) * kTileN;

    auto sLow = cute::make_tensor(
        cute::make_smem_ptr(
            reinterpret_cast<cutlass::uint4b_t*>(storage.a_low)),
        NibbleLayoutA{});
    auto sHigh = cute::make_tensor(
        cute::make_smem_ptr(
            reinterpret_cast<cutlass::int4b_t*>(storage.a_high)),
        NibbleLayoutA{});
    auto sB = cute::make_tensor(
        cute::make_smem_ptr(
            reinterpret_cast<cutlass::int4b_t*>(storage.b)),
        NibbleLayoutB{});

    LowTiledMma low_tiled_mma;
    HighTiledMma high_tiled_mma;
    auto low_thr_mma = low_tiled_mma.get_slice(compute_thread);
    auto high_thr_mma = high_tiled_mma.get_slice(compute_thread);
    auto tCrLowA = low_thr_mma.partition_fragment_A(
        sLow(cute::_, cute::_, cute::Int<0>{}));
    auto tCrHighA = high_thr_mma.partition_fragment_A(
        sHigh(cute::_, cute::_, cute::Int<0>{}));
    auto tCrB = low_thr_mma.partition_fragment_B(
        sB(cute::_, cute::_, cute::Int<0>{}));
    auto cC = cute::make_identity_tensor(
        cute::make_shape(cute::Int<kTileM>{}, cute::Int<kTileN>{}));
    auto tCcC = low_thr_mma.partition_C(cC);
    auto tCrLowPartial = low_thr_mma.make_fragment_C(tCcC);
    auto tCrHighPartial = high_thr_mma.make_fragment_C(tCcC);
    auto tCrAccumulator = cute::make_fragment_like<float>(tCrLowPartial);
    auto tCrRowScale = cute::make_fragment_like<float>(tCrLowPartial);
    cute::clear(tCrAccumulator);
    for (int item = 0; item < cute::size(tCrRowScale); ++item) {
      const int local_row = cute::get<0>(tCcC(item));
      const int global_row = tile_row + local_row;
      tCrRowScale(item) = global_row < m ? a_scale[global_row] : 0.0f;
    }

    using LowSmemCopyAtom =
        cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, cutlass::uint4b_t>;
    using SignedSmemCopyAtom =
        cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, cutlass::int4b_t>;
    using BSignedSmemCopyAtom =
        cute::Copy_Atom<cute::SM75_U32x2_LDSM_N, cutlass::int4b_t>;
    LowSmemCopyAtom low_copy_atom;
    SignedSmemCopyAtom signed_copy_atom;
    BSignedSmemCopyAtom b_copy_atom;
    auto tiled_copy_low_a = cute::make_tiled_copy_A(
        low_copy_atom, low_tiled_mma);
    auto tiled_copy_high_a = cute::make_tiled_copy_A(
        signed_copy_atom, high_tiled_mma);
    auto tiled_copy_b = cute::make_tiled_copy_B(
        b_copy_atom, low_tiled_mma);
    auto low_thr_copy = tiled_copy_low_a.get_slice(compute_thread);
    auto high_thr_copy = tiled_copy_high_a.get_slice(compute_thread);
    auto b_thr_copy = tiled_copy_b.get_slice(compute_thread);
    auto tXsLow = low_thr_copy.partition_S(sLow);
    auto tXsHigh = high_thr_copy.partition_S(sHigh);
    auto tXsB = b_thr_copy.partition_S(sB);
    auto tXrLow = low_thr_copy.retile_D(tCrLowA);
    auto tXrHigh = high_thr_copy.retile_D(tCrHighA);
    auto tXrB = b_thr_copy.retile_D(tCrB);

    typename Pipeline::PipelineState read_state;
    const int pipeline_groups =
        (groups + kGroupsPerStage - 1) / kGroupsPerStage;
    for (int pipeline_group = 0;
         pipeline_group < pipeline_groups;
         ++pipeline_group) {
      pipeline.consumer_wait(read_state);
      const int stage = read_state.index();
      cute::copy(
          low_copy_atom,
          tXsLow(cute::_, cute::_, cute::_, stage),
          tXrLow);
      cute::copy(
          signed_copy_atom,
          tXsHigh(cute::_, cute::_, cute::_, stage),
          tXrHigh);
      cute::copy(
          b_copy_atom,
          tXsB(cute::_, cute::_, cute::_, stage),
          tXrB);
      cute::clear(tCrLowPartial);
      cute::clear(tCrHighPartial);
      cute::gemm(low_tiled_mma, tCrLowA, tCrB, tCrLowPartial);
      cute::gemm(high_tiled_mma, tCrHighA, tCrB, tCrHighPartial);

      float lane_column_scale = 0.0f;
      if (lane < kTileN) {
        lane_column_scale =
            storage.column_scale[stage * kTileN + lane];
      }
      for (int item = 0; item < cute::size(tCrAccumulator); ++item) {
        const int local_column = cute::get<1>(tCcC(item));
        const float column_scale = __shfl_sync(
            0xffffffffu, lane_column_scale, local_column);
        const int32_t partial =
            static_cast<int32_t>(tCrLowPartial(item)) +
            16 * static_cast<int32_t>(tCrHighPartial(item));
        const float scale =
            __fmul_rn(tCrRowScale(item), column_scale);
        tCrAccumulator(item) = __fmaf_rn(
            static_cast<float>(partial), scale, tCrAccumulator(item));
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
        cute::make_coord(
            static_cast<int>(blockIdx.y), static_cast<int>(blockIdx.x)));
    auto tCgC = low_thr_mma.partition_C(gC);
    for (int item = 0; item < cute::size(tCrAccumulator); ++item) {
      const int local_row = cute::get<0>(tCcC(item));
      const int local_column = cute::get<1>(tCcC(item));
      if (tile_row + local_row < m && tile_column + local_column < n) {
        tCgC(item) = tCrAccumulator(item);
      }
    }
    return;
  }

  const int compute_thread = thread - kProducerThreads;
  const int consumer_warp = compute_thread >> 5;
  const int lane_i = lane & 3;
  const int lane_j = lane >> 2;
  const int warp_m = consumer_warp % kMWarps;
  const int warp_n = consumer_warp / kMWarps;
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
    const auto* low_words = reinterpret_cast<const uint32_t*>(
        storage.a_low + stage * kAStageBytes);
    const auto* high_words = reinterpret_cast<const uint32_t*>(
        storage.a_high + stage * kAStageBytes);
    const auto* b_words = reinterpret_cast<const uint32_t*>(
        storage.b + stage * kBStageBytes);
    constexpr int kAWordsPerRow = Config::kAStageRowBytes / 4;
    constexpr int kBWordsPerRow = Config::kBStageRowBytes / 4;
#pragma unroll
    for (int inner_group = 0; inner_group < kGroupsPerStage; ++inner_group) {
      const int group = pipeline_group * kGroupsPerStage + inner_group;
      if (group >= groups) break;
      uint32_t low[kKSubgroups][kNReplicas][4] = {};
      uint32_t high[kKSubgroups][kNReplicas][4] = {};
      if constexpr (Config::kDualK64Chains) {
        // Explicitly preload both independent K64 fragments before issuing MMA.
        // Each K64 subgroup accumulates into its own register chain, removing the
        // read-after-write dependency between the two MMAs.  The chains are
        // reconstructed only after both instructions have completed.
        uint32_t low_a[kKSubgroups][4];
        uint32_t high_a[kKSubgroups][4];
        uint32_t b_fragment[kKSubgroups][kNReplicas][2];
#pragma unroll
        for (int subgroup = 0; subgroup < kKSubgroups; ++subgroup) {
          const int word_base = inner_group * (kPackedK / 4) + subgroup * 8 + lane_i;
          low_a[subgroup][0] = low_words[local_row0 * kAWordsPerRow + word_base];
          low_a[subgroup][1] = low_words[local_row1 * kAWordsPerRow + word_base];
          low_a[subgroup][2] = low_words[local_row0 * kAWordsPerRow + word_base + 4];
          low_a[subgroup][3] = low_words[local_row1 * kAWordsPerRow + word_base + 4];
          high_a[subgroup][0] = high_words[local_row0 * kAWordsPerRow + word_base];
          high_a[subgroup][1] = high_words[local_row1 * kAWordsPerRow + word_base];
          high_a[subgroup][2] = high_words[local_row0 * kAWordsPerRow + word_base + 4];
          high_a[subgroup][3] = high_words[local_row1 * kAWordsPerRow + word_base + 4];
#pragma unroll
          for (int replica = 0; replica < kNReplicas; ++replica) {
            const int atom_n = warp_n * kNReplicas + replica;
            const int b_row = atom_n * 8 + lane_j;
            b_fragment[subgroup][replica][0] =
                b_words[b_row * kBWordsPerRow + word_base];
            b_fragment[subgroup][replica][1] =
                b_words[b_row * kBWordsPerRow + word_base + 4];
          }
        }
#pragma unroll
        for (int subgroup = 0; subgroup < kKSubgroups; ++subgroup) {
#pragma unroll
          for (int replica = 0; replica < kNReplicas; ++replica) {
            LowMma::fma(
                low[subgroup][replica][0], low[subgroup][replica][1],
                low[subgroup][replica][2], low[subgroup][replica][3],
                low_a[subgroup][0], low_a[subgroup][1],
                low_a[subgroup][2], low_a[subgroup][3],
                b_fragment[subgroup][replica][0], b_fragment[subgroup][replica][1],
                0u, 0u, 0u, 0u);
            HighMma::fma(
                high[subgroup][replica][0], high[subgroup][replica][1],
                high[subgroup][replica][2], high[subgroup][replica][3],
                high_a[subgroup][0], high_a[subgroup][1],
                high_a[subgroup][2], high_a[subgroup][3],
                b_fragment[subgroup][replica][0], b_fragment[subgroup][replica][1],
                0u, 0u, 0u, 0u);
          }
        }
      } else {
#pragma unroll
        for (int subgroup = 0; subgroup < kKSubgroups; ++subgroup) {
          const int word_base = inner_group * (kPackedK / 4) + subgroup * 8 + lane_i;
          uint32_t la0, la1, la2, la3;
          uint32_t ha0, ha1, ha2, ha3;
          if constexpr (Config::kUseCuteLdsm) {
            const int matrix = lane >> 3;
            const int matrix_row = lane & 7;
            const int a_row = warp_m * 16 + (matrix & 1) * 8 + matrix_row;
            const int a_byte =
                inner_group * kPackedK + subgroup * 32 + (matrix >> 1) * 16;
            const uint8_t* low_address;
            const uint8_t* high_address;
            if constexpr (Config::kSwizzledSharedRows) {
              const auto a_layout = ByteLayoutA{};
              const int offset = a_layout(
                  cute::make_coord(a_row, a_byte, stage));
              low_address = storage.a_low + offset;
              high_address = storage.a_high + offset;
            } else {
              const int offset = stage * kAStageBytes +
                  a_row * Config::kAStageRowBytes + a_byte;
              low_address = storage.a_low + offset;
              high_address = storage.a_high + offset;
            }
            cute::SM75_U32x4_LDSM_N::copy(
                *reinterpret_cast<const cute::uint128_t*>(low_address),
                la0, la1, la2, la3);
            cute::SM75_U32x4_LDSM_N::copy(
                *reinterpret_cast<const cute::uint128_t*>(high_address),
                ha0, ha1, ha2, ha3);
          } else if constexpr (Config::kSwizzledSharedRows) {
            const auto a_layout = ByteLayoutA{};
            la0 = *reinterpret_cast<const uint32_t*>(
                storage.a_low + a_layout(cute::make_coord(local_row0, word_base * 4, stage)));
            la1 = *reinterpret_cast<const uint32_t*>(
                storage.a_low + a_layout(cute::make_coord(local_row1, word_base * 4, stage)));
            la2 = *reinterpret_cast<const uint32_t*>(
                storage.a_low + a_layout(cute::make_coord(local_row0, (word_base + 4) * 4, stage)));
            la3 = *reinterpret_cast<const uint32_t*>(
                storage.a_low + a_layout(cute::make_coord(local_row1, (word_base + 4) * 4, stage)));
            ha0 = *reinterpret_cast<const uint32_t*>(
                storage.a_high + a_layout(cute::make_coord(local_row0, word_base * 4, stage)));
            ha1 = *reinterpret_cast<const uint32_t*>(
                storage.a_high + a_layout(cute::make_coord(local_row1, word_base * 4, stage)));
            ha2 = *reinterpret_cast<const uint32_t*>(
                storage.a_high + a_layout(cute::make_coord(local_row0, (word_base + 4) * 4, stage)));
            ha3 = *reinterpret_cast<const uint32_t*>(
                storage.a_high + a_layout(cute::make_coord(local_row1, (word_base + 4) * 4, stage)));
          } else {
            la0 = low_words[local_row0 * kAWordsPerRow + word_base];
            la1 = low_words[local_row1 * kAWordsPerRow + word_base];
            la2 = low_words[local_row0 * kAWordsPerRow + word_base + 4];
            la3 = low_words[local_row1 * kAWordsPerRow + word_base + 4];
            ha0 = high_words[local_row0 * kAWordsPerRow + word_base];
            ha1 = high_words[local_row1 * kAWordsPerRow + word_base];
            ha2 = high_words[local_row0 * kAWordsPerRow + word_base + 4];
            ha3 = high_words[local_row1 * kAWordsPerRow + word_base + 4];
          }
#pragma unroll
          for (int replica = 0; replica < kNReplicas; ++replica) {
            const int atom_n = warp_n * kNReplicas + replica;
            const int b_row = atom_n * 8 + lane_j;
            uint32_t b0, b1;
            if constexpr (Config::kUseCuteLdsm) {
              const int matrix = lane >> 3;
              const int matrix_row = lane & 7;
              const int ldsm_b_row = atom_n * 8 + matrix_row;
              const int b_byte =
                  inner_group * kPackedK + subgroup * 32 + (matrix & 1) * 16;
              const uint8_t* b_address;
              if constexpr (Config::kSwizzledSharedRows) {
                const auto b_layout = ByteLayoutB{};
                b_address = storage.b + b_layout(
                    cute::make_coord(ldsm_b_row, b_byte, stage));
              } else {
                b_address = storage.b + stage * kBStageBytes +
                    ldsm_b_row * Config::kBStageRowBytes + b_byte;
              }
              cute::SM75_U32x2_LDSM_N::copy(
                  *reinterpret_cast<const cute::uint128_t*>(b_address), b0, b1);
            } else if constexpr (Config::kSwizzledSharedRows) {
              const auto b_layout = ByteLayoutB{};
              b0 = *reinterpret_cast<const uint32_t*>(
                  storage.b + b_layout(cute::make_coord(b_row, word_base * 4, stage)));
              b1 = *reinterpret_cast<const uint32_t*>(
                  storage.b + b_layout(cute::make_coord(b_row, (word_base + 4) * 4, stage)));
            } else {
              b0 = b_words[b_row * kBWordsPerRow + word_base];
              b1 = b_words[b_row * kBWordsPerRow + word_base + 4];
            }
            LowMma::fma(
                low[0][replica][0], low[0][replica][1],
                low[0][replica][2], low[0][replica][3],
                la0, la1, la2, la3, b0, b1,
                low[0][replica][0], low[0][replica][1],
                low[0][replica][2], low[0][replica][3]);
            HighMma::fma(
                high[0][replica][0], high[0][replica][1],
                high[0][replica][2], high[0][replica][3],
                ha0, ha1, ha2, ha3, b0, b1,
                high[0][replica][0], high[0][replica][1],
                high[0][replica][2], high[0][replica][3]);
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
        const int chain_count = Config::kDualK64Chains ? kKSubgroups : 1;
        int32_t partial[4] = {};
#pragma unroll
        for (int chain = 0; chain < chain_count; ++chain) {
#pragma unroll
          for (int item = 0; item < 4; ++item) {
            partial[item] += static_cast<int32_t>(low[chain][replica][item])
                + 16 * static_cast<int32_t>(high[chain][replica][item]);
          }
        }
        accumulators[replica][0] = __fmaf_rn(
            static_cast<float>(partial[0]), __fmul_rn(row_scale0, column_scale0),
            accumulators[replica][0]);
        accumulators[replica][1] = __fmaf_rn(
            static_cast<float>(partial[1]), __fmul_rn(row_scale0, column_scale1),
            accumulators[replica][1]);
        accumulators[replica][2] = __fmaf_rn(
            static_cast<float>(partial[2]), __fmul_rn(row_scale1, column_scale0),
            accumulators[replica][2]);
        accumulators[replica][3] = __fmaf_rn(
            static_cast<float>(partial[3]), __fmul_rn(row_scale1, column_scale1),
            accumulators[replica][3]);
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
auto make_tma_a(const at::Tensor& split, int row_offset) {
  const int m = static_cast<int>(split.size(0) / 2);
  const int packed_k = static_cast<int>(split.size(1));
  const uint8_t* pointer = split.data_ptr<uint8_t>() + static_cast<int64_t>(row_offset) * packed_k;
  auto tensor = cute::make_tensor(pointer, cute::make_shape(m, packed_k), cute::make_stride(packed_k, cute::_1{}));
  auto layout = typename Config::ByteLayoutA{};
  return cute::make_tma_atom(
      cute::SM90_TMA_LOAD{}, tensor, layout(cute::_, cute::_, cute::Int<0>{}),
      cute::make_shape(
          cute::Int<Config::kTileM>{}, cute::Int<Config::kPipelinePackedK>{}));
}

template <class Config>
auto make_tma_b(const at::Tensor& q4) {
  const int n = static_cast<int>(q4.size(0));
  const int packed_k = static_cast<int>(q4.size(1));
  auto tensor = cute::make_tensor(
      q4.data_ptr<uint8_t>(), cute::make_shape(n, packed_k), cute::make_stride(packed_k, cute::_1{}));
  auto layout = typename Config::ByteLayoutB{};
  return cute::make_tma_atom(
      cute::SM90_TMA_LOAD{}, tensor, layout(cute::_, cute::_, cute::Int<0>{}),
      cute::make_shape(
          cute::Int<Config::kTileN>{}, cute::Int<Config::kPipelinePackedK>{}));
}

template <class Config, class TmaLow, class TmaHigh, class TmaB>
void launch_gemm(
    const at::Tensor& a_scale, const at::Tensor& w_scale, at::Tensor& output,
    int m, int n, int k, TmaLow const& low, TmaHigh const& high, TmaB const& b,
    cudaStream_t stream) {
  using SharedStorage = typename Config::SharedStorage;
  dim3 grid(
      (n + Config::kTileN - 1) / Config::kTileN,
      (m + Config::kTileM - 1) / Config::kTileM);
  // The staged Split pipeline uses more than CUDA's default dynamic
  // shared-memory allowance. Opt this exact template specialization into the
  // device limit before launch; otherwise CUDA reports invalid argument.
  check_cuda(
      cudaFuncSetAttribute(
          adangel_o3_split_tma_ws<Config, TmaLow, TmaHigh, TmaB>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(sizeof(SharedStorage))),
      "O3 set dynamic shared-memory limit");
  adangel_o3_split_tma_ws<Config><<<grid, Config::kThreads, sizeof(SharedStorage), stream>>>(
      low, high, b, a_scale.data_ptr<float>(), w_scale.data_ptr<uint8_t>(),
      output.data_ptr<float>(), m, n, k, k / kGroupSize);
  check_cuda(cudaGetLastError(), "O3 Split TMA warp-specialized launch");
}

template <class Config>
py::dict kernel_metadata(
    int groups, const char* implementation_key, const char* kernel_symbol) {
  py::dict result;
  result["library"] = "CUTLASS CuTe + CUDA";
  result["implementation"] =
      "paper_split_g128_tma_warp_specialized_register_partial_candidate_matrix";
  result["implementation_key"] = implementation_key;
  result["production_selected"] =
      std::string(implementation_key) == kProductionO3Implementation;
  result["kernel_symbol"] = kernel_symbol;
  result["tensor_core"] = true;
  result["mma_family"] = "IMMA_INT4";
  result["mma_api"] = "cute::arch MMA wrapper with trait-derived lane mapping";
  result["mma_atoms"] = py::make_tuple(
      "SM80_16x8x64_S32U4S4S32_TN", "SM80_16x8x64_S32S4S4S32_TN");
  result["mma_shape"] = "m16n8k64";
  result["data_movement"] = "TMA";
  result["kernel_schedule"] = "cooperative_warp_specialized";
  result["cta_tile"] = py::make_tuple(
      Config::kTileM, Config::kTileN, Config::kPipelineK);
  result["pipeline_stages"] = Config::kStages;
  result["groups_per_pipeline_stage"] = Config::kGroupsPerStage;
  result["dynamic_shared_memory_bytes"] = sizeof(typename Config::SharedStorage);
  result["instruction_double_buffer"] = Config::kDualK64Chains;
  result["shared_row_stride_bytes"] = Config::kAStageRowBytes;
  result["shared_bank_conflict_mitigation"] =
      Config::kSwizzledSharedRows ? "tma_swizzle_64b" : "none";
  result["shared_to_register_copy"] =
      Config::kUseCuteLdsm ? "explicit_ldmatrix_fragment" : "scalar_ld_shared";
  result["independent_k64_accumulator_chains"] =
      Config::kDualK64Chains ? kKSubgroups : 1;
  result["producer_warps"] = 1;
  result["consumer_warps"] = Config::kConsumerWarps;
  result["group_size"] = kGroupSize;
  result["group_count"] = groups;
  result["split_formula"] = "A8=A_low_u4+16*A_high_s4";
  result["scale_formula"] = "A_scale*decode_ue8m0(W_scale_g128)";
  result["partial_storage"] = "register";
  result["global_partial_buffer"] = false;
  result["output_stores_per_element"] = 1;
  result["accumulation_dtype"] = "fp32";
  result["output_dtype"] = "fp32";
  return result;
}

}  // namespace

bool adangel_o3_is_implemented() { return true; }

template <class Config>
py::dict benchmark_o3_config(
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
              "invalid O3 timing repetition count");
  c10::cuda::CUDAGuard guard(a_int8.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(a_int8.get_device()).stream();
  const int m = static_cast<int>(a_int8.size(0));
  const int k = static_cast<int>(a_int8.size(1));
  const int n = static_cast<int>(w_mxfp4_g128.size(0));
  auto byte_options = a_int8.options().dtype(at::kByte);
  auto q4 = at::empty({n, k / 2}, byte_options);
  auto split = at::empty({2 * m, k / 2}, byte_options);
  auto output = at::empty({m, n}, a_scale.options().dtype(at::kFloat));
  auto convert_w = [&]() { adangel_launch_mxfp4_to_q4(w_mxfp4_g128, q4, stream); };
  auto convert_a = [&]() { adangel_launch_split_int8_to_int4(a_int8, split, stream); };
  convert_w();
  convert_a();
  auto tma_low = make_tma_a<Config>(split, 0);
  auto tma_high = make_tma_a<Config>(split, m);
  auto tma_b = make_tma_b<Config>(q4);
  auto gemm = [&]() {
    launch_gemm<Config>(
        a_scale, w_scale_g128, output, m, n, k,
        tma_low, tma_high, tma_b, stream);
  };
  const TimingMode mode = parse_mode(mode_name);
  for (int i = 0; i < warmup; ++i) {
    if (mode == TimingMode::kConversionOnly) { convert_w(); convert_a(); }
    else if (mode == TimingMode::kCold) { convert_w(); convert_a(); gemm(); }
    else if (mode == TimingMode::kSteadyState) { convert_a(); gemm(); }
    else gemm();
  }
  check_cuda(cudaStreamSynchronize(stream), "O3 warmup synchronization");

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
  check_cuda(cudaEventSynchronize(final_event), "O3 timing synchronization");

  std::vector<float> isolated_w, isolated_a;
  if (mode == TimingMode::kConversionOnly || mode == TimingMode::kCold) {
    isolated_w = measure_batched(convert_w, repeats, conversion_inner_repeats, stream,
                                 "O3 weight conversion synchronization");
  }
  if (mode != TimingMode::kComputeOnly) {
    isolated_a = measure_batched(convert_a, repeats, conversion_inner_repeats, stream,
                                 "O3 activation conversion synchronization");
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
  result["converted_weight"] = q4;
  result["converted_activation"] = split;
  result["timings_ms"] = timings;
  result["timing_method"] = timing_metadata(mode_name, conversion_inner_repeats);
  result["kernel"] =
      kernel_metadata<Config>(k / kGroupSize, implementation_key, kernel_symbol);
  return result;
}

py::dict adangel_benchmark_o3_impl(
    const std::string& implementation,
    const std::string& mode_name,
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4_g128,
    const at::Tensor& w_scale_g128,
    int warmup,
    int repeats,
    int conversion_inner_repeats) {
  const O3Implementation selected = parse_o3_implementation(implementation);
  if (selected == O3Implementation::kN16K256) {
    return benchmark_o3_config<O3N16K256Config>(
        a_int8, a_scale, w_mxfp4_g128, w_scale_g128,
        "n16_k256", "adangel_o3_split_tma_ws<O3Config<16,2,false>>",
        mode_name, warmup, repeats, conversion_inner_repeats);
  }
  if (selected == O3Implementation::kN32K128) {
    return benchmark_o3_config<O3N32K128Config>(
        a_int8, a_scale, w_mxfp4_g128, w_scale_g128,
        "n32_k128", "adangel_o3_split_tma_ws<O3Config<32,1,false>>",
        mode_name, warmup, repeats, conversion_inner_repeats);
  }
  if (selected == O3Implementation::kN16K128Dual) {
    return benchmark_o3_config<O3N16K128DualConfig>(
        a_int8, a_scale, w_mxfp4_g128, w_scale_g128,
        "n16_k128_dual", "adangel_o3_split_tma_ws<O3Config<16,1,true>>",
        mode_name, warmup, repeats, conversion_inner_repeats);
  }
  if (selected == O3Implementation::kN32K256Dual) {
    return benchmark_o3_config<O3N32K256DualConfig>(
        a_int8, a_scale, w_mxfp4_g128, w_scale_g128,
        "n32_k256_dual", "adangel_o3_split_tma_ws<O3Config<32,2,true>>",
        mode_name, warmup, repeats, conversion_inner_repeats);
  }
  if (selected == O3Implementation::kN16K128Swizzle) {
    return benchmark_o3_config<O3N16K128SwizzleConfig>(
        a_int8, a_scale, w_mxfp4_g128, w_scale_g128,
        "n16_k128_swizzle", "adangel_o3_split_tma_ws<O3SwizzledConfig<16>>",
        mode_name, warmup, repeats, conversion_inner_repeats);
  }
  if (selected == O3Implementation::kN32K128Swizzle) {
    return benchmark_o3_config<O3N32K128SwizzleConfig>(
        a_int8, a_scale, w_mxfp4_g128, w_scale_g128,
        "n32_k128_swizzle", "adangel_o3_split_tma_ws<O3SwizzledConfig<32>>",
        mode_name, warmup, repeats, conversion_inner_repeats);
  }
  if (selected == O3Implementation::kM64N16K128) {
    return benchmark_o3_config<O3M64N16K128Config>(
        a_int8, a_scale, w_mxfp4_g128, w_scale_g128,
        "m64_n16_k128", "adangel_o3_split_tma_ws<O3Config<16,1,false,64>>",
        mode_name, warmup, repeats, conversion_inner_repeats);
  }
  if (selected == O3Implementation::kM64N32K128) {
    return benchmark_o3_config<O3M64N32K128Config>(
        a_int8, a_scale, w_mxfp4_g128, w_scale_g128,
        "m64_n32_k128", "adangel_o3_split_tma_ws<O3Config<32,1,false,64>>",
        mode_name, warmup, repeats, conversion_inner_repeats);
  }
  if (selected == O3Implementation::kM64N16K128CuteLdsm) {
    return benchmark_o3_config<O3M64N16K128CuteLdsmConfig>(
        a_int8, a_scale, w_mxfp4_g128, w_scale_g128,
        "m64_n16_k128_cute_ldsm",
        "adangel_o3_split_tma_ws<O3Config<16,1,false,64,true>>",
        mode_name, warmup, repeats, conversion_inner_repeats);
  }
  if (selected == O3Implementation::kN16K128CuteLdsm) {
    return benchmark_o3_config<O3N16K128CuteLdsmConfig>(
        a_int8, a_scale, w_mxfp4_g128, w_scale_g128,
        "n16_k128_cute_ldsm",
        "adangel_o3_split_tma_ws<O3Config<16,1,false,128,true>>",
        mode_name, warmup, repeats, conversion_inner_repeats);
  }
  if (selected == O3Implementation::kN32K128CuteLdsm) {
    return benchmark_o3_config<O3N32K128CuteLdsmConfig>(
        a_int8, a_scale, w_mxfp4_g128, w_scale_g128,
        "n32_k128_cute_ldsm",
        "adangel_o3_split_tma_ws<O3Config<32,1,false,128,true>>",
        mode_name, warmup, repeats, conversion_inner_repeats);
  }
  if (selected == O3Implementation::kN16K128LdsmSwizzle) {
    return benchmark_o3_config<O3N16K128LdsmSwizzleConfig>(
        a_int8, a_scale, w_mxfp4_g128, w_scale_g128,
        "n16_k128_ldsm_swizzle",
        "adangel_o3_split_tma_ws<O3SwizzledConfig<16,true>>",
        mode_name, warmup, repeats, conversion_inner_repeats);
  }
  return benchmark_o3_config<O3N16K128Config>(
      a_int8, a_scale, w_mxfp4_g128, w_scale_g128,
      "n16_k128", "adangel_o3_split_tma_ws<O3Config<16,1,false>>",
      mode_name, warmup, repeats, conversion_inner_repeats);
}

py::dict adangel_benchmark_o3(
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4_g128,
    const at::Tensor& w_scale_g128,
    const std::string& mode_name,
    int warmup,
    int repeats,
    int conversion_inner_repeats) {
  return adangel_benchmark_o3_impl(
      kProductionO3Implementation, mode_name,
      a_int8, a_scale, w_mxfp4_g128, w_scale_g128,
      warmup, repeats, conversion_inner_repeats);
}

py::dict adangel_run_o3(
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4_g128,
    const at::Tensor& w_scale_g128,
    const std::string& mode) {
  py::dict measured = adangel_benchmark_o3(
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
