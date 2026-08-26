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
#include <vector>

#include "adangel/data_types.cuh"
#include "adangel/kernel_api.h"

namespace py = pybind11;

namespace {

constexpr int kGroupSize = 128;
constexpr int kMmaK = 64;
constexpr int kKSubgroups = 2;
constexpr int kTileM = 128;
constexpr int kTileN = 16;
constexpr int kNReplicas = kTileN / 16;
constexpr int kGroupsPerStage = 1;
constexpr int kStages = 2;
constexpr int kProducerThreads = 32;
constexpr int kConsumerThreads = 512;
constexpr int kThreads = kProducerThreads + kConsumerThreads;
constexpr int kPackedK = kGroupSize / 2;
constexpr int kPipelinePackedK = kGroupsPerStage * kPackedK;
constexpr int kPipelineK = kGroupsPerStage * kGroupSize;
constexpr int kAStageBytes = kTileM * kPipelinePackedK;
constexpr int kBStageBytes = kTileN * kPipelinePackedK;

using Pipeline = cutlass::PipelineTmaAsync<kStages>;
using ByteLayoutA = decltype(cute::make_layout(
    cute::make_shape(cute::_128{}, cute::_64{}, cute::_2{}),
    cute::make_stride(cute::_64{}, cute::_1{}, cute::Int<kAStageBytes>{})));
using ByteLayoutB = decltype(cute::make_layout(
    cute::make_shape(cute::_16{}, cute::_64{}, cute::_2{}),
    cute::make_stride(cute::_64{}, cute::_1{}, cute::Int<kBStageBytes>{})));
using LowMma = cute::SM80_16x8x64_S32U4S4S32_TN;
using HighMma = cute::SM80_16x8x64_S32S4S4S32_TN;

struct alignas(128) SharedStorage {
  alignas(128) uint8_t a_low[kStages * kAStageBytes];
  alignas(128) uint8_t a_high[kStages * kAStageBytes];
  alignas(128) uint8_t b[kStages * kBStageBytes];
  alignas(128) float column_scale[kStages * kGroupsPerStage * kTileN];
  alignas(16) Pipeline::SharedStorage pipeline;
};

static_assert(kThreads == 544);

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

template <class TmaLow, class TmaHigh, class TmaB>
__global__ __launch_bounds__(kThreads) void adangel_o3_split_tma_ws(
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

  auto mLow = tma_low.get_tma_tensor(cute::make_shape(m, k / 2));
  auto mHigh = tma_high.get_tma_tensor(cute::make_shape(m, k / 2));
  auto mB = tma_b.get_tma_tensor(cute::make_shape(n, k / 2));
  auto byte_tiler = cute::make_shape(cute::_128{}, cute::_64{});
  auto a_coord = cute::make_coord(static_cast<int>(blockIdx.y), cute::_);
  auto b_coord = cute::make_coord(static_cast<int>(blockIdx.x), cute::_);
  auto gLow = cute::local_tile(mLow, byte_tiler, a_coord);
  auto gHigh = cute::local_tile(mHigh, byte_tiler, a_coord);
  auto gB = cute::local_tile(mB, cute::make_shape(cute::_16{}, cute::_64{}), b_coord);
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
  Pipeline::Params params;
  params.num_consumers = kConsumerThreads;
  params.transaction_bytes = 2 * kAStageBytes + kBStageBytes;
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

  const int compute_thread = thread - kProducerThreads;
  const int consumer_warp = compute_thread >> 5;
  const int lane_i = lane & 3;
  const int lane_j = lane >> 2;
  const int warp_m = consumer_warp & 7;
  const int warp_n = consumer_warp >> 3;
  const int tile_row = static_cast<int>(blockIdx.y) * kTileM;
  const int tile_column = static_cast<int>(blockIdx.x) * kTileN;
  const int local_row0 = warp_m * 16 + lane_j;
  const int local_row1 = local_row0 + 8;
  const int global_row0 = tile_row + local_row0;
  const int global_row1 = tile_row + local_row1;
  const float row_scale0 = global_row0 < m ? a_scale[global_row0] : 0.0f;
  const float row_scale1 = global_row1 < m ? a_scale[global_row1] : 0.0f;
  float accumulators[kNReplicas][4] = {};

  Pipeline::PipelineState read_state;
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
    constexpr int kAWordsPerRow = kPipelinePackedK / 4;
    constexpr int kBWordsPerRow = kPipelinePackedK / 4;
#pragma unroll
    for (int inner_group = 0; inner_group < kGroupsPerStage; ++inner_group) {
      const int group = pipeline_group * kGroupsPerStage + inner_group;
      if (group >= groups) break;
      uint32_t low[kNReplicas][4] = {};
      uint32_t high[kNReplicas][4] = {};
#pragma unroll
      for (int subgroup = 0; subgroup < kKSubgroups; ++subgroup) {
        const int word_base = inner_group * (kPackedK / 4) + subgroup * 8 + lane_i;
        const uint32_t la0 = low_words[local_row0 * kAWordsPerRow + word_base];
        const uint32_t la1 = low_words[local_row1 * kAWordsPerRow + word_base];
        const uint32_t la2 = low_words[local_row0 * kAWordsPerRow + word_base + 4];
        const uint32_t la3 = low_words[local_row1 * kAWordsPerRow + word_base + 4];
        const uint32_t ha0 = high_words[local_row0 * kAWordsPerRow + word_base];
        const uint32_t ha1 = high_words[local_row1 * kAWordsPerRow + word_base];
        const uint32_t ha2 = high_words[local_row0 * kAWordsPerRow + word_base + 4];
        const uint32_t ha3 = high_words[local_row1 * kAWordsPerRow + word_base + 4];
#pragma unroll
        for (int replica = 0; replica < kNReplicas; ++replica) {
          const int atom_n = warp_n * kNReplicas + replica;
          const int b_row = atom_n * 8 + lane_j;
          const uint32_t b0 = b_words[b_row * kBWordsPerRow + word_base];
          const uint32_t b1 = b_words[b_row * kBWordsPerRow + word_base + 4];
          LowMma::fma(
              low[replica][0], low[replica][1], low[replica][2], low[replica][3],
              la0, la1, la2, la3, b0, b1,
              low[replica][0], low[replica][1], low[replica][2], low[replica][3]);
          HighMma::fma(
              high[replica][0], high[replica][1], high[replica][2], high[replica][3],
              ha0, ha1, ha2, ha3, b0, b1,
              high[replica][0], high[replica][1], high[replica][2], high[replica][3]);
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
        const int32_t partial0 = static_cast<int32_t>(low[replica][0])
            + 16 * static_cast<int32_t>(high[replica][0]);
        const int32_t partial1 = static_cast<int32_t>(low[replica][1])
            + 16 * static_cast<int32_t>(high[replica][1]);
        const int32_t partial2 = static_cast<int32_t>(low[replica][2])
            + 16 * static_cast<int32_t>(high[replica][2]);
        const int32_t partial3 = static_cast<int32_t>(low[replica][3])
            + 16 * static_cast<int32_t>(high[replica][3]);
        accumulators[replica][0] = __fmaf_rn(
            static_cast<float>(partial0), __fmul_rn(row_scale0, column_scale0),
            accumulators[replica][0]);
        accumulators[replica][1] = __fmaf_rn(
            static_cast<float>(partial1), __fmul_rn(row_scale0, column_scale1),
            accumulators[replica][1]);
        accumulators[replica][2] = __fmaf_rn(
            static_cast<float>(partial2), __fmul_rn(row_scale1, column_scale0),
            accumulators[replica][2]);
        accumulators[replica][3] = __fmaf_rn(
            static_cast<float>(partial3), __fmul_rn(row_scale1, column_scale1),
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

auto make_tma_a(const at::Tensor& split, int row_offset) {
  const int m = static_cast<int>(split.size(0) / 2);
  const int packed_k = static_cast<int>(split.size(1));
  const uint8_t* pointer = split.data_ptr<uint8_t>() + static_cast<int64_t>(row_offset) * packed_k;
  auto tensor = cute::make_tensor(pointer, cute::make_shape(m, packed_k), cute::make_stride(packed_k, cute::_1{}));
  auto layout = ByteLayoutA{};
  return cute::make_tma_atom(
      cute::SM90_TMA_LOAD{}, tensor, layout(cute::_, cute::_, cute::Int<0>{}),
      cute::make_shape(cute::_128{}, cute::_64{}));
}

auto make_tma_b(const at::Tensor& q4) {
  const int n = static_cast<int>(q4.size(0));
  const int packed_k = static_cast<int>(q4.size(1));
  auto tensor = cute::make_tensor(
      q4.data_ptr<uint8_t>(), cute::make_shape(n, packed_k), cute::make_stride(packed_k, cute::_1{}));
  auto layout = ByteLayoutB{};
  return cute::make_tma_atom(
      cute::SM90_TMA_LOAD{}, tensor, layout(cute::_, cute::_, cute::Int<0>{}),
      cute::make_shape(cute::_16{}, cute::_64{}));
}

template <class TmaLow, class TmaHigh, class TmaB>
void launch_gemm(
    const at::Tensor& a_scale, const at::Tensor& w_scale, at::Tensor& output,
    int m, int n, int k, TmaLow const& low, TmaHigh const& high, TmaB const& b,
    cudaStream_t stream) {
  dim3 grid((n + kTileN - 1) / kTileN, (m + kTileM - 1) / kTileM);
  // The staged Split pipeline uses more than CUDA's default dynamic
  // shared-memory allowance. Opt this exact template specialization into the
  // device limit before launch; otherwise CUDA reports invalid argument.
  check_cuda(
      cudaFuncSetAttribute(
          adangel_o3_split_tma_ws<TmaLow, TmaHigh, TmaB>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(sizeof(SharedStorage))),
      "O3 set dynamic shared-memory limit");
  adangel_o3_split_tma_ws<<<grid, kThreads, sizeof(SharedStorage), stream>>>(
      low, high, b, a_scale.data_ptr<float>(), w_scale.data_ptr<uint8_t>(),
      output.data_ptr<float>(), m, n, k, k / kGroupSize);
  check_cuda(cudaGetLastError(), "O3 Split TMA warp-specialized launch");
}

py::dict kernel_metadata(int groups) {
  py::dict result;
  result["library"] = "CUTLASS CuTe + CUDA";
  result["implementation"] =
      "paper_split_g128_tma_warp_specialized_register_partial_occupancy_n16";
  result["kernel_symbol"] = "adangel_o3_split_tma_ws";
  result["tensor_core"] = true;
  result["mma_family"] = "IMMA_INT4";
  result["mma_api"] = "cute::arch MMA wrapper with trait-derived lane mapping";
  result["mma_atoms"] = py::make_tuple(
      "SM80_16x8x64_S32U4S4S32_TN", "SM80_16x8x64_S32S4S4S32_TN");
  result["mma_shape"] = "m16n8k64";
  result["data_movement"] = "TMA";
  result["kernel_schedule"] = "cooperative_warp_specialized";
  result["cta_tile"] = py::make_tuple(kTileM, kTileN, kPipelineK);
  result["pipeline_stages"] = kStages;
  result["groups_per_pipeline_stage"] = kGroupsPerStage;
  result["producer_warps"] = 1;
  result["consumer_warps"] = 16;
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

py::dict adangel_benchmark_o3(
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4_g128,
    const at::Tensor& w_scale_g128,
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
  auto tma_low = make_tma_a(split, 0);
  auto tma_high = make_tma_a(split, m);
  auto tma_b = make_tma_b(q4);
  auto gemm = [&]() {
    launch_gemm(a_scale, w_scale_g128, output, m, n, k, tma_low, tma_high, tma_b, stream);
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
  result["kernel"] = kernel_metadata(k / kGroupSize);
  return result;
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
