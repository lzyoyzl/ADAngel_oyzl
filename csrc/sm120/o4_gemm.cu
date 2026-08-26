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
constexpr int kWordsPerGroup = 4;
constexpr int kAPlanes = 8;
constexpr int kWPlanes = 4;
constexpr int kTileM = 128;
constexpr int kTileN = 64;
constexpr int kStages = 3;
constexpr int kProducerThreads = 32;
constexpr int kConsumerThreads = 512;
constexpr int kThreads = kProducerThreads + kConsumerThreads;
constexpr int kAStageWords = kAPlanes * kTileM * kWordsPerGroup;
constexpr int kBStageWords = kWPlanes * kTileN * kWordsPerGroup;

constexpr int kAWeights[kAPlanes] = {1, 2, 4, 8, 16, 32, 64, -128};
constexpr int kWWeights[kWPlanes] = {1, 2, 4, -8};

using Pipeline = cutlass::PipelineTmaAsync<kStages>;
using WordLayoutA = decltype(cute::make_layout(
    cute::make_shape(cute::_8{}, cute::_128{}, cute::_4{}, cute::_3{}),
    cute::make_stride(
        cute::Int<kTileM * kWordsPerGroup>{}, cute::_4{}, cute::_1{},
        cute::Int<kAStageWords>{})));
using WordLayoutB = decltype(cute::make_layout(
    cute::make_shape(cute::_4{}, cute::_64{}, cute::_4{}, cute::_3{}),
    cute::make_stride(
        cute::Int<kTileN * kWordsPerGroup>{}, cute::_4{}, cute::_1{},
        cute::Int<kBStageWords>{})));
using BitLayoutA = decltype(cute::make_layout(
    cute::make_shape(cute::_8{}, cute::_128{}, cute::_128{}, cute::_3{}),
    cute::make_stride(
        cute::Int<kTileM * kGroupSize>{}, cute::_128{}, cute::_1{},
        cute::Int<kAPlanes * kTileM * kGroupSize>{})));
using BitLayoutB = decltype(cute::make_layout(
    cute::make_shape(cute::_4{}, cute::_64{}, cute::_128{}, cute::_3{}),
    cute::make_stride(
        cute::Int<kTileN * kGroupSize>{}, cute::_128{}, cute::_1{},
        cute::Int<kWPlanes * kTileN * kGroupSize>{})));
using TiledMma = cute::TiledMMA<
    cute::MMA_Atom<cute::SM80_16x8x128_S32U1U1S32_TN_ANDPOPC>,
    cute::Layout<cute::Shape<cute::_8, cute::_2, cute::_1>>,
    cute::Tile<cute::_128, cute::_64, cute::_128>>;

struct alignas(128) SharedStorage {
  alignas(128) uint32_t a[kStages * kAStageWords];
  alignas(128) uint32_t b[kStages * kBStageWords];
  alignas(128) float column_scale[kStages * kTileN];
  alignas(16) Pipeline::SharedStorage pipeline;
};

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

template <class TmaA, class TmaB>
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
  extern __shared__ __align__(128) uint8_t shared_bytes[];
  auto& storage = *reinterpret_cast<SharedStorage*>(shared_bytes);
  auto mA = tma_a.get_tma_tensor(cute::make_shape(kAPlanes, m, k / 32));
  auto mB = tma_b.get_tma_tensor(cute::make_shape(kWPlanes, n, k / 32));
  auto gA = cute::local_tile(
      mA, cute::make_shape(cute::_8{}, cute::_128{}, cute::_4{}),
      cute::make_coord(cute::Int<0>{}, static_cast<int>(blockIdx.y), cute::_));
  auto gB = cute::local_tile(
      mB, cute::make_shape(cute::_4{}, cute::_64{}, cute::_4{}),
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
  Pipeline::Params params;
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
    for (int group = 0; group < groups; ++group) {
      if (lane == 0) pipeline.producer_acquire(write_state);
      __syncwarp();
      const int stage = write_state.index();
      for (int column = lane; column < kTileN; column += 32) {
        const int global_column = static_cast<int>(blockIdx.x) * kTileN + column;
        storage.column_scale[stage * kTileN + column] = global_column < n
            ? adangel::decode_ue8m0(w_scale[global_column * groups + group])
            : 0.0f;
      }
      __threadfence_block();
      __syncwarp();
      if (lane == 0) {
        auto* barrier = pipeline.producer_get_barrier(write_state);
        cute::copy(tma_a.with(*barrier), tAgA(cute::_, group), tAsA(cute::_, stage));
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
  auto sA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<cutlass::uint1b_t*>(storage.a)), BitLayoutA{});
  auto sB = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<cutlass::uint1b_t*>(storage.b)), BitLayoutB{});
  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_slice(compute_thread);
  auto cC = cute::make_identity_tensor(cute::make_shape(cute::_128{}, cute::_64{}));
  auto tCcC = thr_mma.partition_C(cC);
  auto tCrPartial = thr_mma.make_fragment_C(tCcC);
  auto tCrGroup = cute::make_fragment_like<int32_t>(tCrPartial);
  auto tCrAccumulator = cute::make_fragment_like<float>(tCrPartial);
  cute::clear(tCrAccumulator);
  float row_scales[2] = {0.0f, 0.0f};
#pragma unroll
  for (int slot = 0; slot < 2; ++slot) {
    const int local_row = cute::get<0>(tCcC(slot * 2));
    const int global_row = tile_row + local_row;
    row_scales[slot] = global_row < m ? a_scale[global_row] : 0.0f;
  }

  using BitCopy = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, cutlass::uint1b_t>;
  BitCopy bit_copy;
  auto tiled_copy_a = cute::make_tiled_copy_A(bit_copy, tiled_mma);
  auto tiled_copy_b = cute::make_tiled_copy_B(bit_copy, tiled_mma);
  auto copy_a_thr = tiled_copy_a.get_slice(compute_thread);
  auto copy_b_thr = tiled_copy_b.get_slice(compute_thread);

  Pipeline::PipelineState read_state;
  for (int group = 0; group < groups; ++group) {
    pipeline.consumer_wait(read_state);
    const int stage = read_state.index();
    cute::clear(tCrGroup);
#pragma unroll
    for (int a_plane = 0; a_plane < kAPlanes; ++a_plane) {
#pragma unroll
      for (int w_plane = 0; w_plane < kWPlanes; ++w_plane) {
        auto a_frag = thr_mma.partition_fragment_A(sA(a_plane, cute::_, cute::_, stage));
        auto b_frag = thr_mma.partition_fragment_B(sB(w_plane, cute::_, cute::_, stage));
        auto a_src = copy_a_thr.partition_S(sA(a_plane, cute::_, cute::_, stage));
        auto b_src = copy_b_thr.partition_S(sB(w_plane, cute::_, cute::_, stage));
        auto a_dst = copy_a_thr.retile_D(a_frag);
        auto b_dst = copy_b_thr.retile_D(b_frag);
        cute::copy(bit_copy, a_src, a_dst);
        cute::copy(bit_copy, b_src, b_dst);
        cute::clear(tCrPartial);
        cute::gemm(tiled_mma, a_frag, b_frag, tCrPartial);
        const int coefficient = kAWeights[a_plane] * kWWeights[w_plane];
#pragma unroll
        for (int item = 0; item < cute::size(tCrPartial); ++item) {
          tCrGroup(item) += coefficient * static_cast<int32_t>(tCrPartial(item));
        }
      }
    }

    float warp_scales[kTileN / 32];
#pragma unroll
    for (int round = 0; round < kTileN / 32; ++round) {
      warp_scales[round] = storage.column_scale[stage * kTileN + round * 32 + lane];
    }
#pragma unroll
    for (int item = 0; item < cute::size(tCrGroup); ++item) {
      const int local_column = cute::get<1>(tCcC(item));
      float column_scale = 0.0f;
#pragma unroll
      for (int round = 0; round < kTileN / 32; ++round) {
        const float candidate = __shfl_sync(
            0xffffffffu, warp_scales[round], local_column & 31);
        if ((local_column >> 5) == round) column_scale = candidate;
      }
      const float scale = __fmul_rn(row_scales[(item >> 1) & 1], column_scale);
      tCrAccumulator(item) = __fmaf_rn(
          static_cast<float>(tCrGroup(item)), scale, tCrAccumulator(item));
    }
    pipeline.consumer_release(read_state);
    ++read_state;
  }

  auto mC = cute::make_tensor(
      cute::make_gmem_ptr(output), cute::make_shape(m, n), cute::make_stride(n, cute::_1{}));
  auto gC = cute::local_tile(
      mC, cute::make_shape(cute::_128{}, cute::_64{}),
      cute::make_coord(static_cast<int>(blockIdx.y), static_cast<int>(blockIdx.x)));
  auto tCgC = thr_mma.partition_C(gC);
#pragma unroll
  for (int item = 0; item < cute::size(tCrAccumulator); ++item) {
    const int row = cute::get<0>(tCcC(item));
    const int column = cute::get<1>(tCcC(item));
    if (tile_row + row < m && tile_column + column < n) tCgC(item) = tCrAccumulator(item);
  }
}

auto make_tma_a(const at::Tensor& planes) {
  const int m = static_cast<int>(planes.size(1));
  const int words = static_cast<int>(planes.size(2));
  auto tensor = cute::make_tensor(
      reinterpret_cast<const uint32_t*>(planes.data_ptr<int32_t>()),
      cute::make_shape(kAPlanes, m, words),
      cute::make_stride(m * words, words, cute::_1{}));
  auto layout = WordLayoutA{};
  return cute::make_tma_atom(
      cute::SM90_TMA_LOAD{}, tensor,
      layout(cute::_, cute::_, cute::_, cute::Int<0>{}),
      cute::make_shape(cute::_8{}, cute::_128{}, cute::_4{}));
}

auto make_tma_b(const at::Tensor& planes) {
  const int n = static_cast<int>(planes.size(1));
  const int words = static_cast<int>(planes.size(2));
  auto tensor = cute::make_tensor(
      reinterpret_cast<const uint32_t*>(planes.data_ptr<int32_t>()),
      cute::make_shape(kWPlanes, n, words),
      cute::make_stride(n * words, words, cute::_1{}));
  auto layout = WordLayoutB{};
  return cute::make_tma_atom(
      cute::SM90_TMA_LOAD{}, tensor,
      layout(cute::_, cute::_, cute::_, cute::Int<0>{}),
      cute::make_shape(cute::_4{}, cute::_64{}, cute::_4{}));
}

template <class TmaA, class TmaB>
void launch_gemm(
    const at::Tensor& a_scale, const at::Tensor& w_scale, at::Tensor& output,
    int m, int n, int k, TmaA const& tma_a, TmaB const& tma_b, cudaStream_t stream) {
  dim3 grid((n + kTileN - 1) / kTileN, (m + kTileM - 1) / kTileM);
  adangel_o4_bitwise_tma_ws<<<grid, kThreads, sizeof(SharedStorage), stream>>>(
      tma_a, tma_b, a_scale.data_ptr<float>(), w_scale.data_ptr<uint8_t>(),
      output.data_ptr<float>(), m, n, k, k / kGroupSize);
  check_cuda(cudaGetLastError(), "O4 Bitwise TMA warp-specialized launch");
}

py::dict kernel_metadata(int groups) {
  py::dict result;
  result["library"] = "CUTLASS CuTe + CUDA";
  result["implementation"] = "paper_bitwise_twos_complement_g128_tma_warp_specialized";
  result["kernel_symbol"] = "adangel_o4_bitwise_tma_ws";
  result["tensor_core"] = true;
  result["mma_family"] = "BMMA";
  result["mma_api"] = "cute::MMA_Atom";
  result["mma_atom"] = "SM80_16x8x128_S32U1U1S32_TN_ANDPOPC";
  result["mma_shape"] = "m16n8k128";
  result["bit_operator"] = "and.popc";
  result["logical_mma_per_group"] = 32;
  result["activation_bit_weights"] = py::make_tuple(1, 2, 4, 8, 16, 32, 64, -128);
  result["weight_bit_weights"] = py::make_tuple(1, 2, 4, -8);
  result["data_movement"] = "TMA";
  result["kernel_schedule"] = "cooperative_warp_specialized";
  result["cta_tile"] = py::make_tuple(kTileM, kTileN, kGroupSize);
  result["pipeline_stages"] = kStages;
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

py::dict adangel_benchmark_o4(
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
  auto tma_a = make_tma_a(a_planes);
  auto tma_b = make_tma_b(w_planes);
  auto gemm = [&]() {
    launch_gemm(a_scale, w_scale_g128, output, m, n, k, tma_a, tma_b, stream);
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
  result["kernel"] = kernel_metadata(k / kGroupSize);
  return result;
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
      measured[std::string(name) + "_ms"] = samples.front();
    } else {
      measured[std::string(name) + "_ms"] = 0.0f;
    }
  }
  return measured;
}
