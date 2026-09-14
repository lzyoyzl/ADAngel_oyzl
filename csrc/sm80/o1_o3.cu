// Ampere port: ordinary warp-cooperative cp.async double buffering replaces
// SM120 TMA. Both variants use the same output tile and copy/synchronization
// policy. O1 retains K32 scale; O3 retains two's-complement Split and G128/Q4.
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cutlass/integer_subbyte.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <vector>
#include <cstring>
#include "adangel/kernel_api.h"
#include "adangel/data_types.cuh"

namespace py = pybind11;
namespace {
constexpr int TM = 64, TN = 64, THREADS = 256, STAGES = 2;
template<bool Split> struct Config {
  static constexpr int KStage = Split ? 128 : 64;
  static constexpr int KMma = Split ? 64 : 32;
  static constexpr int RowBytes = 64;
  using A = std::conditional_t<Split, cutlass::uint4b_t, int8_t>;
  using B = std::conditional_t<Split, cutlass::int4b_t, int8_t>;
  using Op = std::conditional_t<Split, cute::SM80_16x8x64_S32U4S4S32_TN,
                                      cute::SM80_16x8x32_S32S8S8S32_TN>;
  using HighOp = cute::SM80_16x8x64_S32S4S4S32_TN;
  using Mma = cute::TiledMMA<cute::MMA_Atom<Op>,
      cute::Layout<cute::Shape<cute::_4,cute::_2,cute::_1>>,
      cute::Tile<cute::Int<TM>,cute::Int<TN>,cute::Int<KMma>>>;
  using HighMma = cute::TiledMMA<cute::MMA_Atom<HighOp>,
      cute::Layout<cute::Shape<cute::_4,cute::_2,cute::_1>>,
      cute::Tile<cute::Int<TM>,cute::Int<TN>,cute::Int<64>>>;
};
struct alignas(128) Storage {
  alignas(128) uint8_t a[STAGES][TM * 64];
  alignas(128) uint8_t b[STAGES][TN * 64];
  alignas(128) uint8_t high[STAGES][TM * 64];
};
__device__ void copy16(void* dst, const void* src) {
  const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
      "r"(address), "l"(src) : "memory");
}

#include "o1_optimized.cuh"
#include "o3_optimized.cuh"
template<bool Split>
__device__ void prefetch(Storage& s, int slot, int group,
    const uint8_t* a, const uint8_t* b, int m, int n, int k) {
  int stride = Split ? k/2 : k;
  for (int offset = threadIdx.x * 16; offset < TM*64; offset += THREADS*16) {
    int row = int(blockIdx.y)*TM + offset/64;
    copy16(&s.a[slot][offset], a + row*stride + group*64 + offset%64);
    if constexpr(Split)
      copy16(&s.high[slot][offset], a + m*stride + row*stride + group*64 + offset%64);
  }
  for (int offset = threadIdx.x * 16; offset < TN*64; offset += THREADS*16) {
    int row = int(blockIdx.x)*TN + offset/64;
    copy16(&s.b[slot][offset], b + row*stride + group*64 + offset%64);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template<bool Split>
__global__ __launch_bounds__(THREADS) void adangel_sm80_grouped_gemm(
    const uint8_t* a, const uint8_t* b, const float* as,
    const uint8_t* ws, float* y, int m, int n, int k) {
  using C = Config<Split>;
  __shared__ Storage s;
  typename C::Mma mma;
  auto thr = mma.get_slice(threadIdx.x);
  auto layout = cute::make_layout(cute::make_shape(cute::Int<TM>{},cute::Int<C::KMma>{}),
      cute::make_stride(cute::Int<C::KStage>{},cute::_1{}));
  auto blayout = cute::make_layout(cute::make_shape(cute::Int<TN>{},cute::Int<C::KMma>{}),
      cute::make_stride(cute::Int<C::KStage>{},cute::_1{}));
  // Typed void* overload uses recast_ptr<T>: for 4-bit T this constructs a
  // subbyte iterator. A raw int4b_t* advances by sizeof(T)==1 byte, not a nibble.
  auto sa = cute::make_tensor(cute::make_smem_ptr<typename C::A>(static_cast<void*>(s.a[0])),layout);
  auto sb = cute::make_tensor(cute::make_smem_ptr<typename C::B>(static_cast<void*>(s.b[0])),blayout);
  auto ra = thr.partition_fragment_A(sa);
  auto rb = thr.partition_fragment_B(sb);
  auto coord = thr.partition_C(cute::make_identity_tensor(cute::make_shape(cute::Int<TM>{},cute::Int<TN>{})));
  auto partial = thr.make_fragment_C(coord);
  auto acc = cute::make_fragment_like<float>(partial);
  auto rows = cute::make_fragment_like<float>(partial);
  cute::clear(acc);
  CUTE_UNROLL
  for(int i=0;i<cute::size(rows);++i) rows(i)=as[int(blockIdx.y)*TM+cute::get<0>(coord(i))];
  using ACopy = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,typename C::A>;
  using BCopy = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,typename C::B>;
  auto ca = cute::make_tiled_copy_A(ACopy{},mma).get_slice(threadIdx.x);
  auto cb = cute::make_tiled_copy_B(BCopy{},mma).get_slice(threadIdx.x);
  auto da = ca.retile_D(ra);
  auto db = cb.retile_D(rb);
  const int count = k / C::KStage;
  prefetch<Split>(s,0,0,a,b,m,n,k);
  for(int group=0;group<count;++group) {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    int slot=group%STAGES;
    if(group+1<count) prefetch<Split>(s,(group+1)%STAGES,group+1,a,b,m,n,k);
    if constexpr(Split) {
      typename C::HighMma hm;
      auto ht=hm.get_slice(threadIdx.x);
      auto hs = cute::make_tensor(cute::make_smem_ptr<cutlass::int4b_t>(static_cast<void*>(s.high[slot])),layout);
      auto rh=ht.partition_fragment_A(hs);
      auto high=ht.make_fragment_C(coord);
      using HCopy=cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,cutlass::int4b_t>;
      auto ch=cute::make_tiled_copy_A(HCopy{},hm).get_slice(threadIdx.x);
      auto dh=ch.retile_D(rh);
      cute::clear(partial); cute::clear(high);
      CUTE_UNROLL
      for(int sub=0;sub<2;++sub) {
        auto al = cute::make_tensor(cute::make_smem_ptr<cutlass::uint4b_t>(static_cast<void*>(s.a[slot]+sub*32)),layout);
        auto ah = cute::make_tensor(cute::make_smem_ptr<cutlass::int4b_t>(static_cast<void*>(s.high[slot]+sub*32)),layout);
        auto bw = cute::make_tensor(cute::make_smem_ptr<cutlass::int4b_t>(static_cast<void*>(s.b[slot]+sub*32)),blayout);
        cute::copy(ACopy{},ca.partition_S(al),da);
        cute::copy(HCopy{},ch.partition_S(ah),dh);
        cute::copy(BCopy{},cb.partition_S(bw),db);
        cute::gemm(mma,ra,rb,partial);
        cute::gemm(hm,rh,rb,high);
      }
      CUTE_UNROLL
      for(int i=0;i<cute::size(acc);++i) {
        int col=int(blockIdx.x)*TN+cute::get<1>(coord(i));
        float scale=__fmul_rn(rows(i),adangel::decode_ue8m0(ws[col*count+group]));
        acc(i)=__fmaf_rn(float(partial(i)+16*high(i)),scale,acc(i));
      }
    } else {
      CUTE_UNROLL
      for(int sub=0;sub<2;++sub) {
        auto al = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<int8_t*>(s.a[slot]+sub*32)),layout);
        auto bw = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<int8_t*>(s.b[slot]+sub*32)),blayout);
        cute::copy(ACopy{},ca.partition_S(al),da);
        cute::copy(BCopy{},cb.partition_S(bw),db);
        cute::clear(partial); cute::gemm(mma,ra,rb,partial);
        CUTE_UNROLL
        for(int i=0;i<cute::size(acc);++i) {
          int col=int(blockIdx.x)*TN+cute::get<1>(coord(i));
          float wscale=__fmul_rn(adangel::decode_ue8m0(ws[col*(k/32)+group*2+sub]),0.5f);
          acc(i)=__fmaf_rn(float(partial(i)),__fmul_rn(rows(i),wscale),acc(i));
        }
      }
    }
    __syncthreads(); // All consumers finish before a slot may be overwritten.
  }
  CUTE_UNROLL
  for(int i=0;i<cute::size(acc);++i) {
    int row=int(blockIdx.y)*TM+cute::get<0>(coord(i));
    int col=int(blockIdx.x)*TN+cute::get<1>(coord(i));
    y[row*n+col]=acc(i);
  }
}
void check(cudaError_t rc) { TORCH_CHECK(rc==cudaSuccess,cudaGetErrorString(rc)); }
struct Event {
  cudaEvent_t e;
  Event(){check(cudaEventCreate(&e));}
  ~Event(){if(e) cudaEventDestroy(e);}
  Event(Event const&)=delete;
  Event(Event&& v):e(v.e){v.e=nullptr;}
};
struct Mark { Event start,w,a,end; };
float elapsed(Event const& x,Event const& y) {
  float t; check(cudaEventElapsedTime(&t,x.e,y.e)); return t;
}
template<class F> std::vector<float> batch(F f,int repeats,int inner,cudaStream_t stream) {
  std::vector<Mark> marks(repeats);
  for(auto& e:marks) {
    check(cudaEventRecord(e.start.e,stream));
    for(int j=0;j<inner;++j) f();
    check(cudaEventRecord(e.end.e,stream));
  }
  check(cudaEventSynchronize(marks.back().end.e));
  std::vector<float> r; for(auto& e:marks) r.push_back(elapsed(e.start,e.end)/inner);
  return r;
}

py::dict benchmark(std::string variant,std::string mode,at::Tensor a,at::Tensor as,
    at::Tensor w,at::Tensor ws,int warmup,int repeats,int inner,std::string implementation) {
  TORCH_CHECK(variant=="o1"||variant=="o3","variant must be o1 or o3");
  TORCH_CHECK(mode=="conversion_only"||mode=="compute_only"||mode=="cold"||mode=="steady_state","invalid mode");
  TORCH_CHECK(warmup>=0&&repeats>0&&inner>0,"invalid repetitions");
  bool split=variant=="o3";
  const auto requested_implementation=implementation;
  if(implementation=="production") {
    // Fixed, validated SM80 policy. O3 retains its original native INT4 path.
    // Smaller aligned O1 shapes use the corresponding tested exact kernel.
    if(split) implementation="baseline";
    else if(a.dim()==2 && a.size(0)%128==0 && a.size(1)%128==0)
      implementation="swizzle_128x64_k128_magic";
    else if(a.dim()==2 && a.size(1)%128==0)
      implementation="swizzle_64x64_k128_magic";
    else implementation="swizzle_64x64_k64";
  }
  const bool o3_candidate=split&&(implementation=="o3_swizzle_64x64_k128_magic"||
      implementation=="o3_swizzle_64x128_k256_exp_static_stream_bound2"||
      implementation=="o3_swizzle_64x128_k256_exp_static_stream"||
      implementation=="o3_swizzle_64x128_k128_exp_static_stream"||
      implementation=="o3_swizzle_64x128_k128_exp_static_cached"||
      implementation=="o3_swizzle_32x128_k128_exp_static"||
      implementation=="o3_swizzle_64x128_k128_exp_static_phase"||implementation=="o3_swizzle_64x128_k128_exp_merge_static_phase"||
      implementation=="o3_swizzle_64x128_k128_exp_static"||implementation=="o3_swizzle_64x128_k128_exp_merge_static"||
      implementation=="o3_swizzle_64x128_k128_16w_exp_static"||implementation=="o3_swizzle_64x128_k128_16w_exp_merge_static"||
      implementation=="o3_swizzle_64x128_k128_exp_merge"||implementation=="o3_swizzle_64x128_k128_16w_exp_merge"||
      implementation=="o3_swizzle_64x128_k128_16w_magic"||implementation=="o3_swizzle_64x128_k128_16w_exp"||
      implementation=="o3_swizzle_64x128_k128_exp"||implementation=="o3_swizzle_64x64_k128_exp"||
      implementation=="o3_swizzle_64x32_k128_magic"||implementation=="o3_swizzle_64x128_k128_magic"||
      implementation=="o3_swizzle_64x32_k128_magic_cached"||implementation=="o3_swizzle_64x64_k128_magic_cached"||
      implementation=="o3_swizzle_128x64_k128_magic"||implementation=="o3_swizzle_128x64_k256_magic");
  TORCH_CHECK(implementation=="baseline" || o3_candidate || (!split && (
      implementation=="swizzle_64x64_k128" || implementation=="swizzle_128x64_k128" ||
      implementation=="swizzle_64x64_k64" || implementation=="swizzle_64x32_k128" ||
      implementation=="swizzle_64x64_k128_16w" || implementation=="swizzle_128x64_k128_16w" ||
      implementation=="swizzle_64x64_k128_16w_exp" || implementation=="swizzle_128x64_k128_exp" ||
      implementation=="swizzle_64x64_k128_pair" || implementation=="swizzle_64x32_k128_pair" ||
      implementation=="swizzle_64x64_k128_magic" || implementation=="swizzle_128x64_k128_magic" ||
      implementation=="swizzle_64x32_k128_magic" ||
      implementation=="swizzle_64x64_k128_magic_stream" || implementation=="swizzle_128x64_k128_magic_stream" ||
      implementation=="swizzle_128x128_k128" || implementation=="swizzle_128x64_k64")),
      "unknown implementation or O1-only candidate requested for O3");
  TORCH_CHECK(a.is_cuda()&&as.is_cuda()&&w.is_cuda()&&ws.is_cuda(),"CUDA tensors required");
  TORCH_CHECK(a.device()==as.device()&&a.device()==w.device()&&a.device()==ws.device(),"device mismatch");
  TORCH_CHECK(a.dim()==2&&w.dim()==2&&as.dim()==1&&ws.dim()==2,"invalid ranks");
  TORCH_CHECK(a.scalar_type()==at::kChar&&as.scalar_type()==at::kFloat&&w.scalar_type()==at::kByte&&ws.scalar_type()==at::kByte,"invalid dtypes");
  TORCH_CHECK(a.is_contiguous()&&as.is_contiguous()&&w.is_contiguous()&&ws.is_contiguous(),"contiguous required");
  int m=a.size(0),k=a.size(1),n=w.size(0),g=split?128:32;
  TORCH_CHECK(m>0&&n>0&&k>0&&m%TM==0&&n%TN==0&&k%(split?128:64)==0,"aligned M/N64 and K64(O1)/K128(O3) required");
  TORCH_CHECK(as.size(0)==m&&w.size(1)==k/2&&ws.size(0)==n&&ws.size(1)==k/g,"shape mismatch");
  int tile_m=TM,tile_n=TN,tile_k=split?128:64;
  if(implementation!="baseline") {
    tile_m=(implementation.rfind("swizzle_64x",0)==0||implementation.rfind("o3_swizzle_64x",0)==0)?64:128;
    if(implementation.rfind("o3_swizzle_32x",0)==0) tile_m=32;
    tile_n=implementation=="swizzle_128x128_k128"?128:(implementation.rfind("swizzle_64x32_",0)==0?32:64);
    if(implementation.rfind("o3_swizzle_64x32_",0)==0) tile_n=32;
    if(implementation.rfind("o3_swizzle_64x128_",0)==0||implementation.rfind("o3_swizzle_32x128_",0)==0) tile_n=128;
    tile_k=implementation=="swizzle_128x64_k64"||implementation=="swizzle_64x64_k64"?64:128;
    if(o3_candidate&&implementation.find("_k256_")!=std::string::npos) tile_k=256;
    TORCH_CHECK(m%tile_m==0&&n%tile_n==0&&k%tile_k==0,"candidate tile alignment required");
    if(implementation.find("_cached")!=std::string::npos) TORCH_CHECK(k<=4096,"cached scale panel requires K<=4096");
    if(implementation.find("_bound2")!=std::string::npos)
      TORCH_CHECK(a.numel()<=0xffffffffLL&&w.numel()<=0xffffffffLL,
          "bound2 packed buffers must fit unsigned 32-bit byte offsets");
  }
  c10::cuda::CUDAGuard guard(a.device());
  cudaDeviceProp prop; check(cudaGetDeviceProperties(&prop,a.get_device()));
  TORCH_CHECK(prop.major==8&&prop.minor==0,"this experiment requires SM80 A100");
  auto stream=c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
  // Validation and allocation are outside every timing range.
  TORCH_CHECK(at::isfinite(as).all().item<bool>()&&as.ge(0).all().item<bool>(),"invalid activation scale");
  TORCH_CHECK(ws.ne(255).all().item<bool>(),"UE8M0 code 255 is invalid");
  bool exponent_scale=false;
  if(implementation.find("_exp")!=std::string::npos||implementation.find("_magic")!=std::string::npos) {
    float amin=as.min().item<float>(),amax=as.max().item<float>();
    uint32_t lo,hi; std::memcpy(&lo,&amin,4);std::memcpy(&hi,&amax,4);
    int emin=(lo>>23)&255,emax=(hi>>23)&255;
    int wmin=ws.min().item<int>(),wmax=ws.max().item<int>();
    const int bias=split?127:128;
    exponent_scale=amin>0&&emin>0&&emin+wmin-bias>=1&&emax+wmax-bias<=254;
  }
  auto out=at::empty({m,n},as.options());
  auto wa=at::empty({n,split?k/2:k},w.options().dtype(split?at::kByte:at::kChar));
  auto aa=split?at::empty({2*m,k/2},w.options()):a;
  if(implementation=="o3_swizzle_64x128_k128_exp_static_stream") {
    if(exponent_scale) o3_configure<64,128,128,true,false,false,2,false,true,false,true>(); else o3_configure<64,128,128,false,false,false,2,false,true,false,true>();
  }
  if(implementation=="o3_swizzle_64x128_k256_exp_static_stream") {
    if(exponent_scale) o3_configure<64,128,256,true,false,false,2,false,true,false,true>(); else o3_configure<64,128,256,false,false,false,2,false,true,false,true>();
  }
  if(implementation=="o3_swizzle_64x128_k256_exp_static_stream_bound2") {
    if(exponent_scale) o3_configure<64,128,256,true,false,false,2,false,true,false,true,true>(); else o3_configure<64,128,256,false,false,false,2,false,true,false,true,true>();
  }
  if(implementation=="o3_swizzle_64x128_k128_exp_static_cached") {
    if(exponent_scale) o3_configure<64,128,128,true,true,false,2,false,true>(); else o3_configure<64,128,128,false,true,false,2,false,true>();
  }
  if(implementation=="o3_swizzle_32x128_k128_exp_static") {
    if(exponent_scale) o3_configure<32,128,128,true,false,false,4,false,true>(); else o3_configure<32,128,128,false,false,false,4,false,true>();
  }
  if(implementation=="o3_swizzle_64x128_k128_exp_static_phase") {
    if(exponent_scale) o3_configure<64,128,128,true,false,false,2,false,true,true>(); else o3_configure<64,128,128,false,false,false,2,false,true,true>();
  }
  if(implementation=="o3_swizzle_64x128_k128_exp_merge_static_phase") {
    if(exponent_scale) o3_configure<64,128,128,true,false,false,2,true,true,true>(); else o3_configure<64,128,128,false,false,false,2,true,true,true>();
  }
  if(implementation=="o3_swizzle_64x128_k128_16w_exp_static") {
    if(exponent_scale) o3_configure<64,128,128,true,false,false,4,false,true>(); else o3_configure<64,128,128,false,false,false,4,false,true>();
  }
  if(implementation=="o3_swizzle_64x128_k128_16w_exp_merge_static") {
    if(exponent_scale) o3_configure<64,128,128,true,false,false,4,true,true>(); else o3_configure<64,128,128,false,false,false,4,true,true>();
  }
  if(implementation=="o3_swizzle_64x128_k128_exp_static") {
    if(exponent_scale) o3_configure<64,128,128,true,false,false,2,false,true>(); else o3_configure<64,128,128,false,false,false,2,false,true>();
  }
  if(implementation=="o3_swizzle_64x128_k128_exp_merge_static") {
    if(exponent_scale) o3_configure<64,128,128,true,false,false,2,true,true>(); else o3_configure<64,128,128,false,false,false,2,true,true>();
  }
  if(implementation=="o3_swizzle_64x128_k128_exp_merge") {
    if(exponent_scale) o3_configure<64,128,128,true,false,false,2,true>(); else o3_configure<64,128,128,false,false,false,2,true>();
  }
  if(implementation=="o3_swizzle_64x128_k128_16w_exp_merge") {
    if(exponent_scale) o3_configure<64,128,128,true,false,false,4,true>(); else o3_configure<64,128,128,false,false,false,4,true>();
  }
  if(implementation=="o3_swizzle_64x128_k128_16w_magic") {
    if(exponent_scale) o3_configure<64,128,128,true,false,true,4>(); else o3_configure<64,128,128,false,false,false,4>();
  }
  if(implementation=="o3_swizzle_64x128_k128_16w_exp") {
    if(exponent_scale) o3_configure<64,128,128,true,false,false,4>(); else o3_configure<64,128,128,false,false,false,4>();
  }
  if(implementation=="o3_swizzle_64x128_k128_exp") {
    if(exponent_scale) o3_configure<64,128,128,true,false,false>(); else o3_configure<64,128,128,false>();
  }
  if(implementation=="o3_swizzle_64x64_k128_exp") {
    if(exponent_scale) o3_configure<64,64,128,true,false,false>(); else o3_configure<64,64,128,false>();
  }
  if(implementation=="o3_swizzle_64x32_k128_magic_cached") {
    if(exponent_scale) o3_configure<64,32,128,true,true>(); else o3_configure<64,32,128,false,true>();
  }
  if(implementation=="o3_swizzle_64x64_k128_magic_cached") {
    if(exponent_scale) o3_configure<64,64,128,true,true>(); else o3_configure<64,64,128,false,true>();
  }
  if(implementation=="o3_swizzle_64x32_k128_magic") {
    if(exponent_scale) o3_configure<64,32,128,true>(); else o3_configure<64,32,128,false>();
  }
  if(implementation=="o3_swizzle_64x128_k128_magic") {
    if(exponent_scale) o3_configure<64,128,128,true>(); else o3_configure<64,128,128,false>();
  }
  if(implementation=="o3_swizzle_64x64_k128_magic") {
    if(exponent_scale) o3_configure<64,64,128,true>(); else o3_configure<64,64,128,false>();
  }
  if(implementation=="o3_swizzle_128x64_k128_magic") {
    if(exponent_scale) o3_configure<128,64,128,true>(); else o3_configure<128,64,128,false>();
  }
  if(implementation=="o3_swizzle_128x64_k256_magic") {
    if(exponent_scale) o3_configure<128,64,256,true>(); else o3_configure<128,64,256,false>();
  }
  if(implementation=="swizzle_64x64_k128") o1_ampere_configure<64,64,128>();
  if(implementation=="swizzle_64x64_k64") o1_ampere_configure<64,64,64>();
  if(implementation=="swizzle_64x32_k128") o1_ampere_configure<64,32,128>();
  if(implementation=="swizzle_64x64_k128_16w") o1_ampere_configure<64,64,128,4,4>();
  if(implementation=="swizzle_128x64_k128_16w") o1_ampere_configure<128,64,128,8,2>();
  if(implementation=="swizzle_64x64_k128_16w_exp") {
    if(exponent_scale) o1_ampere_configure<64,64,128,4,4,true>();
    else o1_ampere_configure<64,64,128,4,4>();
  }
  if(implementation=="swizzle_128x64_k128_exp") {
    if(exponent_scale) o1_ampere_configure<128,64,128,4,2,true>();
    else o1_ampere_configure<128,64,128>();
  }
  if(implementation=="swizzle_64x64_k128_pair") o1_ampere_configure<64,64,128,4,2,false,true>();
  if(implementation=="swizzle_64x32_k128_pair") o1_ampere_configure<64,32,128,4,2,false,true>();
  if(implementation=="swizzle_64x64_k128_magic") {
    if(exponent_scale) o1_ampere_configure<64,64,128,4,2,true,false,true>();
    else o1_ampere_configure<64,64,128>();
  }
  if(implementation=="swizzle_128x64_k128_magic") {
    if(exponent_scale) o1_ampere_configure<128,64,128,4,2,true,false,true>();
    else o1_ampere_configure<128,64,128>();
  }
  if(implementation=="swizzle_64x32_k128_magic") {
    if(exponent_scale) o1_ampere_configure<64,32,128,4,2,true,false,true>();
    else o1_ampere_configure<64,32,128>();
  }
  if(implementation=="swizzle_128x64_k128") o1_ampere_configure<128,64,128>();
  if(implementation=="swizzle_64x64_k128_magic_stream") {
    if(exponent_scale) o1_ampere_configure<64,64,128,4,2,true,false,true,true>();
    else o1_ampere_configure<64,64,128>();
  }
  if(implementation=="swizzle_128x64_k128_magic_stream") {
    if(exponent_scale) o1_ampere_configure<128,64,128,4,2,true,false,true,true>();
    else o1_ampere_configure<128,64,128>();
  }
  if(implementation=="swizzle_128x128_k128") o1_ampere_configure<128,128,128>();
  if(implementation=="swizzle_128x64_k64") o1_ampere_configure<128,64,64>();
  auto cvw=[&](){if(split) adangel_launch_mxfp4_to_q4(w,wa,stream); else adangel_launch_mxfp4_to_int8(w,wa,stream);};
  auto cva=[&](){if(split) adangel_launch_split_int8_to_int4(a,aa,stream);};
  auto gemm=[&](){
    dim3 grid(n/TN,m/TM);
    auto ap=reinterpret_cast<uint8_t*>(aa.data_ptr()); auto bp=reinterpret_cast<uint8_t*>(wa.data_ptr());
    if(implementation=="o3_swizzle_64x128_k128_exp_static_stream") {
      if(exponent_scale) o3_launch<64,128,128,true,false,false,2,false,true,false,true>(aa,wa,as,ws,out,stream); else o3_launch<64,128,128,false,false,false,2,false,true,false,true>(aa,wa,as,ws,out,stream);
    }
    else if(implementation=="o3_swizzle_64x128_k256_exp_static_stream") {
      if(exponent_scale) o3_launch<64,128,256,true,false,false,2,false,true,false,true>(aa,wa,as,ws,out,stream); else o3_launch<64,128,256,false,false,false,2,false,true,false,true>(aa,wa,as,ws,out,stream);
    }
    else if(implementation=="o3_swizzle_64x128_k256_exp_static_stream_bound2") {
      if(exponent_scale) o3_launch<64,128,256,true,false,false,2,false,true,false,true,true>(aa,wa,as,ws,out,stream); else o3_launch<64,128,256,false,false,false,2,false,true,false,true,true>(aa,wa,as,ws,out,stream);
    }
    else if(implementation=="o3_swizzle_64x128_k128_exp_static_cached") {
      if(exponent_scale) o3_launch<64,128,128,true,true,false,2,false,true>(aa,wa,as,ws,out,stream); else o3_launch<64,128,128,false,true,false,2,false,true>(aa,wa,as,ws,out,stream);
    }
    else if(implementation=="o3_swizzle_32x128_k128_exp_static") {
      if(exponent_scale) o3_launch<32,128,128,true,false,false,4,false,true>(aa,wa,as,ws,out,stream); else o3_launch<32,128,128,false,false,false,4,false,true>(aa,wa,as,ws,out,stream);
    }
    else if(implementation=="o3_swizzle_64x128_k128_exp_static_phase") {
      if(exponent_scale) o3_launch<64,128,128,true,false,false,2,false,true,true>(aa,wa,as,ws,out,stream); else o3_launch<64,128,128,false,false,false,2,false,true,true>(aa,wa,as,ws,out,stream);
    }
    else if(implementation=="o3_swizzle_64x128_k128_exp_merge_static_phase") {
      if(exponent_scale) o3_launch<64,128,128,true,false,false,2,true,true,true>(aa,wa,as,ws,out,stream); else o3_launch<64,128,128,false,false,false,2,true,true,true>(aa,wa,as,ws,out,stream);
    }
    else if(implementation=="o3_swizzle_64x128_k128_16w_exp_static") {
      if(exponent_scale) o3_launch<64,128,128,true,false,false,4,false,true>(aa,wa,as,ws,out,stream); else o3_launch<64,128,128,false,false,false,4,false,true>(aa,wa,as,ws,out,stream);
    }
    else if(implementation=="o3_swizzle_64x128_k128_16w_exp_merge_static") {
      if(exponent_scale) o3_launch<64,128,128,true,false,false,4,true,true>(aa,wa,as,ws,out,stream); else o3_launch<64,128,128,false,false,false,4,true,true>(aa,wa,as,ws,out,stream);
    }
    else if(implementation=="o3_swizzle_64x128_k128_exp_static") {
      if(exponent_scale) o3_launch<64,128,128,true,false,false,2,false,true>(aa,wa,as,ws,out,stream); else o3_launch<64,128,128,false,false,false,2,false,true>(aa,wa,as,ws,out,stream);
    }
    else if(implementation=="o3_swizzle_64x128_k128_exp_merge_static") {
      if(exponent_scale) o3_launch<64,128,128,true,false,false,2,true,true>(aa,wa,as,ws,out,stream); else o3_launch<64,128,128,false,false,false,2,true,true>(aa,wa,as,ws,out,stream);
    }
    else if(implementation=="o3_swizzle_64x128_k128_exp_merge") {
      if(exponent_scale) o3_launch<64,128,128,true,false,false,2,true>(aa,wa,as,ws,out,stream); else o3_launch<64,128,128,false,false,false,2,true>(aa,wa,as,ws,out,stream);
    }
    else if(implementation=="o3_swizzle_64x128_k128_16w_exp_merge") {
      if(exponent_scale) o3_launch<64,128,128,true,false,false,4,true>(aa,wa,as,ws,out,stream); else o3_launch<64,128,128,false,false,false,4,true>(aa,wa,as,ws,out,stream);
    }
    else if(implementation=="o3_swizzle_64x128_k128_16w_magic") {
      if(exponent_scale) o3_launch<64,128,128,true,false,true,4>(aa,wa,as,ws,out,stream); else o3_launch<64,128,128,false,false,false,4>(aa,wa,as,ws,out,stream);
    }
    else if(implementation=="o3_swizzle_64x128_k128_16w_exp") {
      if(exponent_scale) o3_launch<64,128,128,true,false,false,4>(aa,wa,as,ws,out,stream); else o3_launch<64,128,128,false,false,false,4>(aa,wa,as,ws,out,stream);
    }
    else if(implementation=="o3_swizzle_64x128_k128_exp") {
      if(exponent_scale) o3_launch<64,128,128,true,false,false>(aa,wa,as,ws,out,stream); else o3_launch<64,128,128,false>(aa,wa,as,ws,out,stream);
    }
    else if(implementation=="o3_swizzle_64x64_k128_exp") {
      if(exponent_scale) o3_launch<64,64,128,true,false,false>(aa,wa,as,ws,out,stream); else o3_launch<64,64,128,false>(aa,wa,as,ws,out,stream);
    }
    else if(implementation=="o3_swizzle_64x32_k128_magic_cached") {
      if(exponent_scale) o3_launch<64,32,128,true,true>(aa,wa,as,ws,out,stream); else o3_launch<64,32,128,false,true>(aa,wa,as,ws,out,stream);
    }
    else if(implementation=="o3_swizzle_64x64_k128_magic_cached") {
      if(exponent_scale) o3_launch<64,64,128,true,true>(aa,wa,as,ws,out,stream); else o3_launch<64,64,128,false,true>(aa,wa,as,ws,out,stream);
    }
    else if(implementation=="o3_swizzle_64x32_k128_magic") {
      if(exponent_scale) o3_launch<64,32,128,true>(aa,wa,as,ws,out,stream); else o3_launch<64,32,128,false>(aa,wa,as,ws,out,stream);
    }
    else if(implementation=="o3_swizzle_64x128_k128_magic") {
      if(exponent_scale) o3_launch<64,128,128,true>(aa,wa,as,ws,out,stream); else o3_launch<64,128,128,false>(aa,wa,as,ws,out,stream);
    }
    else if(implementation=="o3_swizzle_64x64_k128_magic") {
      if(exponent_scale) o3_launch<64,64,128,true>(aa,wa,as,ws,out,stream); else o3_launch<64,64,128,false>(aa,wa,as,ws,out,stream);
    }
    else if(implementation=="o3_swizzle_128x64_k128_magic") {
      if(exponent_scale) o3_launch<128,64,128,true>(aa,wa,as,ws,out,stream); else o3_launch<128,64,128,false>(aa,wa,as,ws,out,stream);
    }
    else if(implementation=="o3_swizzle_128x64_k256_magic") {
      if(exponent_scale) o3_launch<128,64,256,true>(aa,wa,as,ws,out,stream); else o3_launch<128,64,256,false>(aa,wa,as,ws,out,stream);
    }
    else if(split) adangel_sm80_grouped_gemm<true><<<grid,THREADS,0,stream>>>(ap,bp,as.data_ptr<float>(),ws.data_ptr<uint8_t>(),out.data_ptr<float>(),m,n,k);
    else if(implementation=="swizzle_64x64_k128") o1_ampere_launch<64,64,128>(aa,wa,as,ws,out,stream);
    else if(implementation=="swizzle_64x64_k64") o1_ampere_launch<64,64,64>(aa,wa,as,ws,out,stream);
    else if(implementation=="swizzle_64x32_k128") o1_ampere_launch<64,32,128>(aa,wa,as,ws,out,stream);
    else if(implementation=="swizzle_64x64_k128_16w") o1_ampere_launch<64,64,128,4,4>(aa,wa,as,ws,out,stream);
    else if(implementation=="swizzle_128x64_k128_16w") o1_ampere_launch<128,64,128,8,2>(aa,wa,as,ws,out,stream);
    else if(implementation=="swizzle_64x64_k128_16w_exp") {
      if(exponent_scale) o1_ampere_launch<64,64,128,4,4,true>(aa,wa,as,ws,out,stream);
      else o1_ampere_launch<64,64,128,4,4>(aa,wa,as,ws,out,stream);
    }
    else if(implementation=="swizzle_128x64_k128_exp") {
      if(exponent_scale) o1_ampere_launch<128,64,128,4,2,true>(aa,wa,as,ws,out,stream);
      else o1_ampere_launch<128,64,128>(aa,wa,as,ws,out,stream);
    }
    else if(implementation=="swizzle_64x64_k128_pair") o1_ampere_launch<64,64,128,4,2,false,true>(aa,wa,as,ws,out,stream);
    else if(implementation=="swizzle_64x32_k128_pair") o1_ampere_launch<64,32,128,4,2,false,true>(aa,wa,as,ws,out,stream);
    else if(implementation=="swizzle_64x64_k128_magic") {
      if(exponent_scale) o1_ampere_launch<64,64,128,4,2,true,false,true>(aa,wa,as,ws,out,stream);
      else o1_ampere_launch<64,64,128>(aa,wa,as,ws,out,stream);
    }
    else if(implementation=="swizzle_128x64_k128_magic") {
      if(exponent_scale) o1_ampere_launch<128,64,128,4,2,true,false,true>(aa,wa,as,ws,out,stream);
      else o1_ampere_launch<128,64,128>(aa,wa,as,ws,out,stream);
    }
    else if(implementation=="swizzle_64x32_k128_magic") {
      if(exponent_scale) o1_ampere_launch<64,32,128,4,2,true,false,true>(aa,wa,as,ws,out,stream);
      else o1_ampere_launch<64,32,128>(aa,wa,as,ws,out,stream);
    }
    else if(implementation=="swizzle_128x64_k128") o1_ampere_launch<128,64,128>(aa,wa,as,ws,out,stream);
    else if(implementation=="swizzle_64x64_k128_magic_stream") {
      if(exponent_scale) o1_ampere_launch<64,64,128,4,2,true,false,true,true>(aa,wa,as,ws,out,stream);
      else o1_ampere_launch<64,64,128>(aa,wa,as,ws,out,stream);
    }
    else if(implementation=="swizzle_128x64_k128_magic_stream") {
      if(exponent_scale) o1_ampere_launch<128,64,128,4,2,true,false,true,true>(aa,wa,as,ws,out,stream);
      else o1_ampere_launch<128,64,128>(aa,wa,as,ws,out,stream);
    }
    else if(implementation=="swizzle_128x128_k128") o1_ampere_launch<128,128,128>(aa,wa,as,ws,out,stream);
    else if(implementation=="swizzle_128x64_k64") o1_ampere_launch<128,64,64>(aa,wa,as,ws,out,stream);
    else adangel_sm80_grouped_gemm<false><<<grid,THREADS,0,stream>>>(ap,bp,as.data_ptr<float>(),ws.data_ptr<uint8_t>(),out.data_ptr<float>(),m,n,k);
    check(cudaGetLastError());
  };
  bool weight=mode=="cold"||mode=="conversion_only";
  bool activation=split&&mode!="compute_only";
  bool compute=mode!="conversion_only";
  cvw(); cva();
  for(int j=0;j<warmup;++j){if(weight) cvw();if(activation)cva();if(compute)gemm();}
  check(cudaStreamSynchronize(stream));
  std::vector<Mark> markers(repeats);
  for(auto& e:markers) {
    check(cudaEventRecord(e.start.e,stream));
    if(weight)cvw(); check(cudaEventRecord(e.w.e,stream));
    if(activation)cva(); check(cudaEventRecord(e.a.e,stream));
    if(compute)gemm(); check(cudaEventRecord(e.end.e,stream));
  }
  check(cudaEventSynchronize(markers.back().end.e));
  auto wt=weight?batch(cvw,repeats,inner,stream):std::vector<float>{};
  auto at=activation?batch(cva,repeats,inner,stream):std::vector<float>{};
  std::vector<float> gt,total;
  for(int j=0;j<repeats;++j){
    if(compute)gt.push_back(elapsed(markers[j].a,markers[j].end));
    total.push_back(compute?elapsed(markers[j].start,markers[j].end):wt[j]+(activation?at[j]:0.0f));
  }
  if(!compute) {gemm();check(cudaStreamSynchronize(stream));}
  py::dict timings;
  if(weight)timings["weight_conversion"]=wt;
  if(activation)timings["activation_conversion"]=at;
  if(compute)timings["gemm"]=gt;
  timings["total"]=total;
  py::dict meta;
  meta["architecture"]="sm80";meta["variant"]=variant;
  meta["cta_tile"]=py::make_tuple(tile_m,tile_n,tile_k);
  meta["implementation"]=implementation;
  meta["requested_implementation"]=requested_implementation;
  meta["exponent_scale_fast_path"]=exponent_scale;
  meta["paired_mma_issue"]=implementation.find("_pair")!=std::string::npos;
  meta["stream_atom_partial"]=exponent_scale&&implementation.find("_stream")!=std::string::npos;
  if(o3_candidate) meta["stream_atom_partial"]=implementation.find("_stream")!=std::string::npos;
  if(o3_candidate) meta["operand_register_double_buffer"]=implementation.find("_stream")!=std::string::npos;
  meta["integer_conversion"]=(exponent_scale&&implementation.find("_magic")!=std::string::npos)?"exact_bias_bits_fadd":"i2f";
  if(split) meta["integer_partial_reconstruction"]=implementation.find("_merge")!=std::string::npos?
      "high_times16_then_low_mma":"separate_low_high_then_imad";
  if(split) meta["copy_loop"]=implementation.find("_static")!=std::string::npos?"compile_time":"runtime";
  if(o3_candidate) meta["pipeline_phase_pair"]=implementation.find("_phase")!=std::string::npos;
  if(o3_candidate) meta["minimum_blocks_launch_bound"]=implementation.find("_bound2")!=std::string::npos?"2":"unspecified";
  if(o3_candidate) meta["group_loop"]="unrolled";
  if(o3_candidate&&implementation.find("_stream")!=std::string::npos)
    meta["weight_register_slice_n"]=implementation.find("_bound2")!=std::string::npos?16:32;
  if(o3_candidate) meta["packed_addressing"]=implementation.find("_bound2")!=std::string::npos?"guarded_u32_offset":"pointer_expression";
  meta["scale_storage"]=implementation=="baseline"?"global_per_fragment":"shared_per_cta_column_group";
  meta["smem_swizzle"]=implementation!="baseline";
  meta["whole_scale_panel_cached"]=implementation.find("_cached")!=std::string::npos;
  meta["pipeline_stages"]=STAGES;meta["threads"]=implementation.find("16w")!=std::string::npos?512:THREADS;
  meta["data_movement"]="cp.async";meta["scheduling"]="warp_cooperative";
  meta["partial_storage"]="register";meta["output_dtype"]="fp32";
  meta["group_size"]=g;
  meta["kernel_symbol"]=implementation=="baseline"?"adangel_sm80_grouped_gemm":"adangel_sm80_o1_swizzled";
  if(o3_candidate) meta["kernel_symbol"]="adangel_sm80_o3_swizzled";
  if(o3_candidate&&implementation.find("_bound2")!=std::string::npos) meta["kernel_symbol"]="adangel_sm80_o3_swizzled_bound2";
  meta["mma"]=split?"m16n8k64.u4.s4 + m16n8k64.s4.s4":"m16n8k32.s8.s8";
  py::dict r;r["output"]=out;r["timings_ms"]=timings;r["kernel"]=meta;
  r["converted_weight"]=wa;r["converted_activation"]=aa;
  return r;
}
} // namespace
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("benchmark",&benchmark,py::arg("variant"),py::arg("mode"),py::arg("a"),py::arg("a_scale"),py::arg("w"),py::arg("w_scale"),py::arg("warmup")=50,py::arg("repeats")=200,py::arg("inner")=100,py::arg("implementation")="production");
  m.def("benchmark_o0",&adangel_benchmark_o0);
}
