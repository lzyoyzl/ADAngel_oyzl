// Candidates39/40: pair G128 products inside a narrow output fragment.
// This TU owns independent device IR; no production or prior kernel changes.
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cutlass/integer_subbyte.h>
#include <stdexcept>
#include "roof_pipeline_api.h"

namespace {
template<int I,int End,class F>
__device__ __forceinline__ void static_for(F const& f) {
  if constexpr(I<End) { f(cute::Int<I>{}); static_for<I+1,End>(f); }
}
__device__ __forceinline__ void copy16(void* dst,const void* src) {
  auto address=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
      "r"(address),"l"(src):"memory");
}
constexpr int M=64,N=128,K=128,Threads=128,Stages=3;
template<int Rows> using ByteLayout=decltype(cute::composition(
    cute::Swizzle<2,4,3>{},cute::Layout<cute::Shape<cute::Int<Rows>,cute::_64>,
    cute::Stride<cute::_64,cute::_1>>{}));
template<int Rows> using NibbleLayout=decltype(cute::composition(
    cute::Swizzle<2,5,3>{},cute::Layout<cute::Shape<cute::Int<Rows>,cute::_128>,
    cute::Stride<cute::_128,cute::_1>>{}));
template<bool Dual> struct ActivationScales {};
template<> struct ActivationScales<true> {alignas(16) float rows[Stages*M];};
template<bool Dual> struct alignas(128) Storage : ActivationScales<Dual> {
  alignas(128) uint8_t low[Stages][M*K/2],high[Stages][M*K/2],weight[Stages][N*K/2];
  float columns[Stages*N];
};
static_assert(sizeof(Storage<false>)==50688 && sizeof(Storage<true>)==51456);
using LowAtom=cute::MMA_Atom<cute::SM80_16x8x64_S32U4S4S32_TN>;
using HighAtom=cute::MMA_Atom<cute::SM80_16x8x64_S32S4S4S32_TN>;
using WarpLayout=cute::Layout<cute::Shape<cute::_2,cute::_2,cute::_1>>;
template<int Rows,int Columns,class Atom>
using TiledMma=cute::TiledMMA<Atom,WarpLayout,
    cute::Tile<cute::Int<Rows>,cute::Int<Columns>,cute::_64>>;

template<bool Dual,bool Fast>
__device__ __forceinline__ void prefetch(Storage<Dual>& s,int group,
    const uint8_t* a,const uint8_t* w,const float* as,const uint8_t* ws,int m,int n,int k) {
  static_assert(!Dual || !Fast);
  const int slot=group%Stages;
  ByteLayout<M> la;ByteLayout<N> lb;
  static_for<0,M*K/2/(Threads*16)>([&](auto chunk) {
    unsigned off=threadIdx.x*16+chunk*Threads*16,row=off/64,col=off%64;
    auto src=a+(blockIdx.y*M+row)*(k/2)+group*64+col;
    copy16(s.low[slot]+la(row,col),src);
    copy16(s.high[slot]+la(row,col),src+m*(k/2));
  });
  static_for<0,N*K/2/(Threads*16)>([&](auto chunk) {
    unsigned off=threadIdx.x*16+chunk*Threads*16,row=off/64,col=off%64;
    copy16(s.weight[slot]+lb(row,col),w+(blockIdx.x*N+row)*(k/2)+group*64+col);
  });
  if constexpr(Dual) {
    if(threadIdx.x<M) s.rows[slot*M+threadIdx.x]=as[group*m+blockIdx.y*M+threadIdx.x];
    if(threadIdx.x<N) s.columns[slot*N+threadIdx.x]=
        reinterpret_cast<const float*>(ws)[group*n+blockIdx.x*N+threadIdx.x];
  } else if(threadIdx.x<N) {
    uint32_t code=ws[(blockIdx.x*N+threadIdx.x)*(k/128)+group];
    uint32_t bits=Fast?((code-127u)<<23):(code?code<<23:0x00400000u);
    s.columns[slot*N+threadIdx.x]=__uint_as_float(bits);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template<bool Dual,bool Fast,int Tune>
__global__ __launch_bounds__(128,3)
void adangel_sm80_roof_candidate(const uint8_t* a,const uint8_t* w,const float* as,
    const uint8_t* ws,float* y,int m,int n,int k) {
  static_assert(Tune==39 || Tune==40);
  static_assert(!Dual || !Fast);
  constexpr int SliceN=Tune==39?64:32;
  extern __shared__ __align__(128) uint8_t buf[];
  auto& s=*reinterpret_cast<Storage<Dual>*>(buf);
  TiledMma<M,N,LowAtom> full_mma;
  auto thr=full_mma.get_slice(threadIdx.x);
  auto coords=thr.partition_C(cute::make_identity_tensor(cute::make_shape(cute::_64{},cute::_128{})));
  auto shape_fragment=thr.make_fragment_C(coords);
  auto acc=cute::make_fragment_like<float>(shape_fragment);
  cute::clear(acc);
  static_assert(decltype(cute::size(acc))::value==64);
  static_assert(decltype(cute::size<1>(acc))::value==2);
  auto fixed_rows=cute::make_tensor<float>(cute::make_shape(cute::_4{},cute::_2{}));
  if constexpr(!Dual) static_for<0,2>([&](auto mi) {
    static_for<0,4>([&](auto vi) {
      fixed_rows(vi,mi)=as[blockIdx.y*M+cute::get<0>(coords(vi,mi,cute::_0{}))];
    });
  });
  auto make_low=[&](int slot) {return cute::make_tensor(cute::make_smem_ptr<cutlass::uint4b_t>(
      static_cast<void*>(s.low[slot])),NibbleLayout<M>{});};
  auto make_high=[&](int slot) {return cute::make_tensor(cute::make_smem_ptr<cutlass::int4b_t>(
      static_cast<void*>(s.high[slot])),NibbleLayout<M>{});};
  auto make_weight=[&](int slot) {return cute::make_tensor(cute::make_smem_ptr<cutlass::int4b_t>(
      static_cast<void*>(s.weight[slot])),NibbleLayout<N>{});};
  auto a_tile=[&](auto tensor,auto mi,auto half) {return cute::local_tile(tensor,
      cute::make_shape(cute::_32{},cute::_64{}),cute::make_coord(mi,half));};
  auto b_tile=[&](int slot,auto nb,auto half) {return cute::local_tile(make_weight(slot),
      cute::make_shape(cute::Int<SliceN>{},cute::_64{}),cute::make_coord(nb,half));};
  TiledMma<32,SliceN,LowAtom> small_mma;
  TiledMma<32,SliceN,HighAtom> high_mma;
  auto small_thr=small_mma.get_slice(threadIdx.x);
  auto high_thr=high_mma.get_slice(threadIdx.x);
  auto ra00=small_thr.partition_fragment_A(a_tile(make_low(0),cute::_0{},cute::_0{}));
  auto rh00=high_thr.partition_fragment_A(a_tile(make_high(0),cute::_0{},cute::_0{}));
  auto ra01=cute::make_fragment_like(ra00);auto rh01=cute::make_fragment_like(rh00);
  auto ra10=cute::make_fragment_like(ra00);auto rh10=cute::make_fragment_like(rh00);
  auto ra11=cute::make_fragment_like(ra00);auto rh11=cute::make_fragment_like(rh00);
  auto br0=small_thr.partition_fragment_B(b_tile(0,cute::_0{},cute::_0{}));
  auto br1=cute::make_fragment_like(br0);
  constexpr int NAtoms=decltype(cute::size<1>(br0))::value;
  static_assert(NAtoms==SliceN/16);
  using LCopy=cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,cutlass::uint4b_t>;
  using SCopy=cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,cutlass::int4b_t>;
  auto lc=cute::make_tiled_copy_A(LCopy{},small_mma).get_slice(threadIdx.x);
  auto hc=cute::make_tiled_copy_A(SCopy{},high_mma).get_slice(threadIdx.x);
  auto bc=cute::make_tiled_copy_B(SCopy{},small_mma).get_slice(threadIdx.x);
  auto load_a=[&](int slot,auto mi,auto half,auto& ra,auto& rh) {
    auto ld=lc.retile_D(ra);auto hd=hc.retile_D(rh);
    cute::copy(LCopy{},lc.partition_S(a_tile(make_low(slot),mi,half)),ld);
    cute::copy(SCopy{},hc.partition_S(a_tile(make_high(slot),mi,half)),hd);
  };
  const int groups=k/128;
  prefetch<Dual,Fast>(s,0,a,w,as,ws,m,n,k);
  if(groups>1) prefetch<Dual,Fast>(s,1,a,w,as,ws,m,n,k);
  for(int first=0;first<groups;first+=2) {
    const int slot0=first%Stages,slot1=(first+1)%Stages;
    const bool has_second=first+1<groups;
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    // Only the third slot is free. Load next pair's first G128 during compute.
    if(first+2<groups) prefetch<Dual,Fast>(s,first+2,a,w,as,ws,m,n,k);
    static_for<0,2>([&](auto mi) {
      // Two groups for one M atom use the same A-register volume as one
      // group for both M atoms. Reuse these A fragments over every N slice.
      load_a(slot0,mi,cute::_0{},ra00,rh00);load_a(slot0,mi,cute::_1{},ra01,rh01);
      if(has_second) {
        load_a(slot1,mi,cute::_0{},ra10,rh10);load_a(slot1,mi,cute::_1{},ra11,rh11);
      }
      static_for<0,N/SliceN>([&](auto nb) {
        // Only 8 or16 leaf values, not64 for the whole per-thread output.
        auto leaf=cute::make_tensor<float>(cute::make_shape(cute::_4{},cute::Int<NAtoms>{}));
        auto process=[&](auto phase,auto& ra0,auto& rh0,auto& ra1,auto& rh1) {
          const int slot=decltype(phase)::value==0?slot0:slot1;
          auto bd0=bc.retile_D(br0),bd1=bc.retile_D(br1);
          cute::copy(SCopy{},bc.partition_S(b_tile(slot,nb,cute::_0{})),bd0);
          cute::copy(SCopy{},bc.partition_S(b_tile(slot,nb,cute::_1{})),bd1);
          auto pl=cute::make_tensor<int>(cute::make_shape(cute::_4{},cute::Int<NAtoms>{}));
          auto ph=cute::make_tensor<int>(cute::make_shape(cute::_4{},cute::Int<NAtoms>{}));
          cute::clear(pl);cute::clear(ph);
          static_for<0,NAtoms>([&](auto ni) {
            auto l=pl(cute::_,ni),h=ph(cute::_,ni);
            cute::gemm(LowAtom{},l,ra0(cute::_,cute::_0{},cute::_0{}),br0(cute::_,ni,cute::_0{}),l);
            cute::gemm(HighAtom{},h,rh0(cute::_,cute::_0{},cute::_0{}),br0(cute::_,ni,cute::_0{}),h);
          });
          static_for<0,NAtoms>([&](auto ni) {
            auto l=pl(cute::_,ni),h=ph(cute::_,ni);
            cute::gemm(LowAtom{},l,ra1(cute::_,cute::_0{},cute::_0{}),br1(cute::_,ni,cute::_0{}),l);
            cute::gemm(HighAtom{},h,rh1(cute::_,cute::_0{},cute::_0{}),br1(cute::_,ni,cute::_0{}),h);
          });
          static_for<0,NAtoms>([&](auto ni) {
            auto full_ni=nb*cute::Int<NAtoms>{}+ni;
            static_for<0,4>([&](auto vi) {
              const int partial=pl(vi,ni)+16*ph(vi,ni);
              auto coord=coords(vi,mi,full_ni);
              const float column=s.columns[slot*N+cute::get<1>(coord)];
              float row,scale;
              if constexpr(Dual) row=s.rows[slot*M+cute::get<0>(coord)];
              else row=fixed_rows(vi,mi);
              if constexpr(Fast) scale=__uint_as_float(__float_as_uint(row)+__float_as_uint(column));
              else scale=__fmul_rn(row,column);
              const float product=__fmul_rn(float(partial),scale);
              if constexpr(decltype(phase)::value==0) leaf(vi,ni)=product;
              else acc(vi,mi,full_ni)=__fadd_rn(acc(vi,mi,full_ni),__fadd_rn(leaf(vi,ni),product));
            });
          });
        };
        process(cute::_0{},ra00,rh00,ra01,rh01);
        if(has_second) process(cute::_1{},ra10,rh10,ra11,rh11);
        else static_for<0,NAtoms>([&](auto ni) {
          static_for<0,4>([&](auto vi) {
            auto full_ni=nb*cute::Int<NAtoms>{}+ni;
            acc(vi,mi,full_ni)=__fadd_rn(acc(vi,mi,full_ni),leaf(vi,ni));
          });
        });
      });
    });
    // Both consumed slots must be released by every warp before one is reused.
    __syncthreads();
    if(first+3<groups) prefetch<Dual,Fast>(s,first+3,a,w,as,ws,m,n,k);
  }
  static_for<0,32>([&](auto pair) {
    auto i=pair*cute::_2{};
    auto p=coords(i),q=coords(i+cute::_1{});
    int offset=(blockIdx.y*M+cute::get<0>(p))*n+blockIdx.x*N+cute::get<1>(p);
    if(cute::get<0>(p)==cute::get<0>(q) && cute::get<1>(q)==cute::get<1>(p)+1 && (offset&1)==0)
      *reinterpret_cast<float2*>(y+offset)=make_float2(acc(i),acc(i+cute::_1{}));
    else {
      y[offset]=acc(i);
      y[(blockIdx.y*M+cute::get<0>(q))*n+blockIdx.x*N+cute::get<1>(q)]=acc(i+cute::_1{});
    }
  });
}
} // namespace

namespace adangel_sm80_experiment {
Kernel select_fragment_tree_kernel(bool dual,bool fast,int tune) {
  if((tune!=39 && tune!=40) || (dual && fast)) throw std::invalid_argument("expected fragment-tree candidate39/40");
  if(tune==39) return dual?adangel_sm80_roof_candidate<true,false,39>:
      (fast?adangel_sm80_roof_candidate<false,true,39>:adangel_sm80_roof_candidate<false,false,39>);
  return dual?adangel_sm80_roof_candidate<true,false,40>:
      (fast?adangel_sm80_roof_candidate<false,true,40>:adangel_sm80_roof_candidate<false,false,40>);
}
size_t fragment_tree_shared_bytes(bool dual) {return dual?sizeof(Storage<true>):sizeof(Storage<false>);}
} // namespace adangel_sm80_experiment
