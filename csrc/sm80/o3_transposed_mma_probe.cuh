// v124: algebraic operand transpose, not an input/output transpose kernel.
// The physical output CTA remains Y[64,128]. MMA computes W[128,K]*A[64,K]^T.
// Reuse the exact v89 storage, grouped CTA mapping, async prefetch and guard.
#pragma once
namespace o3_transposed_mma_experiment {
using Old=o3_grouped_cta_experiment::C;
using Storage=o3_grouped_cta_experiment::Storage;
using LA=cute::MMA_Atom<cute::SM80_16x8x64_S32S4U4S32_TN>;
using HA=cute::MMA_Atom<cute::SM80_16x8x64_S32S4S4S32_TN>;
using Warps=cute::Layout<cute::Shape<cute::_2,cute::_2,cute::_1>>;
using Mma=cute::TiledMMA<LA,Warps,cute::Tile<cute::_128,cute::_64,cute::_64>>;
using HighMma=cute::TiledMMA<HA,Warps,cute::Tile<cute::_128,cute::_64,cute::_64>>;
using SliceMma=cute::TiledMMA<LA,Warps,cute::Tile<cute::_64,cute::_64,cute::_64>>;

__device__ __forceinline__ void body(const uint8_t* a,const uint8_t* w,
    const float* as,const uint8_t* metadata,float* y,int m,int n,int k) {
  extern __shared__ __align__(128) uint8_t buf[];
  auto& s=*reinterpret_cast<Storage*>(buf);
  Mma mma;HighMma hm;SliceMma sm;
  auto thr=mma.get_slice(threadIdx.x);auto ht=hm.get_slice(threadIdx.x);
  auto wt=sm.get_slice(threadIdx.x);
  auto coords=thr.partition_C(cute::make_identity_tensor(
      cute::make_shape(cute::_128{},cute::_64{})));
  auto acc=cute::make_fragment_like<int>(thr.make_fragment_C(coords));
  cute::clear(acc);
  auto low=[&](int slot) {return cute::make_tensor(
      cute::make_smem_ptr<cutlass::uint4b_t>(static_cast<void*>(s.low[slot])),
      Old::NibbleLayout<64>{});};
  auto high=[&](int slot) {return cute::make_tensor(
      cute::make_smem_ptr<cutlass::int4b_t>(static_cast<void*>(s.high[slot])),
      Old::NibbleLayout<64>{});};
  auto weight=[&](int slot) {return cute::make_tensor(
      cute::make_smem_ptr<cutlass::int4b_t>(static_cast<void*>(s.weight[slot])),
      Old::NibbleLayout<128>{});};
  auto tile_b=[&](auto t,auto half) {return cute::local_tile(t,
      cute::make_shape(cute::_64{},cute::_64{}),cute::make_coord(cute::_0{},half));};
  auto tile_w=[&](int slot,auto mb,auto half) {return cute::local_tile(weight(slot),
      cute::make_shape(cute::_64{},cute::_64{}),cute::make_coord(mb,half));};
  auto l0=thr.partition_fragment_B(tile_b(low(0),cute::_0{}));
  auto l1=cute::make_fragment_like(l0);
  auto h0=ht.partition_fragment_B(tile_b(high(0),cute::_0{}));
  auto h1=cute::make_fragment_like(h0);
  auto w0=wt.partition_fragment_A(tile_w(0,cute::_0{},cute::_0{}));
  auto w1=cute::make_fragment_like(w0);
  using LCopy=cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,cutlass::uint4b_t>;
  using SCopy=cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,cutlass::int4b_t>;
  auto lc=cute::make_tiled_copy_B(LCopy{},mma).get_slice(threadIdx.x);
  auto hc=cute::make_tiled_copy_B(SCopy{},hm).get_slice(threadIdx.x);
  auto wc=cute::make_tiled_copy_A(SCopy{},sm).get_slice(threadIdx.x);
  auto ld0=lc.retile_D(l0),ld1=lc.retile_D(l1);
  auto hd0=hc.retile_D(h0),hd1=hc.retile_D(h1);
  auto wd0=wc.retile_D(w0),wd1=wc.retile_D(w1);
  static_assert(decltype(cute::size(acc))::value==64);
  static_assert(decltype(cute::size<1>(acc))::value==4);
  static_assert(decltype(cute::size<2>(acc))::value==4);
  static_assert(decltype(cute::size<1>(w0))::value==2);
  static_assert(decltype(cute::size<1>(l0))::value==4);
  auto prefetch=[&](int slot,int group) {
    o3_grouped_cta_experiment::prefetch(s,slot,group,a,w,metadata,m,n,k);
  };
  prefetch(0,0);prefetch(1,1);
  for(int group=0;group<32;++group) {
    const int slot=group%3;
    if(group+2<32) asm volatile("cp.async.wait_group 1;" ::: "memory");
    else asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    if(group+2<32) prefetch((group+2)%3,group+2);
    // Activation B fragments are retained; W streams through two M64 slices.
    cute::copy(LCopy{},lc.partition_S(tile_b(low(slot),cute::_0{})),ld0);
    cute::copy(LCopy{},lc.partition_S(tile_b(low(slot),cute::_1{})),ld1);
    cute::copy(SCopy{},hc.partition_S(tile_b(high(slot),cute::_0{})),hd0);
    cute::copy(SCopy{},hc.partition_S(tile_b(high(slot),cute::_1{})),hd1);
    o1_static_for<0,2>([&](auto mb) {
      cute::copy(SCopy{},wc.partition_S(tile_w(slot,mb,cute::_0{})),wd0);
      cute::copy(SCopy{},wc.partition_S(tile_w(slot,mb,cute::_1{})),wd1);
      auto partial=cute::make_tensor<int>(
          cute::make_shape(cute::_4{},cute::_2{},cute::_4{}));
      cute::clear(partial);
      o1_static_for<0,2>([&](auto mi) {o1_static_for<0,4>([&](auto ni) {
        auto p=partial(cute::_,mi,ni);
        cute::gemm(HA{},p,w0(cute::_,mi,cute::_0{}),h0(cute::_,ni,cute::_0{}),p);
      });});
      o1_static_for<0,2>([&](auto mi) {o1_static_for<0,4>([&](auto ni) {
        auto p=partial(cute::_,mi,ni);
        cute::gemm(HA{},p,w1(cute::_,mi,cute::_0{}),h1(cute::_,ni,cute::_0{}),p);
      });});
      o1_static_for<0,32>([&](auto i) {partial(i)*=16;});
      o1_static_for<0,2>([&](auto mi) {o1_static_for<0,4>([&](auto ni) {
        auto p=partial(cute::_,mi,ni);
        cute::gemm(LA{},p,w0(cute::_,mi,cute::_0{}),l0(cute::_,ni,cute::_0{}),p);
      });});
      o1_static_for<0,2>([&](auto mi) {o1_static_for<0,4>([&](auto ni) {
        auto p=partial(cute::_,mi,ni);
        cute::gemm(LA{},p,w1(cute::_,mi,cute::_0{}),l1(cute::_,ni,cute::_0{}),p);
      });});
      o1_static_for<0,2>([&](auto mi) {o1_static_for<0,4>([&](auto ni) {
        auto full_mi=mb*cute::_2{}+mi;
        o1_static_for<0,4>([&](auto vi) {
          const auto c=coords(vi,full_mi,ni);
          // Original output column is the transposed MMA row (not column).
          acc(vi,full_mi,ni)+=partial(vi,mi,ni)*s.factor[slot][cute::get<0>(c)];
        });
      });});
    });
  }
  const auto tile=roof_grouped_cta::tile();
  o1_static_for<0,64>([&](auto i) {
    const auto c=coords(i);
    const unsigned col=tile.x*128+cute::get<0>(c),row=tile.y*64+cute::get<1>(c);
    const auto code=reinterpret_cast<const uint32_t*>(metadata)[32*n+col];
    acc(i)=__float_as_int(__fmul_rn(__fmul_rn(float(acc(i)),
        o3_grouped_cta_experiment::decode(code)),as[row]));
  });
  // No temporary Y^T: preserve the original output shape, one store per element.
  o1_static_for<0,64>([&](auto i) {
    const auto c=coords(i);
    y[(tile.y*64+cute::get<1>(c))*n+tile.x*128+cute::get<0>(c)]=__int_as_float(acc(i));
  });
}
} // namespace o3_transposed_mma_experiment
