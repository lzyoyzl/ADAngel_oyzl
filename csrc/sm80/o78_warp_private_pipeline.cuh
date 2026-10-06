// v109 isolated warp-private in-place G128 pipeline. Not production dispatch.
// Duplicate read-only input panels to remove cross-warp reuse/synchronization.
#pragma once
namespace o78_warp_private_experiment {
using LA=cute::MMA_Atom<cute::SM80_16x8x64_S32U4S4S32_TN>;
using HA=cute::MMA_Atom<cute::SM80_16x8x64_S32S4S4S32_TN>;
using Mma=cute::TiledMMA<LA,cute::Layout<cute::Shape<cute::_1,cute::_1,cute::_1>>,
    cute::Tile<cute::_32,cute::_64,cute::_64>>;
using HighMma=cute::TiledMMA<HA,cute::Layout<cute::Shape<cute::_1,cute::_1,cute::_1>>,
    cute::Tile<cute::_32,cute::_64,cute::_64>>;
using SliceMma=cute::TiledMMA<LA,cute::Layout<cute::Shape<cute::_1,cute::_1,cute::_1>>,
    cute::Tile<cute::_32,cute::_32,cute::_64>>;
template<int Rows> using ByteLayout=decltype(cute::composition(cute::Swizzle<2,4,3>{},
    cute::Layout<cute::Shape<cute::Int<Rows>,cute::_64>,cute::Stride<cute::_64,cute::_1>>{}));
template<int Rows> using NibbleLayout=decltype(cute::composition(cute::Swizzle<2,5,3>{},
    cute::Layout<cute::Shape<cute::Int<Rows>,cute::_128>,cute::Stride<cute::_128,cute::_1>>{}));
struct alignas(128) WarpStorage {
  alignas(128) uint8_t low[32*64],high[32*64],weight[64*64];
  int activation_factors[32],weight_factors[64];
};
struct alignas(128) Storage {WarpStorage warps[4];};
static_assert(sizeof(WarpStorage)==8576 && sizeof(Storage)==34304);

__device__ __forceinline__ void payload(WarpStorage& s,int group,unsigned lane,
    unsigned row_base,unsigned col_base,const uint8_t* a,const uint8_t* w,
    unsigned m,unsigned n,unsigned k) {
  ByteLayout<32> la;ByteLayout<64> lb;
  o1_static_for<0,4>([&](auto chunk) {
    unsigned off=lane*16+chunk*512,row=off/64,col=off%64;
    const auto* src=a+group*m*64+(row_base+row)*64+col;
    copy16(s.low+la(row,col),src);
    copy16(s.high+la(row,col),src+m*(k/2));
  });
  o1_static_for<0,8>([&](auto chunk) {
    unsigned off=lane*16+chunk*512,row=off/64,col=off%64;
    copy16(s.weight+lb(row,col),w+group*n*64+(col_base+row)*64+col);
  });
}

__device__ __forceinline__ void factors(WarpStorage& s,int group,unsigned lane,
    unsigned row_base,unsigned col_base,const int32_t* af,const int32_t* wf,
    unsigned m,unsigned n) {
  if(lane<8)copy16(s.activation_factors+lane*4,af+group*m+row_base+lane*4);
  if(lane<16)copy16(s.weight_factors+lane*4,wf+group*n+col_base+lane*4);
}

__device__ __forceinline__ void body(const uint8_t* a,const uint8_t* w,
    const int32_t* af,const int32_t* wf,const float* base_a,const float* base_w,
    float* y,unsigned m,unsigned n,unsigned k) {
  extern __shared__ __align__(128) uint8_t buf[];
  unsigned warp=threadIdx.x/32,lane=threadIdx.x%32;
  auto& s=reinterpret_cast<Storage*>(buf)->warps[warp];
  unsigned row_base=blockIdx.y*64+(warp%2)*32;
  unsigned col_base=blockIdx.x*128+(warp/2)*64;
  Mma mma;HighMma high_mma;SliceMma slice_mma;
  auto thr=mma.get_slice(lane);auto ht=high_mma.get_slice(lane);
  auto st=slice_mma.get_slice(lane);
  auto coords=thr.partition_C(cute::make_identity_tensor(cute::make_shape(cute::_32{},cute::_64{})));
  auto acc=cute::make_fragment_like<int>(thr.make_fragment_C(coords));cute::clear(acc);
  auto low=cute::make_tensor(cute::make_smem_ptr<cutlass::uint4b_t>(static_cast<void*>(s.low)),NibbleLayout<32>{});
  auto high=cute::make_tensor(cute::make_smem_ptr<cutlass::int4b_t>(static_cast<void*>(s.high)),NibbleLayout<32>{});
  auto weight=cute::make_tensor(cute::make_smem_ptr<cutlass::int4b_t>(static_cast<void*>(s.weight)),NibbleLayout<64>{});
  auto tile_a=[&](auto t,auto half) {return cute::local_tile(t,
      cute::make_shape(cute::_32{},cute::_64{}),cute::make_coord(cute::_0{},half));};
  auto tile_b=[&](auto nb,auto half) {return cute::local_tile(weight,
      cute::make_shape(cute::_32{},cute::_64{}),cute::make_coord(nb,half));};
  auto a0=thr.partition_fragment_A(tile_a(low,cute::_0{}));auto a1=cute::make_fragment_like(a0);
  auto h0=ht.partition_fragment_A(tile_a(high,cute::_0{}));auto h1=cute::make_fragment_like(h0);
  auto b0=st.partition_fragment_B(tile_b(cute::_0{},cute::_0{}));auto b1=cute::make_fragment_like(b0);
  using LCopy=cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,cutlass::uint4b_t>;
  using SCopy=cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,cutlass::int4b_t>;
  auto lc=cute::make_tiled_copy_A(LCopy{},mma).get_slice(lane);
  auto hc=cute::make_tiled_copy_A(SCopy{},high_mma).get_slice(lane);
  auto bc=cute::make_tiled_copy_B(SCopy{},slice_mma).get_slice(lane);
  auto ld0=lc.retile_D(a0),ld1=lc.retile_D(a1);
  auto hd0=hc.retile_D(h0),hd1=hc.retile_D(h1);
  auto bd0=bc.retile_D(b0),bd1=bc.retile_D(b1);
  static_assert(decltype(cute::size(acc))::value==64);
  static_assert(decltype(cute::size<1>(acc))::value==2);
  static_assert(decltype(cute::size<1>(b0))::value==4);
  payload(s,0,lane,row_base,col_base,a,w,m,n,k);
  factors(s,0,lane,row_base,col_base,af,wf,m,n);
  asm volatile("cp.async.commit_group;" ::: "memory");
  for(int group=0;group<32;++group) {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncwarp(); // Per-thread waits become visible to this private warp.
    cute::copy(LCopy{},lc.partition_S(tile_a(low,cute::_0{})),ld0);
    cute::copy(SCopy{},hc.partition_S(tile_a(high,cute::_0{})),hd0);
    cute::copy(LCopy{},lc.partition_S(tile_a(low,cute::_1{})),ld1);
    cute::copy(SCopy{},hc.partition_S(tile_a(high,cute::_1{})),hd1);
    o1_static_for<0,2>([&](auto nb) {
      cute::copy(SCopy{},bc.partition_S(tile_b(nb,cute::_0{})),bd0);
      cute::copy(SCopy{},bc.partition_S(tile_b(nb,cute::_1{})),bd1);
      auto partial=cute::make_tensor<int>(cute::make_shape(cute::_4{},cute::_2{},cute::_4{}));
      cute::clear(partial);
      o1_static_for<0,2>([&](auto mi) {o1_static_for<0,4>([&](auto ni) {
        auto p=partial(cute::_,mi,ni);
        cute::gemm(HA{},p,h0(cute::_,mi,cute::_0{}),b0(cute::_,ni,cute::_0{}),p);
      });});
      o1_static_for<0,2>([&](auto mi) {o1_static_for<0,4>([&](auto ni) {
        auto p=partial(cute::_,mi,ni);
        cute::gemm(HA{},p,h1(cute::_,mi,cute::_0{}),b1(cute::_,ni,cute::_0{}),p);
      });});
      if constexpr(decltype(nb)::value==1) {
        if(group+1<32) {
          // High MMA consumed both final B halves: all payload reads are done.
          // No next-group factor writes yet; current factors still in use.
          __syncwarp();
          payload(s,group+1,lane,row_base,col_base,a,w,m,n,k);
        }
      }
      o1_static_for<0,32>([&](auto i) {partial(i)*=16;});
      o1_static_for<0,2>([&](auto mi) {o1_static_for<0,4>([&](auto ni) {
        auto p=partial(cute::_,mi,ni);
        cute::gemm(LA{},p,a0(cute::_,mi,cute::_0{}),b0(cute::_,ni,cute::_0{}),p);
      });});
      o1_static_for<0,2>([&](auto mi) {o1_static_for<0,4>([&](auto ni) {
        auto p=partial(cute::_,mi,ni);
        cute::gemm(LA{},p,a1(cute::_,mi,cute::_0{}),b1(cute::_,ni,cute::_0{}),p);
      });});
      o1_static_for<0,2>([&](auto mi) {o1_static_for<0,4>([&](auto ni) {
        auto full_ni=nb*cute::_4{}+ni;
        o1_static_for<0,4>([&](auto vi) {
          auto p=coords(vi,mi,full_ni);
          int coefficient=s.activation_factors[cute::get<0>(p)]*s.weight_factors[cute::get<1>(p)];
          acc(vi,mi,full_ni)+=partial(vi,mi,ni)*coefficient;
        });
      });});
    });
    if(group+1<32) {
      __syncwarp(); // Current factor readers finish before factor reuse.
      factors(s,group+1,lane,row_base,col_base,af,wf,m,n);
      asm volatile("cp.async.commit_group;" ::: "memory");
    }
  }
  o1_static_for<0,64>([&](auto i) {
    auto p=coords(i);
    float row=base_a[row_base+cute::get<0>(p)],col=base_w[col_base+cute::get<1>(p)];
    acc(i)=__float_as_int(__fmul_rn(__fmul_rn(float(acc(i)),row),col));
  });
  o1_static_for<0,32>([&](auto pair) {
    auto i=pair*cute::_2{};auto p=coords(i),q=coords(i+cute::_1{});
    unsigned offset=(row_base+cute::get<0>(p))*n+col_base+cute::get<1>(p);
    if(cute::get<0>(p)==cute::get<0>(q) && cute::get<1>(q)==cute::get<1>(p)+1 && (offset&1u)==0)
      *reinterpret_cast<float2*>(y+offset)=make_float2(__int_as_float(acc(i)),__int_as_float(acc(i+cute::_1{})));
    else {
      y[offset]=__int_as_float(acc(i));
      y[(row_base+cute::get<0>(q))*n+col_base+cute::get<1>(q)]=__int_as_float(acc(i+cute::_1{}));
    }
  });
}
} // namespace o78_warp_private_experiment
