// Generated v78; same guard/quantization/epilogue, different MMA schedule.
// Isolated O7/O8 full-K candidate. No production binding or default change.
// Host proves coefficient/product/prefix INT32 bounds before selecting this.
#pragma once
namespace o78_output_streaming_experiment {
using C=O78::O3AmpereConfig<64,128,128,false,2,true,2>;
struct alignas(128) Storage {
  alignas(16) int activation_factors[2][64];
  alignas(128) uint8_t low[2][64*64],high[2][64*64],weight[2][128*64];
  int weight_factors[2][128];
};
static_assert(sizeof(Storage)==34304);
static_assert(sizeof(Storage)==sizeof(C::Storage));

__device__ __forceinline__ void prefetch(Storage& s,int slot,int group,
    const uint8_t* a,const uint8_t* w,const int32_t* af,const int32_t* wf,
    uint32_t m,uint32_t n,uint32_t k) {
  C::ByteLayout<64> la;C::ByteLayout<128> lb;
  o1_static_for<0,2>([&](auto chunk) {
    unsigned off=threadIdx.x*16+chunk*2048,row=off/64,col=off%64;
    const auto* src=a+group*m*64+(blockIdx.y*64+row)*64+col;
    copy16(s.low[slot]+la(row,col),src);
    copy16(s.high[slot]+la(row,col),src+m*(k/2));
  });
  o1_static_for<0,4>([&](auto chunk) {
    unsigned off=threadIdx.x*16+chunk*2048,row=off/64,col=off%64;
    copy16(s.weight[slot]+lb(row,col),w+group*n*64+(blockIdx.x*128+row)*64+col);
  });
  // Factors share the old scale-panel layout, commit/wait and CTA barrier.
  const unsigned first=threadIdx.x*4;
  if(threadIdx.x<16) copy16(s.activation_factors[slot]+first,
      af+group*m+blockIdx.y*64+first);
  if(threadIdx.x<32) copy16(s.weight_factors[slot]+first,
      wf+group*n+blockIdx.x*128+first);
  asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ void body(const uint8_t* a,const uint8_t* w,
    const int32_t* af,const int32_t* wf,const float* base_a,const float* base_w,
    float* y,uint32_t m,uint32_t n,uint32_t k) {
  // Host guards K4096, M%64==0, N%128==0 and finite representable base scales.
  // Bases combine dyadic anchors with the existing tensor/fixed-point scale.
  extern __shared__ __align__(128) uint8_t buf[];
  auto& s=*reinterpret_cast<Storage*>(buf);
  C::Mma mma;C::HighMma high_mma;
  auto thr=mma.get_slice(threadIdx.x);auto ht=high_mma.get_slice(threadIdx.x);
  auto coords=thr.partition_C(cute::make_identity_tensor(cute::make_shape(cute::_64{},cute::_128{})));
  auto acc=cute::make_fragment_like<int>(thr.make_fragment_C(coords));cute::clear(acc);
  auto low=[&](int slot) {return cute::make_tensor(cute::make_smem_ptr<cutlass::uint4b_t>(
      static_cast<void*>(s.low[slot])),C::NibbleLayout<64>{});};
  auto high=[&](int slot) {return cute::make_tensor(cute::make_smem_ptr<cutlass::int4b_t>(
      static_cast<void*>(s.high[slot])),C::NibbleLayout<64>{});};
  auto weight=[&](int slot) {return cute::make_tensor(cute::make_smem_ptr<cutlass::int4b_t>(
      static_cast<void*>(s.weight[slot])),C::NibbleLayout<128>{});};
  auto tile_a=[&](auto t,auto half) {return cute::local_tile(t,
      cute::make_shape(cute::_64{},cute::_64{}),cute::make_coord(cute::_0{},half));};
  auto a0=thr.partition_fragment_A(tile_a(low(0),cute::_0{}));auto a1=cute::make_fragment_like(a0);
  auto h0=ht.partition_fragment_A(tile_a(high(0),cute::_0{}));auto h1=cute::make_fragment_like(h0);
  using LCopy=cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,cutlass::uint4b_t>;
  using SCopy=cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,cutlass::int4b_t>;
  auto lc=cute::make_tiled_copy_A(LCopy{},mma).get_slice(threadIdx.x);
  auto hc=cute::make_tiled_copy_A(SCopy{},high_mma).get_slice(threadIdx.x);
  auto ld0=lc.retile_D(a0),ld1=lc.retile_D(a1);
  auto hd0=hc.retile_D(h0),hd1=hc.retile_D(h1);
  using SliceMma=O78::O3AmpereConfig<64,64,128,false,2>::Mma;
  SliceMma slice_mma;auto st=slice_mma.get_slice(threadIdx.x);
  auto tile_b=[&](int slot,auto nb,auto half) {return cute::local_tile(weight(slot),
      cute::make_shape(cute::_64{},cute::_64{}),cute::make_coord(nb,half));};
  auto b0=st.partition_fragment_B(tile_b(0,cute::_0{},cute::_0{}));auto b1=cute::make_fragment_like(b0);
  auto bc=cute::make_tiled_copy_B(SCopy{},slice_mma).get_slice(threadIdx.x);
  auto bd0=bc.retile_D(b0),bd1=bc.retile_D(b1);
  static_assert(decltype(cute::size<1>(b0))::value==4);
  static_assert(decltype(cute::size<1>(acc))::value==2);
  static_assert(decltype(cute::size(acc))::value==64);
  using LA=cute::MMA_Atom<cute::SM80_16x8x64_S32U4S4S32_TN>;
  using HA=cute::MMA_Atom<cute::SM80_16x8x64_S32S4S4S32_TN>;
  constexpr int Groups=32;
  prefetch(s,0,0,a,w,af,wf,m,n,k);
  for(int group=0;group<Groups;++group) {
    const int slot=group%2;
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads(); // All prior readers finish before a slot is reused.
    if(group+1<Groups) prefetch(s,1-slot,group+1,a,w,af,wf,m,n,k);
    cute::copy(LCopy{},lc.partition_S(tile_a(low(slot),cute::_0{})),ld0);
    cute::copy(SCopy{},hc.partition_S(tile_a(high(slot),cute::_0{})),hd0);
    cute::copy(LCopy{},lc.partition_S(tile_a(low(slot),cute::_1{})),ld1);
    cute::copy(SCopy{},hc.partition_S(tile_a(high(slot),cute::_1{})),hd1);
    // Preserve N64 operand reuse; merge eight M/N MMA chains below.
    o1_static_for<0,2>([&](auto nb) {
      cute::copy(SCopy{},bc.partition_S(tile_b(slot,nb,cute::_0{})),bd0);
      cute::copy(SCopy{},bc.partition_S(tile_b(slot,nb,cute::_1{})),bd1);
      // Eight independent chains: two M atoms x four N atoms.
      // Same32 logical partial registers; retain all original A/B reuse.
      auto partial=cute::make_tensor<int>(
          cute::make_shape(cute::_4{},cute::_2{},cute::_4{}));
      cute::clear(partial);
      o1_static_for<0,2>([&](auto mi) {
        o1_static_for<0,4>([&](auto ni) {
          auto p=partial(cute::_,mi,ni);
          cute::gemm(HA{},p,h0(cute::_,mi,cute::_0{}),b0(cute::_,ni,cute::_0{}),p);
        });
      });
      o1_static_for<0,2>([&](auto mi) {
        o1_static_for<0,4>([&](auto ni) {
          auto p=partial(cute::_,mi,ni);
          cute::gemm(HA{},p,h1(cute::_,mi,cute::_0{}),b1(cute::_,ni,cute::_0{}),p);
        });
      });
      // A G128 signed-high dot is at most8192 in magnitude. Multiplication,
      // not signed left-shift, is defined for every intermediate here.
      o1_static_for<0,32>([&](auto i) { partial(i)*=16; });
      o1_static_for<0,2>([&](auto mi) {
        o1_static_for<0,4>([&](auto ni) {
          auto p=partial(cute::_,mi,ni);
          cute::gemm(LA{},p,a0(cute::_,mi,cute::_0{}),b0(cute::_,ni,cute::_0{}),p);
        });
      });
      o1_static_for<0,2>([&](auto mi) {
        o1_static_for<0,4>([&](auto ni) {
          auto p=partial(cute::_,mi,ni);
          cute::gemm(LA{},p,a1(cute::_,mi,cute::_0{}),b1(cute::_,ni,cute::_0{}),p);
        });
      });
      o1_static_for<0,2>([&](auto mi) {
        o1_static_for<0,4>([&](auto ni) {
          auto full_ni=nb*cute::_4{}+ni;
          o1_static_for<0,4>([&](auto vi) {
            const auto coord=coords(vi,mi,full_ni);
            const int coefficient=s.activation_factors[slot][cute::get<0>(coord)]*
                                  s.weight_factors[slot][cute::get<1>(coord)];
            acc(vi,mi,full_ni)+=partial(vi,mi,ni)*coefficient;
          });
        });
      });
    });
  }
  // Reuse the 64 INT32 register slots as FP32 bits before any output store.
  o1_static_for<0,decltype(cute::size(acc))::value>([&](auto i) {
    const auto p=coords(i);
    const float row=base_a[blockIdx.y*64+cute::get<0>(p)];
    const float column=base_w[blockIdx.x*128+cute::get<1>(p)];
    acc(i)=__float_as_int(__fmul_rn(__fmul_rn(float(acc(i)),row),column));
  });
  o1_static_for<0,decltype(cute::size(acc))::value/2>([&](auto pair) {
    auto i=pair*cute::_2{};const auto p=coords(i),q=coords(i+cute::_1{});
    uint32_t offset=(blockIdx.y*64+cute::get<0>(p))*n+blockIdx.x*128+cute::get<1>(p);
    if(cute::get<0>(p)==cute::get<0>(q) && cute::get<1>(q)==cute::get<1>(p)+1 && (offset&1u)==0)
      __stcs(reinterpret_cast<float2*>(y+offset),make_float2(__int_as_float(acc(i)),__int_as_float(acc(i+cute::_1{}))));
    else {
      __stcs(y+offset,__int_as_float(acc(i)));
      __stcs(y+((blockIdx.y*64+cute::get<0>(q))*n+blockIdx.x*128+cute::get<1>(q)),__int_as_float(acc(i+cute::_1{})));
    }
  });
}
} // namespace o78_output_streaming_experiment
