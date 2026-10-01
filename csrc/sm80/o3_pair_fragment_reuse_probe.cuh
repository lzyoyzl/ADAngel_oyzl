// v52 paired path: reuse B, or A+B fragments across the two groups.
// Host guard proves every adjacent exponent difference <=13.
// Unsafe inputs dispatch to the unchanged v50 general kernel. Never production dispatch.
#pragma once
namespace o3_pair_fragment_reuse_experiment {
using C=O3::O3AmpereConfig<64,128,128,false,2,false,3>;
struct alignas(128) Storage {
  alignas(128) uint8_t low[4][64*64],high[4][64*64],weight[4][128*64];
  int factor0[2][128],factor1[2][128];
  float anchor_scale[2][128];
};
static_assert(sizeof(Storage)==68608);

__device__ __forceinline__ float decode(uint32_t code) {
  return __uint_as_float(code?code<<23:0x00400000u);
}

template<bool First>
__device__ __forceinline__ void prefetch(Storage& s,int slot,int group,
    const uint8_t* a,const uint8_t* w,const uint8_t* ws,int m,int n,int k) {
  C::ByteLayout<64> la;C::ByteLayout<128> lb;
  o1_static_for<0,2>([&](auto chunk) {
    unsigned off=threadIdx.x*16+chunk*2048;
    unsigned row=off/64,col=off%64;
    const auto* src=a+group*m*64+(blockIdx.y*64+row)*64+col;
    copy16(s.low[slot]+la(row,col),src);
    copy16(s.high[slot]+la(row,col),src+m*(k/2));
  });
  o1_static_for<0,4>([&](auto chunk) {
    unsigned off=threadIdx.x*16+chunk*2048;
    unsigned row=off/64,col=off%64;
    copy16(s.weight[slot]+lb(row,col),w+group*n*64+(blockIdx.x*128+row)*64+col);
  });
  if constexpr(First) {
    // One producer thread per column, once for the whole pair. Parameters are
    // broadcast/reused by all consumer rows; not recomputed per output element.
    const uint32_t c0=ws[group*n+blockIdx.x*128+threadIdx.x];
    const uint32_t c1=group+1<k/128?ws[(group+1)*n+blockIdx.x*128+threadIdx.x]:c0;
    const uint32_t anchor=c0<c1?c0:c1;
    s.factor0[slot/2][threadIdx.x]=int(1u<<(c0-anchor));
    s.factor1[slot/2][threadIdx.x]=int(1u<<(c1-anchor));
    s.anchor_scale[slot/2][threadIdx.x]=decode(anchor);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template<bool ReuseA>
__device__ __forceinline__ void body(const uint8_t* a,const uint8_t* w,
    const float* as,const uint8_t* ws,float* y,int m,int n,int k) {
  extern __shared__ __align__(128) uint8_t buf[];
  auto& s=*reinterpret_cast<Storage*>(buf);
  C::Mma mma;C::HighMma high_mma;
  auto thr=mma.get_slice(threadIdx.x);auto ht=high_mma.get_slice(threadIdx.x);
  auto coords=thr.partition_C(cute::make_identity_tensor(cute::make_shape(cute::_64{},cute::_128{})));
  auto acc=cute::make_fragment_like<float>(thr.make_fragment_C(coords));cute::clear(acc);
  auto low=[&](int slot) {return cute::make_tensor(cute::make_smem_ptr<cutlass::uint4b_t>(
      static_cast<void*>(s.low[slot])),C::NibbleLayout<64>{});};
  auto high=[&](int slot) {return cute::make_tensor(cute::make_smem_ptr<cutlass::int4b_t>(
      static_cast<void*>(s.high[slot])),C::NibbleLayout<64>{});};
  auto weight=[&](int slot) {return cute::make_tensor(cute::make_smem_ptr<cutlass::int4b_t>(
      static_cast<void*>(s.weight[slot])),C::NibbleLayout<128>{});};
  auto tile_a=[&](auto t,auto half) {return cute::local_tile(t,
      cute::make_shape(cute::_64{},cute::_64{}),cute::make_coord(cute::_0{},half));};
  auto a00=thr.partition_fragment_A(tile_a(low(0),cute::_0{}));
  auto a01=cute::make_fragment_like(a00),a10=cute::make_fragment_like(a00),a11=cute::make_fragment_like(a00);
  auto h00=ht.partition_fragment_A(tile_a(high(0),cute::_0{}));
  auto h01=cute::make_fragment_like(h00),h10=cute::make_fragment_like(h00),h11=cute::make_fragment_like(h00);
  using LCopy=cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,cutlass::uint4b_t>;
  using SCopy=cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,cutlass::int4b_t>;
  auto lc=cute::make_tiled_copy_A(LCopy{},mma).get_slice(threadIdx.x);
  auto hc=cute::make_tiled_copy_A(SCopy{},high_mma).get_slice(threadIdx.x);
  auto load_a=[&](int slot,auto& al0,auto& al1,auto& ah0,auto& ah1) {
    auto dl0=lc.retile_D(al0),dl1=lc.retile_D(al1);
    auto dh0=hc.retile_D(ah0),dh1=hc.retile_D(ah1);
    cute::copy(LCopy{},lc.partition_S(tile_a(low(slot),cute::_0{})),dl0);
    cute::copy(LCopy{},lc.partition_S(tile_a(low(slot),cute::_1{})),dl1);
    cute::copy(SCopy{},hc.partition_S(tile_a(high(slot),cute::_0{})),dh0);
    cute::copy(SCopy{},hc.partition_S(tile_a(high(slot),cute::_1{})),dh1);
  };
  using SliceMma=O3::O3AmpereConfig<64,64,128,false,2>::Mma;
  SliceMma slice_mma;auto st=slice_mma.get_slice(threadIdx.x);
  auto tile_b=[&](int slot,auto nb,auto half) {return cute::local_tile(weight(slot),
      cute::make_shape(cute::_64{},cute::_64{}),cute::make_coord(nb,half));};
  auto b00=st.partition_fragment_B(tile_b(0,cute::_0{},cute::_0{}));
  auto b01=cute::make_fragment_like(b00);
  auto bc=cute::make_tiled_copy_B(SCopy{},slice_mma).get_slice(threadIdx.x);
  auto load_b=[&](int slot,auto nb,auto& b0,auto& b1) {
    auto d0=bc.retile_D(b0),d1=bc.retile_D(b1);
    cute::copy(SCopy{},bc.partition_S(tile_b(slot,nb,cute::_0{})),d0);
    cute::copy(SCopy{},bc.partition_S(tile_b(slot,nb,cute::_1{})),d1);
  };
  static_assert(decltype(cute::size<1>(b00))::value==4);
  static_assert(decltype(cute::size<1>(acc))::value==2);
  using LA=cute::MMA_Atom<cute::SM80_16x8x64_S32U4S4S32_TN>;
  using HA=cute::MMA_Atom<cute::SM80_16x8x64_S32S4S4S32_TN>;
  auto integer_group=[&](auto mi,auto& al0,auto& al1,auto& ah0,auto& ah1,
                         auto& bl0,auto& bl1,auto& pl,auto& ph) {
    cute::clear(pl);cute::clear(ph);
    o1_static_for<0,4>([&](auto ni) {
      auto l=pl(cute::_,ni),h=ph(cute::_,ni);
      cute::gemm(LA{},l,al0(cute::_,mi,cute::_0{}),bl0(cute::_,ni,cute::_0{}),l);
      cute::gemm(HA{},h,ah0(cute::_,mi,cute::_0{}),bl0(cute::_,ni,cute::_0{}),h);
    });
    o1_static_for<0,4>([&](auto ni) {
      auto l=pl(cute::_,ni),h=ph(cute::_,ni);
      cute::gemm(LA{},l,al1(cute::_,mi,cute::_0{}),bl1(cute::_,ni,cute::_0{}),l);
      cute::gemm(HA{},h,ah1(cute::_,mi,cute::_0{}),bl1(cute::_,ni,cute::_0{}),h);
    });
  };
  const int groups=k/128;
  prefetch<true>(s,0,0,a,w,ws,m,n,k);
  if(groups>1) prefetch<false>(s,1,1,a,w,ws,m,n,k);
  for(int group=0;group<groups;group+=2) {
    const int slot=group%4;const bool second=group+1<groups;
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    // Covers both completed inputs, scale stores, and old slots before reuse.
    __syncthreads();
    if(group+2<groups) prefetch<true>(s,(group+2)%4,group+2,a,w,ws,m,n,k);
    if(group+3<groups) prefetch<false>(s,(group+3)%4,group+3,a,w,ws,m,n,k);
    if constexpr(!ReuseA) {
      load_a(slot,a00,a01,h00,h01);
      if(second) load_a(slot+1,a10,a11,h10,h11);
    }
    o1_static_for<0,2>([&](auto nb) {
      // Only one B group is live. Retain first-group partials for both M atoms
      // so its B fragment is still reused across M, not reloaded per M atom.
      auto first=cute::make_tensor<int>(cute::make_shape(cute::_4{},cute::_2{},cute::_4{}));
      if constexpr(ReuseA) load_a(slot,a00,a01,h00,h01);
      load_b(slot,nb,b00,b01);
      o1_static_for<0,2>([&](auto mi) {
        auto pl=cute::make_tensor<int>(cute::make_shape(cute::_4{},cute::_4{}));
        auto ph=cute::make_fragment_like(pl);
        integer_group(mi,a00,a01,h00,h01,b00,b01,pl,ph);
        o1_static_for<0,4>([&](auto ni) {
          o1_static_for<0,4>([&](auto vi) {
            first(vi,mi,ni)=pl(vi,ni)+16*ph(vi,ni);
          });
        });
      });
      if(second) {
        if constexpr(ReuseA) load_a(slot+1,a00,a01,h00,h01);
        load_b(slot+1,nb,b00,b01);
      }
      o1_static_for<0,2>([&](auto mi) {
        auto pl=cute::make_tensor<int>(cute::make_shape(cute::_4{},cute::_4{}));
        auto ph=cute::make_fragment_like(pl);
        cute::clear(pl);cute::clear(ph);
        if(second) {
          if constexpr(ReuseA)
            integer_group(mi,a00,a01,h00,h01,b00,b01,pl,ph);
          else
            integer_group(mi,a10,a11,h10,h11,b00,b01,pl,ph);
        }
        o1_static_for<0,4>([&](auto ni) {
          auto full_ni=nb*cute::_4{}+ni;
          o1_static_for<0,4>([&](auto vi) {
            auto coord=coords(vi,mi,full_ni);const int col=cute::get<1>(coord);
            const int p0=first(vi,mi,ni),p1=pl(vi,ni)+16*ph(vi,ni);
            const int sum=p0*s.factor0[slot/2][col]+p1*s.factor1[slot/2][col];
            acc(vi,mi,full_ni)=__fmaf_rn(float(sum),s.anchor_scale[slot/2][col],acc(vi,mi,full_ni));
          });
        });
      });
    });
  }
  o1_static_for<0,decltype(cute::size(acc))::value>([&](auto i) {
    acc(i)=__fmul_rn(acc(i),as[blockIdx.y*64+cute::get<0>(coords(i))]);
  });
  o1_static_for<0,decltype(cute::size(acc))::value/2>([&](auto pair) {
    auto i=pair*cute::_2{};auto p=coords(i),q=coords(i+cute::_1{});
    int offset=(blockIdx.y*64+cute::get<0>(p))*n+blockIdx.x*128+cute::get<1>(p);
    if(cute::get<0>(p)==cute::get<0>(q) && cute::get<1>(q)==cute::get<1>(p)+1 && (offset&1)==0)
      *reinterpret_cast<float2*>(y+offset)=make_float2(acc(i),acc(i+cute::_1{}));
    else {
      y[offset]=acc(i);
      y[(blockIdx.y*64+cute::get<0>(q))*n+blockIdx.x*128+cute::get<1>(q)]=acc(i+cute::_1{});
    }
  });
}
} // namespace o3_pair_fragment_reuse_experiment
