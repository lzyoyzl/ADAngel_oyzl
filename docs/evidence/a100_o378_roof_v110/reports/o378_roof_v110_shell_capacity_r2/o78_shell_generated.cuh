// Generated v78; same guard/quantization/epilogue, different MMA schedule.
// Isolated O7/O8 full-K candidate. No production binding or default change.
// Host proves coefficient/product/prefix INT32 bounds before selecting this.
#pragma once
namespace o78_shell_capacity_experiment {
using C=O78::O3AmpereConfig<64,128,128,false,2,true,2>;
struct alignas(128) Storage {
  alignas(16) int activation_factors[32][64];
  alignas(128) uint8_t low[1][64*64],high[1][64*64],weight[1][128*64];
  int weight_factors[32][128];
};
static_assert(sizeof(Storage)==40960);

template<bool Scale>
__device__ __forceinline__ void initialize(Storage& s,int seed) {
  C::ByteLayout<64> la;C::ByteLayout<128> lb;
  for(int off=threadIdx.x;off<64*64;off+=128) {
    int row=off/64,col=off%64;
    int value=(int(blockIdx.y)*64+row+seed)%5-2;
    s.low[0][la(row,col)]=uint8_t((value&15)*17);
    s.high[0][la(row,col)]=uint8_t(((value<0)?15:0)*17);
  }
  for(int off=threadIdx.x;off<128*64;off+=128) {
    int row=off/64,col=off%64;
    int value=(int(blockIdx.x)*128+row+seed)%7-3;
    s.weight[0][lb(row,col)]=uint8_t((value&15)*17);
  }
  if constexpr(Scale) {
    for(int off=threadIdx.x;off<32*64;off+=128) {
      int g=off/64,row=off%64;
      s.activation_factors[g][row]=1+(int(blockIdx.y)*64+row+g+seed)%3;
    }
    for(int off=threadIdx.x;off<32*128;off+=128) {
      int g=off/128,col=off%128;
      s.weight_factors[g][col]=1+(int(blockIdx.x)*128+col+2*g+seed)%3;
    }
  }
  __syncthreads(); // Read-only shared data afterwards; no unsafe slot reuse.
}

template<int Mode>
__device__ __forceinline__ void body(float* y,int groups,int seed) {
  static_assert(Mode>=0 && Mode<=2);
  constexpr uint32_t m=4096,n=4096,k=4096;
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
  auto b20=cute::make_fragment_like(b0),b21=cute::make_fragment_like(b1);
  auto load_a=[&]() {
    cute::copy(LCopy{},lc.partition_S(tile_a(low(0),cute::_0{})),ld0);
    cute::copy(SCopy{},hc.partition_S(tile_a(high(0),cute::_0{})),hd0);
    cute::copy(LCopy{},lc.partition_S(tile_a(low(0),cute::_1{})),ld1);
    cute::copy(SCopy{},hc.partition_S(tile_a(high(0),cute::_1{})),hd1);
  };
  initialize<Mode==2>(s,seed);
  if constexpr(Mode==0) {
    load_a();
    cute::copy(SCopy{},bc.partition_S(tile_b(0,cute::_0{},cute::_0{})),bd0);
    cute::copy(SCopy{},bc.partition_S(tile_b(0,cute::_0{},cute::_1{})),bd1);
    auto bd20=bc.retile_D(b20),bd21=bc.retile_D(b21);
    cute::copy(SCopy{},bc.partition_S(tile_b(0,cute::_1{},cute::_0{})),bd20);
    cute::copy(SCopy{},bc.partition_S(tile_b(0,cute::_1{},cute::_1{})),bd21);
  }
  #pragma unroll 1
  for(int group=0;group<groups;++group) {
    constexpr int slot=0;
    if constexpr(Mode!=0) load_a();
    // Preserve N64 operand reuse; merge eight M/N MMA chains below.
    o1_static_for<0,2>([&](auto nb) {
      if constexpr(Mode!=0) {
        cute::copy(SCopy{},bc.partition_S(tile_b(slot,nb,cute::_0{})),bd0);
        cute::copy(SCopy{},bc.partition_S(tile_b(slot,nb,cute::_1{})),bd1);
      }
      auto use_b0=[&]() {
        if constexpr(Mode==0 && decltype(nb)::value==1) return b20(cute::_,cute::_,cute::_);
        else return b0(cute::_,cute::_,cute::_);
      }();
      auto use_b1=[&]() {
        if constexpr(Mode==0 && decltype(nb)::value==1) return b21(cute::_,cute::_,cute::_);
        else return b1(cute::_,cute::_,cute::_);
      }();
      // Eight independent chains: two M atoms x four N atoms.
      // Same32 logical partial registers; retain all original A/B reuse.
      auto partial=cute::make_tensor<int>(
          cute::make_shape(cute::_4{},cute::_2{},cute::_4{}));
      cute::clear(partial);
      o1_static_for<0,2>([&](auto mi) {
        o1_static_for<0,4>([&](auto ni) {
          auto p=partial(cute::_,mi,ni);
          cute::gemm(HA{},p,h0(cute::_,mi,cute::_0{}),use_b0(cute::_,ni,cute::_0{}),p);
        });
      });
      o1_static_for<0,2>([&](auto mi) {
        o1_static_for<0,4>([&](auto ni) {
          auto p=partial(cute::_,mi,ni);
          cute::gemm(HA{},p,h1(cute::_,mi,cute::_0{}),use_b1(cute::_,ni,cute::_0{}),p);
        });
      });
      // A G128 signed-high dot is at most8192 in magnitude. Multiplication,
      // not signed left-shift, is defined for every intermediate here.
      o1_static_for<0,32>([&](auto i) { partial(i)*=16; });
      o1_static_for<0,2>([&](auto mi) {
        o1_static_for<0,4>([&](auto ni) {
          auto p=partial(cute::_,mi,ni);
          cute::gemm(LA{},p,a0(cute::_,mi,cute::_0{}),use_b0(cute::_,ni,cute::_0{}),p);
        });
      });
      o1_static_for<0,2>([&](auto mi) {
        o1_static_for<0,4>([&](auto ni) {
          auto p=partial(cute::_,mi,ni);
          cute::gemm(LA{},p,a1(cute::_,mi,cute::_0{}),use_b1(cute::_,ni,cute::_0{}),p);
        });
      });
      o1_static_for<0,2>([&](auto mi) {
        o1_static_for<0,4>([&](auto ni) {
          auto full_ni=nb*cute::_4{}+ni;
          o1_static_for<0,4>([&](auto vi) {
            const auto coord=coords(vi,mi,full_ni);
            if constexpr(Mode==2) {
              const int coefficient=s.activation_factors[group&31][cute::get<0>(coord)]*
                                    s.weight_factors[group&31][cute::get<1>(coord)];
              acc(vi,mi,full_ni)+=partial(vi,mi,ni)*coefficient;
            } else acc(vi,mi,full_ni)+=partial(vi,mi,ni);
          });
        });
      });
    });
  }
  // Reuse the 64 INT32 register slots as FP32 bits before any output store.
  o1_static_for<0,decltype(cute::size(acc))::value>([&](auto i) {
    const auto p=coords(i);
    // Synthetic |integer|<=256*128*2*3*9=1769472, exactly FP32.
    acc(i)=__float_as_int(float(acc(i)));
  });
  o1_static_for<0,decltype(cute::size(acc))::value/2>([&](auto pair) {
    auto i=pair*cute::_2{};const auto p=coords(i),q=coords(i+cute::_1{});
    uint32_t offset=(blockIdx.y*64+cute::get<0>(p))*n+blockIdx.x*128+cute::get<1>(p);
    if(cute::get<0>(p)==cute::get<0>(q) && cute::get<1>(q)==cute::get<1>(p)+1 && (offset&1u)==0)
      *reinterpret_cast<float2*>(y+offset)=make_float2(__int_as_float(acc(i)),__int_as_float(acc(i+cute::_1{})));
    else {
      y[offset]=__int_as_float(acc(i));
      y[(blockIdx.y*64+cute::get<0>(q))*n+blockIdx.x*128+cute::get<1>(q)]=__int_as_float(acc(i+cute::_1{}));
    }
  });
}
} // namespace o78_shell_capacity_experiment
