// v112: fixed public cache-fed experiment; no production dispatch.
#pragma once
namespace o78_direct_fragment_experiment {
using C=O78::O3AmpereConfig<64,128,128,false,2,true,2>;
template<class Scale> struct alignas(128) Storage {
  alignas(16) Scale activation_factors[32][64];
  alignas(16) Scale weight_factors[32][128];
};
static_assert(sizeof(Storage<int>)==24576 && sizeof(Storage<float>)==24576);
__device__ __forceinline__ uint4 load128(const void* p) {
  uint4 v;
  asm volatile("ld.global.ca.v4.b32 {%0,%1,%2,%3}, [%4];"
      : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p) : "memory");
  return v;
}
template<class Scale>
__device__ __forceinline__ void cache_metadata(Storage<Scale>& s,
    const Scale* af,const Scale* wf,unsigned m,unsigned n) {
  for(unsigned off=threadIdx.x*4;off<32*64;off+=128*4) {
    auto v=load128(af+(off/64)*m+blockIdx.y*64+off%64);
    *reinterpret_cast<uint4*>(&s.activation_factors[0][0]+off)=v;
  }
  for(unsigned off=threadIdx.x*4;off<32*128;off+=128*4) {
    auto v=load128(wf+(off/128)*n+blockIdx.x*128+off%128);
    *reinterpret_cast<uint4*>(&s.weight_factors[0][0]+off)=v;
  }
  __syncthreads(); // Single publish barrier; cache is read-only thereafter.
}
template<bool Integer>
__device__ __forceinline__ void body(const uint8_t* a,const uint8_t* w,
    const std::conditional_t<Integer,int,float>* af,
    const std::conditional_t<Integer,int,float>* wf,const float* base_a,const float* base_w,
    float* y,uint32_t m,uint32_t n,uint32_t k) {
  // Host guards K4096, M%64==0, N%128==0 and finite representable base scales.
  // Bases combine dyadic anchors with the existing tensor/fixed-point scale.
  extern __shared__ __align__(128) uint8_t buf[];
  auto& s=*reinterpret_cast<Storage<std::conditional_t<Integer,int,float>>*>(buf);
  C::Mma mma;C::HighMma high_mma;
  auto thr=mma.get_slice(threadIdx.x);auto ht=high_mma.get_slice(threadIdx.x);
  auto coords=thr.partition_C(cute::make_identity_tensor(cute::make_shape(cute::_64{},cute::_128{})));
  auto acc=cute::make_fragment_like<std::conditional_t<Integer,int,float>>(thr.make_fragment_C(coords));cute::clear(acc);
  auto low=[&](int slot) {return cute::make_tensor(cute::make_smem_ptr<cutlass::uint4b_t>(
      static_cast<void*>(nullptr)),C::NibbleLayout<64>{});};
  auto high=[&](int slot) {return cute::make_tensor(cute::make_smem_ptr<cutlass::int4b_t>(
      static_cast<void*>(nullptr)),C::NibbleLayout<64>{});};
  auto weight=[&](int slot) {return cute::make_tensor(cute::make_smem_ptr<cutlass::int4b_t>(
      static_cast<void*>(nullptr)),C::NibbleLayout<128>{});};
  auto tile_a=[&](auto t,auto half) {return cute::local_tile(t,
      cute::make_shape(cute::_64{},cute::_64{}),cute::make_coord(cute::_0{},half));};
  // A single warp atom; derive its M tile from CuTe C coordinates.
  using LA=cute::MMA_Atom<cute::SM80_16x8x64_S32U4S4S32_TN>;
  using HA=cute::MMA_Atom<cute::SM80_16x8x64_S32S4S4S32_TN>;
  auto atom_low=cute::make_tiled_mma(LA{});
  auto atom_high=cute::make_tiled_mma(HA{});
  auto lt=atom_low.get_slice(threadIdx.x%32);
  auto at=atom_high.get_slice(threadIdx.x%32);
  using LCopy=cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,cutlass::uint4b_t>;
  using SCopy=cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,cutlass::int4b_t>;
  auto ac_low=cute::make_tiled_copy_A(LCopy{},atom_low).get_slice(threadIdx.x%32);
  auto ac_high=cute::make_tiled_copy_A(SCopy{},atom_high).get_slice(threadIdx.x%32);
  auto atom_tile_a=[&](auto tensor,int atom_m,auto half) {
    return cute::local_tile(tensor,cute::make_shape(cute::_16{},cute::_64{}),
                           cute::make_coord(atom_m,half));
  };
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
  constexpr int Groups=32;
  cache_metadata(s,af,wf,m,n);
  for(int group=0;group<Groups;++group) {
    const int slot=0; // Type-only dummy fragment view.
    // Preserve N64 operand reuse; merge eight M/N MMA chains below.
    o1_static_for<0,2>([&](auto nb) {
      auto load_b=[&](auto half,auto& bv) {
        auto br=cute::recast<uint32_t>(bv);
        o1_static_for<0,2>([&](auto pair) {
          const int off=o78_register_layout_mapping::w_offset(
              cute::get<1>(coords(cute::_0{},cute::_0{},nb*cute::_4{}+pair*cute::_2{})),half,threadIdx.x&31);
          const uint4 v=load128(w+size_t(group)*n*64+blockIdx.x*128*64+off);
          br(pair*4+0)=v.x;br(pair*4+1)=v.y;br(pair*4+2)=v.z;br(pair*4+3)=v.w;
        });
      };
      load_b(cute::_0{},b0);load_b(cute::_1{},b1);
      // Keep the original B0/B1 fragments across both M atoms.
      o1_static_for<0,2>([&](auto mi) {
        const int atom_m=cute::get<0>(coords(cute::_0{},mi,nb*cute::_4{}))/16;
        auto partial=cute::make_tensor<int>(cute::make_shape(cute::_4{},cute::_4{}));
        cute::clear(partial);
        auto ar=at.partition_fragment_A(atom_tile_a(high(slot),atom_m,cute::_0{}));
        // Same-width signed/unsigned view, not conversion or new storage.
        auto al=cute::recast<cutlass::uint4b_t>(ar);
        static_assert(decltype(cute::size(ar))::value==32);
        auto load_a=[&](bool high_plane,int half) {
          const int off=o78_register_layout_mapping::a_offset(atom_m*16,half,threadIdx.x&31);
          const auto* src=a+size_t(group)*m*64+blockIdx.y*64*64+off;
          if(high_plane)src+=size_t(m)*(k/2);
          const uint4 v=load128(src);auto words=cute::recast<uint32_t>(ar);
          words(0)=v.x;words(1)=v.y;words(2)=v.z;words(3)=v.w;
        };
        load_a(true,0);
        o1_static_for<0,4>([&](auto ni) {
          auto p=partial(cute::_,ni);
          cute::gemm(HA{},p,ar(cute::_,cute::_0{},cute::_0{}),b0(cute::_,ni,cute::_0{}),p);
        });
        load_a(true,1);
        o1_static_for<0,4>([&](auto ni) {
          auto p=partial(cute::_,ni);
          cute::gemm(HA{},p,ar(cute::_,cute::_0{},cute::_0{}),b1(cute::_,ni,cute::_0{}),p);
        });
        o1_static_for<0,16>([&](auto i) { partial(i)*=16; });
        load_a(false,0);
        o1_static_for<0,4>([&](auto ni) {
          auto p=partial(cute::_,ni);
          cute::gemm(LA{},p,al(cute::_,cute::_0{},cute::_0{}),b0(cute::_,ni,cute::_0{}),p);
        });
        load_a(false,1);
        o1_static_for<0,4>([&](auto ni) {
          auto p=partial(cute::_,ni);
          cute::gemm(LA{},p,al(cute::_,cute::_0{},cute::_0{}),b1(cute::_,ni,cute::_0{}),p);
        });
        o1_static_for<0,4>([&](auto ni) {
          auto full_ni=nb*cute::_4{}+ni;
          o1_static_for<0,4>([&](auto vi) {
            const auto coord=coords(vi,mi,full_ni);
            if constexpr(Integer) {
              const int coefficient=s.activation_factors[group][cute::get<0>(coord)]*
                                    s.weight_factors[group][cute::get<1>(coord)];
              acc(vi,mi,full_ni)+=partial(vi,ni)*coefficient;
            } else {
              const float scale=__fmul_rn(s.activation_factors[group][cute::get<0>(coord)],
                                       s.weight_factors[group][cute::get<1>(coord)]);
              acc(vi,mi,full_ni)=__fmaf_rn(float(partial(vi,ni)),scale,acc(vi,mi,full_ni));
            }
          });
        });
      });
    });
  }
  if constexpr(Integer) {
  // Reuse the 64 INT32 register slots as FP32 bits before any output store.
  o1_static_for<0,decltype(cute::size(acc))::value>([&](auto i) {
    const auto p=coords(i);
    const float row=base_a[blockIdx.y*64+cute::get<0>(p)];
    const float column=base_w[blockIdx.x*128+cute::get<1>(p)];
    acc(i)=__float_as_int(__fmul_rn(__fmul_rn(float(acc(i)),row),column));
  });
  }
  auto output_value=[&](auto i) {
    if constexpr(Integer)return __int_as_float(acc(i));
    else return acc(i);
  };
  o1_static_for<0,decltype(cute::size(acc))::value/2>([&](auto pair) {
    auto i=pair*cute::_2{};const auto p=coords(i),q=coords(i+cute::_1{});
    uint32_t offset=(blockIdx.y*64+cute::get<0>(p))*n+blockIdx.x*128+cute::get<1>(p);
    if(cute::get<0>(p)==cute::get<0>(q) && cute::get<1>(q)==cute::get<1>(p)+1 && (offset&1u)==0)
      *reinterpret_cast<float2*>(y+offset)=make_float2(output_value(i),output_value(i+cute::_1{}));
    else {
      y[offset]=output_value(i);
      y[(blockIdx.y*64+cute::get<0>(q))*n+blockIdx.x*128+cute::get<1>(q)]=output_value(i+cute::_1{});
    }
  });
}
} // namespace o78_direct_fragment_experiment
