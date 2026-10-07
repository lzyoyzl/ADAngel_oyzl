// v137: two warp cohorts keep route sums for all K; one shared epilogue merge.
// Experimental only. Original full-K guard and FP32 fallback remain mandatory.
#pragma once
namespace o3_route_cohort_experiment {
using C = O3::O3AmpereConfig<64,128,128,false,2,false,3>;
using S = o3_grouped_cta_experiment::Storage;

__device__ __forceinline__ void prefetch(S& s, int slot, int group,
    const uint8_t* a, const uint8_t* w, const uint8_t* metadata,
    int m, int n, int k) {
  C::ByteLayout<64> la; C::ByteLayout<128> lb;
  // The256 cooperative threads copy the same bytes as the128-thread control.
  unsigned off = threadIdx.x*16;
  unsigned row = off/64, col = off%64;
  const auto* src = a+group*m*64+(roof_grouped_cta::tile().y*64+row)*64+col;
  copy16(s.low[slot]+la(row,col),src);
  copy16(s.high[slot]+la(row,col),src+m*(k/2));
  o1_static_for<0,2>([&](auto chunk) {
    unsigned pos=off+chunk*4096, r=pos/64, c=pos%64;
    copy16(s.weight[slot]+lb(r,c),w+group*n*64+(roof_grouped_cta::tile().x*128+r)*64+c);
  });
  if(threadIdx.x<32) {
    unsigned first=threadIdx.x*4;
    copy16(s.factor[slot]+first,reinterpret_cast<const int*>(metadata)+
        group*n+roof_grouped_cta::tile().x*128+first);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ void body(const uint8_t* a,const uint8_t* w,
    const float* as,const uint8_t* metadata,float* y,int m,int n,int k) {
  extern __shared__ __align__(128) uint8_t buf[];
  auto& s=*reinterpret_cast<S*>(buf);
  const bool high_route=threadIdx.x>=128;
  const int math_thread=threadIdx.x%128;
  C::Mma mma; auto thr=mma.get_slice(math_thread);
  auto coords=thr.partition_C(cute::make_identity_tensor(cute::make_shape(cute::_64{},cute::_128{})));
  auto acc=cute::make_fragment_like<uint32_t>(thr.make_fragment_C(coords)); cute::clear(acc);
  auto activation=[&](int slot) {
    return cute::make_tensor(cute::make_smem_ptr<cutlass::uint4b_t>(
        static_cast<void*>(high_route?s.high[slot]:s.low[slot])),C::NibbleLayout<64>{});
  };
  auto tile_a=[&](int slot,auto half) {
    return cute::local_tile(activation(slot),cute::make_shape(cute::_64{},cute::_64{}),
        cute::make_coord(cute::_0{},half));
  };
  auto a0=thr.partition_fragment_A(tile_a(0,cute::_0{})); auto a1=cute::make_fragment_like(a0);
  using AC=cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,cutlass::uint4b_t>;
  auto ac=cute::make_tiled_copy_A(AC{},mma).get_slice(math_thread);
  auto load_a=[&](int slot) {
    auto d0=ac.retile_D(a0),d1=ac.retile_D(a1);
    cute::copy(AC{},ac.partition_S(tile_a(slot,cute::_0{})),d0);
    cute::copy(AC{},ac.partition_S(tile_a(slot,cute::_1{})),d1);
  };
  using Slice=O3::O3AmpereConfig<64,32,128,false,2>::Mma;
  Slice slice; auto st=slice.get_slice(math_thread);
  auto weight=[&](int slot) {return cute::make_tensor(cute::make_smem_ptr<cutlass::int4b_t>(
      static_cast<void*>(s.weight[slot])),C::NibbleLayout<128>{});};
  auto tile_b=[&](int slot,auto nb,auto half) {return cute::local_tile(weight(slot),
      cute::make_shape(cute::_32{},cute::_64{}),cute::make_coord(nb,half));};
  auto b0=st.partition_fragment_B(tile_b(0,cute::_0{},cute::_0{})); auto b1=cute::make_fragment_like(b0);
  using BC=cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,cutlass::int4b_t>;
  auto bc=cute::make_tiled_copy_B(BC{},slice).get_slice(math_thread);
  auto load_b=[&](int slot,auto nb) {
    auto d0=bc.retile_D(b0),d1=bc.retile_D(b1);
    cute::copy(BC{},bc.partition_S(tile_b(slot,nb,cute::_0{})),d0);
    cute::copy(BC{},bc.partition_S(tile_b(slot,nb,cute::_1{})),d1);
  };
  using LA=cute::MMA_Atom<cute::SM80_16x8x64_S32U4S4S32_TN>;
  using HA=cute::MMA_Atom<cute::SM80_16x8x64_S32S4S4S32_TN>;
  static_assert(decltype(cute::size<1>(b0))::value==2);
  static_assert(decltype(cute::size<1>(acc))::value==2);
  prefetch(s,0,0,a,w,metadata,m,n,k); prefetch(s,1,1,a,w,metadata,m,n,k);
  for(int group=0;group<32;++group) {
    const int slot=group%3;
    if(group+2<32) asm volatile("cp.async.wait_group 1;" ::: "memory");
    else asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    if(group+2<32) prefetch(s,(group+2)%3,group+2,a,w,metadata,m,n,k);
    load_a(slot);
    o1_static_for<0,4>([&](auto nb) {
      load_b(slot,nb);
      auto partial=cute::make_tensor<int>(cute::make_shape(cute::_4{},cute::_2{},cute::_2{}));
      cute::clear(partial);
      // Only the MMA region is warp-uniform conditional. All256 threads
      // execute the SAME loop barriers/async-copy commits at the same PCs.
      if(high_route) {
        auto ah0=cute::recast<cutlass::int4b_t>(a0),ah1=cute::recast<cutlass::int4b_t>(a1);
        o1_static_for<0,2>([&](auto mi) {o1_static_for<0,2>([&](auto ni) {
          auto p=partial(cute::_,mi,ni);
          cute::gemm(HA{},p,ah0(cute::_,mi,cute::_0{}),b0(cute::_,ni,cute::_0{}),p);
        });});
        o1_static_for<0,2>([&](auto mi) {o1_static_for<0,2>([&](auto ni) {
          auto p=partial(cute::_,mi,ni);
          cute::gemm(HA{},p,ah1(cute::_,mi,cute::_0{}),b1(cute::_,ni,cute::_0{}),p);
        });});
      } else {
        o1_static_for<0,2>([&](auto mi) {o1_static_for<0,2>([&](auto ni) {
          auto p=partial(cute::_,mi,ni);
          cute::gemm(LA{},p,a0(cute::_,mi,cute::_0{}),b0(cute::_,ni,cute::_0{}),p);
        });});
        o1_static_for<0,2>([&](auto mi) {o1_static_for<0,2>([&](auto ni) {
          auto p=partial(cute::_,mi,ni);
          cute::gemm(LA{},p,a1(cute::_,mi,cute::_0{}),b1(cute::_,ni,cute::_0{}),p);
        });});
      }
      o1_static_for<0,2>([&](auto mi) {o1_static_for<0,2>([&](auto ni) {
        auto full_ni=nb*cute::_2{}+ni;
        o1_static_for<0,4>([&](auto vi) {
          const int col=cute::get<1>(coords(vi,mi,full_ni));
          // uint32 arithmetic is exact modulo2^32, including route sums that
          // individually wrap. The original guard bounds the reconstructed sum.
          acc(vi,mi,full_ni)+=uint32_t(partial(vi,mi,ni))*uint32_t(s.factor[slot][col]);
        });
      });});
    });
  }
  __syncthreads(); // All consumers finished reading the last pipeline stage.
  uint32_t* low_sum=reinterpret_cast<uint32_t*>(buf); // Reuse storage, not extra shared memory.
  if(!high_route) {
    o1_static_for<0,decltype(cute::size(acc))::value>([&](auto i) {
      auto p=coords(i);low_sum[cute::get<0>(p)*128+cute::get<1>(p)]=acc(i);
    });
  }
  __syncthreads(); // One-time route handoff only, never perG128.
  if(high_route) {
    o1_static_for<0,decltype(cute::size(acc))::value>([&](auto i) {
      auto p=coords(i);
      uint32_t sum=low_sum[cute::get<0>(p)*128+cute::get<1>(p)]+16u*acc(i);
      int signed_sum; asm("mov.b32 %0,%1;":"=r"(signed_sum):"r"(sum));
      const uint32_t code=reinterpret_cast<const uint32_t*>(metadata)[32*n+
          roof_grouped_cta::tile().x*128+cute::get<1>(p)];
      float result=__fmul_rn(__fmul_rn(float(signed_sum),o3_grouped_cta_experiment::decode(code)),
          as[roof_grouped_cta::tile().y*64+cute::get<0>(p)]);
      acc(i)=__float_as_uint(result);
    });
    o1_static_for<0,decltype(cute::size(acc))::value/2>([&](auto pair) {
      auto i=pair*cute::_2{};auto p=coords(i),q=coords(i+cute::_1{});
      const int offset=(roof_grouped_cta::tile().y*64+cute::get<0>(p))*n+
          roof_grouped_cta::tile().x*128+cute::get<1>(p);
      if(cute::get<0>(p)==cute::get<0>(q) && cute::get<1>(q)==cute::get<1>(p)+1 && (offset&1)==0)
        *reinterpret_cast<float2*>(y+offset)=make_float2(__uint_as_float(acc(i)),__uint_as_float(acc(i+cute::_1{})));
      else {
        y[offset]=__uint_as_float(acc(i));
        y[(roof_grouped_cta::tile().y*64+cute::get<0>(q))*n+roof_grouped_cta::tile().x*128+cute::get<1>(q)]
            =__uint_as_float(acc(i+cute::_1{}));
      }
    });
  }
}
} // namespace o3_route_cohort_experiment
