// Generated from the exact v78 four-warp MMA/payload schedule.
__device__ __forceinline__ void producer(const uint8_t* a,const uint8_t* w,
    const int32_t* af,const int32_t* wf,const float* base_a,const float* base_w,
    float* y,uint32_t m,uint32_t n,uint32_t k) {
  // Host guards K4096, M%64==0, N%128==0 and finite representable base scales.
  // Bases combine dyadic anchors with the existing tensor/fixed-point scale.
  extern __shared__ __align__(128) uint8_t buf[];
  auto& h=*reinterpret_cast<Shared*>(buf);auto& s=h.matrix;
  C::Mma mma;C::HighMma high_mma;
  auto thr=mma.get_slice(threadIdx.x);auto ht=high_mma.get_slice(threadIdx.x);
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
  using LA=cute::MMA_Atom<cute::SM80_16x8x64_S32U4S4S32_TN>;
  using HA=cute::MMA_Atom<cute::SM80_16x8x64_S32S4S4S32_TN>;
  constexpr int Groups=32;
  prefetch(s,0,0,a,w,af,wf,m,n,k);
  for(int group=0;group<Groups;++group) {
    const int slot=group%2;
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    wait<0,128>(); // All prior readers finish before a slot is reused.
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
      constexpr int buffer=decltype(nb)::value;
      wait<2+2*buffer,256>();
      o1_static_for<0,4>([&](auto q) {
        const uint4 v=make_uint4(pack(partial(q*8),partial(q*8+1)),
            pack(partial(q*8+2),partial(q*8+3)),
            pack(partial(q*8+4),partial(q*8+5)),
            pack(partial(q*8+6),partial(q*8+7)));
        h.partial[buffer][q][threadIdx.x]=v;
      });
      h.factors[buffer][threadIdx.x]=threadIdx.x<64 ?
          s.activation_factors[slot][threadIdx.x] :
          s.weight_factors[slot][buffer*64+threadIdx.x-64];
      arrive<1+2*buffer>();
    });
  }
}
