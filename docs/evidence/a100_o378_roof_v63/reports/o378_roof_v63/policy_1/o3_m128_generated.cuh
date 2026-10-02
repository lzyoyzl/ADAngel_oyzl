// Candidates51/52: factor the O3 row scale after the FP32 G128 reduction.
// Separate device IR: never alter production or older candidate codegen.
#pragma once
namespace o3_row_scale_epilogue_experiment {
template<int N,bool Cached> struct O3ScaleCodeScratch {};
template<int N> struct O3ScaleCodeScratch<N,true> { uint8_t scale_codes[N*32]; };
template<int M,int Groups,bool Enabled,int Stages=2> struct SplitActivationScales {};
template<int M,int Groups,int Stages> struct SplitActivationScales<M,Groups,true,Stages> {
  alignas(16) float activation_scales[Stages*Groups*M];
};

template<int M,int N,int K,bool Cached=false,int WN=2,bool DualScale=false,int Stages=2>
struct O3AmpereConfig {
  static_assert(K==128||K==256);
  static_assert(Stages==2 || Stages==3);
  static constexpr int WM=2;
  static_assert(M==64 && WN==2 && K==128);
  static constexpr int Threads=32*WM*WN, Groups=K/128, Bytes=K/2;
  using Low=cutlass::uint4b_t;
  using Signed=cutlass::int4b_t;
  using Mma=cute::TiledMMA<cute::MMA_Atom<cute::SM80_16x8x64_S32U4S4S32_TN>,
      cute::Layout<cute::Shape<cute::Int<WM>,cute::Int<WN>,cute::_1>>,
      cute::Tile<cute::Int<M>,cute::Int<N>,cute::_64>>;
  using HighMma=cute::TiledMMA<cute::MMA_Atom<cute::SM80_16x8x64_S32S4S4S32_TN>,
      cute::Layout<cute::Shape<cute::Int<WM>,cute::Int<WN>,cute::_1>>,
      cute::Tile<cute::Int<M>,cute::Int<N>,cute::_64>>;
  template<int Rows> using ByteLayout=decltype(cute::composition(
      cute::Swizzle<K==256?3:2,4,3>{},cute::Layout<cute::Shape<cute::Int<Rows>,cute::Int<Bytes>>,
      cute::Stride<cute::Int<Bytes>,cute::_1>>{}));
  // The nibble layout is the byte layout with every bit position shifted by1.
  template<int Rows> using NibbleLayout=decltype(cute::composition(
      cute::Swizzle<K==256?3:2,5,3>{},cute::Layout<cute::Shape<cute::Int<Rows>,cute::Int<K>>,
      cute::Stride<cute::Int<K>,cute::_1>>{}));
  struct alignas(128) Storage : O3ScaleCodeScratch<N,Cached>,
      SplitActivationScales<M,Groups,DualScale,Stages> {
    alignas(128) uint8_t low[Stages][M*Bytes], high[Stages][M*Bytes], weight[Stages][N*Bytes];
    float scales[(Cached?32:Stages*Groups)*N];
  };
};

template<int M,int N,int K,bool Fast,bool Cached,int WN,bool StaticCopy,bool VectorScale=false,bool DualScale=false,bool GroupMajorScale=false,bool PrebiasActivationScale=false,bool AsyncScale=false,bool CombinedScalePanels=false,int Stages=2>
__device__ __forceinline__ void o3_prefetch(typename O3AmpereConfig<M,N,K,Cached,WN,DualScale,Stages>::Storage& s,
    int slot,int stage,const uint8_t* a,const uint8_t* w,const uint8_t* ws,int m,int k,
    const float* grouped_as=nullptr,int total_n=0) {
  using C=O3AmpereConfig<M,N,K,Cached,WN,DualScale,Stages>;
  static_assert(!DualScale || (!Fast && !Cached && !VectorScale));
  static_assert(!GroupMajorScale || DualScale || (!Cached && !VectorScale));
  static_assert(!PrebiasActivationScale || DualScale);
  static_assert(!AsyncScale || (DualScale && GroupMajorScale && !PrebiasActivationScale));
  static_assert(!CombinedScalePanels || AsyncScale);
  typename C::template ByteLayout<M> la;
  typename C::template ByteLayout<N> lb;
  auto copy_a=[&](unsigned off) {
    unsigned row=off/C::Bytes,col=off%C::Bytes;
    auto src=a+stage*m*C::Bytes+(blockIdx.y*M+row)*C::Bytes+col;
    copy16(s.low[slot]+la(row,col),src);
    copy16(s.high[slot]+la(row,col),src+m*(k/2));
  };
  auto copy_b=[&](unsigned off) {
    unsigned row=off/C::Bytes,col=off%C::Bytes;
    copy16(s.weight[slot]+lb(row,col),w+stage*total_n*C::Bytes+(blockIdx.x*N+row)*C::Bytes+col);
  };
  if constexpr(StaticCopy) {
    // The per-thread copy count is known from the CTA shape. Do not make
    // ptxas infer a runtime loop trip count from threadIdx.x/launch bounds.
    constexpr int Step=C::Threads*16;
    o1_static_for<0,(M*C::Bytes+Step-1)/Step>([&](auto chunk) {
      unsigned off=threadIdx.x*16+chunk*Step;
      if constexpr(M*C::Bytes%Step==0) copy_a(off);
      else if(off<M*C::Bytes) copy_a(off);
    });
    o1_static_for<0,(N*C::Bytes+Step-1)/Step>([&](auto chunk) {
      unsigned off=threadIdx.x*16+chunk*Step;
      if constexpr(N*C::Bytes%Step==0) copy_b(off);
      else if(off<N*C::Bytes) copy_b(off);
    });
  } else {
    for(unsigned off=threadIdx.x*16;off<M*C::Bytes;off+=C::Threads*16) copy_a(off);
    for(unsigned off=threadIdx.x*16;off<N*C::Bytes;off+=C::Threads*16) copy_b(off);
  }
  if constexpr(AsyncScale) {
    // Candidate14: group-major FP32 panels are contiguous and 16B aligned.
    // Copy their bits directly, with no scale arithmetic or temporary GPRs.
    // The existing commit/wait_group + CTA barrier covers payload AND scales;
    // do not consume this slot before the next process_stage wait/barrier.
    static_assert(M%4==0 && N%4==0 && alignof(typename C::Storage)>=16);
    const auto* grouped_ws=reinterpret_cast<const float*>(ws);
    if constexpr(CombinedScalePanels) {
      // Candidate15: flatten the two physical G128 panels, not their math.
      // A uses one full warp and W two full warps at64x128x256. Each lane
      // still copies16B, but there is no half-active A-copy warp.
      static_assert((M*C::Groups)%128==0 && (N*C::Groups)%128==0);
      unsigned off=threadIdx.x*4;
      if(threadIdx.x<M*C::Groups/4) {
        unsigned group=off/M,row=off%M;
        copy16(s.activation_scales+slot*C::Groups*M+off,
               grouped_as+(stage*C::Groups+group)*m+blockIdx.y*M+row);
      }
      if(threadIdx.x<N*C::Groups/4) {
        unsigned group=off/N,col=off%N;
        copy16(s.scales+slot*C::Groups*N+off,
               grouped_ws+(stage*C::Groups+group)*total_n+blockIdx.x*N+col);
      }
    } else {
    o1_static_for<0,C::Groups>([&](auto group) {
      int g=stage*C::Groups+group;
      unsigned first=threadIdx.x*4;
      if(threadIdx.x<M/4)
        copy16(s.activation_scales+(slot*C::Groups+group)*M+first,
               grouped_as+g*m+blockIdx.y*M+first);
      if(threadIdx.x<N/4)
        copy16(s.scales+(slot*C::Groups+group)*N+first,
               grouped_ws+g*total_n+blockIdx.x*N+first);
    });
    }
  } else if constexpr(DualScale) {
    // Both sides vary across G128. Only a host-guarded candidate may prebias A.
    const auto* grouped_ws=reinterpret_cast<const float*>(ws);
    o1_static_for<0,C::Groups>([&](auto group) {
      int g=stage*C::Groups+group;
      if(threadIdx.x<M) {
        float row_scale=grouped_as[GroupMajorScale ? g*m+blockIdx.y*M+threadIdx.x
                                      : (blockIdx.y*M+threadIdx.x)*(k/128)+g];
        // Bit payload only: after this transformation the shared value must
        // never participate in FP arithmetic. Each group/row is prepared once
        // per CTA instead of subtracting the FP32 exponent bias per output.
        if constexpr(PrebiasActivationScale)
          row_scale=__uint_as_float(__float_as_uint(row_scale)-0x3f800000u);
        s.activation_scales[(slot*C::Groups+group)*M+threadIdx.x]=row_scale;
      }
      if(threadIdx.x<N)
        s.scales[(slot*C::Groups+group)*N+threadIdx.x]=
            grouped_ws[GroupMajorScale ? g*total_n+blockIdx.x*N+threadIdx.x
                                      : (blockIdx.x*N+threadIdx.x)*(k/128)+g];
    });
  } else if(!Cached && threadIdx.x<N) {
    uint32_t packed_codes=0;
    if constexpr(VectorScale) {
      static_assert(K==256 && !Cached);
      // K is a multiple of256: row stride k/128 and stage*2 are even.
      // Read the two adjacent UE8M0 codes once; keep both G128 scales separate.
      const auto* src=ws+(blockIdx.x*N+threadIdx.x)*(k/128)+stage*2;
      packed_codes=*reinterpret_cast<const uint16_t*>(src);
    }
    o1_static_for<0,C::Groups>([&](auto group) {
      uint32_t code;
      if constexpr(VectorScale) code=(packed_codes>>(8*group))&255u;
      else code=ws[GroupMajorScale ? (stage*C::Groups+group)*total_n+blockIdx.x*N+threadIdx.x
                                  : (blockIdx.x*N+threadIdx.x)*(k/128)+stage*C::Groups+group];
      uint32_t bits=code?code<<23:0x00400000u;
      s.scales[(slot*C::Groups+group)*N+threadIdx.x]=__uint_as_float(Fast?((code-127u)<<23):bits);
    });
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template<int M,int N,int K,bool Fast,bool Cached=false,bool Magic=Fast,int WN=2,bool Merge=false,bool StaticCopy=false,bool PhasePair=false,bool Stream=false,bool BoundedOperands=false,bool VectorStore=false,bool VectorScale=false,bool DualScale=false,bool GroupMajorScale=false,int RoofTune=0,bool ActivationPower2=false,bool PrebiasActivationScale=false,bool AsyncScale=false,bool CombinedScalePanels=false,int Stages=2>
__device__ __forceinline__ void o3_body(
    const uint8_t* a,const uint8_t* w,const float* as,const uint8_t* ws,float* y,int m,int n,int k) {
  using C=O3AmpereConfig<M,N,K,Cached,WN,DualScale,Stages>;
  static_assert((Stages==2 || Stages==3) && K==128 && !PhasePair && !Cached && !AsyncScale);
  static_assert(!DualScale && !Fast && !Magic && RoofTune==6 && Stream && VectorStore);
  static_assert(RoofTune==2 || RoofTune==6);
  static_assert(RoofTune>=0 && RoofTune<=7);
  static_assert(RoofTune==0 || (Stream && !Merge && !Magic));
  static_assert(!DualScale || (!Fast && !Cached && !Magic && Stream));
  static_assert(!ActivationPower2 || (DualScale && RoofTune!=0));
  static_assert(!PrebiasActivationScale || ActivationPower2);
  static_assert(!AsyncScale || (DualScale && GroupMajorScale && !ActivationPower2));
  static_assert(!CombinedScalePanels || AsyncScale);
  extern __shared__ __align__(128) uint8_t buf[];
  auto& s=*reinterpret_cast<typename C::Storage*>(buf);
  if constexpr(Cached) {
    // First stage tightly packed codes in a bank-swizzled byte layout, then
    // decode in group-major thread order. Direct vector-load -> group-major
    // float stores made eight lanes contend for each bank during initialization.
    using CodesLayout=decltype(cute::composition(cute::Swizzle<3,2,5>{},
        cute::Layout<cute::Shape<cute::Int<N>,cute::_32>,cute::Stride<cute::_32,cute::_1>>{}));
    CodesLayout codes_layout;
    int groups=k/128;
    for(unsigned off=threadIdx.x*4;off<N*32;off+=C::Threads*4) {
      unsigned col=off/32,first_group=off%32;
      if(first_group<groups) {
        const auto* src=ws+(blockIdx.x*N+col)*groups+first_group;
        uint32_t codes=0;
        // Vector loads need an aligned row stride; small/non-four group
        // counts use scalar bytes and never read outside the scale panel.
        if(groups%4==0 && first_group+4<=groups) {
          codes=*reinterpret_cast<const uint32_t*>(src);
        } else {
          o1_static_for<0,4>([&](auto j) {
            if(first_group+j<groups) codes|=uint32_t(src[j])<<(8*j);
          });
        }
        *reinterpret_cast<uint32_t*>(s.scale_codes+codes_layout(col,first_group))=codes;
      }
    }
    __syncthreads();
    for(unsigned off=threadIdx.x;off<N*32;off+=C::Threads) {
      unsigned col=off%N,group=off/N;
      if(group<groups) {
        uint32_t code=s.scale_codes[codes_layout(col,group)];
        uint32_t bits=code?code<<23:0x00400000u;
        s.scales[group*N+col]=__uint_as_float(Fast?((code-127u)<<23):bits);
      }
    }
  }
  typename C::Mma mma; typename C::HighMma high_mma;
  auto thr=mma.get_slice(threadIdx.x); auto ht=high_mma.get_slice(threadIdx.x);
  auto coords=thr.partition_C(cute::make_identity_tensor(cute::make_shape(cute::Int<M>{},cute::Int<N>{})));
  auto low=thr.make_fragment_C(coords), high=ht.make_fragment_C(coords);
  auto acc=cute::make_fragment_like<float>(low), rows=cute::make_fragment_like<float>(low);
  cute::clear(acc);
  // A row scale is invariant across G128; load it only in the epilogue.
  auto make_low=[&](int slot) { return cute::make_tensor(cute::make_smem_ptr<cutlass::uint4b_t>(
      static_cast<void*>(s.low[slot])),typename C::template NibbleLayout<M>{}); };
  auto make_high=[&](int slot) { return cute::make_tensor(cute::make_smem_ptr<cutlass::int4b_t>(
      static_cast<void*>(s.high[slot])),typename C::template NibbleLayout<M>{}); };
  auto make_weight=[&](int slot) { return cute::make_tensor(cute::make_smem_ptr<cutlass::int4b_t>(
      static_cast<void*>(s.weight[slot])),typename C::template NibbleLayout<N>{}); };
  auto tile_a=[&](auto t,auto sub) {return cute::local_tile(t,cute::make_shape(cute::Int<M>{},cute::_64{}),cute::make_coord(cute::_0{},sub));};
  auto tile_b=[&](auto t,auto sub) {return cute::local_tile(t,cute::make_shape(cute::Int<N>{},cute::_64{}),cute::make_coord(cute::_0{},sub));};
  auto ra=thr.partition_fragment_A(tile_a(make_low(0),cute::_0{}));
  auto rh=ht.partition_fragment_A(tile_a(make_high(0),cute::_0{}));
  auto rb=thr.partition_fragment_B(tile_b(make_weight(0),cute::_0{}));
  auto rb1=cute::make_fragment_like(rb);
  auto ra1=cute::make_fragment_like(ra);auto rh1=cute::make_fragment_like(rh);
  using LCopy=cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,cutlass::uint4b_t>;
  using SCopy=cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,cutlass::int4b_t>;
  auto lc=cute::make_tiled_copy_A(LCopy{},mma).get_slice(threadIdx.x);
  auto hc=cute::make_tiled_copy_A(SCopy{},high_mma).get_slice(threadIdx.x);
  auto bc=cute::make_tiled_copy_B(SCopy{},mma).get_slice(threadIdx.x);
  auto ld=lc.retile_D(ra); auto hd=hc.retile_D(rh); auto bd=bc.retile_D(rb);
  auto bd1=bc.retile_D(rb1);
  auto ld1=lc.retile_D(ra1);auto hd1=hc.retile_D(rh1);
  if constexpr(Stages==2) {
    // Keep the historical call structure for the existing binary controls.
    o3_prefetch<M,N,K,Fast,Cached,WN,StaticCopy,VectorScale,DualScale,GroupMajorScale,PrebiasActivationScale,AsyncScale,CombinedScalePanels>(s,0,0,a,w,ws,m,k,as,n);
  } else {
    o3_prefetch<M,N,K,Fast,Cached,WN,StaticCopy,VectorScale,DualScale,GroupMajorScale,PrebiasActivationScale,AsyncScale,CombinedScalePanels,Stages>(s,0,0,a,w,ws,m,k,as,n);
  }
  // Three-stage candidate: prime a second G128. Each copy stage has its own
  // commit group. The prologue/drain also covers short/odd stage counts.
  if constexpr(Stages==3) {
    if(k/K>1) o3_prefetch<M,N,K,Fast,Cached,WN,StaticCopy,VectorScale,DualScale,GroupMajorScale,PrebiasActivationScale,AsyncScale,CombinedScalePanels,Stages>(s,1,1,a,w,ws,m,k,as,n);
  }
  auto process_stage=[&](int stage,auto slot) {
    if constexpr(Stages==3) {
      if(stage+2<k/K) asm volatile("cp.async.wait_group 1;" ::: "memory");
      else asm volatile("cp.async.wait_group 0;" ::: "memory");
    } else {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    }
    // The completed current slot is visible to every warp. This barrier also
    // protects the previous slot before any warp reuses it for stage+2.
    __syncthreads();
    if constexpr(Stages==3) {
      if(stage+2<k/K) o3_prefetch<M,N,K,Fast,Cached,WN,StaticCopy,VectorScale,DualScale,GroupMajorScale,PrebiasActivationScale,AsyncScale,CombinedScalePanels,Stages>(s,(stage+2)%Stages,stage+2,a,w,ws,m,k,as,n);
    } else {
      if(stage+1<k/K) o3_prefetch<M,N,K,Fast,Cached,WN,StaticCopy,VectorScale,DualScale,GroupMajorScale,PrebiasActivationScale,AsyncScale,CombinedScalePanels>(s,1-slot,stage+1,a,w,ws,m,k,as,n);
    }
    auto process_group=[&](auto group) {
      if constexpr(Stream) {
        // Preload both K64 A sets, but retain only a narrow N slice of B.
        // Only 4 low + 4 high INT32 partials remain live per thread; FP32
        // output accumulators and their G128 FMA order are unchanged.
        auto sub=group*cute::_2{};
        cute::copy(LCopy{},lc.partition_S(tile_a(make_low(slot),sub)),ld);
        cute::copy(SCopy{},hc.partition_S(tile_a(make_high(slot),sub)),hd);
        cute::copy(LCopy{},lc.partition_S(tile_a(make_low(slot),sub+cute::_1{})),ld1);
        cute::copy(SCopy{},hc.partition_S(tile_a(make_high(slot),sub+cute::_1{})),hd1);
        // Roof bit 2 doubles the streamed N slice, exposing four independent
        // MMA atoms per warp instead of two without changing the CTA or G128.
        constexpr int SliceN=WN*((RoofTune&4)?32:16);
        using SmallCopy=SCopy;
        using SliceMma=typename O3AmpereConfig<M,SliceN,K,Cached,WN>::Mma;
        SliceMma slice_mma;
        auto slice_thr=slice_mma.get_slice(threadIdx.x);
        auto slice_b=[&](auto nb,auto half) {return cute::local_tile(make_weight(slot),
            cute::make_shape(cute::Int<SliceN>{},cute::_64{}),cute::make_coord(nb,half));};
        auto br0=slice_thr.partition_fragment_B(slice_b(cute::_0{},sub));
        auto br1=cute::make_fragment_like(br0);
        auto sbc=cute::make_tiled_copy_B(SmallCopy{},slice_mma).get_slice(threadIdx.x);
        auto sd0=sbc.retile_D(br0);auto sd1=sbc.retile_D(br1);
        constexpr int NAtoms=decltype(cute::size<1>(br0))::value;
        static_assert(NAtoms*(N/SliceN)==decltype(cute::size<2>(acc))::value);
        constexpr int MAtoms=decltype(cute::size<1>(acc))::value;
        // Candidate-only: identity coordinates determine each owned row. Keep
        // row scales live across N slices instead of rediscovering them after
        // every MMA. No group may borrow another group's scale.
        auto group_rows=cute::make_tensor<float>(cute::make_shape(cute::_4{},cute::Int<MAtoms>{}));
        const int roof_scale_group=(Cached?stage:slot)*C::Groups+group;
        if constexpr(RoofTune&1) {
          o1_static_for<0,MAtoms>([&](auto mi) {
            o1_static_for<0,4>([&](auto vi) {
              if constexpr(DualScale)
                group_rows(vi,mi)=s.activation_scales[roof_scale_group*M+cute::get<0>(coords(vi,mi,cute::_0{}))];
              else group_rows(vi,mi)=rows(vi,mi,cute::_0{});
            });
          });
        }
        o1_static_for<0,N/SliceN>([&](auto nb) {
          cute::copy(SmallCopy{},sbc.partition_S(slice_b(nb,sub)),sd0);
          cute::copy(SmallCopy{},sbc.partition_S(slice_b(nb,sub+cute::_1{})),sd1);
          o1_static_for<0,decltype(cute::size<1>(acc))::value>([&](auto mi) {
          if constexpr(RoofTune!=0) {
            using LA=cute::MMA_Atom<cute::SM80_16x8x64_S32U4S4S32_TN>;
            using HA=cute::MMA_Atom<cute::SM80_16x8x64_S32S4S4S32_TN>;
            auto columns=cute::make_tensor<float>(cute::make_shape(cute::_4{},cute::Int<NAtoms>{}));
            if constexpr(RoofTune&1) {
              o1_static_for<0,NAtoms>([&](auto ni) {
                auto full_ni=nb*cute::Int<NAtoms>{}+ni;
                o1_static_for<0,4>([&](auto vi) {
                  columns(vi,ni)=s.scales[roof_scale_group*N+cute::get<1>(coords(vi,mi,full_ni))];
                });
              });
            }
            auto finish=[&](auto ni,auto& pl,auto& ph) {
              auto full_ni=nb*cute::Int<NAtoms>{}+ni;
              o1_static_for<0,4>([&](auto vi) {
                const int partial=pl(vi);
                const auto coord=coords(vi,mi,full_ni);
                const float column=s.scales[roof_scale_group*N+cute::get<1>(coord)];
                // Independent original UE8M0 scale, no group coalescing.
                acc(vi,mi,full_ni)=__fmaf_rn(float(partial),column,acc(vi,mi,full_ni));
              });
            };
            if constexpr(RoofTune&2) {
              // Independent output atoms have separate integer dependency
              // chains. Interleave them before conversion; FP32 G128 order
              // remains unchanged for every output element.
              // Four independent N atoms preserve inter-atom ILP. Each
              // fragment becomes 16*high first, then receives both low MMAs.
              // All integer intermediates are < 2^18 for a G128 INT8 x INT4
              // dot; this is exact, including negative high and -128 inputs.
              auto pls=cute::make_tensor<int>(cute::make_shape(cute::_4{},cute::Int<NAtoms>{}));
              cute::clear(pls);
              o1_static_for<0,NAtoms>([&](auto ni) {
                auto pl=pls(cute::_,ni);
                cute::gemm(HA{},pl,rh(cute::_,mi,cute::_0{}),br0(cute::_,ni,cute::_0{}),pl);
              });
              o1_static_for<0,NAtoms>([&](auto ni) {
                auto pl=pls(cute::_,ni);
                cute::gemm(HA{},pl,rh1(cute::_,mi,cute::_0{}),br1(cute::_,ni,cute::_0{}),pl);
              });
              o1_static_for<0,NAtoms>([&](auto ni) {
                o1_static_for<0,4>([&](auto vi) { pls(vi,ni)*=16; });
              });
              o1_static_for<0,NAtoms>([&](auto ni) {
                auto pl=pls(cute::_,ni);
                cute::gemm(LA{},pl,ra(cute::_,mi,cute::_0{}),br0(cute::_,ni,cute::_0{}),pl);
              });
              o1_static_for<0,NAtoms>([&](auto ni) {
                auto pl=pls(cute::_,ni);
                cute::gemm(LA{},pl,ra1(cute::_,mi,cute::_0{}),br1(cute::_,ni,cute::_0{}),pl);
              });
              o1_static_for<0,NAtoms>([&](auto ni) {
                auto pl=pls(cute::_,ni);finish(ni,pl,pl);
              });
            } else {
              o1_static_for<0,NAtoms>([&](auto ni) {
                auto pl=cute::make_tensor<int>(cute::make_shape(cute::_4{}));
                auto ph=cute::make_tensor<int>(cute::make_shape(cute::_4{}));
                cute::clear(pl);cute::clear(ph);
                cute::gemm(LA{},pl,ra(cute::_,mi,cute::_0{}),br0(cute::_,ni,cute::_0{}),pl);
                cute::gemm(HA{},ph,rh(cute::_,mi,cute::_0{}),br0(cute::_,ni,cute::_0{}),ph);
                cute::gemm(LA{},pl,ra1(cute::_,mi,cute::_0{}),br1(cute::_,ni,cute::_0{}),pl);
                cute::gemm(HA{},ph,rh1(cute::_,mi,cute::_0{}),br1(cute::_,ni,cute::_0{}),ph);
                finish(ni,pl,ph);
              });
            }
          } else {
          o1_static_for<0,NAtoms>([&](auto ni) {
            auto full_ni=nb*cute::Int<NAtoms>{}+ni;
            auto pl=cute::make_tensor<int>(cute::make_shape(cute::_4{}));
            auto ph=cute::make_tensor<int>(cute::make_shape(cute::_4{}));
            cute::clear(pl);cute::clear(ph);
            using LA=cute::MMA_Atom<cute::SM80_16x8x64_S32U4S4S32_TN>;
            using HA=cute::MMA_Atom<cute::SM80_16x8x64_S32S4S4S32_TN>;
            if constexpr(Merge) {
            // Exact G128 reconstruction using four INT32 registers, not
            // a full output-tile partial. W fragments still serve both paths.
            cute::gemm(HA{},pl,rh(cute::_,mi,cute::_0{}),br0(cute::_,ni,cute::_0{}),pl);
            cute::gemm(HA{},pl,rh1(cute::_,mi,cute::_0{}),br1(cute::_,ni,cute::_0{}),pl);
            o1_static_for<0,4>([&](auto vi) { pl(vi)*=16; });
            cute::gemm(LA{},pl,ra(cute::_,mi,cute::_0{}),br0(cute::_,ni,cute::_0{}),pl);
            cute::gemm(LA{},pl,ra1(cute::_,mi,cute::_0{}),br1(cute::_,ni,cute::_0{}),pl);
            } else {
            cute::gemm(LA{},pl,ra(cute::_,mi,cute::_0{}),br0(cute::_,ni,cute::_0{}),pl);
            cute::gemm(HA{},ph,rh(cute::_,mi,cute::_0{}),br0(cute::_,ni,cute::_0{}),ph);
            cute::gemm(LA{},pl,ra1(cute::_,mi,cute::_0{}),br1(cute::_,ni,cute::_0{}),pl);
            cute::gemm(HA{},ph,rh1(cute::_,mi,cute::_0{}),br1(cute::_,ni,cute::_0{}),ph);
            }
            o1_static_for<0,4>([&](auto vi) {
              int partial;
              if constexpr(Merge) partial=pl(vi);
              else partial=pl(vi)+16*ph(vi);
              auto coord=coords(vi,mi,full_ni);
              int scale_group=(Cached?stage:slot)*C::Groups+group;
              float column=s.scales[scale_group*N+cute::get<1>(coord)];
              float scale;
              if constexpr(DualScale)
                scale=__fmul_rn(s.activation_scales[scale_group*M+cute::get<0>(coord)],column);
              else if constexpr(Fast) scale=__uint_as_float(__float_as_uint(rows(vi,mi,full_ni))+__float_as_uint(column));
              else scale=__fmul_rn(rows(vi,mi,full_ni),column);
              float value;
              if constexpr(Magic) value=__fadd_rn(__int_as_float(0x4b400000+partial),-12582912.0f);
              else value=float(partial);
              acc(vi,mi,full_ni)=__fmaf_rn(value,scale,acc(vi,mi,full_ni));
            });
          });
          }
        });
          // All lanes take every slice. Keep next-slice shared loads after
          // this warp-scoped boundary, limiting cross-slice operand hoisting.
          // Keep this boundary only for the merged candidate. The independent
          // partial candidate restores the v15 schedule (no extra warp fence).
          if constexpr(BoundedOperands && Merge) __syncwarp();
        });
      } else {
      cute::clear(low);
      if constexpr(Merge) {
        // Exact integer reassociation INSIDE one G128 only. Compute the high
        // path, multiply its INT32 accumulator by16, then add both low MMAs
        // directly into it. One partial fragment instead of two; keep both
        // weight fragments to avoid reloading them for the low path.
        o1_static_for<0,2>([&](auto half) {
          auto sub=group*cute::_2{}+half;
          cute::copy(SCopy{},hc.partition_S(tile_a(make_high(slot),sub)),hd);
          if constexpr(decltype(half)::value==0) {
            cute::copy(SCopy{},bc.partition_S(tile_b(make_weight(slot),sub)),bd);
            cute::gemm(high_mma,rh,rb,low);
          } else {
            cute::copy(SCopy{},bc.partition_S(tile_b(make_weight(slot),sub)),bd1);
            cute::gemm(high_mma,rh,rb1,low);
          }
        });
        o1_static_for<0,decltype(cute::size(low))::value>([&](auto i) {low(i)*=16;});
        o1_static_for<0,2>([&](auto half) {
          auto sub=group*cute::_2{}+half;
          cute::copy(LCopy{},lc.partition_S(tile_a(make_low(slot),sub)),ld);
          if constexpr(decltype(half)::value==0) cute::gemm(mma,ra,rb,low);
          else cute::gemm(mma,ra,rb1,low);
        });
      } else {
        cute::clear(high);
        o1_static_for<0,2>([&](auto half) {
          auto sub=group*cute::_2{}+half;
          cute::copy(LCopy{},lc.partition_S(tile_a(make_low(slot),sub)),ld);
          cute::copy(SCopy{},hc.partition_S(tile_a(make_high(slot),sub)),hd);
          cute::copy(SCopy{},bc.partition_S(tile_b(make_weight(slot),sub)),bd);
          cute::gemm(mma,ra,rb,low);
          cute::gemm(high_mma,rh,rb,high);
        });
      }
      o1_static_for<0,decltype(cute::size(acc))::value>([&](auto i) {
        int partial;
        if constexpr(Merge) partial=low(i);
        else partial=low(i)+16*high(i);
        // Reconstructed INT8 x signed INT4 G128 bound:128*128*8=131072.
        // Exact bias conversion stays well inside the unit-ULP FP32 binade.
        float value,scale;
        int scale_group=(Cached?stage:slot)*C::Groups+group;
        float column=s.scales[scale_group*N+cute::get<1>(coords(i))];
        if constexpr(Magic) {
          value=__fadd_rn(__int_as_float(0x4b400000+partial),-12582912.0f);
        } else {
          value=float(partial);
        }
        if constexpr(Fast) {
          scale=__uint_as_float(__float_as_uint(rows(i))+__float_as_uint(column));
        } else {
          scale=__fmul_rn(rows(i),column);
        }
        acc(i)=__fmaf_rn(value,scale,acc(i));
      });
      }
    };
    o1_static_for<0,C::Groups>(process_group);
    // Next iteration's barrier protects this slot before it is overwritten.
  };
  if constexpr(PhasePair) {
    // The slot is a CuTe compile-time constant, while G128/group ordering
    // stays sequential. The final odd stage needs no second invocation.
    for(int stage=0;stage<k/K;stage+=2) {
      process_stage(stage,cute::_0{});
      if(stage+1<k/K) process_stage(stage+1,cute::_1{});
    }
  } else if constexpr(Stages==2) {
    for(int stage=0;stage<k/K;++stage) process_stage(stage,stage%2);
  } else {
    for(int stage=0;stage<k/K;++stage) process_stage(stage,stage%Stages);
  }
  // FP32 across-group accumulator remains FP32. Only the invariant factor moves.
  o1_static_for<0,decltype(cute::size(acc))::value>([&](auto i) {
    acc(i)=__fmul_rn(acc(i),as[blockIdx.y*M+cute::get<0>(coords(i))]);
  });
  if constexpr(VectorStore) {
    // SM80_16x8_Row pairs values 0/1 and 2/3 along N. Addresses still come
    // from partition_C(identity), never from a hand-written lane mapping.
    static_assert(decltype(cute::size<0>(acc))::value==4);
    o1_static_for<0,decltype(cute::size(acc))::value/2>([&](auto pair) {
      auto i=pair*cute::_2{};
      auto p=coords(i),q=coords(i+cute::_1{});
      int offset=(blockIdx.y*M+cute::get<0>(p))*n+blockIdx.x*N+cute::get<1>(p);
      // Guard also keeps this safe if a future CuTe layout changes adjacency.
      if(cute::get<0>(p)==cute::get<0>(q) && cute::get<1>(q)==cute::get<1>(p)+1 && (offset&1)==0) {
        *reinterpret_cast<float2*>(y+offset)=make_float2(acc(i),acc(i+cute::_1{}));
      } else {
        y[offset]=acc(i);
        y[(blockIdx.y*M+cute::get<0>(q))*n+blockIdx.x*N+cute::get<1>(q)]=acc(i+cute::_1{});
      }
    });
  } else {
  o1_static_for<0,decltype(cute::size(acc))::value>([&](auto i) {
    y[(blockIdx.y*M+cute::get<0>(coords(i)))*n+blockIdx.x*N+cute::get<1>(coords(i))]=acc(i);
  });
  }
}

} // namespace o3_row_scale_epilogue_experiment
