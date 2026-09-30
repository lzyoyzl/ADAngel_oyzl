// Candidate29: paired G128 copies into a three-slot ring; independent scale/math.
// Separate device IR: never alter production or older candidate codegen.
#pragma once
namespace o3_paired_pipeline_experiment {
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
    // Each slot retains128B alignment; odd logical groups rotate by64B.
    // Padding keeps the full logical tile in bounds after that rotation.
    alignas(128) uint8_t low[Stages][M*Bytes+128], high[Stages][M*Bytes+128], weight[Stages][N*Bytes+128];
    float scales[(Cached?32:Stages*Groups)*N];
  };
};

// Issue one async group for a pair of adjacent logical G128 blocks.
// Global per-row span is128B; each logical group still has its own ring slot
// and scale. An odd tail predication never reads the missing second half.
template<int M,int N,bool Fast,bool DualScale>
__device__ __forceinline__ void o3_prefetch_pair(
    typename O3AmpereConfig<M,N,128,false,2,DualScale,3>::Storage& s,
    int first,const uint8_t* a,const uint8_t* w,const uint8_t* ws,
    int m,int n,int k,const float* as) {
  using C=O3AmpereConfig<M,N,128,false,2,DualScale,3>;
  static_assert(!DualScale || !Fast);
  typename C::template ByteLayout<M> la;
  typename C::template ByteLayout<N> lb;
  constexpr int PairBytes=128,Step=C::Threads*16;
  o1_static_for<0,M*PairBytes/Step>([&](auto chunk) {
    unsigned off=threadIdx.x*16+chunk*Step;
    unsigned row=off/PairBytes,col=off%PairBytes;
    int group=first+col/64,slot=group%3;
    if(group<k/128) {
      auto src=a+(blockIdx.y*M+row)*(k/2)+first*64+col;
      unsigned dst=(group&1)*64+la(row,col%64);
      copy16(s.low[slot]+dst,src);
      copy16(s.high[slot]+dst,src+m*(k/2));
    }
  });
  o1_static_for<0,N*PairBytes/Step>([&](auto chunk) {
    unsigned off=threadIdx.x*16+chunk*Step;
    unsigned row=off/PairBytes,col=off%PairBytes;
    int group=first+col/64,slot=group%3;
    if(group<k/128)
      copy16(s.weight[slot]+(group&1)*64+lb(row,col%64),
             w+(blockIdx.x*N+row)*(k/2)+first*64+col);
  });
  o1_static_for<0,2>([&](auto half) {
    int group=first+half,slot=group%3;
    if(group<k/128) {
      if constexpr(DualScale) {
        if(threadIdx.x<M) s.activation_scales[slot*M+threadIdx.x]=
            as[group*m+blockIdx.y*M+threadIdx.x];
        if(threadIdx.x<N) s.scales[slot*N+threadIdx.x]=
            reinterpret_cast<const float*>(ws)[group*n+blockIdx.x*N+threadIdx.x];
      } else if(threadIdx.x<N) {
        uint32_t code=ws[(blockIdx.x*N+threadIdx.x)*(k/128)+group];
        uint32_t bits=code?code<<23:0x00400000u;
        s.scales[slot*N+threadIdx.x]=__uint_as_float(Fast?((code-127u)<<23):bits);
      }
    }
  });
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template<int M,int N,int K,bool Fast,bool Cached=false,bool Magic=Fast,int WN=2,bool Merge=false,bool StaticCopy=false,bool PhasePair=false,bool Stream=false,bool BoundedOperands=false,bool VectorStore=false,bool VectorScale=false,bool DualScale=false,bool GroupMajorScale=false,int RoofTune=0,bool ActivationPower2=false,bool PrebiasActivationScale=false,bool AsyncScale=false,bool CombinedScalePanels=false,int Stages=2>
__device__ __forceinline__ void o3_body(
    const uint8_t* a,const uint8_t* w,const float* as,const uint8_t* ws,float* y,int m,int n,int k) {
  using C=O3AmpereConfig<M,N,K,Cached,WN,DualScale,Stages>;
  static_assert(Stages==3 && K==128 && !PhasePair && !Cached && !AsyncScale);
  static_assert(!DualScale || GroupMajorScale);
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
  o1_static_for<0,decltype(cute::size(rows))::value>([&](auto i) {
    if constexpr(!DualScale) rows(i)=as[blockIdx.y*M+cute::get<0>(coords(i))];
  });
  int physical_stage=0;
  auto make_low=[&](int slot) { return cute::make_tensor(cute::make_smem_ptr<cutlass::uint4b_t>(
      static_cast<void*>(s.low[slot]+(physical_stage&1)*64)),typename C::template NibbleLayout<M>{}); };
  auto make_high=[&](int slot) { return cute::make_tensor(cute::make_smem_ptr<cutlass::int4b_t>(
      static_cast<void*>(s.high[slot]+(physical_stage&1)*64)),typename C::template NibbleLayout<M>{}); };
  auto make_weight=[&](int slot) { return cute::make_tensor(cute::make_smem_ptr<cutlass::int4b_t>(
      static_cast<void*>(s.weight[slot]+(physical_stage&1)*64)),typename C::template NibbleLayout<N>{}); };
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
  o3_prefetch_pair<M,N,Fast,DualScale>(s,0,a,w,ws,m,n,k,as);
  auto process_stage=[&](int stage,auto slot) {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    // All warps finished the preceding group before its slot can be reused.
    __syncthreads();
    physical_stage=stage;
    // On odd groups two non-current slots are free. Fill the next pair while
    // computing this group. On even groups the remaining slot holds the next
    // ready group, so no pair may overwrite it.
    if((stage&1) && stage+1<k/128)
      o3_prefetch_pair<M,N,Fast,DualScale>(s,stage+1,a,w,ws,m,n,k,as);
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
                const int partial=pl(vi)+16*ph(vi);
                auto coord=coords(vi,mi,full_ni);
                float column,row;
                if constexpr(RoofTune&1) {column=columns(vi,ni);row=group_rows(vi,mi);}
                else {
                  column=s.scales[roof_scale_group*N+cute::get<1>(coord)];
                  if constexpr(DualScale) row=s.activation_scales[roof_scale_group*M+cute::get<0>(coord)];
                  else row=rows(vi,mi,full_ni);
                }
                float scale;
                if constexpr(Fast) scale=__uint_as_float(__float_as_uint(row)+__float_as_uint(column));
                else if constexpr(ActivationPower2) {
                  // Guarded positive normal A=2^e, normal W and normal product:
                  // exactly the original round_fp32(A*W), not magic-bias I2F.
                  if constexpr(PrebiasActivationScale)
                    scale=__uint_as_float(__float_as_uint(column)+__float_as_uint(row));
                  else scale=__uint_as_float(__float_as_uint(column)+__float_as_uint(row)-0x3f800000u);
                }
                else scale=__fmul_rn(row,column);
                acc(vi,mi,full_ni)=__fmaf_rn(float(partial),scale,acc(vi,mi,full_ni));
              });
            };
            if constexpr(RoofTune&2) {
              // Independent output atoms have separate integer dependency
              // chains. Interleave them before conversion; FP32 G128 order
              // remains unchanged for every output element.
              auto pls=cute::make_tensor<int>(cute::make_shape(cute::_4{},cute::Int<NAtoms>{}));
              auto phs=cute::make_tensor<int>(cute::make_shape(cute::_4{},cute::Int<NAtoms>{}));
              cute::clear(pls);cute::clear(phs);
              o1_static_for<0,NAtoms>([&](auto ni) {
                auto pl=pls(cute::_,ni),ph=phs(cute::_,ni);
                cute::gemm(LA{},pl,ra(cute::_,mi,cute::_0{}),br0(cute::_,ni,cute::_0{}),pl);
                cute::gemm(HA{},ph,rh(cute::_,mi,cute::_0{}),br0(cute::_,ni,cute::_0{}),ph);
              });
              o1_static_for<0,NAtoms>([&](auto ni) {
                auto pl=pls(cute::_,ni),ph=phs(cute::_,ni);
                cute::gemm(LA{},pl,ra1(cute::_,mi,cute::_0{}),br1(cute::_,ni,cute::_0{}),pl);
                cute::gemm(HA{},ph,rh1(cute::_,mi,cute::_0{}),br1(cute::_,ni,cute::_0{}),ph);
              });
              o1_static_for<0,NAtoms>([&](auto ni) {
                auto pl=pls(cute::_,ni),ph=phs(cute::_,ni);finish(ni,pl,ph);
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

} // namespace o3_paired_pipeline_experiment
