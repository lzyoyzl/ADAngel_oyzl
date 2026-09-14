// SM80 O3: preserve two native U4/S4 MMA paths and the ordered G128 FMA.
#pragma once
template<int N,bool Cached> struct O3ScaleCodeScratch {};
template<int N> struct O3ScaleCodeScratch<N,true> { uint8_t scale_codes[N*32]; };
template<int M,int N,int K,bool Cached=false,int WN=2>
struct O3AmpereConfig {
  static_assert(K==128||K==256);
  static constexpr int WM=M==32?2:4;
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
  struct alignas(128) Storage : O3ScaleCodeScratch<N,Cached> {
    alignas(128) uint8_t low[2][M*Bytes], high[2][M*Bytes], weight[2][N*Bytes];
    float scales[(Cached?32:2*Groups)*N];
  };
};

template<int M,int N,int K,bool Fast,bool Cached,int WN,bool StaticCopy>
__device__ __forceinline__ void o3_prefetch(typename O3AmpereConfig<M,N,K,Cached,WN>::Storage& s,
    int slot,int stage,const uint8_t* a,const uint8_t* w,const uint8_t* ws,int m,int k) {
  using C=O3AmpereConfig<M,N,K,Cached,WN>;
  typename C::template ByteLayout<M> la;
  typename C::template ByteLayout<N> lb;
  auto copy_a=[&](unsigned off) {
    unsigned row=off/C::Bytes,col=off%C::Bytes;
    auto src=a+(blockIdx.y*M+row)*(k/2)+stage*C::Bytes+col;
    copy16(s.low[slot]+la(row,col),src);
    copy16(s.high[slot]+la(row,col),src+m*(k/2));
  };
  auto copy_b=[&](unsigned off) {
    unsigned row=off/C::Bytes,col=off%C::Bytes;
    copy16(s.weight[slot]+lb(row,col),w+(blockIdx.x*N+row)*(k/2)+stage*C::Bytes+col);
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
  if(!Cached && threadIdx.x<N) {
    o1_static_for<0,C::Groups>([&](auto group) {
      uint32_t code=ws[(blockIdx.x*N+threadIdx.x)*(k/128)+stage*C::Groups+group];
      uint32_t bits=code?code<<23:0x00400000u;
      s.scales[(slot*C::Groups+group)*N+threadIdx.x]=__uint_as_float(Fast?((code-127u)<<23):bits);
    });
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template<int M,int N,int K,bool Fast,bool Cached=false,bool Magic=Fast,int WN=2,bool Merge=false,bool StaticCopy=false,bool PhasePair=false,bool Stream=false,bool BoundedOperands=false>
__device__ __forceinline__ void o3_body(
    const uint8_t* a,const uint8_t* w,const float* as,const uint8_t* ws,float* y,int m,int n,int k) {
  using C=O3AmpereConfig<M,N,K,Cached,WN>;
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
    rows(i)=as[blockIdx.y*M+cute::get<0>(coords(i))];
  });
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
  o3_prefetch<M,N,K,Fast,Cached,WN,StaticCopy>(s,0,0,a,w,ws,m,k);
  auto process_stage=[&](int stage,auto slot) {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    if(stage+1<k/K) o3_prefetch<M,N,K,Fast,Cached,WN,StaticCopy>(s,1-slot,stage+1,a,w,ws,m,k);
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
        constexpr int SliceN=WN*16;
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
        o1_static_for<0,N/SliceN>([&](auto nb) {
          cute::copy(SmallCopy{},sbc.partition_S(slice_b(nb,sub)),sd0);
          cute::copy(SmallCopy{},sbc.partition_S(slice_b(nb,sub+cute::_1{})),sd1);
          o1_static_for<0,decltype(cute::size<1>(acc))::value>([&](auto mi) {
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
              if constexpr(Fast) scale=__uint_as_float(__float_as_uint(rows(vi,mi,full_ni))+__float_as_uint(column));
              else scale=__fmul_rn(rows(vi,mi,full_ni),column);
              float value;
              if constexpr(Magic) value=__fadd_rn(__int_as_float(0x4b400000+partial),-12582912.0f);
              else value=float(partial);
              acc(vi,mi,full_ni)=__fmaf_rn(value,scale,acc(vi,mi,full_ni));
            });
          });
        });
          // All lanes take every slice. Keep next-slice shared loads after
          // this warp-scoped boundary, limiting cross-slice operand hoisting.
          if constexpr(BoundedOperands) __syncwarp();
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
  } else {
    for(int stage=0;stage<k/K;++stage) process_stage(stage,stage%2);
  }
  o1_static_for<0,decltype(cute::size(acc))::value>([&](auto i) {
    y[(blockIdx.y*M+cute::get<0>(coords(i)))*n+blockIdx.x*N+cute::get<1>(coords(i))]=acc(i);
  });
}

template<int M,int N,int K,bool Fast,bool Cached=false,bool Magic=Fast,int WN=2,bool Merge=false,bool StaticCopy=false,bool PhasePair=false,bool Stream=false>
__global__ __launch_bounds__(32*(M==32?2:4)*WN) void adangel_sm80_o3_swizzled(
    const uint8_t* a,const uint8_t* w,const float* as,const uint8_t* ws,float* y,int m,int n,int k) {
  o3_body<M,N,K,Fast,Cached,Magic,WN,Merge,StaticCopy,PhasePair,Stream>(a,w,as,ws,y,m,n,k);
}

// A distinct entry keeps the original one-argument launch policy unchanged.
template<int M,int N,int K,bool Fast,bool Cached,bool Magic,int WN,bool Merge,bool StaticCopy,bool PhasePair,bool Stream>
__global__ __launch_bounds__(256,2) void adangel_sm80_o3_swizzled_bound2(
    const uint8_t* a,const uint8_t* w,const float* as,const uint8_t* ws,float* y,int m,int n,int k) {
  static_assert(M==64 && N==128 && K==256 && WN==2 && (Stream || Merge));
  o3_body<M,N,K,Fast,Cached,Magic,WN,Merge,StaticCopy,PhasePair,Stream,true>(a,w,as,ws,y,m,n,k);
}

template<int M,int N,int K,bool Fast,bool Cached=false,bool Magic=Fast,int WN=2,bool Merge=false,bool StaticCopy=false,bool PhasePair=false,bool Stream=false,bool Bound2=false>
void o3_configure() {
  auto f=[]() {
    if constexpr(Bound2) return adangel_sm80_o3_swizzled_bound2<M,N,K,Fast,Cached,Magic,WN,Merge,StaticCopy,PhasePair,Stream>;
    else return adangel_sm80_o3_swizzled<M,N,K,Fast,Cached,Magic,WN,Merge,StaticCopy,PhasePair,Stream>;
  }();
  TORCH_CHECK(cudaFuncSetAttribute(f,cudaFuncAttributeMaxDynamicSharedMemorySize,
      sizeof(typename O3AmpereConfig<M,N,K,Cached,WN>::Storage))==cudaSuccess,"O3 shared memory opt-in failed");
  TORCH_CHECK(cudaFuncSetAttribute(f,cudaFuncAttributePreferredSharedMemoryCarveout,100)==cudaSuccess,"O3 carveout failed");
}
template<int M,int N,int K,bool Fast,bool Cached=false,bool Magic=Fast,int WN=2,bool Merge=false,bool StaticCopy=false,bool PhasePair=false,bool Stream=false,bool Bound2=false>
void o3_launch(const at::Tensor& a,const at::Tensor& w,const at::Tensor& as,const at::Tensor& ws,
    at::Tensor& out,cudaStream_t stream) {
  if constexpr(Bound2) {
  adangel_sm80_o3_swizzled_bound2<M,N,K,Fast,Cached,Magic,WN,Merge,StaticCopy,PhasePair,Stream><<<dim3(out.size(1)/N,out.size(0)/M),32*(M==32?2:4)*WN,
      sizeof(typename O3AmpereConfig<M,N,K,Cached,WN>::Storage),stream>>>(a.data_ptr<uint8_t>(),w.data_ptr<uint8_t>(),as.data_ptr<float>(),ws.data_ptr<uint8_t>(),out.data_ptr<float>(),out.size(0),out.size(1),w.size(1)*2);
  } else {
  adangel_sm80_o3_swizzled<M,N,K,Fast,Cached,Magic,WN,Merge,StaticCopy,PhasePair,Stream><<<dim3(out.size(1)/N,out.size(0)/M),32*(M==32?2:4)*WN,
      sizeof(typename O3AmpereConfig<M,N,K,Cached,WN>::Storage),stream>>>(a.data_ptr<uint8_t>(),w.data_ptr<uint8_t>(),
      as.data_ptr<float>(),ws.data_ptr<uint8_t>(),out.data_ptr<float>(),out.size(0),out.size(1),w.size(1)*2);
  }
}
