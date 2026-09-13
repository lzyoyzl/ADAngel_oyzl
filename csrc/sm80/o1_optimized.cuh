// SM80-only O1 candidates. The legacy implementation remains an A/B baseline.
// No change to E2M1->INT8 mapping or the ordered K32 FMA recurrence.
#pragma once

// Keep fragment indices as CuTe integral constants through layout algebra,
// rather than generating dynamic index arithmetic and only unrolling later.
template<int I,int End,class F>
__device__ __forceinline__ void o1_static_for(F const& f) {
  if constexpr(I<End) {
    f(cute::Int<I>{});
    o1_static_for<I+1,End>(f);
  }
}

template<int M, int N, int K, int WM = 4, int WN = 2>
struct O1AmpereConfig {
  static constexpr int Threads = WM * WN * 32;
  static constexpr int Stages = 2;
  static constexpr int Groups = K / 32;
  static_assert(K == 64 || K == 128);
  using Mma = cute::TiledMMA<cute::MMA_Atom<cute::SM80_16x8x32_S32S8S8S32_TN>,
      cute::Layout<cute::Shape<cute::Int<WM>,cute::Int<WN>,cute::_1>>,
      cute::Tile<cute::Int<M>,cute::Int<N>,cute::_32>>;
  // XOR 16-byte sectors across rows. cp.async and LDSM use the SAME layout.
  using ASmem = decltype(cute::composition(cute::Swizzle<K == 128 ? 3 : 2,4,3>{},
      cute::Layout<cute::Shape<cute::Int<M>,cute::Int<K>>,
                   cute::Stride<cute::Int<K>,cute::_1>>{}));
  using BSmem = decltype(cute::composition(cute::Swizzle<K == 128 ? 3 : 2,4,3>{},
      cute::Layout<cute::Shape<cute::Int<N>,cute::Int<K>>,
                   cute::Stride<cute::Int<K>,cute::_1>>{}));
  struct alignas(128) Storage {
    alignas(128) int8_t a[Stages][M*K];
    alignas(128) int8_t b[Stages][N*K];
    float scales[Stages][Groups*N];
  };
};

template<int M, int N, int K, int WM, int WN, bool ExponentScale=false>
__device__ __forceinline__ void o1_ampere_prefetch(
    typename O1AmpereConfig<M,N,K,WM,WN>::Storage& s, int slot, int stage,
    const int8_t* a, const int8_t* b, const uint8_t* ws, int k) {
  using C = O1AmpereConfig<M,N,K,WM,WN>;
  typename C::ASmem la;
  typename C::BSmem lb;
  for(unsigned offset=threadIdx.x*16;offset<M*K;offset+=C::Threads*16) {
    unsigned row=offset/K, col=offset%K;
    copy16(s.a[slot]+la(row,col),a+(int(blockIdx.y)*M+row)*k+stage*K+col);
  }
  for(unsigned offset=threadIdx.x*16;offset<N*K;offset+=C::Threads*16) {
    unsigned row=offset/K, col=offset%K;
    copy16(s.b[slot]+lb(row,col),b+(int(blockIdx.x)*N+row)*k+stage*K+col);
  }
  // One decode per CTA/column/K32. This small buffer is shared by output rows.
  for(int col=threadIdx.x;col<N;col+=C::Threads) {
    const auto* src=ws+(int(blockIdx.x)*N+col)*(k/32)+stage*C::Groups;
    // K/32 and stage*Groups align this vector load. Avoid generic ldexpf,
    // retaining exact code 0/1 subnormal behavior; code 255 is rejected by host.
    uint32_t codes;
    if constexpr(C::Groups==4) codes=*reinterpret_cast<const uint32_t*>(src);
    else codes=*reinterpret_cast<const uint16_t*>(src);
    CUTE_UNROLL
    for(int sub=0;sub<C::Groups;++sub) {
      uint32_t code=(codes>>(8*sub))&255;
      uint32_t bits=code>=2 ? ((code-1)<<23) : (0x00200000u<<code);
      s.scales[slot][sub*N+col]=__uint_as_float(ExponentScale ? ((code-128u)<<23) : bits);
    }
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template<int M, int N, int K, int WM=4, int WN=2, bool ExponentScale=false, bool PairMma=false, bool MagicCast=false>
__global__ __launch_bounds__(WM*WN*32) void adangel_sm80_o1_swizzled(
    const int8_t* a, const int8_t* b, const float* as,
    const uint8_t* ws, float* y, int k) {
  using C=O1AmpereConfig<M,N,K,WM,WN>;
  extern __shared__ __align__(128) uint8_t storage[];
  auto& s=*reinterpret_cast<typename C::Storage*>(storage);
  typename C::Mma mma;
  auto thr=mma.get_slice(threadIdx.x);
  auto coords=thr.partition_C(cute::make_identity_tensor(
      cute::make_shape(cute::Int<M>{},cute::Int<N>{})));
  auto partial=thr.make_fragment_C(coords);
  auto acc=cute::make_fragment_like<float>(partial);
  auto rows=cute::make_fragment_like<float>(partial);
  cute::clear(acc);
  o1_static_for<0,decltype(cute::size(rows))::value>([&](auto i) {
    rows(i)=as[int(blockIdx.y)*M+cute::get<0>(coords(i))];
  });
  auto sa0=cute::make_tensor(cute::make_smem_ptr(s.a[0]),typename C::ASmem{});
  auto sb0=cute::make_tensor(cute::make_smem_ptr(s.b[0]),typename C::BSmem{});
  auto ta0=cute::local_tile(sa0,cute::make_shape(cute::Int<M>{},cute::_32{}),cute::make_coord(0,0));
  auto tb0=cute::local_tile(sb0,cute::make_shape(cute::Int<N>{},cute::_32{}),cute::make_coord(0,0));
  auto ra=thr.partition_fragment_A(ta0);
  auto rb=thr.partition_fragment_B(tb0);
  using Copy=cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,int8_t>;
  auto ca=cute::make_tiled_copy_A(Copy{},mma).get_slice(threadIdx.x);
  auto cb=cute::make_tiled_copy_B(Copy{},mma).get_slice(threadIdx.x);
  auto da=ca.retile_D(ra);
  auto db=cb.retile_D(rb);
  o1_ampere_prefetch<M,N,K,WM,WN,ExponentScale>(s,0,0,a,b,ws,k);
  for(int stage=0;stage<k/K;++stage) {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    int slot=stage%2;
    if(stage+1<k/K) o1_ampere_prefetch<M,N,K,WM,WN,ExponentScale>(s,1-slot,stage+1,a,b,ws,k);
    auto sa=cute::make_tensor(cute::make_smem_ptr(s.a[slot]),typename C::ASmem{});
    auto sb=cute::make_tensor(cute::make_smem_ptr(s.b[slot]),typename C::BSmem{});
    auto multiply=[&](auto& target,auto sub) {
      auto ta=cute::local_tile(sa,cute::make_shape(cute::Int<M>{},cute::_32{}),cute::make_coord(cute::_0{},sub));
      auto tb=cute::local_tile(sb,cute::make_shape(cute::Int<N>{},cute::_32{}),cute::make_coord(cute::_0{},sub));
      cute::copy(Copy{},ca.partition_S(ta),da);
      cute::copy(Copy{},cb.partition_S(tb),db);
      cute::clear(target);
      cute::gemm(mma,ra,rb,target);
    };
    auto accumulate=[&](auto const& partial,auto sub) {
      o1_static_for<0,decltype(cute::size(acc))::value>([&](auto i) {
        float column=s.scales[slot][sub*N+cute::get<1>(coords(i))];
        // Host proves normal positive row scales and normal finite products.
        // Power-of-two multiplication then changes only the exponent bits,
        // exactly matching FMUL.rn without changing the ordered FMA recurrence.
        float scale;
        if constexpr(ExponentScale)
          scale=__uint_as_float(__float_as_uint(rows(i))+__float_as_uint(column));
        else scale=__fmul_rn(rows(i),column);
        float value;
        if constexpr(MagicCast) {
          // |sum32(A8 * 2*E2M1)| <= 32*128*12 = 49152.
          // Around 1.5*2^23 every float is an integer with ULP 1. Adding the
          // signed partial to its bit pattern and subtracting the bias is
          // EXACT (including negative partials); no I2F/XU instruction needed.
          value=__fadd_rn(__int_as_float(0x4b400000+partial(i)),-12582912.0f);
        } else value=float(partial(i));
        acc(i)=__fmaf_rn(value,scale,acc(i));
      });
    };
    if constexpr(PairMma) {
      auto pending=thr.make_fragment_C(coords);
      o1_static_for<0,C::Groups/2>([&](auto pair) {
        auto sub=pair*cute::_2{};
        multiply(partial,sub);
        multiply(pending,sub+cute::_1{});
        accumulate(partial,sub);
        accumulate(pending,sub+cute::_1{});
      });
    } else {
      o1_static_for<0,C::Groups>([&](auto sub) {
        multiply(partial,sub);
        accumulate(partial,sub);
      });
    }
    // No trailing CTA barrier: the next iteration's wait+barrier executes
    // before prefetch can overwrite this slot, and therefore protects all
    // current readers. The final iteration has no slot reuse.
  }
  o1_static_for<0,decltype(cute::size(acc))::value>([&](auto i) {
    int row=int(blockIdx.y)*M+cute::get<0>(coords(i));
    int col=int(blockIdx.x)*N+cute::get<1>(coords(i));
    y[row*int(gridDim.x)*N+col]=acc(i);
  });
}

template<int M,int N,int K,int WM=4,int WN=2,bool ExponentScale=false,bool PairMma=false,bool MagicCast=false>
void o1_ampere_configure() {
  using C=O1AmpereConfig<M,N,K,WM,WN>;
  auto rc=cudaFuncSetAttribute(adangel_sm80_o1_swizzled<M,N,K,WM,WN,ExponentScale,PairMma,MagicCast>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(typename C::Storage));
  TORCH_CHECK(rc==cudaSuccess,cudaGetErrorString(rc));
  rc=cudaFuncSetAttribute(adangel_sm80_o1_swizzled<M,N,K,WM,WN,ExponentScale,PairMma,MagicCast>,
      cudaFuncAttributePreferredSharedMemoryCarveout,100);
  TORCH_CHECK(rc==cudaSuccess,cudaGetErrorString(rc));
}

template<int M,int N,int K,int WM=4,int WN=2,bool ExponentScale=false,bool PairMma=false,bool MagicCast=false>
void o1_ampere_launch(const at::Tensor& a,const at::Tensor& w,const at::Tensor& as,
    const at::Tensor& ws,at::Tensor& out,cudaStream_t stream) {
  using C=O1AmpereConfig<M,N,K,WM,WN>;
  adangel_sm80_o1_swizzled<M,N,K,WM,WN,ExponentScale,PairMma,MagicCast><<<dim3(out.size(1)/N,out.size(0)/M),C::Threads,sizeof(typename C::Storage),stream>>>(
      a.data_ptr<int8_t>(),w.data_ptr<int8_t>(),as.data_ptr<float>(),ws.data_ptr<uint8_t>(),
      out.data_ptr<float>(),a.size(1));
}
