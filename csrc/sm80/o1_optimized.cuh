// SM80-only O1 candidates. The legacy implementation remains an A/B baseline.
// No change to E2M1->INT8 mapping or the ordered K32 FMA recurrence.
#pragma once

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

template<int M, int N, int K, int WM, int WN>
__device__ __forceinline__ void o1_ampere_prefetch(
    typename O1AmpereConfig<M,N,K,WM,WN>::Storage& s, int slot, int stage,
    const int8_t* a, const int8_t* b, const uint8_t* ws, int k) {
  using C = O1AmpereConfig<M,N,K,WM,WN>;
  typename C::ASmem la;
  typename C::BSmem lb;
  for(int offset=threadIdx.x*16;offset<M*K;offset+=C::Threads*16) {
    int row=offset/K, col=offset%K;
    copy16(s.a[slot]+la(row,col),a+(int(blockIdx.y)*M+row)*k+stage*K+col);
  }
  for(int offset=threadIdx.x*16;offset<N*K;offset+=C::Threads*16) {
    int row=offset/K, col=offset%K;
    copy16(s.b[slot]+lb(row,col),b+(int(blockIdx.x)*N+row)*k+stage*K+col);
  }
  // One decode per CTA/column/K32. This small buffer is shared by output rows.
  for(int i=threadIdx.x;i<N*C::Groups;i+=C::Threads) {
    int col=i/C::Groups, sub=i%C::Groups;
    s.scales[slot][sub*N+col]=__fmul_rn(adangel::decode_ue8m0(
        ws[(int(blockIdx.x)*N+col)*(k/32)+stage*C::Groups+sub]),0.5f);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template<int M, int N, int K, int WM=4, int WN=2>
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
  CUTE_UNROLL
  for(int i=0;i<cute::size(rows);++i)
    rows(i)=as[int(blockIdx.y)*M+cute::get<0>(coords(i))];
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
  o1_ampere_prefetch<M,N,K,WM,WN>(s,0,0,a,b,ws,k);
  for(int stage=0;stage<k/K;++stage) {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    int slot=stage%2;
    if(stage+1<k/K) o1_ampere_prefetch<M,N,K,WM,WN>(s,1-slot,stage+1,a,b,ws,k);
    auto sa=cute::make_tensor(cute::make_smem_ptr(s.a[slot]),typename C::ASmem{});
    auto sb=cute::make_tensor(cute::make_smem_ptr(s.b[slot]),typename C::BSmem{});
    CUTE_UNROLL
    for(int sub=0;sub<C::Groups;++sub) {
      auto ta=cute::local_tile(sa,cute::make_shape(cute::Int<M>{},cute::_32{}),cute::make_coord(0,sub));
      auto tb=cute::local_tile(sb,cute::make_shape(cute::Int<N>{},cute::_32{}),cute::make_coord(0,sub));
      cute::copy(Copy{},ca.partition_S(ta),da);
      cute::copy(Copy{},cb.partition_S(tb),db);
      cute::clear(partial);
      cute::gemm(mma,ra,rb,partial);
      CUTE_UNROLL
      for(int i=0;i<cute::size(acc);++i) {
        float scale=__fmul_rn(rows(i),s.scales[slot][sub*N+cute::get<1>(coords(i))]);
        acc(i)=__fmaf_rn(float(partial(i)),scale,acc(i));
      }
    }
    __syncthreads();
  }
  CUTE_UNROLL
  for(int i=0;i<cute::size(acc);++i) {
    int row=int(blockIdx.y)*M+cute::get<0>(coords(i));
    int col=int(blockIdx.x)*N+cute::get<1>(coords(i));
    y[row*int(gridDim.x)*N+col]=acc(i);
  }
}

template<int M,int N,int K,int WM=4,int WN=2>
void o1_ampere_configure() {
  using C=O1AmpereConfig<M,N,K,WM,WN>;
  auto rc=cudaFuncSetAttribute(adangel_sm80_o1_swizzled<M,N,K,WM,WN>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(typename C::Storage));
  TORCH_CHECK(rc==cudaSuccess,cudaGetErrorString(rc));
}

template<int M,int N,int K,int WM=4,int WN=2>
void o1_ampere_launch(const at::Tensor& a,const at::Tensor& w,const at::Tensor& as,
    const at::Tensor& ws,at::Tensor& out,cudaStream_t stream) {
  using C=O1AmpereConfig<M,N,K,WM,WN>;
  adangel_sm80_o1_swizzled<M,N,K,WM,WN><<<dim3(out.size(1)/N,out.size(0)/M),C::Threads,sizeof(typename C::Storage),stream>>>(
      a.data_ptr<int8_t>(),w.data_ptr<int8_t>(),as.data_ptr<float>(),ws.data_ptr<uint8_t>(),
      out.data_ptr<float>(),a.size(1));
}
