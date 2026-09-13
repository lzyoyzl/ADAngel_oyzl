// SM80 O3: preserve two native U4/S4 MMA paths and the ordered G128 FMA.
#pragma once
template<int M,int N,int K>
struct O3AmpereConfig {
  static_assert(K==128||K==256);
  static constexpr int Threads=256, Groups=K/128, Bytes=K/2;
  using Low=cutlass::uint4b_t;
  using Signed=cutlass::int4b_t;
  using Mma=cute::TiledMMA<cute::MMA_Atom<cute::SM80_16x8x64_S32U4S4S32_TN>,
      cute::Layout<cute::Shape<cute::_4,cute::_2,cute::_1>>,
      cute::Tile<cute::Int<M>,cute::Int<N>,cute::_64>>;
  using HighMma=cute::TiledMMA<cute::MMA_Atom<cute::SM80_16x8x64_S32S4S4S32_TN>,
      cute::Layout<cute::Shape<cute::_4,cute::_2,cute::_1>>,
      cute::Tile<cute::Int<M>,cute::Int<N>,cute::_64>>;
  template<int Rows> using ByteLayout=decltype(cute::composition(
      cute::Swizzle<K==256?3:2,4,3>{},cute::Layout<cute::Shape<cute::Int<Rows>,cute::Int<Bytes>>,
      cute::Stride<cute::Int<Bytes>,cute::_1>>{}));
  // The nibble layout is the byte layout with every bit position shifted by1.
  template<int Rows> using NibbleLayout=decltype(cute::composition(
      cute::Swizzle<K==256?3:2,5,3>{},cute::Layout<cute::Shape<cute::Int<Rows>,cute::Int<K>>,
      cute::Stride<cute::Int<K>,cute::_1>>{}));
  struct alignas(128) Storage {
    alignas(128) uint8_t low[2][M*Bytes], high[2][M*Bytes], weight[2][N*Bytes];
    float scales[2][Groups*N];
  };
};

template<int M,int N,int K,bool Fast>
__device__ __forceinline__ void o3_prefetch(typename O3AmpereConfig<M,N,K>::Storage& s,
    int slot,int stage,const uint8_t* a,const uint8_t* w,const uint8_t* ws,int m,int k) {
  using C=O3AmpereConfig<M,N,K>;
  typename C::template ByteLayout<M> la;
  typename C::template ByteLayout<N> lb;
  for(unsigned off=threadIdx.x*16;off<M*C::Bytes;off+=256*16) {
    unsigned row=off/C::Bytes,col=off%C::Bytes;
    auto src=a+(blockIdx.y*M+row)*(k/2)+stage*C::Bytes+col;
    copy16(s.low[slot]+la(row,col),src);
    copy16(s.high[slot]+la(row,col),src+m*(k/2));
  }
  for(unsigned off=threadIdx.x*16;off<N*C::Bytes;off+=256*16) {
    unsigned row=off/C::Bytes,col=off%C::Bytes;
    copy16(s.weight[slot]+lb(row,col),w+(blockIdx.x*N+row)*(k/2)+stage*C::Bytes+col);
  }
  if(threadIdx.x<N) {
    o1_static_for<0,C::Groups>([&](auto group) {
      uint32_t code=ws[(blockIdx.x*N+threadIdx.x)*(k/128)+stage*C::Groups+group];
      uint32_t bits=code?code<<23:0x00400000u;
      s.scales[slot][group*N+threadIdx.x]=__uint_as_float(Fast?((code-127u)<<23):bits);
    });
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template<int M,int N,int K,bool Fast>
__global__ __launch_bounds__(256) void adangel_sm80_o3_swizzled(
    const uint8_t* a,const uint8_t* w,const float* as,const uint8_t* ws,float* y,int m,int n,int k) {
  using C=O3AmpereConfig<M,N,K>;
  extern __shared__ __align__(128) uint8_t buf[];
  auto& s=*reinterpret_cast<typename C::Storage*>(buf);
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
  using LCopy=cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,cutlass::uint4b_t>;
  using SCopy=cute::Copy_Atom<cute::SM75_U32x4_LDSM_N,cutlass::int4b_t>;
  auto lc=cute::make_tiled_copy_A(LCopy{},mma).get_slice(threadIdx.x);
  auto hc=cute::make_tiled_copy_A(SCopy{},high_mma).get_slice(threadIdx.x);
  auto bc=cute::make_tiled_copy_B(SCopy{},mma).get_slice(threadIdx.x);
  auto ld=lc.retile_D(ra); auto hd=hc.retile_D(rh); auto bd=bc.retile_D(rb);
  o3_prefetch<M,N,K,Fast>(s,0,0,a,w,ws,m,k);
  for(int stage=0;stage<k/K;++stage) {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    int slot=stage%2;
    if(stage+1<k/K) o3_prefetch<M,N,K,Fast>(s,1-slot,stage+1,a,w,ws,m,k);
    o1_static_for<0,C::Groups>([&](auto group) {
      cute::clear(low);cute::clear(high);
      o1_static_for<0,2>([&](auto half) {
        auto sub=group*cute::_2{}+half;
        cute::copy(LCopy{},lc.partition_S(tile_a(make_low(slot),sub)),ld);
        cute::copy(SCopy{},hc.partition_S(tile_a(make_high(slot),sub)),hd);
        cute::copy(SCopy{},bc.partition_S(tile_b(make_weight(slot),sub)),bd);
        cute::gemm(mma,ra,rb,low);
        cute::gemm(high_mma,rh,rb,high);
      });
      o1_static_for<0,decltype(cute::size(acc))::value>([&](auto i) {
        int partial=low(i)+16*high(i);
        // Reconstructed INT8 x signed INT4 G128 bound:128*128*8=131072.
        // Exact bias conversion stays well inside the unit-ULP FP32 binade.
        float value,scale;
        float column=s.scales[slot][group*N+cute::get<1>(coords(i))];
        if constexpr(Fast) {
          value=__fadd_rn(__int_as_float(0x4b400000+partial),-12582912.0f);
          scale=__uint_as_float(__float_as_uint(rows(i))+__float_as_uint(column));
        } else {
          value=float(partial);scale=__fmul_rn(rows(i),column);
        }
        acc(i)=__fmaf_rn(value,scale,acc(i));
      });
    });
    // Next iteration's barrier protects this slot before it is overwritten.
  }
  o1_static_for<0,decltype(cute::size(acc))::value>([&](auto i) {
    y[(blockIdx.y*M+cute::get<0>(coords(i)))*n+blockIdx.x*N+cute::get<1>(coords(i))]=acc(i);
  });
}

template<int M,int N,int K,bool Fast>
void o3_configure() {
  auto f=adangel_sm80_o3_swizzled<M,N,K,Fast>;
  TORCH_CHECK(cudaFuncSetAttribute(f,cudaFuncAttributeMaxDynamicSharedMemorySize,
      sizeof(typename O3AmpereConfig<M,N,K>::Storage))==cudaSuccess,"O3 shared memory opt-in failed");
  TORCH_CHECK(cudaFuncSetAttribute(f,cudaFuncAttributePreferredSharedMemoryCarveout,100)==cudaSuccess,"O3 carveout failed");
}
template<int M,int N,int K,bool Fast>
void o3_launch(const at::Tensor& a,const at::Tensor& w,const at::Tensor& as,const at::Tensor& ws,
    at::Tensor& out,cudaStream_t stream) {
  adangel_sm80_o3_swizzled<M,N,K,Fast><<<dim3(out.size(1)/N,out.size(0)/M),256,
      sizeof(typename O3AmpereConfig<M,N,K>::Storage),stream>>>(a.data_ptr<uint8_t>(),w.data_ptr<uint8_t>(),
      as.data_ptr<float>(),ws.data_ptr<uint8_t>(),out.data_ptr<float>(),out.size(0),out.size(1),w.size(1)*2);
}
