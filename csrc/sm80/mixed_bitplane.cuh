// O9/O10: source -> signed fixed bitplanes, then native SM80 AND/POPC MMA.
// All G128 scales remain independent, even when a pipeline stage spans K256/512.
#pragma once

template<MixedKind Kind>
__device__ int mixed_fixed_element(const uint8_t* p,const uint8_t* m8,
    const uint8_t* m4,int i) {
  if constexpr(Kind==MixedKind::Nv4) return adangel::e2m1_to_q4((p[i/2]>>(4*(i&1)))&15);
  else if constexpr(Kind==MixedKind::Hif4) {
    const int g=i/128,o=i%128,j8=o/8,j4=o/4;
    const int e8=(m8[g*2+j8/8]>>(j8%8))&1,e4=(m4[g*4+j4/8]>>(j4%8))&1;
    const int code=(p[i/2]>>(4*(i&1)))&15;
    const float v=float((code&7)<<(e8+e4))*.25f;
    return __float2int_rn(code&8?-v:v);
  } else if constexpr(Kind==MixedKind::Mx8) return __float2int_rn(__fmul_rn(mixed_e4m3(p[i]),.25f));
  else {
    const int code=p[i],e=(code&31)>>3,m=code&7;
    const float v=e?float((8+m)<<e)*.25f:float(m)*.5f;
    return __float2int_rn(code&32?-v:v);
  }
}

template<MixedKind Kind,int Planes,bool Major>
__global__ void adangel_sm80_mixed_to_bitplanes(const uint8_t* p,const uint8_t* s,
    const float* ts,const uint8_t* m8,const uint8_t* m4,uint32_t* out,float* eff,
    int rows,int k) {
  const int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
  // K is a multiple of 128, so no partly active warp reaches ballot.
  if(i>=rows*k) return;
  const int q=mixed_fixed_element<Kind>(p,m8,m4,i),lane=threadIdx.x&31;
  const uint32_t u=uint32_t(q);  // low Q bits are the exact two's-complement code
  #pragma unroll
  for(int bit=0;bit<Planes;++bit) {
    const uint32_t word=__ballot_sync(0xffffffffu,(u>>bit)&1u);
    if(lane==0) out[bit*(rows*k/32)+i/32]=word;
  }
  if(i%128==0) {
    const int group=i/128,groups=k/128;
    const int dst=Major?(group%groups)*rows+group/groups:group;
    if constexpr(Kind==MixedKind::Hif4) eff[dst]=mixed_hif_scale(s[group]);
    else if constexpr(Kind==MixedKind::Mx8) eff[dst]=__fmul_rn(mixed_ue8m0(s[group]),4.f);
    else {
      float v=__fmul_rn(mixed_e4m3(s[group]),ts[0]);
      if constexpr(Kind==MixedKind::Nv6) v=__fmul_rn(v,.25f);
      eff[dst]=v;
    }
  }
}

struct MixedBitplanes {
  at::Tensor packed,scale;
  int planes;
  bool major;
  MixedBitplanes(const MixedSource& s,bool gm):planes(s.weight?4:(s.kind==MixedKind::Mx8?8:6)),major(gm) {
    packed=at::empty({planes,s.rows,s.k/32},s.payload.options().dtype(at::kInt));
    auto opt=s.payload.options().dtype(at::kFloat);
    scale=gm?at::empty({s.k/128,s.rows},opt).transpose(0,1):at::empty({s.rows,s.k/128},opt);
  }
};

void launch_mixed_bitplanes(const MixedSource& s,MixedBitplanes& d,cudaStream_t stream) {
  auto kind=[&](auto tag) {
    constexpr auto K=decltype(tag)::value;
    constexpr int P=(K==MixedKind::Nv4 || K==MixedKind::Hif4)?4:(K==MixedKind::Mx8?8:6);
    auto layout=[&](auto gm) {
      adangel_sm80_mixed_to_bitplanes<K,P,decltype(gm)::value><<<(s.rows*s.k+255)/256,256,0,stream>>>(
          s.payload.data_ptr<uint8_t>(),s.scale.data_ptr<uint8_t>(),
          s.tensor_scale.defined()?s.tensor_scale.data_ptr<float>():nullptr,
          s.micro8.defined()?s.micro8.data_ptr<uint8_t>():nullptr,
          s.micro4.defined()?s.micro4.data_ptr<uint8_t>():nullptr,
          reinterpret_cast<uint32_t*>(d.packed.data_ptr<int32_t>()),d.scale.data_ptr<float>(),s.rows,s.k);
    };
    if(d.major) layout(std::true_type{});else layout(std::false_type{});
  };
  switch(s.kind) {
    case MixedKind::Nv4:kind(std::integral_constant<MixedKind,MixedKind::Nv4>{});break;
    case MixedKind::Mx8:kind(std::integral_constant<MixedKind,MixedKind::Mx8>{});break;
    case MixedKind::Hif4:kind(std::integral_constant<MixedKind,MixedKind::Hif4>{});break;
    case MixedKind::Nv6:kind(std::integral_constant<MixedKind,MixedKind::Nv6>{});break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int AP,int N,int TK> struct MixedBinaryConfig {
  static constexpr int PW=TK/32, AW=AP*64*PW, BW=4*N*PW,G=TK/128;
  struct alignas(128) Storage {
    alignas(128) uint32_t a[2][AW],b[2][BW];
    float as[2][G][64],ws[2][G][N];
  };
};

py::dict convert_mixed_bitplanes(const py::dict& source,std::string scale_layout) {
  TORCH_CHECK(scale_layout=="row_major" || scale_layout=="group_major","invalid scale layout");
  const MixedSource s(source);c10::cuda::CUDAGuard guard(s.payload.device());
  cudaDeviceProp prop;check(cudaGetDeviceProperties(&prop,s.payload.get_device()));
  TORCH_CHECK(prop.major==8 && prop.minor==0,"requires A100 SM80");
  MixedBitplanes d(s,scale_layout=="group_major");
  launch_mixed_bitplanes(s,d,c10::cuda::getCurrentCUDAStream(s.payload.get_device()).stream());
  py::dict result;result["packed"]=d.packed;result["scale"]=d.scale;result["planes"]=d.planes;
  return result;
}

template<int AP,int N,int TK,bool Major,int Threads=256>
__device__ void mixed_binary_prefetch(typename MixedBinaryConfig<AP,N,TK>::Storage& s,
    int slot,int step,const uint32_t* a,const uint32_t* w,const float* as,const float* ws,
    int m,int n,int k) {
  using C=MixedBinaryConfig<AP,N,TK>;
  const int t=threadIdx.x,words=k/32,groups=k/128;
  for(int x=t*4;x<C::AW;x+=Threads*4) {
    int plane=x/(64*C::PW),row=(x/C::PW)%64,word=x%C::PW;
    copy16(&s.a[slot][x],a+(plane*m+blockIdx.y*64+row)*words+step*C::PW+word);
  }
  for(int x=t*4;x<C::BW;x+=Threads*4) {
    int plane=x/(N*C::PW),row=(x/C::PW)%N,word=x%C::PW;
    copy16(&s.b[slot][x],w+(plane*n+blockIdx.x*N+row)*words+step*C::PW+word);
  }
  for(int x=t;x<C::G*64;x+=Threads) {
    int g=step*C::G+x/64,row=blockIdx.y*64+x%64;
    s.as[slot][x/64][x%64]=as[Major?g*m+row:row*groups+g];
  }
  for(int x=t;x<C::G*N;x+=Threads) {
    int g=step*C::G+x/N,row=blockIdx.x*N+x%N;
    s.ws[slot][x/N][x%N]=ws[Major?g*n+row:row*groups+g];
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template<int AP,int N,int TK,bool Major,bool Horner=false,int Threads=256>
__global__ __launch_bounds__(Threads) void adangel_sm80_mixed_binary(
    const uint32_t* a,const uint32_t* w,const float* as,const float* ws,float* y,int m,int n,int k) {
  using C=MixedBinaryConfig<AP,N,TK>;
  using Op=cute::SM80_16x8x128_S32U1U1S32_TN_ANDPOPC;
  extern __shared__ __align__(128) unsigned char bytes[];
  auto& s=*reinterpret_cast<typename C::Storage*>(bytes);
  constexpr int R=N/(8*(Threads/128));
  const int lane=threadIdx.x&31,warp=threadIdx.x/32,i=lane&3,j=lane/4;
  const int row0=(warp%4)*16+j,row1=row0+8,wn=warp/4;
  float acc[R][4]={};
  mixed_binary_prefetch<AP,N,TK,Major,Threads>(s,0,0,a,w,as,ws,m,n,k);
  for(int step=0;step<k/TK;++step) {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    const int slot=step&1;
    if(step+1<k/TK) mixed_binary_prefetch<AP,N,TK,Major,Threads>(s,slot^1,step+1,a,w,as,ws,m,n,k);
    #pragma unroll
    for(int g=0;g<C::G;++g) {
      int32_t sum[2][R][4]={};
      uint32_t b[4][R];
      #pragma unroll
      for(int wb=0;wb<4;++wb) {
        #pragma unroll
        for(int r=0;r<R;++r) b[wb][r]=s.b[slot][(wb*N+(wn*R+r)*8+j)*C::PW+4*g+i];
      }
      if constexpr(Horner) {
        // Two's-complement Horner reconstruction. B's negative MSB starts
        // -popcount; following planes use the BMMA C addend (2*partial).
        // A planes are then folded identically. All intermediate integers are
        // bounded by G128 * 128 * 8, so no overflow/rounding is introduced.
        #pragma unroll
        for(int ab=AP-1;ab>=0;--ab) {
          const uint32_t a0=s.a[slot][(ab*64+row0)*C::PW+4*g+i];
          const uint32_t a1=s.a[slot][(ab*64+row1)*C::PW+4*g+i];
          #pragma unroll
          for(int r=0;r<R;++r) {
            uint32_t d0,d1,d2,d3;
            Op::fma(d0,d1,d2,d3,a0,a1,b[3][r],0u,0u,0u,0u);
            d0=uint32_t(-int(d0));d1=uint32_t(-int(d1));d2=uint32_t(-int(d2));d3=uint32_t(-int(d3));
            #pragma unroll
            for(int wb=2;wb>=0;--wb)
              Op::fma(d0,d1,d2,d3,a0,a1,b[wb][r],d0<<1,d1<<1,d2<<1,d3<<1);
            if(ab==AP-1) {
              sum[0][r][0]=-int(d0);sum[0][r][1]=-int(d1);sum[0][r][2]=-int(d2);sum[0][r][3]=-int(d3);
            } else {
              sum[0][r][0]=2*sum[0][r][0]+int(d0);sum[0][r][1]=2*sum[0][r][1]+int(d1);
              sum[0][r][2]=2*sum[0][r][2]+int(d2);sum[0][r][3]=2*sum[0][r][3]+int(d3);
            }
          }
        }
      } else {
      #pragma unroll
      for(int ab=0;ab<AP;++ab) {
        const uint32_t a0=s.a[slot][(ab*64+row0)*C::PW+4*g+i];
        const uint32_t a1=s.a[slot][(ab*64+row1)*C::PW+4*g+i];
        #pragma unroll
        for(int wb=0;wb<4;++wb) {
          constexpr int WSign=3;
          const int coeff=(ab==AP-1?-(1<<ab):(1<<ab))*(wb==WSign?-8:(1<<wb));
          #pragma unroll
          for(int r=0;r<R;++r) {
            uint32_t d0,d1,d2,d3;
            Op::fma(d0,d1,d2,d3,a0,a1,b[wb][r],0u,0u,0u,0u);
            sum[ab%2][r][0]+=coeff*int(d0);sum[ab%2][r][1]+=coeff*int(d1);
            sum[ab%2][r][2]+=coeff*int(d2);sum[ab%2][r][3]+=coeff*int(d3);
          }
        }
      }
      }
      #pragma unroll
      for(int r=0;r<R;++r) {
        const int col=(wn*R+r)*8+2*i;
        #pragma unroll
        for(int v=0;v<4;++v) {
          float scale=__fmul_rn(s.as[slot][g][v<2?row0:row1],s.ws[slot][g][col+(v&1)]);
          acc[r][v]=__fmaf_rn(float(sum[0][r][v]+sum[1][r][v]),scale,acc[r][v]);
        }
      }
    }
    // Every consumer finishes reading before a slot can be recycled.
    __syncthreads();
  }
  #pragma unroll
  for(int r=0;r<R;++r) {
    const int col=blockIdx.x*N+(wn*R+r)*8+2*i;
    y[(blockIdx.y*64+row0)*n+col]=acc[r][0];y[(blockIdx.y*64+row0)*n+col+1]=acc[r][1];
    y[(blockIdx.y*64+row1)*n+col]=acc[r][2];y[(blockIdx.y*64+row1)*n+col+1]=acc[r][3];
  }
}

void launch_mixed_binary(const MixedBitplanes& a,const MixedBitplanes& w,at::Tensor& y,
    int m,int n,int k,int tn,int tk,cudaStream_t stream,bool configure=false,bool horner=false,bool wide=false) {
  auto planes=[&](auto ap) {auto layout=[&](auto gm) {auto tile=[&](auto nc,auto kc,auto hor,auto threads) {
    constexpr int A=decltype(ap)::value,N=decltype(nc)::value,K=decltype(kc)::value;
    constexpr bool G=decltype(gm)::value;
    constexpr size_t S=sizeof(typename MixedBinaryConfig<A,N,K>::Storage);
    constexpr int T=decltype(threads)::value;
    auto kernel=adangel_sm80_mixed_binary<A,N,K,G,decltype(hor)::value,T>;
    if(configure) {check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,int(S)));return;}
    kernel<<<dim3(n/N,m/64),T,S,stream>>>(
        reinterpret_cast<const uint32_t*>(a.packed.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(w.packed.data_ptr<int32_t>()),
        a.scale.data_ptr<float>(),w.scale.data_ptr<float>(),y.data_ptr<float>(),m,n,k);
  };
    if(wide) {
      if(horner) tile(std::integral_constant<int,128>{},std::integral_constant<int,256>{},std::true_type{},std::integral_constant<int,512>{});
      else tile(std::integral_constant<int,128>{},std::integral_constant<int,256>{},std::false_type{},std::integral_constant<int,512>{});
    } else if(horner) tile(std::integral_constant<int,128>{},std::integral_constant<int,256>{},std::true_type{},std::integral_constant<int,256>{});
    else if(tn==128) tile(std::integral_constant<int,128>{},std::integral_constant<int,256>{},std::false_type{},std::integral_constant<int,256>{});
    else if(tk==512) tile(std::integral_constant<int,64>{},std::integral_constant<int,512>{},std::false_type{},std::integral_constant<int,256>{});
    else tile(std::integral_constant<int,64>{},std::integral_constant<int,128>{},std::false_type{},std::integral_constant<int,256>{});
  };if(a.major) layout(std::true_type{});else layout(std::false_type{});
  };if(a.planes==8) planes(std::integral_constant<int,8>{});else planes(std::integral_constant<int,6>{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
