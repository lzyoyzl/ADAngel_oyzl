// v68 independent online metadata experiment. No production dispatch changes.
#include "roof_producer_warp_driver.cpp"
#include "roof_vector_conversion_impl.cuh"
#include <climits>

namespace o78_prepare {
constexpr uint64_t Bound=uint64_t(INT32_MAX)*INT32_MAX;
constexpr uint64_t Cap=Bound+1;
__device__ uint64_t sat_add(uint64_t a,uint64_t b) {return min(a+b,Cap);}
__device__ uint64_t sat_term(uint64_t square,uint64_t factor) {
  const uint64_t f2=factor*factor;
  // Detect multiplication overflow BEFORE admitting its low 64 bits.
  const uint64_t hi=__umul64hi(f2,square),lo=f2*square;
  return hi || lo>Cap?Cap:lo;
}
template<int Kind> __device__ void decode(unsigned c,int& mant,int& exp,bool& bad) {
  bad=false;
  if constexpr(Kind==0) {bad=c==255;mant=1;exp=int(c)-127;}
  else if constexpr(Kind==1) {
    bad=c>126;const int field=c>>3,frac=c&7;
    mant=field?8+frac:frac;exp=field?field-10:-9;
  } else {bad=c==255;mant=4+(c&3);exp=int(c>>2)-50;}
  if(mant) {const int z=__ffs(mant)-1;mant>>=z;exp+=z;}
}
__device__ unsigned warp_max(unsigned x) {
  for(int d=16;d;d>>=1) x=max(x,__shfl_down_sync(0xffffffff,x,d));
  return __shfl_sync(0xffffffff,x,0);
}
__device__ uint64_t warp_sum_sat(uint64_t x) {
  for(int d=16;d;d>>=1) x=sat_add(x,__shfl_down_sync(0xffffffff,x,d));
  return x;
}

template<int Kind> __device__ void row(
    const uint8_t* packed,const uint8_t* codes,float multiplier,int32_t* factors,
    float* bases,uint64_t* norms,int32_t* maxima,uint32_t* status,int rows,int activation) {
  __shared__ int f[32],verdict,factor_bad;
  __shared__ uint64_t partial[4];
  const int r=blockIdx.x,lane=threadIdx.x&31,warp=threadIdx.x>>5;
  if(warp==0) {
    int mant,exp;bool bad;decode<Kind>(codes[r*32+lane],mant,exp,bad);
    int amin=mant?exp:INT_MAX;
    for(int d=16;d;d>>=1) amin=min(amin,__shfl_down_sync(0xffffffff,amin,d));
    amin=__shfl_sync(0xffffffff,amin,0);if(amin==INT_MAX) amin=0;
    const int delta=exp-amin;
    bool overflow=mant && (delta>=31 || uint64_t(mant)>(uint64_t(INT32_MAX)>>delta));
    const int factor=(!bad && !overflow && mant)?int(uint64_t(mant)<<delta):0;
    const unsigned bs=__float_as_uint(multiplier),be=(bs>>23)&255;
    const int newexp=int(be)+amin;
    const bool basebad=(bs>>31) || be==0 || be==255 || newexp<=0 || newexp>=255;
    unsigned v=bad?2u:(overflow || basebad?1u:0u);v=warp_max(v);
    if(lane==0) {
      verdict=int(v);status[r]=v;
      bases[r]=v?1.f:__uint_as_float((bs&0x807fffffu)|(unsigned(newexp)<<23));
    }
    f[lane]=factor;
    factors[lane*rows+r]=v?0:factor;
    const unsigned any_overflow=warp_max(unsigned(overflow));
    if(lane==0) factor_bad=int(any_overflow);
    const unsigned fm=warp_max(overflow?unsigned(INT32_MAX):unsigned(factor));
    if(lane==0) maxima[r]=v==2?0:int(fm);
  }
  __syncthreads();
  uint64_t norm=0;
  if(verdict!=2) {
    for(int g=warp;g<32;g+=4) {
      const size_t offset=(size_t(g)*rows+r)*64+lane*2;
      const unsigned low=reinterpret_cast<const uint16_t*>(packed+offset)[0];
      unsigned high=0;
      if(activation) high=reinterpret_cast<const uint16_t*>(packed+size_t(rows)*32*64+offset)[0];
      unsigned sq=0;
      #pragma unroll
      for(int j=0;j<4;++j) {
        int q=int((low>>(4*j))&15);
        if(activation) {q|=((high>>(4*j))&15)<<4;q=(q^128)-128;}
        else q=(q^8)-8;
        sq+=q*q;
      }
      for(int d=16;d;d>>=1) sq+=__shfl_down_sync(0xffffffff,sq,d);
      if(lane==0) norm=sat_add(norm,sat_term(sq,unsigned(f[g])));
    }
  }
  if(lane==0) partial[warp]=norm;
  __syncthreads();
  if(threadIdx.x==0) {
    uint64_t total=0;for(int i=0;i<4;++i) total=sat_add(total,partial[i]);
    // Overflowed factors already select fallback; a capped sentinel is sufficient.
    norms[r]=verdict==2?0:(factor_bad?Cap:total);
  }
}
} // namespace o78_prepare

#define ROW_ENTRY(Name,Kind) \
extern "C" __global__ __launch_bounds__(128) void Name( \
 const uint8_t* p,const uint8_t* c,float mult,int32_t* f,float* b,uint64_t* norm, \
 int32_t* mx,uint32_t* st,int rows,int activation) { \
 o78_prepare::row<Kind>(p,c,mult,f,b,norm,mx,st,rows,activation); }
ROW_ENTRY(adangel_o78_prepare_ue8m0,0)
ROW_ENTRY(adangel_o78_prepare_e4m3,1)
ROW_ENTRY(adangel_o78_prepare_e6m2,2)
#undef ROW_ENTRY

extern "C" __global__ __launch_bounds__(128) void adangel_o78_prepare_cta_guard(
    const uint64_t* an,const uint64_t* wn,const int32_t* am,const int32_t* wm,
    const uint32_t* ast,const uint32_t* wst,uint32_t* result,int m,int n) {
  __shared__ uint64_t as[4],ws[4];
  __shared__ unsigned af[4],wf[4],st[4];
  const int t=threadIdx.x,lane=t&31,warp=t>>5;
  const int r=blockIdx.y*64+t,c=blockIdx.x*128+t;
  uint64_t a=t<64?an[r]:0,w=wn[c];
  unsigned ax=t<64?unsigned(am[r]):0,wx=unsigned(wm[c]);
  unsigned s=max(t<64?ast[r]:0,wst[c]);
  for(int d=16;d;d>>=1) {
    a=max(a,__shfl_down_sync(0xffffffff,a,d));w=max(w,__shfl_down_sync(0xffffffff,w,d));
    ax=max(ax,__shfl_down_sync(0xffffffff,ax,d));wx=max(wx,__shfl_down_sync(0xffffffff,wx,d));
    s=max(s,__shfl_down_sync(0xffffffff,s,d));
  }
  if(lane==0) {as[warp]=a;ws[warp]=w;af[warp]=ax;wf[warp]=wx;st[warp]=s;}
  __syncthreads();
  if(t==0) {
    a=w=0;ax=wx=s=0;
    for(int i=0;i<4;++i) {a=max(a,as[i]);w=max(w,ws[i]);ax=max(ax,af[i]);wx=max(wx,wf[i]);s=max(s,st[i]);}
    if(s==0 && (uint64_t(ax)*wx>INT32_MAX || (w && a>o78_prepare::Bound/w))) s=1;
    result[blockIdx.y*(n/128)+blockIdx.x]=s;
  }
}

namespace {
using Kind=adangel_sm80_experiment::GroupedSourceKind;
void runtime_check(cudaError_t value) {
  if(value!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(value));
}
struct Online {
  const uint64_t *sa,*sw,*v;int variant,m,n;cudaStream_t stream;
  Online(int vv,const uint64_t* aa,const uint64_t* ww,const uint64_t* state,int mm,int nn,void* ss)
    :sa(aa),sw(ww),v(state),variant(vv),m(mm),n(nn),stream(reinterpret_cast<cudaStream_t>(ss)) {
    if((vv!=7 && vv!=8) || !aa || !ww || !state || mm<=0 || nn<=0 || mm%64 || nn%128 ||
       int64_t(mm)*4096>INT32_MAX || int64_t(nn)*4096>INT32_MAX || int64_t(mm)*nn>INT32_MAX ||
       mm/64>65535 || nn/128>65535) throw std::runtime_error("invalid online shape/variant");
    for(int i=0;i<16;++i) if(!v[i]) throw std::runtime_error("null preallocated state");
    for(auto s:{sa,sw}) if(!s[0] || !s[1]) throw std::runtime_error("null source payload/scale");
    if(!sw[2] && vv==7 || !sa[2] && vv==8 || vv==8 && (!sw[3] || !sw[4]))
      throw std::runtime_error("missing source metadata");
  }
  template<Kind K> void convert(const uint64_t* s,bool act) {
    const int rows=act?m:n;
    vector_probe::adangel_sm80_vector_fixed_conversion<K,16>
      <<<dim3((rows+31)/32,32),256,0,stream>>>(
        reinterpret_cast<const uint8_t*>(s[0]),reinterpret_cast<const uint8_t*>(s[1]),
        reinterpret_cast<const float*>(s[2]),reinterpret_cast<const uint8_t*>(s[3]),
        reinterpret_cast<const uint8_t*>(s[4]),reinterpret_cast<uint8_t*>(v[act?0:1]),
        reinterpret_cast<float*>(v[act?2:3]),rows,32);
  }
  template<int K> void metadata(const uint64_t* s,bool act,float mult) {
    const int rows=act?m:n;
    #define PTR(T,I) reinterpret_cast<T*>(v[I])
    #define META_LAUNCH(Function) Function<<<rows,128,0,stream>>>( \
      PTR(const uint8_t,act?0:1),reinterpret_cast<const uint8_t*>(s[1]),mult, \
      PTR(int32_t,act?4:5),PTR(float,act?6:7),PTR(uint64_t,act?8:9), \
      PTR(int32_t,act?10:11),PTR(uint32_t,act?12:13),rows,int(act))
    if constexpr(K==0) {META_LAUNCH(adangel_o78_prepare_ue8m0);}
    else if constexpr(K==1) {META_LAUNCH(adangel_o78_prepare_e4m3);}
    else {META_LAUNCH(adangel_o78_prepare_e6m2);}
    #undef META_LAUNCH
    #undef PTR
  }
  // Scalar tensor multiplier is passed as a separate pre-read host scalar.
  // Source tensor_scale itself is still consumed by the vector converter.
  float a_mult=1.f,w_mult=1.f;
  void weight(bool candidate) {
    if(variant==7) {convert<Kind::Nv4>(sw,false);if(candidate) metadata<1>(sw,false,w_mult);}
    else {convert<Kind::Hif4>(sw,false);if(candidate) metadata<2>(sw,false,w_mult);}
  }
  void activation(bool candidate) {
    if(variant==7) {convert<Kind::Mx8>(sa,true);if(candidate) metadata<0>(sa,true,a_mult);}
    else {convert<Kind::Nv6>(sa,true);if(candidate) metadata<1>(sa,true,a_mult);}
    if(candidate) adangel_o78_prepare_cta_guard<<<dim3(n/128,m/64),128,0,stream>>>(
      reinterpret_cast<const uint64_t*>(v[8]),reinterpret_cast<const uint64_t*>(v[9]),
      reinterpret_cast<const int32_t*>(v[10]),reinterpret_cast<const int32_t*>(v[11]),
      reinterpret_cast<const uint32_t*>(v[12]),reinterpret_cast<const uint32_t*>(v[13]),
      reinterpret_cast<uint32_t*>(v[14]),m,n);
  }
};
}

extern "C" int roof_o78_gpu_prepare(int variant,const uint64_t* sa,const uint64_t* sw,
    const uint64_t* state,int m,int n,float a_mult,float w_mult,void* stream) {
  try {Online x(variant,sa,sw,state,m,n,stream);x.a_mult=a_mult;x.w_mult=w_mult;
    x.weight(true);x.activation(true);runtime_check(cudaGetLastError());return 0;
  } catch(const std::exception& e) {error=e.what();return 1;}
}

extern "C" int roof_o78_gpu_benchmark(void* handle,int variant,int candidate,int mode,
    const uint64_t* sa,const uint64_t* sw,const uint64_t* state,int m,int n,
    float a_mult,float w_mult,int warmup,int repeats,int inner,void* stream_ptr,float* times) {
  try {
    if(!handle || !times || candidate<0 || candidate>1 || mode<0 || mode>3 ||
       warmup<0 || repeats<1 || repeats>100000 || inner<1 || inner>10000)
      throw std::runtime_error("invalid online timing configuration");
    Online x(variant,sa,sw,state,m,n,stream_ptr);x.a_mult=a_mult;x.w_mult=w_mult;
    auto* p=static_cast<Probe*>(handle);auto stream=reinterpret_cast<CUstream>(stream_ptr);
    uint64_t v[16];for(int i=0;i<16;++i) v[i]=state[i];int k=4096;
    void* args[]={v,v+1,v+2,v+3,v+4,v+5,v+6,v+7,v+14,v+15,&m,&n,&k};
    auto gemm=[&]() {check(cuLaunchKernel(p->function,n/128,m/64,1,128,1,1,p->smem,stream,args,nullptr));};
    auto direct=[&]() {if(mode==2)x.weight(candidate);if(mode==2 || mode==3)x.activation(candidate);gemm();};
    // Allocate Events before any timed region, initialize cached inputs once.
    Events main_events(repeats*3),w_events(repeats*2),a_events(repeats*2);
    x.weight(candidate);x.activation(candidate);
    if(mode!=0) {
      for(int i=0;i<warmup;++i) direct();
      for(int i=0;i<repeats;++i) {
        check(cuEventRecord(main_events.handles[3*i],stream));
        if(mode==2)x.weight(candidate);
        if(mode==2 || mode==3)x.activation(candidate);
        check(cuEventRecord(main_events.handles[3*i+1],stream));gemm();
        check(cuEventRecord(main_events.handles[3*i+2],stream));
      }
    }
    // Isolated, amortized conversion after the direct E2E path (dual track).
    const bool weight=mode==0 || mode==2,activation=mode!=1;
    if(weight) {
      for(int i=0;i<warmup;++i)x.weight(candidate);
      for(int i=0;i<repeats;++i) {
        check(cuEventRecord(w_events.handles[2*i],stream));
        for(int j=0;j<inner;++j)x.weight(candidate);
        check(cuEventRecord(w_events.handles[2*i+1],stream));
      }
    }
    if(activation) {
      for(int i=0;i<warmup;++i)x.activation(candidate);
      for(int i=0;i<repeats;++i) {
        check(cuEventRecord(a_events.handles[2*i],stream));
        for(int j=0;j<inner;++j)x.activation(candidate);
        check(cuEventRecord(a_events.handles[2*i+1],stream));
      }
    }
    runtime_check(cudaGetLastError());check(cuStreamSynchronize(stream));
    for(int i=0;i<repeats;++i) {
      for(int stage=0;stage<4;++stage) times[stage*repeats+i]=0;
      if(weight) {check(cuEventElapsedTime(times+i,w_events.handles[2*i],w_events.handles[2*i+1]));times[i]/=inner;}
      if(activation) {check(cuEventElapsedTime(times+repeats+i,a_events.handles[2*i],a_events.handles[2*i+1]));times[repeats+i]/=inner;}
      if(mode==0) times[3*repeats+i]=times[i]+times[repeats+i];
      else {
        check(cuEventElapsedTime(times+2*repeats+i,main_events.handles[3*i+1],main_events.handles[3*i+2]));
        check(cuEventElapsedTime(times+3*repeats+i,main_events.handles[3*i],main_events.handles[3*i+2]));
      }
    }
    return 0;
  } catch(const std::exception& e) {error=e.what();return 1;}
}
