// v69 independent fused conversion/norm experiment. Reuse v68 numerical guards.
#include "roof_o78_gpu_prepare.cu"
#include "roof_vector_norm_conversion.cuh"

template<int Kind> __global__ __launch_bounds__(128) void adangel_o78_factor_from_squares(
    const uint32_t* square,const uint8_t* codes,float multiplier,int32_t* factors,
    float* bases,uint64_t* norms,int32_t* maxima,uint32_t* status,int rows,int activation) {
  using namespace o78_prepare;
  const int r=blockIdx.x*4+(threadIdx.x>>5),lane=threadIdx.x&31;
  if(r>=rows)return; // complete warps only
  int mant,exp;bool bad;decode<Kind>(codes[r*32+lane],mant,exp,bad);
  int anchor=mant?exp:INT_MAX;
  for(int d=16;d;d>>=1)anchor=min(anchor,__shfl_down_sync(0xffffffff,anchor,d));
  anchor=__shfl_sync(0xffffffff,anchor,0);if(anchor==INT_MAX)anchor=0;
  const int delta=exp-anchor;
  const bool overflow=mant && (delta>=31 || uint64_t(mant)>(uint64_t(INT32_MAX)>>delta));
  const int factor=(!bad && !overflow && mant)?int(uint64_t(mant)<<delta):0;
  const unsigned bs=__float_as_uint(multiplier),be=(bs>>23)&255;
  const int newexp=int(be)+anchor;
  const bool basebad=(bs>>31) || be==0 || be==255 || newexp<=0 || newexp>=255 ||
                     (activation && newexp>223);
  const unsigned verdict=warp_max(bad?2u:(overflow || basebad?1u:0u));
  const unsigned factor_bad=warp_max(unsigned(overflow));
  const unsigned maximum=warp_max(overflow?unsigned(INT32_MAX):unsigned(factor));
  factors[lane*rows+r]=verdict?0:factor;
  uint64_t norm=verdict==2?0:sat_term(square[r*32+lane],unsigned(factor));
  norm=warp_sum_sat(norm);
  if(lane==0) {
    bases[r]=verdict?1.f:__uint_as_float((bs&0x807fffffu)|(unsigned(newexp)<<23));
    status[r]=verdict;maxima[r]=verdict==2?0:int(maximum);
    norms[r]=verdict==2?0:(factor_bad?Cap:norm);
  }
}

namespace {
struct FusedOnline:Online {
  FusedOnline(int vv,const uint64_t* aa,const uint64_t* ww,const uint64_t* state,int mm,int nn,void* ss)
    :Online(vv,aa,ww,state,mm,nn,ss) {
    if(!v[16] || !v[17])throw std::runtime_error("preallocated group squared norms required");
  }
  template<Kind K> void convert_norm(const uint64_t* s,bool act) {
    const int rows=act?m:n;
    fused_norm_probe::adangel_sm80_vector_fixed_conversion_with_norm<K,16>
      <<<dim3((rows+31)/32,32),256,0,stream>>>(
        reinterpret_cast<const uint8_t*>(s[0]),reinterpret_cast<const uint8_t*>(s[1]),
        reinterpret_cast<const float*>(s[2]),reinterpret_cast<const uint8_t*>(s[3]),
        reinterpret_cast<const uint8_t*>(s[4]),reinterpret_cast<uint8_t*>(v[act?0:1]),
        reinterpret_cast<float*>(v[act?2:3]),reinterpret_cast<uint32_t*>(v[act?16:17]),rows,32);
  }
  template<int K> void factor_metadata(const uint64_t* s,bool act,float mult) {
    const int rows=act?m:n;
    adangel_o78_factor_from_squares<K><<<(rows+3)/4,128,0,stream>>>(
      reinterpret_cast<const uint32_t*>(v[act?16:17]),reinterpret_cast<const uint8_t*>(s[1]),mult,
      reinterpret_cast<int32_t*>(v[act?4:5]),reinterpret_cast<float*>(v[act?6:7]),
      reinterpret_cast<uint64_t*>(v[act?8:9]),reinterpret_cast<int32_t*>(v[act?10:11]),
      reinterpret_cast<uint32_t*>(v[act?12:13]),rows,int(act));
  }
  void weight(bool candidate) {
    if(!candidate){Online::weight(false);return;}
    if(variant==7){convert_norm<Kind::Nv4>(sw,false);factor_metadata<1>(sw,false,w_mult);}
    else {convert_norm<Kind::Hif4>(sw,false);factor_metadata<2>(sw,false,w_mult);}
  }
  void activation(bool candidate) {
    if(!candidate){Online::activation(false);return;}
    if(variant==7){convert_norm<Kind::Mx8>(sa,true);factor_metadata<0>(sa,true,a_mult);}
    else {convert_norm<Kind::Nv6>(sa,true);factor_metadata<1>(sa,true,a_mult);}
    adangel_o78_prepare_cta_guard<<<dim3(n/128,m/64),128,0,stream>>>(
      reinterpret_cast<const uint64_t*>(v[8]),reinterpret_cast<const uint64_t*>(v[9]),
      reinterpret_cast<const int32_t*>(v[10]),reinterpret_cast<const int32_t*>(v[11]),
      reinterpret_cast<const uint32_t*>(v[12]),reinterpret_cast<const uint32_t*>(v[13]),
      reinterpret_cast<const float*>(v[6]),reinterpret_cast<const float*>(v[7]),
      reinterpret_cast<uint32_t*>(v[14]),m,n);
  }
};
}

extern "C" int roof_o78_fused_prepare(int variant,const uint64_t* sa,const uint64_t* sw,
    const uint64_t* state,int m,int n,float a_mult,float w_mult,void* stream) {
  try {FusedOnline x(variant,sa,sw,state,m,n,stream);x.a_mult=a_mult;x.w_mult=w_mult;
    x.weight(true);x.activation(true);runtime_check(cudaGetLastError());return 0;
  } catch(const std::exception& e) {error=e.what();return 1;}
}
extern "C" int roof_o78_fused_benchmark(void* handle,int variant,int candidate,int mode,
    const uint64_t* sa,const uint64_t* sw,const uint64_t* state,int m,int n,
    float a_mult,float w_mult,int warmup,int repeats,int inner,void* stream_ptr,float* times) {
  try {
    if(!handle || !times || candidate<0 || candidate>1 || mode<0 || mode>3 ||
       warmup<0 || repeats<1 || repeats>100000 || inner<1 || inner>10000)
      throw std::runtime_error("invalid online timing configuration");
    FusedOnline x(variant,sa,sw,state,m,n,stream_ptr);x.a_mult=a_mult;x.w_mult=w_mult;
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
        if(mode==1) times[3*repeats+i]=times[2*repeats+i];
        else check(cuEventElapsedTime(times+3*repeats+i,main_events.handles[3*i],main_events.handles[3*i+2]));
      }
    }
    return 0;
  } catch(const std::exception& e) {error=e.what();return 1;}
}


