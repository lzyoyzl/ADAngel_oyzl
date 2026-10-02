// v73 isolated row conversion + factor/anchor/norm fusion. v69 remains intact.
#include "roof_o78_fused_prepare.cu"
#include "roof_row_fused_conversion.cuh"

namespace {
struct RowFusedOnline:FusedOnline {
  using FusedOnline::FusedOnline;
  template<Kind K> void convert_metadata(const uint64_t* s,bool act,float mult) {
    const int rows=act?m:n;
    row_fused_probe::adangel_sm80_row_conversion_metadata<K,16>
      <<<rows,256,0,stream>>>(
        reinterpret_cast<const uint8_t*>(s[0]),reinterpret_cast<const uint8_t*>(s[1]),
        reinterpret_cast<const float*>(s[2]),reinterpret_cast<const uint8_t*>(s[3]),
        reinterpret_cast<const uint8_t*>(s[4]),reinterpret_cast<uint8_t*>(v[act?0:1]),
        reinterpret_cast<float*>(v[act?2:3]),reinterpret_cast<uint32_t*>(v[act?16:17]),rows,32,mult,
        reinterpret_cast<int32_t*>(v[act?4:5]),reinterpret_cast<float*>(v[act?6:7]),
        reinterpret_cast<uint64_t*>(v[act?8:9]),reinterpret_cast<int32_t*>(v[act?10:11]),
        reinterpret_cast<uint32_t*>(v[act?12:13]));
  }
  void weight(bool candidate) {
    if(!candidate){FusedOnline::weight(true);return;}
    if(variant==7)convert_metadata<Kind::Nv4>(sw,false,w_mult);
    else convert_metadata<Kind::Hif4>(sw,false,w_mult);
  }
  void activation(bool candidate) {
    if(!candidate){FusedOnline::activation(true);return;}
    if(variant==7)convert_metadata<Kind::Mx8>(sa,true,a_mult);
    else convert_metadata<Kind::Nv6>(sa,true,a_mult);
    adangel_o78_prepare_cta_guard<<<dim3(n/128,m/64),128,0,stream>>>(
      reinterpret_cast<const uint64_t*>(v[8]),reinterpret_cast<const uint64_t*>(v[9]),
      reinterpret_cast<const int32_t*>(v[10]),reinterpret_cast<const int32_t*>(v[11]),
      reinterpret_cast<const uint32_t*>(v[12]),reinterpret_cast<const uint32_t*>(v[13]),
      reinterpret_cast<const float*>(v[6]),reinterpret_cast<const float*>(v[7]),
      reinterpret_cast<uint32_t*>(v[14]),m,n);
  }
};
}

extern "C" int roof_o78_row_fused_prepare(int variant,const uint64_t* sa,const uint64_t* sw,
    const uint64_t* state,int m,int n,float a_mult,float w_mult,void* stream) {
  try {RowFusedOnline x(variant,sa,sw,state,m,n,stream);x.a_mult=a_mult;x.w_mult=w_mult;
    x.weight(true);x.activation(true);runtime_check(cudaGetLastError());return 0;
  } catch(const std::exception& e) {error=e.what();return 1;}
}

extern "C" int roof_o78_row_fused_benchmark(void* handle,int variant,int candidate,int mode,
    const uint64_t* sa,const uint64_t* sw,const uint64_t* state,int m,int n,
    float a_mult,float w_mult,int warmup,int repeats,int inner,void* stream_ptr,float* times) {
  try {
    if(!handle || !times || candidate<0 || candidate>1 || mode<0 || mode>3 ||
       warmup<0 || repeats<1 || repeats>100000 || inner<1 || inner>10000)
      throw std::runtime_error("invalid online timing configuration");
    RowFusedOnline x(variant,sa,sw,state,m,n,stream_ptr);x.a_mult=a_mult;x.w_mult=w_mult;
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
