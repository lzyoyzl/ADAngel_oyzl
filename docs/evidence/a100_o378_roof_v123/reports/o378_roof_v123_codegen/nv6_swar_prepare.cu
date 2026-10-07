// Generated v123; original library entries and Event contract preserved.
#include "roof_o78_row_fused_prepare.cu"
#include "nv6_swar_generated.cuh"
namespace {
struct Nv6SwarOnline:RowFusedOnline {
  using RowFusedOnline::RowFusedOnline;
  void weight(bool) {RowFusedOnline::weight(true);}
  void packed_activation() {
    nv6_row_swar_probe::adangel_sm80_nv6_swar_metadata<Kind::Nv6,16>
      <<<m,256,0,stream>>>(
        reinterpret_cast<const uint8_t*>(sa[0]),reinterpret_cast<const uint8_t*>(sa[1]),
        reinterpret_cast<const float*>(sa[2]),reinterpret_cast<const uint8_t*>(sa[3]),
        reinterpret_cast<const uint8_t*>(sa[4]),reinterpret_cast<uint8_t*>(v[0]),
        reinterpret_cast<float*>(v[2]),reinterpret_cast<uint32_t*>(v[16]),m,32,a_mult,
        reinterpret_cast<int32_t*>(v[4]),reinterpret_cast<float*>(v[6]),
        reinterpret_cast<uint64_t*>(v[8]),reinterpret_cast<int32_t*>(v[10]),
        reinterpret_cast<uint32_t*>(v[12]));
  }
  void activation(bool candidate) {
    if(!candidate || variant!=8){RowFusedOnline::activation(true);return;}
    if(variant==7)convert_metadata<Kind::Mx8>(sa,true,a_mult);
    else packed_activation();
    adangel_o78_prepare_cta_guard<<<dim3(n/128,m/64),128,0,stream>>>(
      reinterpret_cast<const uint64_t*>(v[8]),reinterpret_cast<const uint64_t*>(v[9]),
      reinterpret_cast<const int32_t*>(v[10]),reinterpret_cast<const int32_t*>(v[11]),
      reinterpret_cast<const uint32_t*>(v[12]),reinterpret_cast<const uint32_t*>(v[13]),
      reinterpret_cast<const float*>(v[6]),reinterpret_cast<const float*>(v[7]),
      reinterpret_cast<uint32_t*>(v[14]),m,n);
  }
};
__global__ void adangel_nv6_swar_exhaustive(uint4* out) {
  const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
  const unsigned lo=i&65535u;
  const unsigned hi=i&65536u?(lo^65535u):lo;
  const unsigned word=lo|(hi<<16);
  unsigned packed=0,square=0;
  #pragma unroll
  for(int j=0;j<4;++j) {
    const int q=vector_probe::nv6((word>>(8*j))&255u);
    packed|=(unsigned(q)&255u)<<(8*j);square+=unsigned(q*q);
  }
  const unsigned q=nv6_swar_probe::bytes(word);
  out[i]=make_uint4(packed,q,square,unsigned(nv6_swar_probe::square_add(q,0)));
}
}
extern "C" int roof_nv6_swar_exhaustive(uint4* out,void* stream) {
  if(!out)return 1;
  adangel_nv6_swar_exhaustive<<<512,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(out);
  return int(cudaGetLastError());
}
extern "C" int roof_o78_nv6_swar_benchmark(void* handle,int variant,int candidate,int mode,
    const uint64_t* sa,const uint64_t* sw,const uint64_t* state,int m,int n,
    float a_mult,float w_mult,int warmup,int repeats,int inner,void* stream_ptr,float* times) {
  try {
    if(!handle || !times || candidate<0 || candidate>1 || mode<0 || mode>3 ||
       warmup<0 || repeats<1 || repeats>100000 || inner<1 || inner>10000)
      throw std::runtime_error("invalid online timing configuration");
    Nv6SwarOnline x(variant,sa,sw,state,m,n,stream_ptr);x.a_mult=a_mult;x.w_mult=w_mult;
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
