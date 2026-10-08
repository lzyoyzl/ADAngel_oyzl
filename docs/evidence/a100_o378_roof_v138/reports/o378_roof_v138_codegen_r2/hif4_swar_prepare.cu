// Generated v138; retain every original v123 library entry.
#include "nv6_swar_prepare.cu"
#include "hif4_swar_generated.cuh"
namespace {
struct Hif4SwarOnline:Nv6SwarOnline {
  using Nv6SwarOnline::Nv6SwarOnline;
  void activation(bool) {Nv6SwarOnline::activation(true);}
  void weight(bool candidate) {
    if(!candidate || variant!=8){RowFusedOnline::weight(true);return;}
    hif4_row_swar_probe::adangel_sm80_hif4_swar_metadata<Kind::Hif4,16>
      <<<n,256,0,stream>>>(
        reinterpret_cast<const uint8_t*>(sw[0]),reinterpret_cast<const uint8_t*>(sw[1]),
        reinterpret_cast<const float*>(sw[2]),reinterpret_cast<const uint8_t*>(sw[3]),
        reinterpret_cast<const uint8_t*>(sw[4]),reinterpret_cast<uint8_t*>(v[1]),
        reinterpret_cast<float*>(v[3]),reinterpret_cast<uint32_t*>(v[17]),n,32,w_mult,
        reinterpret_cast<int32_t*>(v[5]),reinterpret_cast<float*>(v[7]),
        reinterpret_cast<uint64_t*>(v[9]),reinterpret_cast<int32_t*>(v[11]),
        reinterpret_cast<uint32_t*>(v[13]));
  }
};
__global__ void adangel_hif4_swar_exhaustive(uint4* out) {
  const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
  const unsigned lo=i&65535u,hi=(i&65536u)?(lo^65535u):lo;
  const unsigned e8=(i>>17)&1u,e4=(i>>18)&3u,word=lo|(hi<<16);
  unsigned packed=0,square=0;
  #pragma unroll
  for(int j=0;j<8;++j) {
    const unsigned c=(word>>(4*j))&15u;
    const int mag=int(vector_probe::rne((c&7u)<<(e8+((e4>>(j/4))&1u)),2));
    const int q=(c&8u)?-mag:mag;
    packed|=(unsigned(q)&15u)<<(4*j);square+=unsigned(q*q);
  }
  const uint2 v=hif4_swar_probe::packed_word(word,e8,e4);
  out[i]=make_uint4(packed,v.x,square,v.y);
}
}
extern "C" int roof_hif4_swar_exhaustive(uint4* out,void* stream) {
  if(!out)return 1;
  adangel_hif4_swar_exhaustive<<<4096,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(out);
  return int(cudaGetLastError());
}
// No kernel launch: inspect actual conversion residency before timing.
extern "C" int roof_hif4_swar_resources(int* values) {
  if(!values)return 1;
  cudaFuncAttributes old{},candidate{};int old_blocks=0,new_blocks=0;
  cudaError_t err=cudaFuncGetAttributes(&old,
      row_fused_probe::adangel_sm80_row_conversion_metadata<Kind::Hif4,16>);
  if(err!=cudaSuccess)return int(err);
  err=cudaFuncGetAttributes(&candidate,
      hif4_row_swar_probe::adangel_sm80_hif4_swar_metadata<Kind::Hif4,16>);
  if(err!=cudaSuccess)return int(err);
  err=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&old_blocks,
      row_fused_probe::adangel_sm80_row_conversion_metadata<Kind::Hif4,16>,256,0);
  if(err!=cudaSuccess)return int(err);
  err=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&new_blocks,
      hif4_row_swar_probe::adangel_sm80_hif4_swar_metadata<Kind::Hif4,16>,256,0);
  if(err!=cudaSuccess)return int(err);
  values[0]=old.numRegs;values[1]=old.localSizeBytes;values[2]=old.sharedSizeBytes;
  values[3]=old.maxThreadsPerBlock;values[4]=old_blocks;
  values[5]=candidate.numRegs;values[6]=candidate.localSizeBytes;values[7]=candidate.sharedSizeBytes;
  values[8]=candidate.maxThreadsPerBlock;values[9]=new_blocks;
  return 0;
}
extern "C" int roof_o78_hif4_swar_benchmark(void* handle,int variant,int candidate,int mode,
    const uint64_t* sa,const uint64_t* sw,const uint64_t* state,int m,int n,
    float a_mult,float w_mult,int warmup,int repeats,int inner,void* stream_ptr,float* times) {
  try {
    if(!handle || !times || candidate<0 || candidate>1 || mode<0 || mode>3 ||
       warmup<0 || repeats<1 || repeats>100000 || inner<1 || inner>10000)
      throw std::runtime_error("invalid online timing configuration");
    Hif4SwarOnline x(variant,sa,sw,state,m,n,stream_ptr);x.a_mult=a_mult;x.w_mult=w_mult;
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
