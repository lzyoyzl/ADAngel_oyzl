// Same cached CUDA Event harness for both N128 and N256. No conversion timing.
#include "roof_producer_warp_driver.cpp"
extern "C" int roof_o78_n256_benchmark(void* handle,const uint64_t* state,
    int m,int n,int k,int tile_n,int warmup,int repeats,void* stream_ptr,float* times) {
  try {
    if(!handle || !state || !times || (tile_n!=128 && tile_n!=256) ||
       m<=0 || n<=0 || m%64 || n%tile_n || k!=4096 || m/64>65535 || n/tile_n>65535 ||
       int64_t(m)*n>INT32_MAX || int64_t(m)*k>INT32_MAX || int64_t(n)*k>INT32_MAX ||
       warmup<0 || repeats<1 || repeats>100000)
      throw std::runtime_error("invalid N-reuse cached launch");
    uint64_t v[16];for(int i=0;i<16;++i) {v[i]=state[i];if(!v[i])throw std::runtime_error("null state");}
    auto* p=static_cast<Probe*>(handle);
    if(p->threads!=128 || p->smem!=(tile_n==256?51712u:34304u))
      throw std::runtime_error("handle does not match tile geometry");
    auto stream=reinterpret_cast<CUstream>(stream_ptr);
    void* args[]={v,v+1,v+2,v+3,v+4,v+5,v+6,v+7,v+14,v+15,&m,&n,&k};
    Events events(repeats*2);
    auto launch=[&]() {check(cuLaunchKernel(p->function,n/tile_n,m/64,1,128,1,1,p->smem,stream,args,nullptr));};
    for(int i=0;i<warmup;++i)launch();
    for(int i=0;i<repeats;++i) {
      check(cuEventRecord(events.handles[2*i],stream));launch();
      check(cuEventRecord(events.handles[2*i+1],stream));
    }
    check(cuStreamSynchronize(stream));
    for(int i=0;i<repeats;++i)check(cuEventElapsedTime(times+i,events.handles[2*i],events.handles[2*i+1]));
    return 0;
  } catch(const std::exception& e) {error=e.what();return 1;}
}
