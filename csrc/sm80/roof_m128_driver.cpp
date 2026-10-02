// Isolated host harness; original open/resource/close helpers remain unchanged.
#include "roof_producer_warp_driver.cpp"
extern "C" int roof_m128_benchmark(void* handle,
    uint64_t a,uint64_t w,uint64_t as,uint64_t ws,uint64_t y,
    int m,int n,int k,int warmup,int repeats,int tile_m,void* stream_ptr,float* times) {
  try {
    if(!handle || !times || !a || !w || !as || !ws || !y || warmup<0 || repeats<1 || repeats>1000000 ||
        (tile_m!=64 && tile_m!=128) || m<=0 || n<=0 || k<=0 || m%tile_m || n%128 || k%128 ||
        int64_t(m)*n>INT32_MAX || int64_t(m)*k>INT32_MAX || int64_t(n)*k>INT32_MAX ||
        m/tile_m>65535 || n/128>65535)
      throw std::runtime_error("invalid bounded M128 probe launch");
    auto* p=static_cast<Probe*>(handle);
    auto stream=reinterpret_cast<CUstream>(stream_ptr);
    void* args[]={&a,&w,&as,&ws,&y,&m,&n,&k};
    Events events(repeats*2);
    auto launch=[&]() {check(cuLaunchKernel(p->function,n/128,m/tile_m,1,p->threads,1,1,
                                           p->smem,stream,args,nullptr));};
    for(int i=0;i<warmup;++i) launch();
    for(int i=0;i<repeats;++i) {
      check(cuEventRecord(events.handles[i*2],stream));launch();
      check(cuEventRecord(events.handles[i*2+1],stream));
    }
    check(cuStreamSynchronize(stream));
    for(int i=0;i<repeats;++i)
      check(cuEventElapsedTime(times+i,events.handles[i*2],events.handles[i*2+1]));
    return 0;
  } catch(const std::exception& e) {error=e.what();return 1;}
}
