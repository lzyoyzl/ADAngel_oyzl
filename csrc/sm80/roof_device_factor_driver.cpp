// Isolated v61 driver. No production dispatch or extension change.
#include "roof_gpu_factor_driver.cpp"

// 0 control, 1 cached v59, 2 cached device-select, 3 prepare + device-select.
// Modes 2/3 have no host decision/synchronization inside their Event interval.
extern "C" int roof_device_factor_benchmark(void* prep,void* control,void* cached,void* device,
    uint64_t a,uint64_t w,uint64_t as,uint64_t ws,uint64_t meta,uint64_t status,uint64_t y,
    int m,int n,int k,int warmup,int repeats,int mode,void* stream_ptr,float* times,int* verdict) {
  try {
    if(!prep || !control || !cached || !device || !a || !w || !as || !ws || !meta || !status || !y ||
       !times || !verdict || m<=0 || n<=0 || m%64 || n%128 || k!=4096 ||
       int64_t(m)*n>INT32_MAX || int64_t(m)*k>INT32_MAX || int64_t(n)*k>INT32_MAX ||
       m/64>65535 || n/128>65535 || warmup<0 || repeats<1 || repeats>1000000 || mode<0 || mode>3)
      throw std::runtime_error("invalid device-factor probe launch");
    auto stream=reinterpret_cast<CUstream>(stream_ptr);
    const int blocks=n/128;PinnedFlags flags(blocks);Events events(repeats*2);
    *verdict=-1;
    auto prepare=[&]() {
      void* args[]={&ws,&meta,&status,&n};
      check(cuLaunchKernel(static_cast<Probe*>(prep)->function,blocks,1,1,128,1,1,0,stream,args,nullptr));
    };
    auto validate_status=[&]() {
      check(cuMemcpyDtoHAsync(flags.values,status,blocks*sizeof(unsigned),stream));
      check(cuStreamSynchronize(stream));
      unsigned all=0;for(int i=0;i<blocks;++i) all|=flags.values[i];
      *verdict=int(all);
      if(all&2u) throw std::runtime_error("invalid UE8M0 code 255");
      if(all&4u) throw std::runtime_error("isolated probe requires normal UE8M0 codes 1..254");
    };
    if(mode!=0) prepare();
    if(mode==1) validate_status(); // cached old signature needs a host selection outside timing
    auto action=[&]() {
      if(mode==3) prepare();
      if(mode>=2) {
        auto* p=static_cast<Probe*>(device);
        void* args[]={&a,&w,&as,&ws,&meta,&status,&y,&m,&n,&k};
        check(cuLaunchKernel(p->function,n/128,m/64,1,128,1,1,p->smem,stream,args,nullptr));
      } else {
        bool fast=mode==1 && *verdict==0;
        auto* p=static_cast<Probe*>(fast?cached:control);uint64_t scales=fast?meta:ws;
        void* args[]={&a,&w,&as,&scales,&y,&m,&n,&k};
        check(cuLaunchKernel(p->function,n/128,m/64,1,128,1,1,p->smem,stream,args,nullptr));
      }
    };
    for(int i=0;i<warmup;++i) action();
    for(int i=0;i<repeats;++i) {
      check(cuEventRecord(events.handles[i*2],stream));action();
      check(cuEventRecord(events.handles[i*2+1],stream));
    }
    check(cuStreamSynchronize(stream));
    // Invalid tiles deliberately don't write output. Reject BEFORE Python can see y.
    if(mode!=0) validate_status();
    for(int i=0;i<repeats;++i)
      check(cuEventElapsedTime(times+i,events.handles[i*2],events.handles[i*2+1]));
    return 0;
  } catch(const std::exception& e) {error=e.what();return 1;}
}
