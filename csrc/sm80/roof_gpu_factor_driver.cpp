// Reuse the identical old CUDA Driver/Event implementation, not a production API.
#include "roof_producer_warp_driver.cpp"

namespace {
struct PinnedFlags {
  unsigned* values=nullptr;
  explicit PinnedFlags(int count) {check(cuMemAllocHost(reinterpret_cast<void**>(&values),count*sizeof(unsigned)));}
  ~PinnedFlags() {if(values) cuMemFreeHost(values);}
};
}

// mode: 0 control GEMM; 1 cached v59 GEMM; 2 prepare/host verdict/GEMM;
// 3 GPU preparation kernel only (batched). Modes 2/3 are NOT full Cold/conversion.
extern "C" int roof_gpu_factor_benchmark(void* prep,void* control,void* candidate,
    uint64_t a,uint64_t w,uint64_t as,uint64_t ws,uint64_t meta,uint64_t status,uint64_t y,
    int m,int n,int k,int warmup,int repeats,int inner,int mode,
    void* stream_ptr,float* times,int* verdict) {
  try {
    if(!prep || !control || !candidate || !a || !w || !as || !ws || !meta || !status || !y ||
       !times || !verdict || m<=0 || n<=0 || m%64 || n%128 || k!=4096 ||
       int64_t(m)*n>INT32_MAX || int64_t(m)*k>INT32_MAX || int64_t(n)*k>INT32_MAX ||
       m/64>65535 || n/128>65535 || warmup<0 || repeats<1 || repeats>1000000 ||
       inner<1 || inner>1000000 || mode<0 || mode>3)
      throw std::runtime_error("invalid GPU-factor probe launch");
    auto* pf=static_cast<Probe*>(prep);
    auto* pc=static_cast<Probe*>(control);
    auto* pv=static_cast<Probe*>(candidate);
    auto stream=reinterpret_cast<CUstream>(stream_ptr);
    const int blocks=(n+127)/128;
    PinnedFlags flags(blocks);Events events(repeats*2);
    *verdict=-1;
    auto prepare=[&]() {
      void* args[]={&ws,&meta,&status,&n};
      check(cuLaunchKernel(pf->function,blocks,1,1,128,1,1,0,stream,args,nullptr));
    };
    auto decide=[&]() {
      check(cuMemcpyDtoHAsync(flags.values,status,blocks*sizeof(unsigned),stream));
      check(cuStreamSynchronize(stream));
      unsigned all=0;for(int i=0;i<blocks;++i) all|=flags.values[i];
      *verdict=int(all);
      if(all&2) throw std::runtime_error("invalid UE8M0 code 255");
      if(all&4) throw std::runtime_error("isolated probe requires normal UE8M0 codes 1..254");
    };
    auto gemm=[&](bool fast) {
      auto* p=fast?pv:pc;uint64_t scales=fast?meta:ws;
      void* args[]={&a,&w,&as,&scales,&y,&m,&n,&k};
      check(cuLaunchKernel(p->function,n/128,m/64,1,128,1,1,p->smem,stream,args,nullptr));
    };
    if(mode==1) {prepare();decide();} // cached preparation is outside GEMM timing
    auto action=[&]() {
      if(mode==3) {for(int j=0;j<inner;++j) prepare();return;}
      if(mode==2) {prepare();decide();} // reset/check each iteration; no cached proof
      gemm(mode!=0 && *verdict==0);
    };
    for(int i=0;i<warmup;++i) action();
    for(int i=0;i<repeats;++i) {
      check(cuEventRecord(events.handles[i*2],stream));
      action();
      check(cuEventRecord(events.handles[i*2+1],stream));
    }
    check(cuStreamSynchronize(stream));
    for(int i=0;i<repeats;++i) {
      check(cuEventElapsedTime(times+i,events.handles[i*2],events.handles[i*2+1]));
      if(mode==3) times[i]/=inner;
    }
    if(mode==3) decide(); // separate from GPU-only preparation interval
    return 0;
  } catch(const std::exception& e) {error=e.what();return 1;}
}
