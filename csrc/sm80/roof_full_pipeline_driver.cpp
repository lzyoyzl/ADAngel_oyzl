// v62 four-mode integration of existing conversion2 and v61. No new GEMM.
#include "roof_device_factor_driver.cpp"

extern "C" int roof_four_open_conversion(const char* path,const char* symbol,void** handle) {
  Probe* p=nullptr;
  try {
    if(!path || !symbol || !handle) throw std::runtime_error("null conversion open argument");
    CUcontext ctx=nullptr;check(cuCtxGetCurrent(&ctx));
    if(!ctx) throw std::runtime_error("Torch CUDA context must be current");
    p=new Probe;check(cuModuleLoad(&p->module,path));check(cuModuleGetFunction(&p->function,p->module,symbol));
    check(cuFuncGetAttribute(&p->threads,CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,p->function));
    if(p->threads<256) throw std::runtime_error("conversion kernel cannot launch 256 threads");
    *handle=p;return 0;
  } catch(const std::exception& e) {delete p;error=e.what();return 1;}
}

// mode 0 conversion_only, 1 compute_only, 2 cold, 3 steady_state; policy 0 old54, 1 v61.
// times is stage-major [weight,activation,gemm,total][repeats]. Inactive stages zero.
extern "C" int roof_four_benchmark(void* prep,void* control,void* device,void* conv_a,void* conv_w,
    uint64_t a,uint64_t w,uint64_t as,uint64_t ws,uint64_t pa,uint64_t pw,uint64_t gws,
    uint64_t meta,uint64_t status,uint64_t y,int m,int n,int k,int warmup,int repeats,int inner,
    int mode,int policy,void* stream_ptr,float* times,int* verdict) {
  try {
    if(!prep || !control || !device || !conv_a || !conv_w || !a || !w || !as || !ws || !pa || !pw ||
       !gws || !meta || !status || !y || !times || !verdict || m<=0 || n<=0 || m%64 || n%128 ||
       k!=4096 || int64_t(m)*n>INT32_MAX || int64_t(m)*k>INT32_MAX || int64_t(n)*k>INT32_MAX ||
       m/64>65535 || n/128>65535 || warmup<0 || repeats<1 || repeats>1000000 || inner<1 || inner>1000000 ||
       mode<0 || mode>3 || policy<0 || policy>1 || (a%8) || (w%4) || (pa%4) || (pw%4))
      throw std::runtime_error("invalid full-pipeline probe launch");
    const bool weight=mode==0 || mode==2,activation=mode!=1,compute=mode!=0;
    const int blocks=n/128;unsigned groups=32;
    auto stream=reinterpret_cast<CUstream>(stream_ptr);
    // Everything, including pinned status and both timing tracks, before warmup.
    PinnedFlags flags(blocks);Events direct(repeats*4),wb(weight?repeats*2:0),ab(activation?repeats*2:0);
    *verdict=-1;for(int i=0;i<repeats*4;++i) times[i]=0;
    auto cvw=[&]() {
      unsigned rows=n;void* args[]={&w,&ws,&pw,&gws,&rows,&groups};
      check(cuLaunchKernel(static_cast<Probe*>(conv_w)->function,(n+15)/16,32,1,256,1,1,0,stream,args,nullptr));
      if(policy) {
        void* prep_args[]={&gws,&meta,&status,&n};
        check(cuLaunchKernel(static_cast<Probe*>(prep)->function,blocks,1,1,128,1,1,0,stream,prep_args,nullptr));
      }
    };
    auto cva=[&]() {
      unsigned rows=m;uint64_t unused=0;void* args[]={&a,&unused,&pa,&unused,&rows,&groups};
      check(cuLaunchKernel(static_cast<Probe*>(conv_a)->function,(m+15)/16,32,1,256,1,1,0,stream,args,nullptr));
    };
    auto gemm=[&]() {
      if(policy) {
        auto* p=static_cast<Probe*>(device);void* args[]={&pa,&pw,&as,&gws,&meta,&status,&y,&m,&n,&k};
        check(cuLaunchKernel(p->function,blocks,m/64,1,128,1,1,p->smem,stream,args,nullptr));
      } else {
        auto* p=static_cast<Probe*>(control);void* args[]={&pa,&pw,&as,&gws,&y,&m,&n,&k};
        check(cuLaunchKernel(p->function,blocks,m/64,1,128,1,1,p->smem,stream,args,nullptr));
      }
    };
    cvw();cva(); // cached preparation is outside measurement, same as native four-mode contract
    for(int i=0;i<warmup;++i) {if(weight)cvw();if(activation)cva();if(compute)gemm();}
    check(cuStreamSynchronize(stream));
    for(int i=0;i<repeats;++i) {
      check(cuEventRecord(direct.handles[4*i],stream));
      if(weight) cvw();check(cuEventRecord(direct.handles[4*i+1],stream));
      if(activation) cva();check(cuEventRecord(direct.handles[4*i+2],stream));
      if(compute) gemm();check(cuEventRecord(direct.handles[4*i+3],stream));
    }
    check(cuStreamSynchronize(stream));
    auto batch=[&](auto& fn,Events& ev,int stage) {
      for(int i=0;i<repeats;++i) {
        check(cuEventRecord(ev.handles[2*i],stream));
        for(int j=0;j<inner;++j) fn();
        check(cuEventRecord(ev.handles[2*i+1],stream));
      }
      check(cuStreamSynchronize(stream));
      for(int i=0;i<repeats;++i) {
        check(cuEventElapsedTime(times+stage*repeats+i,ev.handles[2*i],ev.handles[2*i+1]));
        times[stage*repeats+i]/=inner;
      }
    };
    if(weight)batch(cvw,wb,0);if(activation)batch(cva,ab,1);
    if(!compute) {gemm();check(cuStreamSynchronize(stream));} // diagnostic output only, not conversion timing
    if(policy) {
      check(cuMemcpyDtoHAsync(flags.values,status,blocks*sizeof(unsigned),stream));
      check(cuStreamSynchronize(stream));
      unsigned all=0;for(int i=0;i<blocks;++i) all|=flags.values[i];*verdict=int(all);
      if(all&2u) throw std::runtime_error("invalid UE8M0 code 255");
      if(all&4u) throw std::runtime_error("isolated probe requires normal UE8M0 codes 1..254");
    }
    for(int i=0;i<repeats;++i) {
      if(compute) {
        check(cuEventElapsedTime(times+2*repeats+i,direct.handles[4*i+2],direct.handles[4*i+3]));
        check(cuEventElapsedTime(times+3*repeats+i,direct.handles[4*i],direct.handles[4*i+3]));
      } else times[3*repeats+i]=times[i]+times[repeats+i];
    }
    return 0;
  } catch(const std::exception& e) {error=e.what();return 1;}
}
