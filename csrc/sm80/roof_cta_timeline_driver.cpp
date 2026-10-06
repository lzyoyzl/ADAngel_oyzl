// Isolated profiling ABI. No extension rebuild, production API or dispatch edit.
#include "roof_producer_warp_driver.cpp"
extern "C" int roof_timeline_launch(void* handle,uint64_t* argv,int argc,
    int gx,int gy,int warmup,int repeats,void* stream_ptr,float* times) {
  try {
    if(!handle || !argv || !times || argc<1 || argc>16 || gx<1 || gy<1 ||
        gx>65535 || gy>65535 || warmup<0 || repeats<1 || repeats>1000000)
      throw std::runtime_error("invalid bounded timeline diagnostic launch");
    auto* p=static_cast<Probe*>(handle);
    std::vector<void*> args(argc);
    // uint64_t slots hold pointer parameters and the low32 bits of int params.
    // cuLaunchKernel copies each parameter's actual PTX-declared width.
    for(int i=0;i<argc;++i) args[i]=argv+i;
    auto stream=reinterpret_cast<CUstream>(stream_ptr);
    Events events(repeats*2);
    auto launch=[&]() {check(cuLaunchKernel(p->function,gx,gy,1,p->threads,1,1,
        p->smem,stream,args.data(),nullptr));};
    for(int i=0;i<warmup;++i) launch();
    for(int i=0;i<repeats;++i) {
      check(cuEventRecord(events.handles[2*i],stream));launch();
      check(cuEventRecord(events.handles[2*i+1],stream));
    }
    check(cuStreamSynchronize(stream));
    for(int i=0;i<repeats;++i)
      check(cuEventElapsedTime(times+i,events.handles[2*i],events.handles[2*i+1]));
    return 0;
  } catch(const std::exception& e) {error=e.what();return 1;}
}
