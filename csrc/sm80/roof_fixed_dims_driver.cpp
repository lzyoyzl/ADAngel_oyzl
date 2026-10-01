// Internal v45 fixed4096-only cubin-screen harness. No PyTorch extension or default dispatch edits.
// All launches (including the no-hint control) use this SAME driver/Event loop.
#include <cuda.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string error;
void check(CUresult code) {
  if(code==CUDA_SUCCESS) return;
  const char* message=nullptr;cuGetErrorString(code,&message);
  throw std::runtime_error(message?message:"CUDA driver error");
}
struct Probe {
  CUmodule module=nullptr;CUfunction function=nullptr;unsigned smem=0;int threads=0;
  ~Probe() {if(module) cuModuleUnload(module);}
};
struct Events {
  std::vector<CUevent> handles;
  explicit Events(int count):handles(count,nullptr) {
    try {for(auto& e:handles) check(cuEventCreate(&e,CU_EVENT_DEFAULT));}
    catch(...) {for(auto e:handles) if(e) cuEventDestroy(e);throw;}
  }
  ~Events() {for(auto e:handles) if(e) cuEventDestroy(e);}
};
}
extern "C" const char* roof_probe_error() {return error.c_str();}
extern "C" int roof_probe_open(const char* path,const char* symbol,unsigned smem,void** handle) {
  Probe* p=nullptr;
  try {
    if(!path || !symbol || !handle) throw std::runtime_error("null open argument");
    CUcontext context=nullptr;check(cuCtxGetCurrent(&context));
    if(!context) throw std::runtime_error("Torch CUDA context must be current");
    p=new Probe;check(cuModuleLoad(&p->module,path));
    check(cuModuleGetFunction(&p->function,p->module,symbol));
    check(cuFuncSetAttribute(p->function,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,smem));
    check(cuFuncGetAttribute(&p->threads,CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,p->function));
    if(p->threads!=128 && p->threads!=160) throw std::runtime_error("unsupported compiled producer geometry");
    p->smem=smem;*handle=p;return 0;
  } catch(const std::exception& e) {delete p;error=e.what();return 1;}
}
// Queries use the actual compiled function and actual dynamic shared allocation.
extern "C" int roof_probe_resources(void* handle,int* values) {
  try {
    if(!handle || !values) throw std::runtime_error("null resource argument");
    auto* p=static_cast<Probe*>(handle);
    check(cuFuncGetAttribute(values,CU_FUNC_ATTRIBUTE_NUM_REGS,p->function));
    check(cuFuncGetAttribute(values+1,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,p->function));
    values[2]=p->threads;
    check(cuOccupancyMaxActiveBlocksPerMultiprocessor(values+3,p->function,p->threads,p->smem));
    return 0;
  } catch(const std::exception& e) {error=e.what();return 1;}
}
extern "C" int roof_probe_close(void* handle) {
  delete static_cast<Probe*>(handle);return 0;
}
extern "C" int roof_probe_benchmark(void* handle,
    uint64_t a,uint64_t w,uint64_t as,uint64_t ws,uint64_t y,
    int m,int n,int k,int warmup,int repeats,void* stream_ptr,float* times) {
  try {
    // All policies, including the original control, obey the same shape contract.
    // Reject before creating events or launching a fixed-address kernel.
    if(m!=4096 || n!=4096 || k!=4096)
      throw std::runtime_error("fixed4096 probe requires M=N=K=4096");
    if(!handle || !times || !a || !w || !as || !ws || !y || warmup<0 || repeats<1 || repeats>1000000 ||
        m<=0 || n<=0 || k<=0 || m%64 || n%128 || k%128 ||
        int64_t(m)*n>INT32_MAX || int64_t(m)*k>INT32_MAX || int64_t(n)*k>INT32_MAX ||
        m/64>65535 || n/128>65535)
      throw std::runtime_error("invalid bounded probe launch");
    auto* p=static_cast<Probe*>(handle);
    auto stream=reinterpret_cast<CUstream>(stream_ptr);
    void* args[]={&a,&w,&as,&ws,&y,&m,&n,&k};
    // Events are allocated before warmup and before every timed region.
    Events events(repeats*2);
    auto launch=[&]() {check(cuLaunchKernel(p->function,n/128,m/64,1,p->threads,1,1,
                                           p->smem,stream,args,nullptr));};
    for(int i=0;i<warmup;++i) launch();
    for(int i=0;i<repeats;++i) {
      check(cuEventRecord(events.handles[i*2],stream));
      launch();
      check(cuEventRecord(events.handles[i*2+1],stream));
    }
    check(cuStreamSynchronize(stream));
    for(int i=0;i<repeats;++i)
      check(cuEventElapsedTime(times+i,events.handles[i*2],events.handles[i*2+1]));
    return 0;
  } catch(const std::exception& e) {error=e.what();return 1;}
}
