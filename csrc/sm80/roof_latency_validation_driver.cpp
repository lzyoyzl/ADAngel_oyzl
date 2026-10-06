// v104: exact existing cubin runtime validation; no production/kernel edits.
// A single Event loop for the128-thread best and256-thread unmeasured v98.
#include <cuda.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string error;
void check(CUresult result) {
  if(result==CUDA_SUCCESS) return;
  const char* message=nullptr;cuGetErrorString(result,&message);
  throw std::runtime_error(message?message:"CUDA driver error");
}
struct Handle {
  CUmodule module=nullptr;CUfunction function=nullptr;unsigned shared=0;int threads=0,kind=0;
  ~Handle(){if(module)cuModuleUnload(module);}
};
struct Events {
  std::vector<CUevent> events;
  explicit Events(int count):events(count,nullptr) {
    try {for(auto& event:events)check(cuEventCreate(&event,CU_EVENT_DEFAULT));}
    catch(...) {for(auto event:events)if(event)cuEventDestroy(event);throw;}
  }
  ~Events(){for(auto event:events)if(event)cuEventDestroy(event);}
};
}
extern "C" const char* roof_latency_error(){return error.c_str();}
extern "C" int roof_latency_open(const char* path,const char* symbol,int kind,
                                int threads,unsigned shared,void** handle,int* resources) {
  Handle* p=nullptr;
  try {
    if(!path||!symbol||!handle||!resources||(kind!=3&&kind!=78)||(threads!=128&&threads!=256))
      throw std::runtime_error("invalid exact-cubin open");
    CUcontext context=nullptr;check(cuCtxGetCurrent(&context));
    if(!context)throw std::runtime_error("Torch CUDA context must be current");
    p=new Handle;check(cuModuleLoad(&p->module,path));
    check(cuModuleGetFunction(&p->function,p->module,symbol));
    check(cuFuncGetAttribute(resources+2,CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,p->function));
    if(resources[2]!=threads)throw std::runtime_error("compiled block geometry mismatch");
    check(cuFuncSetAttribute(p->function,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,shared));
    check(cuFuncGetAttribute(resources,CU_FUNC_ATTRIBUTE_NUM_REGS,p->function));
    check(cuFuncGetAttribute(resources+1,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,p->function));
    check(cuOccupancyMaxActiveBlocksPerMultiprocessor(resources+3,p->function,threads,shared));
    p->threads=threads;p->kind=kind;p->shared=shared;*handle=p;return 0;
  } catch(const std::exception& e){delete p;error=e.what();return 1;}
}
extern "C" int roof_latency_close(void* handle){delete static_cast<Handle*>(handle);return 0;}
extern "C" int roof_latency_benchmark(void* handle,const uint64_t* pointers,int pointer_count,
    int m,int n,int k,int warmup,int repeats,void* stream_pointer,float* times) {
  try {
    if(!handle||!pointers||!times||m<=0||n<=0||m%64||n%128||k!=4096||
       int64_t(m)*n>INT32_MAX||m/64>65535||n/128>65535||warmup<0||repeats<1||repeats>1000000)
      throw std::runtime_error("invalid cached fullK launch");
    auto* p=static_cast<Handle*>(handle);
    if(pointer_count!=(p->kind==3?7:10))throw std::runtime_error("GEMM pointer ABI mismatch");
    void* args[13];for(int i=0;i<pointer_count;++i) {
      if(!pointers[i])throw std::runtime_error("null GEMM pointer");
      args[i]=const_cast<uint64_t*>(pointers+i);
    }
    args[pointer_count]=&m;args[pointer_count+1]=&n;args[pointer_count+2]=&k;
    auto stream=reinterpret_cast<CUstream>(stream_pointer);
    Events events(repeats*2); // No allocation inside any Event interval.
    auto launch=[&](){check(cuLaunchKernel(p->function,n/128,m/64,1,p->threads,1,1,p->shared,stream,args,nullptr));};
    for(int i=0;i<warmup;++i)launch();
    for(int i=0;i<repeats;++i) {
      check(cuEventRecord(events.events[2*i],stream));launch();
      check(cuEventRecord(events.events[2*i+1],stream));
    }
    check(cuStreamSynchronize(stream));
    for(int i=0;i<repeats;++i)check(cuEventElapsedTime(times+i,events.events[2*i],events.events[2*i+1]));
    return 0;
  } catch(const std::exception& e){error=e.what();return 1;}
}
