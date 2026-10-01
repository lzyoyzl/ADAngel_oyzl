// Isolated v53: vectorized exact conversion; no production extension changes.
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include "roof_fused_conversion_api.h"
#include "roof_integer_conversion_api.h"

#include "roof_vector_conversion_impl.cuh"

namespace vector_probe {
void check(cudaError_t result) {
  if(result!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
thread_local std::string error;
struct Events {
  cudaEvent_t start=nullptr,stop=nullptr;
  Events() { check(cudaEventCreate(&start));
    try {check(cudaEventCreate(&stop));} catch(...) {cudaEventDestroy(start);throw;}}
  ~Events() {if(start)cudaEventDestroy(start);if(stop)cudaEventDestroy(stop);}
};
}

extern "C" const char* vector_conversion_error() {return vector_probe::error.c_str();}
extern "C" int vector_conversion_benchmark(int format,int policy,
    const uint8_t* payload,const uint8_t* scale,const float* tensor_scale,
    const uint8_t* micro8,const uint8_t* micro4,uint8_t* packed,float* effective,
    int rows,int k,int warmup,int repeats,int inner,cudaStream_t stream,float* times) {
  using namespace vector_probe;
  try {
    if(format<0 || format>3 || policy<0 || policy>2 || rows<=0 || k<=0 || k%128
        || int64_t(rows)*k>INT32_MAX || k/128>65535 || warmup<0 || repeats<1 || inner<1
        || !payload || !scale || !packed || !effective || !times
        || ((format==0 || format==3)&&!tensor_scale) || (format==2&&(!micro8 || !micro4)))
      throw std::runtime_error("invalid vector conversion argument");
    if(reinterpret_cast<uintptr_t>(payload)%16 || reinterpret_cast<uintptr_t>(packed)%16)
      throw std::runtime_error("16-byte aligned payload required");
    auto kind=Kind(format);
    auto launch=[&]() {
      if(policy==0) {
        if(kind==Kind::Mx8) adangel_sm80_experiment::fused_mixed_fixed(kind,payload,scale,tensor_scale,
            micro8,micro4,packed,effective,rows,k,stream);
        else adangel_sm80_experiment::integer_mixed_fixed(kind,true,payload,scale,tensor_scale,
            micro8,micro4,packed,effective,rows,k,stream);
      } else {
        const dim3 grid((rows+(policy==1?15:31))/(policy==1?16:32),k/128);
        #define VLAUNCH(K,E) adangel_sm80_vector_fixed_conversion<Kind::K,E><<<grid,256,0,stream>>>(payload,scale,tensor_scale,micro8,micro4,packed,effective,rows,k/128)
        #define VCASE(K) case Kind::K: if(policy==1) {VLAUNCH(K,8);} else {VLAUNCH(K,16);} break
        switch(kind) {VCASE(Nv4);VCASE(Mx8);VCASE(Hif4);VCASE(Nv6);}
        #undef VCASE
        #undef VLAUNCH
      }
    };
    Events event; // all event allocations precede the timed region
    for(int i=0;i<warmup;++i) launch();
    check(cudaGetLastError());check(cudaStreamSynchronize(stream));
    for(int r=0;r<repeats;++r) {
      check(cudaEventRecord(event.start,stream));
      for(int i=0;i<inner;++i) launch();
      check(cudaEventRecord(event.stop,stream));check(cudaEventSynchronize(event.stop));
      float elapsed;check(cudaEventElapsedTime(&elapsed,event.start,event.stop));
      times[r]=elapsed/inner;
    }
    check(cudaGetLastError());return 0;
  } catch(const std::exception& ex) {error=ex.what();return 1;}
}
