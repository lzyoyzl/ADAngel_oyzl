#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>

// K512/K1024 instances of the accepted plans. K4096 has its original TU/API.
namespace adangel_sm80_production_sized {
void mixed_convert(int variant,bool activation,const uint64_t* source,const uint64_t* state,
                   int m,int n,int k,float multiplier,cudaStream_t stream);
void mixed_gemm(const uint64_t* state,int m,int n,int k,cudaStream_t stream);
void o3_prepare(const uint8_t* scales,int32_t* metadata,uint32_t* status,int n,int k,cudaStream_t stream);
void o3_gemm(const uint8_t* a,const uint8_t* w,const float* as,const uint8_t* ws,
             const int32_t* metadata,const uint32_t* status,float* y,int m,int n,int k,cudaStream_t stream);
cudaError_t resources(bool mixed,int k,cudaFuncAttributes* attributes,int* active_blocks);
}
