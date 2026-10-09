#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>

namespace adangel_sm80_production {
// State: pa,pw,as,ws,af,wf,ab,wb,an,wn,am,wm,ast,wst,status,y,asq,wsq.
void mixed_convert(int variant, bool activation, const uint64_t* source,
                   const uint64_t* state, int m, int n, float multiplier, cudaStream_t stream);
void mixed_gemm(const uint64_t* state, int m, int n, cudaStream_t stream);
void o3_prepare(const uint8_t* scales, int32_t* metadata, uint32_t* status, int n, cudaStream_t stream);
void o3_gemm(const uint8_t* a, const uint8_t* w, const float* as, const uint8_t* ws,
             const int32_t* metadata, const uint32_t* status, float* y, int m, int n, cudaStream_t stream);
// Configure dynamic shared memory and query resources outside timing.
cudaError_t resources(bool mixed, cudaFuncAttributes* attributes, int* active_blocks);
}
