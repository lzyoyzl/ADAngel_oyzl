// SM80-only experiment. No ATen/device templates enter the legacy translation unit.
#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>

namespace adangel_sm80_experiment {
enum class GroupedSourceKind { Nv4, Mx8, Hif4, Nv6 };
void fused_o3_weight(const uint8_t* source,uint8_t* packed,int rows,int k,cudaStream_t stream);
void fused_o3_activation(const int8_t* source,uint8_t* packed,int rows,int k,cudaStream_t stream);
void fused_mixed_fixed(GroupedSourceKind kind,const uint8_t* payload,const uint8_t* scale,
    const float* tensor_scale,const uint8_t* micro8,const uint8_t* micro4,
    uint8_t* packed,float* effective,int rows,int k,cudaStream_t stream);
}
