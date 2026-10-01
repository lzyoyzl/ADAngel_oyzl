// Fixed vector16 conversion, internal SM80 candidate5; defaults unchanged.
#pragma once
#include "roof_fused_conversion_api.h"
namespace adangel_sm80_experiment {
void vector_mixed_fixed(GroupedSourceKind kind,
    const uint8_t* payload,const uint8_t* scale,const float* tensor_scale,
    const uint8_t* micro8,const uint8_t* micro4,uint8_t* packed,float* effective,
    int rows,int k,cudaStream_t stream);
}
