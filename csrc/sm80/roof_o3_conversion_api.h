#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>

namespace adangel_sm80_experiment {
// Internal v36 candidates: exact Q4/split, final G128 layout, no GEMM change.
void tiled_o3_conversion(bool activation,bool vectorized,const uint8_t* source,
    const uint8_t* scale,uint8_t* packed,uint8_t* grouped_scale,
    int rows,int k,cudaStream_t stream);
}
