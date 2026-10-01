// v54: same device body as the numerically audited v53 vector16 probe.
#include "roof_vector_conversion_api.h"
#include "roof_vector_conversion_impl.cuh"

namespace adangel_sm80_experiment {
void vector_mixed_fixed(GroupedSourceKind kind,
    const uint8_t* payload,const uint8_t* scale,const float* tensor_scale,
    const uint8_t* micro8,const uint8_t* micro4,uint8_t* packed,float* effective,
    int rows,int k,cudaStream_t stream) {
  // Shape/range/source/alignment guards run once in the caller before timing.
  const dim3 grid((rows+31)/32,k/128);
  #define VCASE(K) case GroupedSourceKind::K: \
    vector_probe::adangel_sm80_vector_fixed_conversion<GroupedSourceKind::K,16> \
      <<<grid,256,0,stream>>>(payload,scale,tensor_scale,micro8,micro4,packed,effective,rows,k/128);break
  switch(kind) {VCASE(Nv4);VCASE(Mx8);VCASE(Hif4);VCASE(Nv6);}
  #undef VCASE
}
}
