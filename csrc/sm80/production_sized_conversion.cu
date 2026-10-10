#include <cuda_runtime.h>
#include "production_sized_api.h"
#include "roof_vector_conversion_impl.cuh"
#include "production_sized_generated/guard.cuh"
#include "production_sized_generated/nv4.cuh"
#include "production_sized_generated/mx8.cuh"
#include "production_sized_generated/hif4.cuh"
#include "production_sized_generated/nv6.cuh"
#include "production_sized_generated/o3_prepare.cuh"

namespace adangel_sm80_production_sized {
void o3_prepare(const uint8_t* scales,int32_t* metadata,uint32_t* status,int n,int k,cudaStream_t stream) {
  if(k==512)adangel_sized_factor_prepare<4><<<n/128,128,0,stream>>>(scales,metadata,status,n);
  else adangel_sized_factor_prepare<8><<<n/128,128,0,stream>>>(scales,metadata,status,n);
}
void mixed_convert(int variant,bool act,const uint64_t* s,const uint64_t* v,
    int m,int n,int k,float multiplier,cudaStream_t stream) {
  using Kind=adangel_sm80_experiment::GroupedSourceKind;
  const int rows=act?m:n;
  #define P(T,I) reinterpret_cast<T*>(v[I])
  #define LAUNCH(NS,F,KIND) NS::F<KIND,16><<<rows,256,0,stream>>>( \
    reinterpret_cast<const uint8_t*>(s[0]),reinterpret_cast<const uint8_t*>(s[1]), \
    reinterpret_cast<const float*>(s[2]),reinterpret_cast<const uint8_t*>(s[3]), \
    reinterpret_cast<const uint8_t*>(s[4]),P(uint8_t,act?0:1),P(float,act?2:3), \
    P(uint32_t,act?16:17),rows,unsigned(k/128),multiplier,P(int32_t,act?4:5),P(float,act?6:7), \
    P(uint64_t,act?8:9),P(int32_t,act?10:11),P(uint32_t,act?12:13))
  if(variant==7) {
    if(act) {LAUNCH(mx8_warp_lut_sized,adangel_sm80_row_warp_lut_metadata,Kind::Mx8);}
    else {LAUNCH(nv4_row_swar_sized,adangel_sm80_row_swar_metadata,Kind::Nv4);}
  } else {
    if(act) {LAUNCH(nv6_row_swar_sized,adangel_sm80_nv6_swar_metadata,Kind::Nv6);}
    else {LAUNCH(hif4_row_swar_sized,adangel_sm80_hif4_swar_metadata,Kind::Hif4);}
  }
  if(act)adangel_o78_prepare_cta_guard_sized<<<dim3(n/128,m/64),128,0,stream>>>(
    P(const uint64_t,8),P(const uint64_t,9),P(const int32_t,10),P(const int32_t,11),
    P(const uint32_t,12),P(const uint32_t,13),P(const float,6),P(const float,7),P(uint32_t,14),m,n);
  #undef LAUNCH
  #undef P
}
}
