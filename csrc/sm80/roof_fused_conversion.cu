// Conversion candidates43/44 use the EXACT GEMM symbols of41/42. Only conversion
// writes the G128-major destination directly; no intermediate natural payload.
#include <cuda_runtime.h>
#include "adangel/data_types.cuh"
#include "roof_fused_conversion_api.h"

namespace {
using Kind=adangel_sm80_experiment::GroupedSourceKind;

__device__ int grouped_pair(int pair,int rows,int groups) {
  const int row=pair/(groups*64),group=(pair/64)%groups;
  return (group*rows+row)*64+pair%64;
}

template<bool Activation>
__global__ void adangel_sm80_o3_fused_g128_conversion(
    const uint8_t* source,uint8_t* packed,int rows,int groups,int pairs) {
  const int pair=blockIdx.x*blockDim.x+threadIdx.x;
  if(pair>=pairs) return;
  const int dest=grouped_pair(pair,rows,groups);
  if constexpr(Activation) {
    const uchar2 v=reinterpret_cast<const uchar2*>(source)[pair];
    packed[dest]=uint8_t((v.x&15)|((v.y&15)<<4));
    packed[pairs+dest]=uint8_t((v.x>>4)|((v.y>>4)<<4));
  } else {
    const uint8_t v=source[pair];
    packed[dest]=uint8_t((adangel::e2m1_to_q4(v&15)&15)|
                        ((adangel::e2m1_to_q4(v>>4)&15)<<4));
  }
}

// Match mixed_conversion.cuh exactly, including normal exponent rebiasing,
// signed subnormals and ordered FP32 scale products. Host validation is shared.
__device__ float e4m3(uint8_t code) {
  const int exp=(code&127)>>3,mant=code&7;
  if(exp) return __uint_as_float((uint32_t(code&128)<<24)|
      (uint32_t(exp+120)<<23)|(uint32_t(mant)<<20));
  const float value=float(mant)*.001953125f;
  return code&128?-value:value;
}
__device__ float ue8m0(uint8_t code) {
  return __uint_as_float(code?uint32_t(code)<<23:0x00400000u);
}
__device__ float hif_scale(uint8_t code) {
  return __uint_as_float((uint32_t((code>>2)+79)<<23)|(uint32_t(code&3)<<21));
}

template<Kind Format>
__global__ void adangel_sm80_mixed_fused_g128_conversion(
    const uint8_t* payload,const uint8_t* scale,const float* tensor_scale,
    const uint8_t* micro8,const uint8_t* micro4,uint8_t* packed,float* effective,
    int rows,int groups,int pairs) {
  const int pair=blockIdx.x*blockDim.x+threadIdx.x;
  if(pair>=pairs) return;
  const int group=pair/64,in_group=pair%64;
  int q[2];
  if constexpr(Format==Kind::Nv4) {
    const uint8_t byte=payload[pair];
    q[0]=adangel::e2m1_to_q4(byte&15);
    q[1]=adangel::e2m1_to_q4(byte>>4);
  } else if constexpr(Format==Kind::Hif4) {
    const int i8=in_group/4,i4=in_group/2;
    const int e8=(micro8[group*2+i8/8]>>(i8%8))&1;
    const int e4=(micro4[group*4+i4/8]>>(i4%8))&1;
    const uint8_t byte=payload[pair];
    #pragma unroll
    for(int j=0;j<2;++j) {
      const int code=(byte>>(4*j))&15;
      const float mag=float((code&7)<<(e8+e4))*.25f;
      q[j]=__float2int_rn(code&8?-mag:mag);
    }
  } else {
    #pragma unroll
    for(int j=0;j<2;++j) {
      const uint8_t code=payload[2*pair+j];
      if constexpr(Format==Kind::Mx8) {
        q[j]=__float2int_rn(__fmul_rn(e4m3(code),.25f));
      } else {
        const int exp=(code&31)>>3,mant=code&7;
        const float mag=exp?float((8+mant)<<exp)*.25f:float(mant)*.5f;
        q[j]=__float2int_rn(code&32?-mag:mag);
      }
    }
  }
  const int dest=grouped_pair(pair,rows,groups);
  if constexpr(Format==Kind::Nv4 || Format==Kind::Hif4) {
    packed[dest]=uint8_t((q[0]&15)|((q[1]&15)<<4));
  } else {
    const uint8_t a=uint8_t(q[0]),b=uint8_t(q[1]);
    packed[dest]=uint8_t((a&15)|((b&15)<<4));
    packed[pairs+dest]=uint8_t((a>>4)|((b>>4)<<4));
  }
  if(in_group==0) {
    const int index=(group%groups)*rows+group/groups;
    if constexpr(Format==Kind::Hif4) effective[index]=hif_scale(scale[group]);
    else if constexpr(Format==Kind::Mx8) effective[index]=__fmul_rn(ue8m0(scale[group]),4.f);
    else {
      float value=__fmul_rn(e4m3(scale[group]),tensor_scale[0]);
      if constexpr(Format==Kind::Nv6) value=__fmul_rn(value,.25f);
      effective[index]=value;
    }
  }
}
}

namespace adangel_sm80_experiment {
void fused_o3_weight(const uint8_t* source,uint8_t* packed,int rows,int k,cudaStream_t stream) {
  const int pairs=rows*(k/2);
  adangel_sm80_o3_fused_g128_conversion<false><<<(pairs+255)/256,256,0,stream>>>(source,packed,rows,k/128,pairs);
}
void fused_o3_activation(const int8_t* source,uint8_t* packed,int rows,int k,cudaStream_t stream) {
  const int pairs=rows*(k/2);
  adangel_sm80_o3_fused_g128_conversion<true><<<(pairs+255)/256,256,0,stream>>>(
      reinterpret_cast<const uint8_t*>(source),packed,rows,k/128,pairs);
}
void fused_mixed_fixed(Kind kind,const uint8_t* payload,const uint8_t* scale,
    const float* tensor_scale,const uint8_t* micro8,const uint8_t* micro4,
    uint8_t* packed,float* effective,int rows,int k,cudaStream_t stream) {
  const int pairs=rows*(k/2);
  const dim3 grid((pairs+255)/256);
  #define LAUNCH(K) adangel_sm80_mixed_fused_g128_conversion<Kind::K><<<grid,256,0,stream>>>( \
      payload,scale,tensor_scale,micro8,micro4,packed,effective,rows,k/128,pairs)
  switch(kind) {
    case Kind::Nv4: LAUNCH(Nv4); break;
    case Kind::Mx8: LAUNCH(Mx8); break;
    case Kind::Hif4: LAUNCH(Hif4); break;
    case Kind::Nv6: LAUNCH(Nv6); break;
  }
  #undef LAUNCH
}
}
