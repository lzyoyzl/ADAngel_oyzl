// v34: exact integer payload RNE and optional destination-oriented traversal.
// The scale decoder/products are intentionally identical to v25, including
// FP32 subnormal UE8M0 handling and NV6's ordered second multiply by 1/4.
#include <cuda_runtime.h>
#include "adangel/data_types.cuh"
#include "roof_integer_conversion_api.h"

namespace {
using Kind=adangel_sm80_experiment::GroupedSourceKind;

__device__ __forceinline__ unsigned right_rne(unsigned x,unsigned bits) {
  // Callers guarantee 1<=bits<=11. All significands are small/nonnegative.
  const unsigned whole=x>>bits,rem=x&((1u<<bits)-1),half=1u<<(bits-1);
  return whole+unsigned(rem>half || (rem==half && (whole&1)));
}
__device__ __forceinline__ int payload_nv4(uint8_t code) {
  // Exact E2M1 F=0 RNE values [0,0,1,2,2,3,4,6], packed in one register.
  const int q=(0x64322100u>>(4*(code&7)))&15u;
  return code&8?-q:q;
}
__device__ __forceinline__ int payload_mx8(uint8_t code) {
  const unsigned e=(code&127)>>3,m=code&7;
  // E4M3 * 2^-2: normals (8+m)*2^(e-12), subnormals m*2^-11.
  // NaN payloads127/255 are rejected by the existing host contract.
  const unsigned q=e>=12 ? ((8+m)<<(e-12)) : right_rne(e?8+m:m,e?12-e:11);
  return code&128?-int(q):int(q);
}
__device__ __forceinline__ int payload_nv6(uint8_t code) {
  const unsigned e=(code&31)>>3,m=code&7;
  // E2M3 * 2^2: normals (8+m)*2^(e-2), subnormals m/2.
  const unsigned q=e>=2 ? ((8+m)<<(e-2)) : right_rne(e?8+m:m,1);
  return code&32?-int(q):int(q);
}
__device__ float e4m3_scale(uint8_t code) {
  const int exp=(code&127)>>3,mant=code&7;
  if(exp) return __uint_as_float((uint32_t(code&128)<<24)|
      (uint32_t(exp+120)<<23)|(uint32_t(mant)<<20));
  const float value=float(mant)*.001953125f;
  return code&128?-value:value;
}
__device__ float ue8m0_scale(uint8_t code) {
  return __uint_as_float(code?uint32_t(code)<<23:0x00400000u);
}
__device__ float hif_scale(uint8_t code) {
  return __uint_as_float((uint32_t((code>>2)+79)<<23)|(uint32_t(code&3)<<21));
}

template<Kind Format,bool Tiled>
__global__ void adangel_sm80_integer_fixed_conversion(
    const uint8_t* payload,const uint8_t* scale,const float* tensor_scale,
    const uint8_t* micro8,const uint8_t* micro4,uint8_t* packed,float* effective,
    int rows,int groups,int pairs) {
  int pair,row,g,in_group;
  if constexpr(Tiled) {
    // One CTA covers four rows of ONE G128. Both source and destination
    // have contiguous64-byte spans per row; no general division is needed.
    // X counts row tiles, Y counts groups (host checks grid.y<=65535).
    row=int(blockIdx.x)*4+int(threadIdx.x)/64;
    if(row>=rows) return;
    g=int(blockIdx.y);in_group=int(threadIdx.x)%64;
    pair=(row*groups+g)*64+in_group;
  } else {
    // Preserve the v25 flat/source-major traversal as the RNE-only control.
    pair=int(blockIdx.x)*blockDim.x+threadIdx.x;
    if(pair>=pairs) return;
    row=pair/(groups*64);g=(pair/64)%groups;in_group=pair%64;
  }
  const int group=pair/64,dest=(g*rows+row)*64+in_group;
  int q[2];
  if constexpr(Format==Kind::Nv4) {
    const uint8_t byte=payload[pair];
    q[0]=payload_nv4(byte&15);q[1]=payload_nv4(byte>>4);
  } else if constexpr(Format==Kind::Hif4) {
    const int i8=in_group/4,i4=in_group/2;
    const int e8=(micro8[group*2+i8/8]>>(i8%8))&1;
    const int e4=(micro4[group*4+i4/8]>>(i4%8))&1;
    const uint8_t byte=payload[pair];
    #pragma unroll
    for(int j=0;j<2;++j) {
      const unsigned code=(byte>>(4*j))&15;
      const int value=int(right_rne((code&7)<<(e8+e4),2));
      q[j]=code&8?-value:value;
    }
  } else {
    #pragma unroll
    for(int j=0;j<2;++j) {
      const uint8_t code=payload[2*pair+j];
      if constexpr(Format==Kind::Mx8) q[j]=payload_mx8(code);
      else q[j]=payload_nv6(code);
    }
  }
  if constexpr(Format==Kind::Nv4 || Format==Kind::Hif4) {
    packed[dest]=uint8_t((q[0]&15)|((q[1]&15)<<4));
  } else {
    const uint8_t a=uint8_t(q[0]),b=uint8_t(q[1]);
    packed[dest]=uint8_t((a&15)|((b&15)<<4));
    packed[pairs+dest]=uint8_t((a>>4)|((b>>4)<<4));
  }
  if(in_group==0) {
    const int index=g*rows+row;
    if constexpr(Format==Kind::Hif4) effective[index]=hif_scale(scale[group]);
    else if constexpr(Format==Kind::Mx8) effective[index]=__fmul_rn(ue8m0_scale(scale[group]),4.f);
    else {
      float value=__fmul_rn(e4m3_scale(scale[group]),tensor_scale[0]);
      if constexpr(Format==Kind::Nv6) value=__fmul_rn(value,.25f);
      effective[index]=value;
    }
  }
}
}

namespace adangel_sm80_experiment {
void integer_mixed_fixed(Kind kind,bool tiled,const uint8_t* payload,const uint8_t* scale,
    const float* tensor_scale,const uint8_t* micro8,const uint8_t* micro4,
    uint8_t* packed,float* effective,int rows,int k,cudaStream_t stream) {
  const int pairs=rows*(k/2);
  const dim3 grid=tiled?dim3((rows+3)/4,k/128):dim3((pairs+255)/256);
  #define LAUNCH(K,T) adangel_sm80_integer_fixed_conversion<Kind::K,T><<<grid,256,0,stream>>>( \
      payload,scale,tensor_scale,micro8,micro4,packed,effective,rows,k/128,pairs)
  #define CASE(K) case Kind::K: if(tiled) {LAUNCH(K,true);} else {LAUNCH(K,false);} break
  switch(kind) {CASE(Nv4);CASE(Mx8);CASE(Hif4);CASE(Nv6);}
  #undef CASE
  #undef LAUNCH
}
}
