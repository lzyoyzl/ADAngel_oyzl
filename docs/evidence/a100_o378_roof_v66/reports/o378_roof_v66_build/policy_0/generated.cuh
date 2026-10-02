// Exact v53 device conversion shared by the isolated probe and opt-in v54 pipeline.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "roof_fused_conversion_api.h"

namespace vector_probe {
using Kind=adangel_sm80_experiment::GroupedSourceKind;
__device__ __forceinline__ unsigned rne(unsigned x,unsigned bits) {
  const unsigned whole=x>>bits,rem=x&((1u<<bits)-1),half=1u<<(bits-1);
  return whole+unsigned(rem>half || (rem==half && (whole&1)));
}
__device__ __forceinline__ int nv4(unsigned c) {
  const int q=(0x64322100u>>(4*(c&7)))&15;
  return c&8?-q:q;
}
__device__ __forceinline__ float e4m3(unsigned c) {
  const unsigned e=(c&127)>>3,m=c&7;
  if(e) return __uint_as_float(((c&128)<<24)|((e+120)<<23)|(m<<20));
  const float v=float(m)*.001953125f;
  return c&128?-v:v;
}
__device__ __forceinline__ int nv6(unsigned c) {
  const unsigned e=(c&31)>>3,m=c&7;
  const int q=e>=2?int((8+m)<<(e-2)):int(rne(e?8+m:m,1));
  return c&32?-q:q;
}

template<Kind Format,int Elements>
__global__ void adangel_sm80_vector_fixed_conversion(
    const uint8_t* payload,const uint8_t* scale,const float* tensor_scale,
    const uint8_t* micro8,const uint8_t* micro4,uint8_t* packed,float* effective,
    unsigned rows,unsigned groups) {
  constexpr bool Weight=Format==Kind::Nv4 || Format==Kind::Hif4;
  constexpr unsigned Lanes=128/Elements;
  const unsigned row=blockIdx.x*(256/Lanes)+threadIdx.x/Lanes;
  if(row>=rows) return;
  const unsigned g=blockIdx.y,lane=threadIdx.x%Lanes;
  const unsigned src_group=row*groups+g,dst_group=g*rows+row;
  unsigned words[Elements/4]={};
  if constexpr(Weight && Elements==8) {
    words[0]=reinterpret_cast<const unsigned*>(payload)[src_group*16+lane];
  } else if constexpr(Weight || Elements==8) {
    const uint2 v=reinterpret_cast<const uint2*>(payload)[src_group*(Weight?8:16)+lane];
    words[0]=v.x;words[1]=v.y;
  } else {
    const uint4 v=reinterpret_cast<const uint4*>(payload)[src_group*8+lane];
    words[0]=v.x;words[1]=v.y;words[2]=v.z;words[3]=v.w;
  }
  unsigned low[Elements/8]={},high[Elements/8]={};
  // Micro8 is shared by eight elements; micro4 by four. Decode once per four,
  // rather than reloading each metadata bit for every two source elements.
  #pragma unroll
  for(int j4=0;j4<Elements/4;++j4) {
    unsigned micro=0;
    if constexpr(Format==Kind::Hif4) {
      const unsigned element=lane*Elements+j4*4,i8=element/8,i4=element/4;
      micro=((micro8[src_group*2+i8/8]>>(i8%8))&1)
          +((micro4[src_group*4+i4/8]>>(i4%8))&1);
    }
    #pragma unroll
    for(int t=0;t<4;++t) {
      constexpr unsigned Width=Weight?4:8;
      const int j=j4*4+t;
      const unsigned c=(words[j/(32/Width)]>>((j%(32/Width))*Width))&((1u<<Width)-1);
      int q;
      if constexpr(Format==Kind::Nv4) q=nv4(c);
      else if constexpr(Format==Kind::Hif4) {
        const int v=int(rne((c&7)<<micro,2));q=c&8?-v:v;
      } else if constexpr(Format==Kind::Mx8) q=__float2int_rn(__fmul_rn(e4m3(c),.25f));
      else q=nv6(c);
      low[j/8]|=(unsigned(q)&15u)<<(4*(j%8));
      if constexpr(!Weight) high[j/8]|=((unsigned(q)>>4)&15u)<<(4*(j%8));
    }
  }
  const unsigned dest=dst_group*(128/Elements)+lane;
  if constexpr(Elements==8) {
    reinterpret_cast<unsigned*>(packed)[dest]=low[0];
    if constexpr(!Weight) reinterpret_cast<unsigned*>(packed)[rows*groups*16+dest]=high[0];
  } else {
    reinterpret_cast<uint2*>(packed)[dest]=make_uint2(low[0],low[1]);
    if constexpr(!Weight) reinterpret_cast<uint2*>(packed)[rows*groups*8+dest]=make_uint2(high[0],high[1]);
  }
  if(lane==0) {
    const unsigned c=scale[src_group];float value;
    if constexpr(Format==Kind::Hif4) value=__uint_as_float(((c/4+79)<<23)|((c&3)<<21));
    else if constexpr(Format==Kind::Mx8) value=__fmul_rn(__uint_as_float(c?c<<23:0x00400000u),4.f);
    else {
      value=__fmul_rn(e4m3(c),tensor_scale[0]);
      if constexpr(Format==Kind::Nv6) value=__fmul_rn(value,.25f);
    }
    effective[dst_group]=value;
  }
}

}  // namespace vector_probe

