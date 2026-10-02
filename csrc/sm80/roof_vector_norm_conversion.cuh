// v69: exact vector16 conversion plus group squared norm, no payload reread.
#pragma once
#include "roof_vector_conversion_impl.cuh"
namespace fused_norm_probe {
using namespace vector_probe;
template<Kind Format,int Elements>
__global__ void adangel_sm80_vector_fixed_conversion_with_norm(
    const uint8_t* payload,const uint8_t* scale,const float* tensor_scale,
    const uint8_t* micro8,const uint8_t* micro4,uint8_t* packed,float* effective,
    uint32_t* group_squares,unsigned rows,unsigned groups) {
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
  unsigned square=0;
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
      square+=unsigned(q*q);
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
  // One subwarp covers exactly one G128; source rows are M64/N128 aligned.
  for(int d=Lanes/2;d;d>>=1) square+=__shfl_down_sync(0xffffffff,square,d,Lanes);
  if(lane==0) {
    group_squares[src_group]=square;
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


} // namespace fused_norm_probe

