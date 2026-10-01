// v36: fuse exact O3 conversion and payload/scale permutation. Independent TU
// prevents changing the established MMA templates or SM120 source.
#include <cuda_runtime.h>
#include "roof_o3_conversion_api.h"

namespace {
__device__ __forceinline__ unsigned q4(unsigned code) {
  const int mag=(0x64322100u>>(4*(code&7)))&15u;
  return unsigned(code&8?-mag:mag)&15u;
}
__device__ __forceinline__ unsigned low_nibbles(unsigned v) {
  // Four little-endian INT8 elements -> two packed U4 bytes.
  return (v&15u)|((v>>4)&0xf0u)|((v>>8)&0xf00u)|((v>>12)&0xf000u);
}

template<bool Activation,bool Vectorized>
__global__ void adangel_sm80_o3_tiled_conversion(const uint8_t* source,
    const uint8_t* scale,uint8_t* packed,uint8_t* grouped_scale,
    unsigned rows,unsigned groups) {
  constexpr unsigned Lanes=Vectorized?16:64;
  const unsigned row=blockIdx.x*(256/Lanes)+threadIdx.x/Lanes;
  if(row>=rows) return;
  const unsigned g=blockIdx.y,i=threadIdx.x%Lanes;
  const unsigned source_group=row*groups+g,dest_group=g*rows+row;
  if constexpr(Vectorized) {
    // Each lane owns eight original elements (4 packed bytes). Input alignment
    // is explicitly checked by the opt-in host path; all G128 strides align.
    const unsigned dst=dest_group*16+i;
    if constexpr(Activation) {
      const uint2 v=reinterpret_cast<const uint2*>(source)[source_group*16+i];
      reinterpret_cast<unsigned*>(packed)[dst]=low_nibbles(v.x)|(low_nibbles(v.y)<<16);
      reinterpret_cast<unsigned*>(packed)[rows*groups*16+dst]=
          low_nibbles(v.x>>4)|(low_nibbles(v.y>>4)<<16);
    } else {
      const unsigned v=reinterpret_cast<const unsigned*>(source)[source_group*16+i];
      unsigned result=0;
      #pragma unroll
      for(int j=0;j<8;++j) result|=q4((v>>(4*j))&15)<<(4*j);
      reinterpret_cast<unsigned*>(packed)[dst]=result;
    }
  } else {
    const unsigned src=source_group*64+i,dst=dest_group*64+i;
    if constexpr(Activation) {
      const uint8_t a=source[2*src],b=source[2*src+1];
      packed[dst]=(a&15)|((b&15)<<4);
      packed[rows*groups*64+dst]=(a>>4)|((b>>4)<<4);
    } else {
      const unsigned v=source[src];
      packed[dst]=q4(v&15)|(q4(v>>4)<<4);
    }
  }
  // Exactly one writer for each scale; byte codes (including code0) unchanged.
  if constexpr(!Activation) if(i==0) grouped_scale[dest_group]=scale[source_group];
}
}

namespace adangel_sm80_experiment {
void tiled_o3_conversion(bool activation,bool vectorized,const uint8_t* source,
    const uint8_t* scale,uint8_t* packed,uint8_t* grouped_scale,
    int rows,int k,cudaStream_t stream) {
  const unsigned tile_rows=vectorized?16:4;
  const dim3 grid((unsigned(rows)+tile_rows-1)/tile_rows,unsigned(k)/128);
  #define LAUNCH(A,V) adangel_sm80_o3_tiled_conversion<A,V><<<grid,256,0,stream>>>(source,scale,packed,grouped_scale,rows,k/128)
  if(activation) {if(vectorized) {LAUNCH(true,true);} else {LAUNCH(true,false);}}
  else {if(vectorized) {LAUNCH(false,true);} else {LAUNCH(false,false);}}
  #undef LAUNCH
}
}
