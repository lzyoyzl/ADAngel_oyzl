// v123: four independent byte-held E2M3 codes -> signed Q6/INT8 bytes.
// Exact old F=2/RNE semantics; unused upper input bits are ignored.
#pragma once
#include <cuda_runtime.h>

namespace nv6_swar_probe {
__device__ __forceinline__ unsigned bytes(unsigned word) {
  constexpr unsigned Lsb=0x01010101u;
  const unsigned e0=(word>>3)&Lsb,e1=(word>>4)&Lsb;
  const unsigned sign=(word>>5)&Lsb;
  // e<2: RNE of the four independent four-bit magnitudes divided by2.
  const unsigned rounded=((word>>1)&0x07070707u)+((word&(word>>1))&Lsb);
  // e>=2: (8+mantissa) * (e==3 ? 2 : 1).
  const unsigned base=(word&0x07070707u)+0x08080808u;
  const unsigned e0mask=e0*255u,e1mask=e1*255u;
  const unsigned large=(base&~e0mask)|((base<<1)&e0mask);
  const unsigned magnitude=(rounded&~e1mask)|(large&e1mask);
  // Each byte's intermediate <=128: no cross-byte carry, even negative0.
  return ((magnitude^(sign*127u))+sign)^(sign<<7);
}

__device__ __forceinline__ unsigned compact_nibbles(unsigned word) {
  const unsigned low=word&0x0f0f0f0fu;
  const unsigned pairs=(low|(low>>4))&0x00ff00ffu;
  return (pairs|(pairs>>8))&0xffffu;
}

__device__ __forceinline__ int square_add(unsigned q,int sum) {
  int out;
  asm("dp4a.s32.s32 %0,%1,%1,%2;" : "=r"(out) : "r"(q),"r"(sum));
  return out;
}
} // namespace nv6_swar_probe
