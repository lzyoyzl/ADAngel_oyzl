// v138: independent HiF4 G128 conversion candidate, not a GEMM change.
// Decode eight signed-magnitude payload nibbles together. The micro8 bit
// is common to the word, while its two micro4 bits each cover four nibbles.
#pragma once
#include <cuda_runtime.h>

namespace hif4_swar_probe {
__device__ __forceinline__ uint2 packed_word(unsigned word,unsigned micro8,unsigned micro4) {
  constexpr unsigned Lsb=0x11111111u;
  const unsigned magnitude=word&0x77777777u;
  const unsigned half=((word>>1)&0x33333333u)+((word&(word>>1))&Lsb);
  const unsigned quarter=((word>>2)&Lsb)+(((word>>1)&(word|(word>>2)))&Lsb);
  // RNE(m*2^(micro8+micro4)/4), with m in [0,7]. No nibble overflows.
  const unsigned mask=(micro4&1u)*0xffffu+((micro4>>1)&1u)*0xffff0000u;
  const unsigned hi=micro8?magnitude:half;
  const unsigned lo=micro8?half:quarter;
  const unsigned rounded=(hi&mask)|(lo&~mask);
  const unsigned sign=(word>>3)&Lsb;
  // The inner per-nibble addition is <=8, including negative zero.
  const unsigned packed=((rounded^(sign*7u))+sign)^(sign<<3);
  const unsigned even=rounded&0x07070707u,odd=(rounded>>4)&0x07070707u;
  unsigned square;
  // These are scalar integer dot instructions, NOT INT8 Tensor Core MMA.
  asm("dp4a.u32.u32 %0,%1,%1,0;" : "=r"(square) : "r"(even));
  asm("dp4a.u32.u32 %0,%1,%1,%0;" : "+r"(square) : "r"(odd));
  return make_uint2(packed,square);
}
} // namespace hif4_swar_probe
