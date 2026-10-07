// v118: exact eight-nibble E2M1 -> signed Q4 conversion and squared norm.
// Unlike the scalar decoder or a warp lookup, every Boolean op processes
// eight independent nibbles. No payload shuffle or new global table.
#pragma once
#include <cuda_runtime.h>

namespace nv4_swar_probe {
template<unsigned Lut>
__device__ __forceinline__ unsigned truth(unsigned a,unsigned b,unsigned c) {
  unsigned out;
  asm("lop3.b32 %0,%1,%2,%3,%4;" : "=r"(out)
      : "r"(a),"r"(b),"r"(c),"n"(Lut));
  return out;
}

__device__ __forceinline__ uint2 packed_word(unsigned word) {
  constexpr unsigned Lsb=0x11111111u;
  const unsigned x0=word&Lsb,x1=(word>>1)&Lsb,x2=(word>>2)&Lsb;
  const unsigned sign=(word>>3)&Lsb;
  // RNE magnitudes for E2M1 codes0..7 are {0,0,1,2,2,3,4,6}.
  // LOP3 input order is x2,x1,x0; index=(x2<<2)|(x1<<1)|x0.
  const unsigned m0=truth<0x24>(x2,x1,x0);
  const unsigned m1=truth<0xb8>(x2,x1,x0);
  const unsigned m2=truth<0xc0>(x2,x1,x0);
  const unsigned magnitude=m0|(m1<<1)|(m2<<2);
  // Per-nibble addition never exceeds8, so no carry crosses a nibble.
  // Negative zero and negative0.5 must both become the same Q4 zero.
  const unsigned low=((magnitude^(sign*7u))+sign)&0x77777777u;
  const unsigned negative_nonzero=sign&(x1|x2);
  const unsigned packed=low|(negative_nonzero<<3);
  // Squares for magnitude codes0..7: {0,0,1,4,4,9,16,36}.
  // Their five nonzero bit planes give an exact sum of eight q*q values.
  const unsigned square=unsigned(__popc(m0))
      +4u*unsigned(__popc(truth<0x98>(x2,x1,x0)))
      +8u*unsigned(__popc(truth<0x20>(x2,x1,x0)))
      +16u*unsigned(__popc(truth<0x40>(x2,x1,x0)))
      +32u*unsigned(__popc(truth<0x80>(x2,x1,x0)));
  return make_uint2(packed,square);
}
} // namespace nv4_swar_probe
