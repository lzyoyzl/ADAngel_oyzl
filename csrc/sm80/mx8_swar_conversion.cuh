// v126: four byte-held E4M3FN values -> signed Q8 bytes, F=-2/RNE.
// Source validation rejects NaN codes. Even the unchecked legacy decoder's
// two reserved-code numeric results are preserved; this does not legalize NaN.
#pragma once
#include "nv6_swar_conversion.cuh"
namespace mx8_swar_probe {
using nv6_swar_probe::compact_nibbles;
using nv6_swar_probe::square_add;

__device__ __forceinline__ unsigned bytes(unsigned word) {
  constexpr unsigned Lsb=0x01010101u;
  const unsigned mant=(word&0x07070707u)|0x08080808u;
  const unsigned e0=((word>>3)&Lsb)*255u;
  const unsigned e1=((word>>4)&Lsb)*255u;
  const unsigned e2=((word>>5)&Lsb)*255u;
  const unsigned e3=((word>>6)&Lsb)*255u;
  // For exponent fields8..11, round mantissa / {16,8,4,2} to nearest even.
  // All additions fit in each byte; masks remove adjacent-byte shift bits.
  const unsigned q0=((mant+0x07070707u)>>4)&Lsb;
  const unsigned q1=((mant+0x04040404u)>>3)&0x03030303u;
  const unsigned q2=((mant+Lsb+((mant>>2)&Lsb))>>2)&0x07070707u;
  const unsigned q3=((mant+((mant>>1)&Lsb))>>1)&0x0f0f0f0fu;
  const unsigned small01=(q0&~e0)|(q1&e0);
  const unsigned small23=(q2&~e0)|(q3&e0);
  const unsigned small=(small01&~e1)|(small23&e1);
  // Fields12..15 are exact integer shifts of8+fraction. Largest lane=120.
  const unsigned large01=(mant&~e0)|((mant<<1)&e0);
  const unsigned large=(large01&~e1)|((large01<<2)&e1);
  const unsigned magnitude=((small&~e2)|(large&e2))&e3;
  const unsigned sign=(word>>7)&Lsb;
  // Per-byte intermediate <=128, including negative zero: no cross-byte carry.
  return ((magnitude^(sign*127u))+sign)^(sign<<7);
}
} // namespace mx8_swar_probe
