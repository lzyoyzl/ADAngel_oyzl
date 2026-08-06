#pragma once

#include <stdint.h>

namespace adangel {

// Canonical PTX scale_vec::2X selector-{0,0} packing. A and B are separate .b32 operands.
__device__ inline uint32_t pack_sfa_word(int lane, const uint8_t* sfa, int lda) {
  int q = lane >> 2;
  switch (lane & 3) {
    case 0:
      return static_cast<uint32_t>(sfa[q * lda]) |
             (static_cast<uint32_t>(sfa[q * lda + 1]) << 8);
    case 1:
      return static_cast<uint32_t>(sfa[(q + 8) * lda]) |
             (static_cast<uint32_t>(sfa[(q + 8) * lda + 1]) << 8);
    default:
      return 0;
  }
}

__device__ inline uint32_t pack_sfb_word(int lane, const uint8_t* sfb, int ldb) {
  if ((lane & 3) != 0) return 0;
  int q = lane >> 2;
  return static_cast<uint32_t>(sfb[q]) |
         (static_cast<uint32_t>(sfb[ldb + q]) << 8);
}

}  // namespace adangel
