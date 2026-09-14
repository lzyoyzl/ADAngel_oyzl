#pragma once

#include <cstdint>

// Only use with a proved integer partial bound. This changes conversion
// instructions, not quantization, scale arithmetic or the ordered FP32 FMA.
template <bool MagicBias, int AbsoluteBound>
__device__ __forceinline__ float adangel_exact_partial_cast(int32_t partial) {
  static_assert(AbsoluteBound > 0 && AbsoluteBound < (1 << 22),
                "biased float must stay within the ULP=1 exponent interval");
  if constexpr (MagicBias) {
    return __fadd_rn(__int_as_float(0x4b400000 + partial), -12582912.0f);
  } else {
    return static_cast<float>(partial);
  }
}
