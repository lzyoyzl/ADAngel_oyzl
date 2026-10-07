// Generated from pinned CuTe operations; only asm volatility changes.
#pragma once
#include <cute/atom/mma_traits_sm80.hpp>
struct AdangelRegisterOnlyU4S4
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm(
      "mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use AdangelRegisterOnlyU4S4 without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};
struct AdangelRegisterOnlyS4S4
{
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1, uint32_t      & d2, uint32_t      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1, uint32_t const& c2, uint32_t const& c3)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm(
      "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1),  "r"(c2),  "r"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use AdangelRegisterOnlyS4S4 without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};
namespace cute {
template<> struct MMA_Traits<::AdangelRegisterOnlyU4S4> : MMA_Traits<SM80_16x8x64_S32U4S4S32_TN> {};
template<> struct MMA_Traits<::AdangelRegisterOnlyS4S4> : MMA_Traits<SM80_16x8x64_S32S4S4S32_TN> {};
} // namespace cute
