// v101 diagnostic only: four warp intervals, not exact CTA residency events.
// Scratch is AFTER the complete FP32 output, never in the numerical payload.
#pragma once
#include <cstdint>
namespace roof_cta_timeline {
template<bool End>
__device__ __forceinline__ void stamp(float* y,int m,int n) {
  if((threadIdx.x&31u)!=0u) return;
  unsigned long long now;
  unsigned sm;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now) :: "memory");
  asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
  const unsigned cta=blockIdx.y*gridDim.x+blockIdx.x;
  auto* record=reinterpret_cast<unsigned long long*>(y+size_t(m)*n)
      +(size_t(cta)*4+(threadIdx.x>>5))*4;
  // Write immediately. Never keep a timestamp live across the hot G128 loop.
  record[End?1:0]=now;
  record[End?3:2]=sm;
}
}
