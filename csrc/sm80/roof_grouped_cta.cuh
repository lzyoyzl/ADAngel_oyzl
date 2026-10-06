// v89 isolated experiment: one-to-one grouped CTA ordering, not an SMEM swizzle.
// Source: Triton 03-matrix-multiplication, grouped L2 ordering. No autotuning.
#pragma once
namespace roof_grouped_cta {
struct Tile { unsigned x, y; };
__device__ __forceinline__ Tile tile() {
  const unsigned columns=gridDim.x, rows=gridDim.y;
  // The actual 4096^3 case is 32 columns x64 rows. Use only bit operations
  // and one uniform multiplication; no division in its group loop.
  if ((columns&7u)==0u && (rows&7u)==0u) {
    return {(blockIdx.y&7u)*(columns>>3)+(blockIdx.x>>3),
            (blockIdx.y&~7u)+(blockIdx.x&7u)};
  }
  // Small validation shapes and a partial final group retain exact coverage.
  const unsigned linear=blockIdx.y*columns+blockIdx.x;
  const unsigned span=8u*columns, first=(linear/span)*8u;
  const unsigned height=(rows-first<8u)?rows-first:8u;
  const unsigned within=linear-(first/8u)*span;
  return {within/height,first+within%height};
}
} // namespace roof_grouped_cta
