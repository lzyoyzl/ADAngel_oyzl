// Exact storage addressing shared by the v85 consumer and host CuTe verifier.
#pragma once
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm80.hpp>

namespace o78_register_layout_mapping {
using Atom=cute::MMA_Atom<cute::SM80_16x8x64_S32U4S4S32_TN>;
using PackA=decltype(cute::make_tiled_mma(Atom{}));
// Neighboring N atoms owned by one warp are 16 columns apart, not 8.
// Two warps together pack N32; each warp emits its own two N8 fragments.
using PackW=decltype(cute::make_tiled_mma(Atom{},
    cute::Layout<cute::Shape<cute::_1,cute::_2,cute::_1>>{}));
using FullMma=cute::TiledMMA<Atom,cute::Layout<cute::Shape<cute::_2,cute::_2,cute::_1>>,
    cute::Tile<cute::_64,cute::_128,cute::_64>>;
using SliceMma=cute::TiledMMA<Atom,cute::Layout<cute::Shape<cute::_2,cute::_2,cute::_1>>,
    cute::Tile<cute::_64,cute::_64,cute::_64>>;

CUTE_HOST_DEVICE constexpr int a_offset(int row,int half,int lane) {
  return (row/16)*1024+half*512+lane*16;
}
CUTE_HOST_DEVICE constexpr int w_offset(int column,int half,int lane) {
  return (column/32)*2048+((column%32)/8)*1024+half*512+lane*16;
}
} // namespace o78_register_layout_mapping
