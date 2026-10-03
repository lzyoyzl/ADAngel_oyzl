// v85: exact nibble permutation only. CuTe owns the MMA thread/value mapping.
#pragma once
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cutlass/numeric_types.h>

namespace o78_register_pack {
template<bool Weight>
__global__ void pack(const uint8_t* src,uint8_t* dst,int rows) {
  // One warp owns a16-row G128 tile. W uses two adjacent N8 atoms, A one M16.
  // Each lane writes one contiguous16-byte fragment for each K64 half.
  using Atom=cute::MMA_Atom<cute::SM80_16x8x64_S32U4S4S32_TN>;
  auto mma=cute::make_tiled_mma(Atom{});
  auto thr=mma.get_slice(threadIdx.x);
  const int atom=blockIdx.x,group=blockIdx.y;
  constexpr int Planes=Weight?1:2;
  #pragma unroll
  for(int plane=0;plane<Planes;++plane) {
    #pragma unroll
    for(int half=0;half<2;++half) {
      const uint8_t* p=src+size_t(plane)*rows*2048+(group*rows+atom*16)*64+half*32;
      uint8_t* out=dst+size_t(plane)*rows*2048+(group*(rows/16)+atom)*1024+half*512+threadIdx.x*16;
      auto layout=cute::make_layout(cute::make_shape(cute::_16{},cute::_64{}),
                                    cute::make_stride(cute::_128{},cute::_1{}));
      if constexpr(Weight) {
        auto tensor=cute::make_tensor(cute::make_gmem_ptr<cutlass::int4b_t>(
            static_cast<const void*>(p)),layout);
        auto fragment=thr.partition_fragment_B(tensor);
        cute::copy(thr.partition_B(tensor),fragment);
        auto words=cute::recast<uint32_t>(fragment);
        static_assert(decltype(cute::size(words))::value==4);
        *reinterpret_cast<uint4*>(out)=make_uint4(words(0),words(1),words(2),words(3));
      } else {
        auto tensor=cute::make_tensor(cute::make_gmem_ptr<cutlass::uint4b_t>(
            static_cast<const void*>(p)),layout);
        auto fragment=thr.partition_fragment_A(tensor);
        cute::copy(thr.partition_A(tensor),fragment);
        auto words=cute::recast<uint32_t>(fragment);
        static_assert(decltype(cute::size(words))::value==4);
        *reinterpret_cast<uint4*>(out)=make_uint4(words(0),words(1),words(2),words(3));
      }
    }
  }
}
} // namespace o78_register_pack
