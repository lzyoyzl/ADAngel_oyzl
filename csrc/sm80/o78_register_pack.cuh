// v85: exact nibble permutation only. CuTe owns the MMA thread/value mapping.
#pragma once
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cutlass/numeric_types.h>
#include "o78_register_layout_mapping.cuh"

namespace o78_register_pack {
template<bool Weight>
__global__ void pack(const uint8_t* src,uint8_t* dst,int rows) {
  // A: one warp/M16. W: two warps/N32, each owns N8 atoms 16 columns apart.
  // Each lane writes one contiguous16-byte fragment for each K64 half.
  using MMA=std::conditional_t<Weight,o78_register_layout_mapping::PackW,
      o78_register_layout_mapping::PackA>;
  MMA mma;
  auto thr=mma.get_slice(threadIdx.x);
  const int atom=blockIdx.x,group=blockIdx.y;
  constexpr int Rows=Weight?32:16;
  constexpr int Planes=Weight?1:2;
  #pragma unroll
  for(int plane=0;plane<Planes;++plane) {
    #pragma unroll
    for(int half=0;half<2;++half) {
      const uint8_t* p=src+size_t(plane)*rows*2048+(group*rows+atom*Rows)*64+half*32;
      uint8_t* out=dst+size_t(plane)*rows*2048+(group*(rows/Rows)+atom)*Rows*64+
          (threadIdx.x/32)*1024+half*512+(threadIdx.x%32)*16;
      auto layout=cute::make_layout(cute::make_shape(cute::Int<Rows>{},cute::_64{}),
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
