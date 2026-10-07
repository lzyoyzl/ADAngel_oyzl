// CPU-only verification of source addresses and register word destinations.
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cutlass/numeric_types.h>
#include <array>
#include <cstdio>
#include <stdexcept>

using namespace cute;
using Mma=TiledMMA<MMA_Atom<SM80_16x8x64_S32U4S4S32_TN>,
    Layout<Shape<_2,_2,_1>>,Tile<_64,_128,_64>>;
using High=TiledMMA<MMA_Atom<SM80_16x8x64_S32S4S4S32_TN>,
    Layout<Shape<_2,_2,_1>>,Tile<_64,_128,_64>>;
using NibbleLayout=decltype(composition(Swizzle<2,5,3>{},
    Layout<Shape<_64,_128>,Stride<_128,_1>>{}));

int main() {
  alignas(128) std::array<unsigned char,24576> bytes{};
  int addresses=0,words=0;
  for(int tid=0;tid<128;++tid) {
    Mma mma;High hm;
    auto lc=make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N,cutlass::uint4b_t>{},mma).get_slice(tid);
    auto hc=make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N,cutlass::int4b_t>{},hm).get_slice(tid);
    auto tile=[&](auto tensor,auto half) {
      return local_tile(tensor,make_shape(_64{},_64{}),make_coord(_0{},half));
    };
    auto lo=[&](int slot) {
      return make_tensor(make_smem_ptr<cutlass::uint4b_t>(bytes.data()+slot*4096),NibbleLayout{});
    };
    auto hi=[&](int slot) {
      return make_tensor(make_smem_ptr<cutlass::int4b_t>(bytes.data()+12288+slot*4096),NibbleLayout{});
    };
    // The kernel's four cached addresses, expressed here as byte offsets.
    std::array<std::array<ptrdiff_t,2>,2> cached{};
    for_each(make_seq<2>{},[&](auto half) {
      auto src=recast<uint128_t>(lc.partition_S(tile(lo(0),half)));
      static_assert(decltype(size<0>(src))::value==1);
      static_assert(decltype(size<1>(src))::value==2);
      for(int mi=0;mi<2;++mi)
        cached[half][mi]=reinterpret_cast<unsigned char*>(&src(0,mi,0))-bytes.data();
    });
    for(int slot=0;slot<3;++slot) for_each(make_seq<2>{},[&](auto half) {
      auto sl=recast<uint128_t>(lc.partition_S(tile(lo(slot),half)));
      auto sh=recast<uint128_t>(hc.partition_S(tile(hi(slot),half)));
      for(int mi=0;mi<2;++mi) {
        ptrdiff_t expected=cached[half][mi]+slot*4096;
        auto l=reinterpret_cast<unsigned char*>(&sl(0,mi,0))-bytes.data();
        auto h=reinterpret_cast<unsigned char*>(&sh(0,mi,0))-bytes.data();
        if(l!=expected || h!=expected+12288 || l%16 || h%16 ||
            l<slot*4096 || l+16>(slot+1)*4096 || h+16>24576)
          throw std::runtime_error("CuTe source address mismatch");
        addresses+=2;
      }
    });
    auto fl=mma.get_slice(tid).partition_fragment_A(tile(lo(0),_0{}));
    auto fh=hm.get_slice(tid).partition_fragment_A(tile(hi(0),_0{}));
    auto check_words=[&](auto& frag,auto const& copier) {
      auto d=recast<uint32_t>(copier.retile_D(frag));
      auto raw=recast<uint32_t>(frag);
      static_assert(decltype(size<0>(d))::value==4);
      static_assert(decltype(size<1>(d))::value==2);
      static_assert(decltype(size<2>(d))::value==1);
      static_assert(decltype(size(raw))::value==8);
      std::array<bool,8> seen{};
      for(int mi=0;mi<2;++mi) {
        // Original Copy_Atom recasts each 32-nibble slice to four U32s.
        auto original=recast<uint32_t>(copier.retile_D(frag)(_,mi,_0{}));
        for(int vi=0;vi<4;++vi) {
          ptrdiff_t index=&d(vi,mi,0)-&raw(0);
          if(index<0 || index>=8 || seen[index] || &original(vi)!=&d(vi,mi,0))
            throw std::runtime_error("CuTe destination word mismatch");
          seen[index]=true;++words;
        }
      }
    };
    check_words(fl,lc);check_words(fh,hc);
  }
  if(addresses!=3072 || words!=2048) return 2;
  std::printf("{\"passed\":true,\"source_addresses\":%d,\"destination_words\":%d}\n",addresses,words);
}
