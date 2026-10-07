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
#include <vector>

// Keep NVIDIA's exact Copy_Atom recursion/layout; replace only execution with
// a CPU callback that records the original instruction's operands.
struct AuditCopy {
  using SRegisters=cute::uint128_t[1];
  using DRegisters=uint32_t[4];
  struct Record { const cute::uint128_t* source; std::array<uint32_t*,4> destination; };
  static inline std::vector<Record> records;
  static void copy(cute::uint128_t const& s,uint32_t& a,uint32_t& b,uint32_t& c,uint32_t& d) {
    records.push_back({&s,{&a,&b,&c,&d}});
  }
};
namespace cute {
template<> struct Copy_Traits<AuditCopy>:Copy_Traits<SM75_U32x4_LDSM_N> {};
}

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
      static_assert(decltype(size(src))::value==2);
      for(int mi=0;mi<2;++mi)
        cached[half][mi]=reinterpret_cast<unsigned char*>(&src(mi))-bytes.data();
    });
    auto fl=mma.get_slice(tid).partition_fragment_A(tile(lo(0),_0{}));
    auto fh=hm.get_slice(tid).partition_fragment_A(tile(hi(0),_0{}));
    auto check=[&](auto& frag,auto const& copier,auto source,int slot,int plane,int half,auto audit_atom) {
      auto ds=copier.retile_D(frag);
      AuditCopy::records.clear();
      cute::copy(audit_atom,copier.partition_S(source),ds);
      if(AuditCopy::records.size()!=2) throw std::runtime_error("unexpected copy atom count");
      auto d=recast<uint32_t>(ds);
      auto raw=recast<uint32_t>(frag);
      static_assert(decltype(size(d))::value==8);
      static_assert(decltype(size(raw))::value==8);
      std::array<bool,8> seen{};
      for(int mi=0;mi<2;++mi) {
        const auto& original=AuditCopy::records[mi];
        ptrdiff_t expected=cached[half][mi]+slot*4096+plane*12288;
        ptrdiff_t actual=reinterpret_cast<const unsigned char*>(original.source)-bytes.data();
        if(actual!=expected || actual%16 || actual<slot*4096+plane*12288 ||
            actual+16>(slot+1)*4096+plane*12288)
          throw std::runtime_error("CuTe source address mismatch");
        ++addresses;
        for(int vi=0;vi<4;++vi) {
          ptrdiff_t index=&d(4*mi+vi)-&raw(0);
          if(index<0 || index>=8 || seen[index] || original.destination[vi]!=&d(4*mi+vi))
            throw std::runtime_error("CuTe destination word mismatch");
          seen[index]=true;++words;
        }
      }
    };
    for(int slot=0;slot<3;++slot) for_each(make_seq<2>{},[&](auto half) {
      check(fl,lc,tile(lo(slot),half),slot,0,half,Copy_Atom<AuditCopy,cutlass::uint4b_t>{});
      check(fh,hc,tile(hi(slot),half),slot,1,half,Copy_Atom<AuditCopy,cutlass::int4b_t>{});
    });
  }
  if(addresses!=3072 || words!=12288) return 2;
  std::printf("{\"passed\":true,\"source_addresses\":%d,\"destination_words\":%d}\n",addresses,words);
}
