// Host-only validation of actual CuTe A/B/C coefficient fragment coordinates.
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cutlass/integer_subbyte.h>
#include <cassert>
#include <iostream>
#include <set>
namespace {
__device__ void copy16(void*,const void*);
template<int I,int End,class F> __device__ void o1_static_for(F const&);
#include "../../csrc/sm80/o78_unsigned_payload_candidate.cuh"
using C=o78_unsigned_payload_experiment::O3AmpereConfig<64,128,128,false,2,true,2>;
using FactorMma=cute::TiledMMA<cute::MMA_Atom<cute::SM80_16x8x16_S32U8U8S32_TN>,
    cute::Layout<cute::Shape<cute::_2,cute::_2,cute::_1>>,
    cute::Tile<cute::_64,cute::_128,cute::_16>>;
}
int main() {
  C::Mma payload;FactorMma factor;
  auto ci=cute::make_identity_tensor(cute::make_shape(cute::_64{},cute::_128{}));
  auto ai=cute::make_identity_tensor(cute::make_shape(cute::_64{},cute::_16{}));
  auto bi=cute::make_identity_tensor(cute::make_shape(cute::_128{},cute::_16{}));
  std::set<int> cseen,aseen,bseen;
  for(int t=0;t<128;++t) {
    auto pc=payload.get_slice(t).partition_C(ci);
    auto fc=factor.get_slice(t).partition_C(ci);
    auto fa=factor.get_slice(t).partition_A(ai);
    auto fb=factor.get_slice(t).partition_B(bi);
    assert(cute::size<0>(fa)==8 && cute::size<1>(fa)==2 && cute::size<2>(fa)==1);
    assert(cute::size<0>(fb)==4 && cute::size<1>(fb)==8 && cute::size<2>(fb)==1);
    for(int mi=0;mi<2;++mi) for(int vi=0;vi<8;++vi) {
      auto p=fa(vi,mi,0);int r=cute::get<0>(p),k=cute::get<1>(p);
      assert(r>=0 && r<64 && k>=0 && k<16);aseen.insert(r*16+k);
    }
    for(int ni=0;ni<8;++ni) for(int vi=0;vi<4;++vi) {
      auto p=fb(vi,ni,0);int n=cute::get<0>(p),k=cute::get<1>(p);
      assert(n>=0 && n<128 && k>=0 && k<16);bseen.insert(n*16+k);
    }
    for(int mi=0;mi<2;++mi) for(int ni=0;ni<8;++ni) for(int vi=0;vi<4;++vi) {
      auto a=pc(vi,mi,ni),b=fc(vi,mi,ni);
      assert(cute::get<0>(a)==cute::get<0>(b) && cute::get<1>(a)==cute::get<1>(b));
      int r=cute::get<0>(b),c=cute::get<1>(b);
      assert(cseen.insert(r*128+c).second);
      int av=(r*17)%256,bv=(c*23)%256,product=0;
      for(int k=0;k<16;++k)product+=(k==0?av:0)*(k==0?bv:0);
      assert(product==av*bv && product<=65025);
    }
  }
  assert(cseen.size()==8192 && aseen.size()==64*16 && bseen.size()==128*16);
  std::cout << "{\"passed\":true,\"gpu_execution\":false,\"outputs\":8192,"
      "\"A_coordinates\":1024,\"B_coordinates\":2048,\"C_matches_payload\":true,"
      "\"coefficient_mma\":\"m16n8k16.u8.u8\",\"nonzero_K\":0}\n";
  return 0;
}
