// Host-only CuTe A/B/C mapping proof, no GPU launch or timing claim.
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cutlass/integer_subbyte.h>
#include <cassert>
#include <iostream>
#include <vector>
namespace {
__device__ void copy16(void*,const void*);
template<int I,int End,class F> __device__ void o1_static_for(F const&);
#include "../../csrc/sm80/o78_unsigned_payload_candidate.cuh"
namespace O78=o78_unsigned_payload_experiment;
}
int main() {
  using C=O78::O3AmpereConfig<64,128,128,false,2,true,2>;
  C::Mma mma;
  using SliceMma=cute::TiledMMA<cute::MMA_Atom<cute::SM80_16x8x64_S32U4S4S32_TN>,
      cute::Layout<cute::Shape<cute::_2,cute::_2,cute::_1>>,
      cute::Tile<cute::_32,cute::_128,cute::_64>>;
  SliceMma slice_mma;
  auto ia=cute::make_identity_tensor(cute::make_shape(cute::_64{},cute::_128{}));
  auto ib=cute::make_identity_tensor(cute::make_shape(cute::_128{},cute::_128{}));
  std::vector<int> owners(64*128,0);
  unsigned compared=0;
  for(int tid=0;tid<128;++tid) {
    auto thr=mma.get_slice(tid);
    auto athr=slice_mma.get_slice(tid);
    auto full_a=thr.partition_A(ia);
    auto full_b=thr.partition_B(ib);
    auto c=thr.partition_C(ia);
    assert(cute::size<1>(c)==2 && cute::size<2>(c)==8 && cute::size(c)==64);
    for(int mb=0;mb<2;++mb) {
      for(int half=0;half<2;++half) {
        auto ta=cute::local_tile(ia,cute::make_shape(cute::_32{},cute::_64{}),cute::make_coord(mb,half));
        auto sa=athr.partition_A(ta);
        assert(cute::size<1>(sa)==1 && cute::size<2>(sa)==1);
        for(int v=0;v<cute::size<0>(sa);++v) {
          auto x=sa(v,0,0),y=full_a(v,mb,half);
          assert(cute::get<0>(x)==cute::get<0>(y) && cute::get<1>(x)==cute::get<1>(y));++compared;
        }
      }
      for(int ni=0;ni<8;++ni) for(int vi=0;vi<4;++vi) {
        auto p=c(vi,mb,ni);
        assert(cute::get<0>(p)>=mb*32 && cute::get<0>(p)<(mb+1)*32);
        assert(cute::get<1>(p)>=0 && cute::get<1>(p)<128);
        ++owners[cute::get<0>(p)*128+cute::get<1>(p)];
      }
    }
    for(int half=0;half<2;++half) {
      auto tb=cute::local_tile(ib,cute::make_shape(cute::_128{},cute::_64{}),cute::make_coord(0,half));
      auto sb=thr.partition_B(tb);
      assert(cute::size<1>(sb)==8 && cute::size<2>(sb)==1);
      for(int ni=0;ni<8;++ni) for(int v=0;v<cute::size<0>(sb);++v) {
        auto x=sb(v,ni,0),y=full_b(v,ni,half);
        assert(cute::get<0>(x)==cute::get<0>(y) && cute::get<1>(x)==cute::get<1>(y));++compared;
      }
    }
  }
  for(int count:owners) assert(count==1);
  std::cout<<"{\"passed\":true,\"threads\":128,\"outputs\":8192,\"operand_coordinates_compared\":"
           <<compared<<",\"gpu_execution\":false}\n";
}
