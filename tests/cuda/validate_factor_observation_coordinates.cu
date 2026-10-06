// v105 host-only: use the actual best O7/O8 CuTe type, no GPU kernel launch.
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cutlass/integer_subbyte.h>
#include <cassert>
#include <iostream>
#include <map>
#include <set>
namespace {
__device__ void copy16(void*,const void*);
template<int I,int End,class F> __device__ void o1_static_for(F const&);
#include "../../csrc/sm80/o78_unsigned_payload_candidate.cuh"
using C=o78_unsigned_payload_experiment::O3AmpereConfig<64,128,128,false,2,true,2>;
}
int main() {
  C::Mma mma;
  auto identity=cute::make_identity_tensor(cute::make_shape(cute::_64{},cute::_128{}));
  std::set<int> covered;
  int bases[4]={};
  for(int warp=0;warp<4;++warp) {
    std::map<int,int> quads;
    for(int lane=0;lane<32;++lane) {
      auto coords=mma.get_slice(warp*32+lane).partition_C(identity);
      std::set<int> rows;
      for(int mi=0;mi<2;++mi) for(int ni=0;ni<8;++ni) for(int vi=0;vi<4;++vi) {
        const auto p=coords(vi,mi,ni);
        rows.insert(int(cute::get<0>(p)));
        assert(covered.insert(int(cute::get<0>(p))*128+int(cute::get<1>(p))).second);
      }
      assert(rows.size()==4);
      const int first=*rows.begin(),base=(first/32)*32,offset=first%32;
      assert(offset<8);
      assert(rows==std::set<int>({first,first+8,first+16,first+24}));
      if(lane==0) bases[warp]=base;
      assert(base==bases[warp]);++quads[offset];
    }
    assert(quads.size()==8);
    for(auto [offset,count]:quads) assert(count==4);
  }
  assert(covered.size()==8192);
  assert(std::multiset<int>(bases,bases+4)==std::multiset<int>({0,0,32,32}));
  std::cout << "{\"passed\":true,\"gpu_execution\":false,\"outputs\":8192,"
    "\"rows_per_thread\":4,\"quad_offsets\":[0,8,16,24],\"quad_bases_per_warp\":8,"
    "\"lanes_per_quad\":4,\"warp_m_bases\":[";
  for(int i=0;i<4;++i) std::cout << (i?",":"") << bases[i];
  std::cout << "]}\n";
}
