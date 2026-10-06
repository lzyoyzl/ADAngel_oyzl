// Host-only CuTe ownership/copy proof; no CUDA context or kernel execution.
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cutlass/integer_subbyte.h>
#include <cstdio>
#include <vector>
#include <cstdlib>
template<int I,int End,class F>
__device__ __forceinline__ void o1_static_for(F const& f) {
  if constexpr(I<End) {f(cute::Int<I>{});o1_static_for<I+1,End>(f);}
}
__device__ void copy16(void*,const void*);
#include "o78_cooperative_reuse_generated.cuh"

int main() {
  using namespace o78_cooperative_reuse_experiment;
  C::Mma mma;C::SliceMma slice;
  auto identity=cute::make_identity_tensor(cute::make_shape(cute::_128{},cute::_192{}));
  auto small_identity=cute::make_identity_tensor(cute::make_shape(cute::_128{},cute::_96{}));
  std::vector<int> owners(128*192),a_bytes(128*64),w_bytes(192*64);
  int pairs=0;
  for(int thread=0;thread<384;++thread) {
    auto coords=mma.get_slice(thread).partition_C(identity);
    auto part=slice.get_slice(thread).partition_C(small_identity);
    if(cute::size(coords)!=64 || cute::size(part)!=32)return 2;
    for(int i=0;i<64;++i) {
      auto p=coords(i);int m=cute::get<0>(p),n=cute::get<1>(p);
      if(m<0 || m>=128 || n<0 || n>=192)return 3;
      if(++owners[m*192+n]!=1)return 4;
    }
    for(int mi=0;mi<2;++mi)for(int ni=0;ni<4;++ni)for(int vi=0;vi<4;++vi) {
      auto p=part(vi,mi,ni);
      for(int nb=0;nb<2;++nb) {
        auto q=coords(vi,mi,nb*4+ni);
        if(cute::get<0>(p)!=cute::get<0>(q) || cute::get<1>(p)+nb*96!=cute::get<1>(q))return 5;
      }
    }
    for(int i=0;i<64;i+=2) {
      auto p=coords(i),q=coords(i+1);
      if(cute::get<0>(p)!=cute::get<0>(q) || cute::get<1>(q)!=cute::get<1>(p)+1)return 6;
      if((cute::get<1>(p)&1)!=0)return 7;++pairs;
    }
    for(int chunk=0;chunk<2;++chunk) {
      unsigned off=thread*16+chunk*6144;
      if(off<128*64)for(int b=0;b<16;++b)++a_bytes[off+b];
      for(int b=0;b<16;++b)++w_bytes[off+b];
    }
  }
  for(int count:owners)if(count!=1)return 8;
  for(int count:a_bytes)if(count!=1)return 9;
  for(int count:w_bytes)if(count!=1)return 10;
  // Both nibble addressing and16B vector copy must match their byte view.
  int checks=0;
  auto layout=[&](auto rows) {
    C::ByteLayout<decltype(rows)::value> bytes;
    C::NibbleLayout<decltype(rows)::value> nibbles;
    for(int r=0;r<rows;++r)for(int k=0;k<128;++k) {
      if(nibbles(r,k)!=2*bytes(r,k/2)+k%2)std::exit(11);++checks;
    }
    for(int r=0;r<rows;++r)for(int c=0;c<64;c+=16)for(int b=0;b<16;++b)
      if(bytes(r,c+b)!=bytes(r,c)+b)std::exit(12);
  };
  layout(cute::_128{});layout(cute::_192{});
  std::printf("{\"passed\":true,\"gpu_execution\":false,\"output_elements\":24576,"
      "\"unique_output_owners\":true,\"same32_partial_mapping\":true,\"vector_pairs\":%d,"
      "\"copy_byte_owners\":20480,\"nibble_layout_checks\":%d,\"shared_bytes\":%zu}\n",
      pairs,checks,sizeof(Storage));
}
