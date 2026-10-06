// Host-only CuTe ownership and byte/nibble proof; no CUDA launch.
#include <cuda_runtime.h>
#include <stdint.h>
#include <iostream>
#include <array>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cutlass/integer_subbyte.h>
__device__ void copy16(void*,const void*);
template<int I,int End,class F> __device__ __forceinline__ void o1_static_for(F const& f) {
  if constexpr(I<End){f(cute::Int<I>{});o1_static_for<I+1,End>(f);}
}
#include "o78_warp_private_pipeline.cuh"
int main() {
  namespace P=o78_warp_private_experiment;
  std::array<int,8192> old{},actual{};
  using Old=cute::TiledMMA<P::LA,cute::Layout<cute::Shape<cute::_2,cute::_2,cute::_1>>,
      cute::Tile<cute::_64,cute::_128,cute::_64>>;
  Old original;P::Mma private_mma;
  for(int t=0;t<128;++t) {
    auto oc=original.get_slice(t).partition_C(cute::make_identity_tensor(cute::make_shape(cute::_64{},cute::_128{})));
    for(int i=0;i<cute::size(oc);++i) {
      auto p=oc(i);int r=cute::get<0>(p),c=cute::get<1>(p);
      if(r<0||r>=64||c<0||c>=128)return 1;
      old[r*128+c]++;
    }
    int warp=t/32,row_base=(warp%2)*32,col_base=(warp/2)*64;
    auto pc=private_mma.get_slice(t%32).partition_C(
        cute::make_identity_tensor(cute::make_shape(cute::_32{},cute::_64{})));
    if(cute::size(pc)!=64)return 2;
    for(int i=0;i<cute::size(pc);++i) {
      auto p=pc(i);int r=row_base+cute::get<0>(p),c=col_base+cute::get<1>(p);
      if(r<0||r>=64||c<0||c>=128)return 3;
      actual[r*128+c]++;
    }
  }
  for(int i=0;i<8192;++i)if(old[i]!=1||actual[i]!=1)return 4;
  int copy_checks=0;
  auto verify=[&](auto rows) {
    constexpr int R=decltype(rows)::value;
    P::ByteLayout<R> bytes;P::NibbleLayout<R> nibbles;
    std::array<int,R*64> owned{};
    for(int lane=0;lane<32;++lane)for(int chunk=0;chunk<R/8;++chunk) {
      int off=lane*16+chunk*512,row=off/64,col=off%64;
      int physical=bytes(row,col);
      if(physical%16)return false;
      for(int b=0;b<16;++b) {
        if(bytes(row,col+b)!=physical+b || nibbles(row,2*(col+b))!=2*(physical+b) ||
           nibbles(row,2*(col+b)+1)!=2*(physical+b)+1)return false;
        owned[physical+b]++;copy_checks++;
      }
    }
    for(int n:owned)if(n!=1)return false;
    return true;
  };
  if(!verify(cute::_32{})||!verify(cute::_64{}))return 5;
  std::cout<<"{\"passed\":true,\"gpu_execution\":false,\"output_elements\":8192,"
      "\"old_and_private_unique_owners\":true,\"ownership_changed\":true,"
      "\"copy_byte_checks\":"<<copy_checks<<",\"shared_bytes\":"<<sizeof(P::Storage)<<"}\n";
}
