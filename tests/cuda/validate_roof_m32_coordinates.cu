// Host-only CuTe coordinate/Storage check using the actual candidate definition.
// No device launch: GPU ISA, numerics and memory-safety tests are still required.
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
__device__ void copy16(void* dst,const void* src) {
  uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(address),"l"(src):"memory");
}
template<int I,int End,class F>
__device__ __forceinline__ void o1_static_for(F const& f) {
  if constexpr(I<End) {f(cute::Int<I>{});o1_static_for<I+1,End>(f);}
}
#include "../../csrc/sm80/o78_m32_payload_candidate.cuh"

template<int Stages> void verify() {
  using C=o78_m32_payload_experiment::O3AmpereConfig<32,128,128,false,4,true,Stages>;
  static_assert(C::Threads==128);
  typename C::Mma mma;
  typename C::HighMma high;
  auto identity=cute::make_identity_tensor(cute::make_shape(cute::Int<32>{},cute::Int<128>{}));
  std::vector<int> owners(32*128,0);
  for(int thread=0;thread<C::Threads;++thread) {
    auto coords=mma.get_slice(thread).partition_C(identity);
    auto hcoords=high.get_slice(thread).partition_C(identity);
    assert(int(cute::size(coords))==32);
    for(int value=0;value<cute::size(coords);++value) {
      int row=cute::get<0>(coords(value)),col=cute::get<1>(coords(value));
      assert(row>=0 && row<32 && col>=0 && col<128);
      assert(row==cute::get<0>(hcoords(value)) && col==cute::get<1>(hcoords(value)));
      ++owners[row*128+col];
    }
  }
  for(int count:owners) assert(count==1);
  typename C::Storage storage;
  for(int stage=0;stage<Stages;++stage) {
    assert(reinterpret_cast<uintptr_t>(storage.activation_scales+stage*32)%16==0);
    assert(reinterpret_cast<uintptr_t>(storage.scales+stage*128)%16==0);
  }
  std::cout<<"stages="<<Stages<<" output_elements="<<owners.size()
           <<" unique_owner=true accumulators_per_thread=32 shared_bytes="<<sizeof(storage)<<"\n";
}
}

int main() {verify<2>();verify<3>();}
