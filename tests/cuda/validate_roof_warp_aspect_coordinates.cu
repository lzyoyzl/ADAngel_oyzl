// Host-only checks using actual CuTe types; no GPU launch or numerical claim.
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
  const uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(address),"l"(src):"memory");
}
template<int I,int End,class F>
__device__ __forceinline__ void o1_static_for(F const& f) {
  if constexpr(I<End) {f(cute::Int<I>{});o1_static_for<I+1,End>(f);}
}
#include "../../csrc/sm80/o3_row_scale_epilogue_candidate.cuh"
#include "../../csrc/sm80/o78_unsigned_payload_candidate.cuh"
#include "../../csrc/sm80/o3_warp_aspect_probe.cuh"
#include "../../csrc/sm80/o78_warp_aspect_probe.cuh"

template<class C,class Reference,bool DualScale,int Stages>
void verify(const char* variant,int wn) {
  static_assert(C::Threads==128);
  assert(C::WM*wn==4);
  static_assert(sizeof(typename C::Storage)==sizeof(typename Reference::Storage));
  typename C::Mma mma;
  typename C::HighMma high;
  auto identity=cute::make_identity_tensor(cute::make_shape(cute::Int<64>{},cute::Int<128>{}));
  std::vector<int> owners(64*128,0);
  for(int thread=0;thread<128;++thread) {
    auto coords=mma.get_slice(thread).partition_C(identity);
    auto high_coords=high.get_slice(thread).partition_C(identity);
    assert(int(cute::size(coords))==64);
    for(int value=0;value<cute::size(coords);++value) {
      const int row=cute::get<0>(coords(value)),col=cute::get<1>(coords(value));
      assert(row>=0 && row<64 && col>=0 && col<128);
      assert(row==cute::get<0>(high_coords(value)) && col==cute::get<1>(high_coords(value)));
      ++owners[row*128+col];
    }
  }
  for(int count:owners) assert(count==1);
  typename C::template ByteLayout<64> la;
  typename C::template ByteLayout<128> lb;
  typename Reference::template ByteLayout<64> old_la;
  typename Reference::template ByteLayout<128> old_lb;
  for(int row=0;row<64;++row) for(int col=0;col<64;++col) assert(la(row,col)==old_la(row,col));
  for(int row=0;row<128;++row) for(int col=0;col<64;++col) assert(lb(row,col)==old_lb(row,col));
  typename C::Storage storage;
  for(int stage=0;stage<Stages;++stage) {
    assert(reinterpret_cast<uintptr_t>(storage.scales+stage*128)%16==0);
    if constexpr(DualScale) assert(reinterpret_cast<uintptr_t>(storage.activation_scales+stage*64)%16==0);
  }
  std::cout<<variant<<" warp_layout="<<C::WM<<"x"<<wn<<" threads=128 output_elements="<<owners.size()
           <<" unique_owner=true accumulators_per_thread=64 shared_bytes="<<sizeof(storage)<<"\n";
}
template<int WN> void check_both() {
  using O3=o3_row_scale_epilogue_experiment_v47::O3AmpereConfig<64,128,128,false,WN,false,3>;
  using R3=o3_row_scale_epilogue_experiment::O3AmpereConfig<64,128,128,false,2,false,3>;
  using O78=o78_unsigned_payload_experiment_v47::O3AmpereConfig<64,128,128,false,WN,true,2>;
  using R78=o78_unsigned_payload_experiment::O3AmpereConfig<64,128,128,false,2,true,2>;
  verify<O3,R3,false,3>("o3",WN);
  verify<O78,R78,true,2>("o78",WN);
}
}
int main() {check_both<2>();check_both<4>();check_both<1>();}
