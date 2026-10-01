// Host CuTe coordinate check; no GPU launch or timing claim.
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cutlass/integer_subbyte.h>
#include <cassert>
#include <iostream>
#define ADANGEL_STREAM_WIDTH 1
namespace {
__device__ void copy16(void* dst,const void* src) {
  const uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(address),"l"(src):"memory");
}
template<int I,int End,class F>
__device__ __forceinline__ void o1_static_for(F const& f) {
  if constexpr(I<End) {f(cute::Int<I>{});o1_static_for<I+1,End>(f);}
}
#include "../../csrc/sm80/o3_stream_width_probe.cuh"
#include "../../csrc/sm80/o78_stream_width_probe.cuh"

template<class Full,class Slice,int Width>
void verify(const char* variant) {
  static_assert(Full::Threads==128 && Slice::Threads==128);
  typename Full::Mma full;
  typename Slice::Mma sliced;
  auto identity=cute::make_identity_tensor(cute::make_shape(cute::Int<128>{},cute::Int<128>{}));
  for(int thread=0;thread<128;++thread) for(int half=0;half<2;++half) {
    auto whole=cute::local_tile(identity,cute::make_shape(cute::Int<128>{},cute::_64{}),
                               cute::make_coord(0,half));
    auto original=full.get_slice(thread).partition_B(whole);
    for(int nb=0;nb<128/Width;++nb) {
      auto tile=cute::local_tile(identity,cute::make_shape(cute::Int<Width>{},cute::_64{}),
                                cute::make_coord(nb,half));
      auto coords=sliced.get_slice(thread).partition_B(tile);
      assert(int(cute::size<1>(coords))==Width/16);
      assert(cute::size<0>(coords)==cute::size<0>(original));
      assert(cute::size<2>(coords)==cute::size<2>(original));
      for(int ki=0;ki<cute::size<2>(coords);++ki)
        for(int ni=0;ni<cute::size<1>(coords);++ni)
          for(int vi=0;vi<cute::size<0>(coords);++vi) {
            const auto x=coords(vi,ni,ki);
            const auto y=original(vi,nb*(Width/16)+ni,ki);
            assert(cute::get<0>(x)==cute::get<0>(y) && cute::get<1>(x)==cute::get<1>(y));
          }
    }
  }
  std::cout<<variant<<" stream_n="<<Width<<" atoms="<<Width/16
           <<" slices="<<128/Width<<" partition_B_matches_full=true threads=128\n";
}
template<int W> void check_both() {
  using F3=o3_row_scale_epilogue_experiment_v48::O3AmpereConfig<64,128,128,false,2,false,3>;
  using S3=o3_row_scale_epilogue_experiment_v48::O3AmpereConfig<64,W,128,false,2>;
  using F78=o78_unsigned_payload_experiment_v48::O3AmpereConfig<64,128,128,false,2,true,2>;
  using S78=o78_unsigned_payload_experiment_v48::O3AmpereConfig<64,W,128,false,2>;
  verify<F3,S3,W>("o3");verify<F78,S78,W>("o78");
}
}
int main() {check_both<64>();check_both<32>();check_both<128>();}
