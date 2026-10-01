//57/58: narrow M rather than N; preserve per-warp M32 B-fragment reuse.
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cutlass/integer_subbyte.h>
#include <stdexcept>
#include "roof_m32_payload_api.h"

namespace {
__device__ void copy16(void* dst,const void* src) {
  const uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
      "r"(address),"l"(src):"memory");
}
template<int I,int End,class F>
__device__ __forceinline__ void o1_static_for(F const& f) {
  if constexpr(I<End) {f(cute::Int<I>{});o1_static_for<I+1,End>(f);}
}
#include "o78_m32_payload_candidate.cuh"

template<bool DualScale,bool Fast,int Tune>
__global__ __launch_bounds__(128,4)
void adangel_sm80_roof_candidate(const uint8_t* a,const uint8_t* w,const float* as,
    const uint8_t* ws,float* y,int m,int n,int k) {
  static_assert(DualScale && !Fast && (Tune==57 || Tune==58));
  constexpr int Stages=Tune==57?2:3;
  o78_m32_payload_experiment::o3_body<32,128,128,false,false,false,4,false,true,false,true,true,true,false,
      true,true,6,false,false,true,false,Stages>(a,w,as,ws,y,m,n,k);
}
}
namespace adangel_sm80_experiment {
size_t m32_payload_shared_bytes(int tune) {
  using C2=o78_m32_payload_experiment::O3AmpereConfig<32,128,128,false,4,true,2>;
  using C3=o78_m32_payload_experiment::O3AmpereConfig<32,128,128,false,4,true,3>;
  if(tune==57) return sizeof(typename C2::Storage);
  if(tune==58) return sizeof(typename C3::Storage);
  throw std::invalid_argument("expected M32 candidate57/58");
}
Kernel select_m32_payload_kernel(int tune) {
  if(tune==57) return adangel_sm80_roof_candidate<true,false,57>;
  if(tune==58) return adangel_sm80_roof_candidate<true,false,58>;
  throw std::invalid_argument("expected O7/O8 M32 asynchronous scale candidate57/58");
}
}
