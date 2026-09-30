// Candidate37/38: compile-time phases for eager four/two-product trees; no default change.
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cutlass/integer_subbyte.h>
#include <stdexcept>
#include "roof_pipeline_api.h"
namespace {
__device__ void copy16(void* dst,const void* src) {
  const uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
      "r"(address), "l"(src) : "memory");
}
template<int I,int End,class F>
__device__ __forceinline__ void o1_static_for(F const& f) {
  if constexpr(I<End) {f(cute::Int<I>{});o1_static_for<I+1,End>(f);}
}
#include "o3_static_eager_tree_candidate.cuh"
template<bool DualScale,bool Fast,int Tune>
__global__ __launch_bounds__(128,1)
void adangel_sm80_roof_candidate(
    const uint8_t* a,const uint8_t* w,const float* as,const uint8_t* ws,
    float* y,int m,int n,int k) {
  static_assert(Tune==37 || Tune==38);
  static_assert(!DualScale || !Fast);
  o3_static_eager_tree_experiment::o3_body<64,128,128,Fast,false,false,2,false,true,false,true,true,true,false,
      DualScale,DualScale,6,false,false,false,false,2,Tune==38?2:4>(a,w,as,ws,y,m,n,k);
}
}
namespace adangel_sm80_experiment {
Kernel select_static_eager_tree_kernel(bool dual,bool fast,int tune) {
  if((tune!=37 && tune!=38) || (dual && fast)) throw std::invalid_argument("expected static-tree candidate37/38");
  if(tune==37) return dual ? adangel_sm80_roof_candidate<true,false,37> :
      (fast ? adangel_sm80_roof_candidate<false,true,37> : adangel_sm80_roof_candidate<false,false,37>);
  return dual ? adangel_sm80_roof_candidate<true,false,38> :
      (fast ? adangel_sm80_roof_candidate<false,true,38> : adangel_sm80_roof_candidate<false,false,38>);
}
}
