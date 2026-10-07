// Host-only CuTe mapping proof; no CUDA kernel or context is created.
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
#include <type_traits>
namespace {
__device__ void copy16(void*,const void*);
template<int I,int End,class F> __device__ void o1_static_for(F const&);
#include "../../csrc/sm80/o3_row_scale_epilogue_candidate.cuh"
#include "../../csrc/sm80/o78_unsigned_payload_candidate.cuh"
using C=o3_row_scale_epilogue_experiment::O3AmpereConfig<64,128,128,false,2,false,3>;
using C78=o78_unsigned_payload_experiment::O3AmpereConfig<64,128,128,false,2,true,2>;
static_assert(std::is_same_v<C::Mma,C78::Mma>);
}
int main() {
  C::Mma mma;
  auto identity=cute::make_identity_tensor(cute::make_shape(cute::_64{},cute::_128{}));
  std::set<int> covered;
  for(int warp=0;warp<4;++warp) for(int ni=0;ni<8;++ni) {
    std::set<int> columns;
    std::set<int> rows[2];
    for(int lane=0;lane<32;++lane) {
      auto coords=mma.get_slice(warp*32+lane).partition_C(identity);
      for(int mi=0;mi<2;++mi) for(int vi=0;vi<4;++vi) {
        auto p=coords(vi,mi,ni);
        const int row=cute::get<0>(p),col=cute::get<1>(p);
        columns.insert(col);
        rows[mi].insert(row);
        assert(covered.insert(row*128+col).second);
      }
    }
    std::set<int> expected;
    for(int j=0;j<8;++j)expected.insert((warp/2)*8+ni*16+j);
    if(columns!=expected)return 1;
    for(int mi=0;mi<2;++mi) {
      std::set<int> expected_rows;
      for(int j=0;j<16;++j)expected_rows.insert((warp%2)*16+mi*32+j);
      if(rows[mi]!=expected_rows)return 1;
    }
  }
  assert(covered.size()==8192);
  std::cout << "{\"passed\":true,\"gpu_execution\":false,\"outputs\":8192,"
    "\"threads\":128,\"outputs_per_thread\":64,\"columns_per_native_atom\":8,\"rows_per_native_atom\":16,"
    "\"distinct_N8_panels_per_N128\":16,\"N8_panels_per_thread\":8,\"native_M16_N8_atoms_per_thread\":16,"
    "\"per_thread_outputs_per_N8_panel\":8,\"per_thread_outputs_per_native_M16_N8_atom\":4,"
    "\"O3_O78_TiledMMA_types_equal\":true,"
    "\"column_formula\":\"(warp/2)*8 + ni*16 + j; j=0..7\"}\n";
}
