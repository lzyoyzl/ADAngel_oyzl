// Host-only use of the candidate's exact CuTe type. No GPU kernel is launched.
#include "../../csrc/sm80/roof_o7_shift_coefficient_probe.cu"
#include <cassert>
#include <iostream>
#include <set>

int main() {
  o7_shift_coefficient_experiment::C::Mma mma;
  auto identity=cute::make_identity_tensor(cute::make_shape(cute::_64{},cute::_128{}));
  std::set<int> covered;
  for(int tid=0;tid<128;++tid) {
    auto coords=mma.get_slice(tid).partition_C(identity);
    std::set<int> rows;
    for(int mi=0;mi<2;++mi) for(int ni=0;ni<8;++ni) for(int vi=0;vi<4;++vi) {
      auto p=coords(vi,mi,ni),r=coords((vi/2)*2,mi,0);
      assert(cute::get<0>(p)==cute::get<0>(r));
      rows.insert(int(cute::get<0>(p)));
      assert(covered.insert(int(cute::get<0>(p))*128+int(cute::get<1>(p))).second);
    }
    assert(rows.size()==4);
  }
  assert(covered.size()==8192);
  std::cout << "{\"passed\":true,\"threads\":128,\"outputs\":8192,\"unique_rows_per_thread\":4,\"gpu_execution\":false}\n";
}
