// Host-only CuTe coordinate proof; compiling kernels here does not launch them.
#include "../../csrc/sm80/roof_o7_pow2_probe.cu"
#include <cassert>
#include <iostream>
#include <set>

int main() {
  o7_fullk_pow2_experiment::C::Mma mma;
  auto identity=cute::make_identity_tensor(cute::make_shape(cute::_64{},cute::_128{}));
  std::set<int> coverage;
  for(int thread=0;thread<128;++thread) {
    auto coords=mma.get_slice(thread).partition_C(identity);
    for(int mi=0;mi<2;++mi) for(int ni=0;ni<8;++ni) for(int vi=0;vi<4;++vi) {
      const auto p=coords(vi,mi,ni), row=coords((vi/2)*2,mi,0);
      assert(cute::get<0>(p)==cute::get<0>(row));
      assert(cute::get<0>(p)>=0 && cute::get<0>(p)<64);
      assert(cute::get<1>(p)>=0 && cute::get<1>(p)<128);
      assert(coverage.insert(int(cute::get<0>(p))*128+int(cute::get<1>(p))).second);
    }
  }
  assert(coverage.size()==8192);
  std::cout << "128 threads / 8192 outputs / four row shifts per thread: exact CuTe mapping passed\n";
}
