// Host-only proof that direct U32 words equal every v129 U8 fragment byte.
// Reuse its full CuTe coordinate proof without writing another lane map.
#define main original_coordinate_check
#include "validate_tensor_factor_coordinates.cu"
#undef main
#include <sstream>
int main() {
  std::ostringstream original;
  auto* saved=std::cout.rdbuf(original.rdbuf());
  const int result=original_coordinate_check();
  std::cout.rdbuf(saved);
  assert(result==0);
  FactorMma mma;
  auto ai=cute::make_identity_tensor(cute::make_shape(cute::_64{},cute::_16{}));
  auto bi=cute::make_identity_tensor(cute::make_shape(cute::_128{},cute::_16{}));
  int checked=0;
  for(int t=0;t<128;++t) {
    auto fa=mma.get_slice(t).partition_A(ai);
    auto fb=mma.get_slice(t).partition_B(bi);
    for(int salt: {0,1,17,127,255}) {
      auto val=[&](int row) {return static_cast<uint32_t>((row*23+salt)%256);};
      for(int mi=0;mi<2;++mi) {
        uint32_t words[2];
        for(int word=0;word<2;++word) {
          auto p=fa(word*4,mi,0);
          words[word]=cute::get<1>(p)==0?val(cute::get<0>(p)):0u;
        }
        for(int vi=0;vi<8;++vi) {
          auto p=fa(vi,mi,0);
          if(vi%4!=0) assert(cute::get<1>(p)!=0);
          uint32_t expected=cute::get<1>(p)==0?val(cute::get<0>(p)):0u;
          assert(((words[vi/4]>>(8*(vi%4)))&255u)==expected);++checked;
        }
      }
      for(int ni=0;ni<8;++ni) {
        auto first=fb(0,ni,0);
        uint32_t word=cute::get<1>(first)==0?val(cute::get<0>(first)):0u;
        for(int vi=0;vi<4;++vi) {
          auto p=fb(vi,ni,0);
          if(vi!=0) assert(cute::get<1>(p)!=0);
          uint32_t expected=cute::get<1>(p)==0?val(cute::get<0>(p)):0u;
          assert(((word>>(8*vi))&255u)==expected);++checked;
        }
      }
    }
  }
  assert(checked==30720);
  std::cout<<"{\"passed\":true,\"gpu_execution\":false,\"v129_all_coordinates_passed\":true,"
      "\"same_layout\":true,\"packed_bytes_checked\":30720,\"outputs\":8192}\n";
}
