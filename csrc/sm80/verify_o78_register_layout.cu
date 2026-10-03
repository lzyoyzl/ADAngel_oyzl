// No GPU work: exhaustive actual-CuTe coordinate check of producer/consumer.
#include "o78_register_layout_mapping.cuh"
#include <array>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace R=o78_register_layout_mapping;
using Point=std::pair<int,int>;
template<bool Weight,bool Legacy=false>
auto packed_coordinates() {
  constexpr int Rows=Weight?128:64;
  constexpr int Tile=(Weight && !Legacy)?32:16;
  using MMA=std::conditional_t<Weight && !Legacy,R::PackW,R::PackA>;
  std::array<Point,Rows*128> result;
  result.fill({-1,-1});
  for(int tile=0;tile<Rows/Tile;++tile)
    for(int half=0;half<2;++half)
      for(int thread=0;thread<(Weight && !Legacy?64:32);++thread) {
        auto tensor=cute::make_identity_tensor(cute::make_shape(cute::Int<Tile>{},cute::_64{}));
        auto thr=MMA{}.get_slice(thread);
        auto coords=[&]() {
          if constexpr(Weight)return thr.partition_B(tensor);
          else return thr.partition_A(tensor);
        }();
        int byte=tile*Tile*64+(thread/32)*1024+half*512+(thread%32)*16;
        for(int v=0;v<32;++v) {
          auto p=coords(v);
          auto& target=result.at(byte*2+v);
          if(target.first!=-1)throw std::runtime_error("duplicate packed nibble");
          target={tile*Tile+int(cute::get<0>(p)),half*64+int(cute::get<1>(p))};
        }
      }
  for(auto p:result)if(p.first<0)throw std::runtime_error("missing packed nibble");
  return result;
}
int main() {
  auto pa=packed_coordinates<false>();
  auto pw=packed_coordinates<true>(),oldw=packed_coordinates<true,true>();
  int acount=0,bcount=0,legacy_bad=0;
  for(int t=0;t<128;++t) {
    auto thr=R::FullMma{}.get_slice(t);
    auto st=R::SliceMma{}.get_slice(t);
    auto c=thr.partition_C(cute::make_identity_tensor(cute::make_shape(cute::_64{},cute::_128{})));
    auto a=thr.partition_A(cute::make_identity_tensor(cute::make_shape(cute::_64{},cute::_64{})));
    auto b=st.partition_B(cute::make_identity_tensor(cute::make_shape(cute::_64{},cute::_64{})));
    for(int half=0;half<2;++half) {
      for(int mi=0;mi<2;++mi)for(int v=0;v<32;++v) {
        auto p=a(v,mi,0);
        Point expected={int(cute::get<0>(p)),half*64+int(cute::get<1>(p))};
        int off=R::a_offset(cute::get<0>(c(0,mi,0)),half,t%32);
        if(pa.at(off*2+v)!=expected)throw std::runtime_error("A consumer mismatch");
        ++acount;
      }
      for(int nb=0;nb<2;++nb)for(int ni=0;ni<4;++ni)for(int v=0;v<16;++v) {
        auto p=b(v,ni,0);
        Point expected={nb*64+int(cute::get<0>(p)),half*64+int(cute::get<1>(p))};
        int column=cute::get<1>(c(0,0,nb*4+(ni/2)*2));
        int index=R::w_offset(column,half,t%32)*2+(ni%2)*16+v;
        if(pw.at(index)!=expected)throw std::runtime_error("B consumer mismatch");
        int oldoff=(column/16)*1024+half*512+(t%32)*16;
        legacy_bad+=oldw.at(oldoff*2+(ni%2)*16+v)!=expected;
        ++bcount;
      }
    }
  }
  if(!legacy_bad)throw std::runtime_error("regression failed to detect old adjacent-N8 mapping");
  std::printf("{\"passed\":true,\"a_register_nibbles\":%d,\"b_register_nibbles\":%d,"
              "\"legacy_adjacent_b_mismatches\":%d,\"gpu_execution\":false}\n",acount,bcount,legacy_bad);
}
