// Host-only real CuTe coordinate checks, not GPU numerical acceptance.
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cassert>
#include <iostream>
#include <set>
#include <vector>
int main() {
  using Atom=cute::MMA_Atom<cute::SM80_16x8x64_S32S4U4S32_TN>;
  using Warps=cute::Layout<cute::Shape<cute::_2,cute::_2,cute::_1>>;
  using MMA=cute::TiledMMA<Atom,Warps,cute::Tile<cute::_128,cute::_64,cute::_64>>;
  using Slice=cute::TiledMMA<Atom,Warps,cute::Tile<cute::_64,cute::_64,cute::_64>>;
  MMA mma;Slice sm;
  auto iw=cute::make_identity_tensor(cute::make_shape(cute::_128{},cute::_128{}));
  auto ia=cute::make_identity_tensor(cute::make_shape(cute::_64{},cute::_128{}));
  auto iy=cute::make_identity_tensor(cute::make_shape(cute::_128{},cute::_64{}));
  std::vector<int> owners(64*128);unsigned compared=0;
  for(int tid=0;tid<128;++tid) {
    auto thr=mma.get_slice(tid);auto st=sm.get_slice(tid);
    auto fw=thr.partition_A(iw);auto fa=thr.partition_B(ia);auto c=thr.partition_C(iy);
    assert(cute::size(c)==64 && cute::size<1>(c)==4 && cute::size<2>(c)==4);
    std::set<int> weight_columns,activation_rows;
    for(int i=0;i<64;++i) {
      auto p=c(i);int col=cute::get<0>(p),row=cute::get<1>(p);
      assert(0<=col && col<128 && 0<=row && row<64);
      ++owners[row*128+col];weight_columns.insert(col);activation_rows.insert(row);
    }
    assert(weight_columns.size()==8 && activation_rows.size()==8);
    for(int mb=0;mb<2;++mb) for(int half=0;half<2;++half) {
      auto tw=cute::local_tile(iw,cute::make_shape(cute::_64{},cute::_64{}),cute::make_coord(mb,half));
      auto sw=st.partition_A(tw);
      for(int mi=0;mi<2;++mi) for(int v=0;v<cute::size<0>(sw);++v) {
        auto x=sw(v,mi,0),z=fw(v,mb*2+mi,half);
        assert(cute::get<0>(x)==cute::get<0>(z) && cute::get<1>(x)==cute::get<1>(z));++compared;
      }
    }
    for(int half=0;half<2;++half) {
      auto ta=cute::local_tile(ia,cute::make_shape(cute::_64{},cute::_64{}),cute::make_coord(0,half));
      auto sa=thr.partition_B(ta);
      for(int ni=0;ni<4;++ni) for(int v=0;v<cute::size<0>(sa);++v) {
        auto x=sa(v,ni,0),z=fa(v,ni,half);
        assert(cute::get<0>(x)==cute::get<0>(z) && cute::get<1>(x)==cute::get<1>(z));++compared;
      }
    }
  }
  for(int n:owners) assert(n==1);
  std::cout<<"{\"passed\":true,\"gpu_execution\":false,\"outputs\":8192,\"threads\":128,"
      <<"\"weight_scales_per_thread\":8,\"activation_rows_per_thread\":8,"
      <<"\"operand_coordinates_compared\":"<<compared<<"}\n";
}
