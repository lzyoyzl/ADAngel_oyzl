//Host-only CuTe ownership proof; no kernel launch/performance claim.
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cassert>
#include <iostream>
#include <vector>

template<int WN,typename Atom>
using Mma=cute::TiledMMA<cute::MMA_Atom<Atom>,
    cute::Layout<cute::Shape<cute::_2,cute::Int<WN>,cute::_1>>,
    cute::Tile<cute::_64,cute::_128,cute::_64>>;

template<int WN> void verify() {
  using Low=Mma<WN,cute::SM80_16x8x64_S32U4S4S32_TN>;
  using High=Mma<WN,cute::SM80_16x8x64_S32S4S4S32_TN>;
  Low low;High high;
  auto identity=cute::make_identity_tensor(cute::make_shape(cute::_64{},cute::_128{}));
  auto weights=cute::make_identity_tensor(cute::make_shape(cute::_128{},cute::_128{}));
  std::vector<int> seen(64*128);
  constexpr int Threads=32*2*WN;
  for(int thread=0;thread<Threads;++thread) {
    auto l=low.get_slice(thread).partition_C(identity);
    auto h=high.get_slice(thread).partition_C(identity);
    assert(cute::size(l)==64*128/Threads);
    assert(cute::size<1>(l)==2 && cute::size<2>(l)==16/WN);
    for(int i=0;i<cute::size(l);++i) {
      const auto p=l(i),q=h(i);
      const int row=cute::get<0>(p),col=cute::get<1>(p);
      assert(row==cute::get<0>(q) && col==cute::get<1>(q));
      assert(row>=0 && row<64 && col>=0 && col<128);++seen[row*128+col];
    }
    if constexpr(WN==4) {
      for(int half=0;half<2;++half) {
        auto tile=cute::local_tile(weights,cute::make_shape(cute::_128{},cute::_64{}),
                                  cute::make_coord(0,half));
        auto b=low.get_slice(thread).partition_B(tile);
        auto bh=high.get_slice(thread).partition_B(tile);
        assert(cute::size<1>(b)==4 && cute::size(b)==cute::size(bh));
        for(int i=0;i<cute::size(b);++i) {
          assert(cute::get<0>(b(i))==cute::get<0>(bh(i)));
          assert(cute::get<1>(b(i))==cute::get<1>(bh(i)));
          assert(cute::get<0>(b(i))>=0 && cute::get<0>(b(i))<128);
          assert(cute::get<1>(b(i))>=half*64 && cute::get<1>(b(i))<(half+1)*64);
        }
      }
    }
  }
  for(auto count:seen) assert(count==1);
  std::cout<<"threads="<<Threads<<" unique_outputs=8192 accumulator_per_thread="
           <<64*128/Threads<<" native_low_high_coordinates_match=true\n";
}
int main() {verify<2>();verify<4>();}
