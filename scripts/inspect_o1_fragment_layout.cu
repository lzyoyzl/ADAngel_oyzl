#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/tensor.hpp>

#include <iostream>

namespace {

template <class TiledMma, class Shape>
void print_thread_zero_fragment(const char* name, TiledMma tiled_mma, Shape shape) {
  auto identity = cute::make_identity_tensor(shape);
  auto thread_mma = tiled_mma.get_slice(0);
  auto coordinates = thread_mma.partition_C(identity);
  std::cout << name << " size=" << cute::size(coordinates) << '\n';
  for (int item = 0; item < cute::size(coordinates); ++item) {
    auto coordinate = coordinates(item);
    std::cout << "  item=" << item
              << " row=" << static_cast<int>(cute::get<0>(coordinate))
              << " column=" << static_cast<int>(cute::get<1>(coordinate))
              << '\n';
  }
}

}  // namespace

int main() {
  using Atom = cute::MMA_Atom<cute::SM80_16x8x32_S32S8S8S32_TN>;

  using Tiled64x32 = cute::TiledMMA<
      Atom,
      cute::Layout<cute::Shape<cute::_4, cute::_2, cute::_1>>,
      cute::Tile<cute::_64, cute::_32, cute::_32>>;
  print_thread_zero_fragment(
      "64x32", Tiled64x32{}, cute::make_shape(cute::_64{}, cute::_32{}));

  using Tiled128x32 = cute::TiledMMA<
      Atom,
      cute::Layout<cute::Shape<cute::_8, cute::_2, cute::_1>>,
      cute::Tile<cute::_128, cute::_32, cute::_32>>;
  print_thread_zero_fragment(
      "128x32", Tiled128x32{}, cute::make_shape(cute::_128{}, cute::_32{}));

  using Tiled64x64 = cute::TiledMMA<
      Atom,
      cute::Layout<cute::Shape<cute::_4, cute::_4, cute::_1>>,
      cute::Tile<cute::_64, cute::_64, cute::_32>>;
  print_thread_zero_fragment(
      "64x64", Tiled64x64{}, cute::make_shape(cute::_64{}, cute::_64{}));

  using Tiled128x64 = cute::TiledMMA<
      Atom,
      cute::Layout<cute::Shape<cute::_8, cute::_2, cute::_1>>,
      cute::Tile<cute::_128, cute::_64, cute::_32>>;
  print_thread_zero_fragment(
      "128x64", Tiled128x64{}, cute::make_shape(cute::_128{}, cute::_64{}));
}
