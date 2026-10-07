// v122 independent candidate: four MMA warps + four integer-scale warps.
// The host/online guard must prove ALL G128 partials in this CTA fit INT16.
// Narrow storage changes no quantization, arithmetic, full-K guard or output.
#pragma once
namespace o78_partial_handoff {
using C=o78_eight_chain_experiment::C;
using Storage=o78_eight_chain_experiment::Storage;
using o78_eight_chain_experiment::prefetch;
struct alignas(128) Shared {
  Storage matrix;
  // Two N64 panels; four vector stores/thread, contiguous/bank-balanced.
  alignas(128) uint4 partial[2][4][128];
  alignas(128) int factors[2][128]; // 64 A-row +64 W-column snapshots.
};
static_assert(sizeof(Shared)==51712);

template<int Id,int Count> __device__ __forceinline__ void wait() {
  asm volatile("bar.sync %0, %1;" :: "n"(Id),"n"(Count):"memory");
}
template<int Id> __device__ __forceinline__ void arrive() {
  asm volatile("bar.arrive %0, 256;" :: "n"(Id):"memory");
}
__device__ __forceinline__ unsigned pack(int a,int b) {
  unsigned r;asm("prmt.b32 %0, %1, %2, 0x5410;":"=r"(r):"r"(a),"r"(b));return r;
}
template<int High> __device__ __forceinline__ int unpack(unsigned x) {
  int r;
  if constexpr(High) asm("shr.s32 %0, %1, 16;":"=r"(r):"r"(x));
  else asm("prmt.b32 %0, %1, 0, 0x9910;":"=r"(r):"r"(x));
  return r;
}

#include "o78_partial_handoff_producer_generated.cuh"

__device__ __forceinline__ void consumer(const float* base_a,const float* base_w,
                                        float* y,uint32_t n) {
  extern __shared__ __align__(128) uint8_t buf[];
  auto& h=*reinterpret_cast<Shared*>(buf);
  const unsigned tid=threadIdx.x-128;
  C::Mma mma;auto thr=mma.get_slice(tid);
  auto coords=thr.partition_C(cute::make_identity_tensor(cute::make_shape(cute::_64{},cute::_128{})));
  auto acc=cute::make_fragment_like<int>(thr.make_fragment_C(coords));cute::clear(acc);
  static_assert(decltype(cute::size(acc))::value==64);
  // Empty handoff slots are primed once. Ready/empty are separate phases.
  arrive<2>();arrive<4>();
  for(int group=0;group<32;++group) {
    o1_static_for<0,2>([&](auto nb) {
      constexpr int b=decltype(nb)::value;
      wait<1+2*b,256>();
      auto packed=cute::make_tensor<unsigned>(cute::make_shape(cute::_16{}));
      o1_static_for<0,4>([&](auto q) {
        const uint4 v=h.partial[b][q][tid];
        packed(q*4)=v.x;packed(q*4+1)=v.y;packed(q*4+2)=v.z;packed(q*4+3)=v.w;
      });
      // Coordinates come from CuTe, not assumed lane mappings.
      auto rows=cute::make_tensor<int>(cute::make_shape(cute::_4{},cute::_2{}));
      auto columns=cute::make_tensor<int>(cute::make_shape(cute::_4{},cute::_4{}));
      o1_static_for<0,2>([&](auto mi) {
        o1_static_for<0,4>([&](auto vi) {
          rows(vi,mi)=h.factors[b][cute::get<0>(coords(vi,mi,nb*cute::_4{}))];
        });
      });
      o1_static_for<0,4>([&](auto ni) {
        o1_static_for<0,4>([&](auto vi) {
          columns(vi,ni)=h.factors[b][64+cute::get<1>(coords(vi,cute::_0{},nb*cute::_4{}+ni))-b*64];
        });
      });
      // All reads precede release. Factors/partials are now thread-private.
      arrive<2+2*b>();
      o1_static_for<0,2>([&](auto mi) {
        o1_static_for<0,4>([&](auto ni) {
          o1_static_for<0,4>([&](auto vi) {
            constexpr int i=decltype(vi)::value+4*decltype(mi)::value+8*decltype(ni)::value;
            const int coefficient=rows(vi,mi)*columns(vi,ni);
            acc(vi,mi,nb*cute::_4{}+ni)+=unpack<i%2>(packed(i/2))*coefficient;
          });
        });
      });
    });
  }
  // Identical FP32 base restore and one final global store per output.
  o1_static_for<0,decltype(cute::size(acc))::value>([&](auto i) {
    const auto c=coords(i);
    acc(i)=__float_as_int(__fmul_rn(__fmul_rn(float(acc(i)),
        base_a[blockIdx.y*64+cute::get<0>(c)]),base_w[blockIdx.x*128+cute::get<1>(c)]));
  });
  o1_static_for<0,decltype(cute::size(acc))::value/2>([&](auto pair) {
    auto i=pair*cute::_2{};const auto p=coords(i),q=coords(i+cute::_1{});
    uint32_t offset=(blockIdx.y*64+cute::get<0>(p))*n+blockIdx.x*128+cute::get<1>(p);
    if(cute::get<0>(p)==cute::get<0>(q) && cute::get<1>(q)==cute::get<1>(p)+1 && (offset&1u)==0)
      *reinterpret_cast<float2*>(y+offset)=make_float2(__int_as_float(acc(i)),__int_as_float(acc(i+cute::_1{})));
    else {
      y[offset]=__int_as_float(acc(i));
      y[(blockIdx.y*64+cute::get<0>(q))*n+blockIdx.x*128+cute::get<1>(q)]=__int_as_float(acc(i+cute::_1{}));
    }
  });
}
} // namespace o78_partial_handoff
