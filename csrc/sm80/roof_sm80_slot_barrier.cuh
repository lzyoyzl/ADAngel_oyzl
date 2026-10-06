// SM80-only full/empty slot protocol. No extra producer warp or TMA claim.
#pragma once
namespace roof_slot_barrier {
__device__ __forceinline__ uint32_t address(const uint64_t& barrier) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
}
__device__ __forceinline__ void init(uint64_t& barrier, unsigned count) {
  const auto addr=address(barrier);
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(addr),"r"(count):"memory");
}
__device__ __forceinline__ void publish(uint64_t& barrier) {
  // Full is initialized to32: each producer lane contributes ONE async
  // completion, including lane16..31 with no activation-factor copy.
  const auto addr=address(barrier);
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(addr):"memory");
}
__device__ __forceinline__ void release(uint64_t& barrier) {
  // Empty is initialized to128. Every consumer thread arrives after its
  // last slot read. Default SM80 arrive is release; do not use SM90 count.
  const auto addr=address(barrier);
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(addr):"memory");
}
__device__ __forceinline__ void wait(uint64_t& barrier,unsigned phase) {
  const auto addr=address(barrier);
  unsigned ready;
  do {
    // SM80 test_wait, NOT SM90 try_wait. Default acquire publishes all
    // tracked copies/reader accesses. Phase is(group/stages)&1.
    asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared.b64 p, [%1], %2; "
                 "selp.u32 %0, 1, 0, p; }"
                 :"=r"(ready):"r"(addr),"r"(phase):"memory");
  } while(!ready);
}
} // namespace roof_slot_barrier
