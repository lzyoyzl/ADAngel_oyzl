// Isolated v91: form each A address at the copy, not as a long-lived pointer.
// Preserve original uint32 component arithmetic and uint64 pointer additions.
#pragma once
namespace roof_address_remat {
template<bool High>
__device__ __forceinline__ void copy16(uint8_t* dst,const uint8_t* base,
    uint32_t group_offset,uint32_t row_offset,uint32_t high_offset) {
  const uint32_t shared=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  if constexpr(High) {
    asm volatile("{ .reg .u64 address;\n"
        "mad.wide.u32 address,%2,1,%1;\n"
        "mad.wide.u32 address,%3,1,address;\n"
        "mad.wide.u32 address,%4,1,address;\n"
        "cp.async.cg.shared.global [%0],[address],16;\n}"
        ::"r"(shared),"l"(base),"r"(group_offset),"r"(row_offset),"r"(high_offset):"memory");
  } else {
    asm volatile("{ .reg .u64 address;\n"
        "mad.wide.u32 address,%2,1,%1;\n"
        "mad.wide.u32 address,%3,1,address;\n"
        "cp.async.cg.shared.global [%0],[address],16;\n}"
        ::"r"(shared),"l"(base),"r"(group_offset),"r"(row_offset):"memory");
  }
}
} // namespace roof_address_remat
