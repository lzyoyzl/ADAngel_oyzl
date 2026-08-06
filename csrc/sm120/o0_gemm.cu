#include <cuda_fp16.h>
#include <stdint.h>

// O0 production uses cublasLt with FP16 A/B, FP32 compute and FP32 D. This translation unit is
// kept separate so its algorithm ID/workspace can be pinned after tuning on the target RTX 5090.
// The Python capability gate remains false until that target-specific selection is checked in.
extern "C" __global__ void adangel_o0_fp16_mma_probe(
    const uint32_t* a, const uint32_t* b, float* d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  float x0 = 0, x1 = 0, x2 = 0, x3 = 0;
  uint32_t a0 = a[threadIdx.x * 4 + 0], a1 = a[threadIdx.x * 4 + 1];
  uint32_t a2 = a[threadIdx.x * 4 + 2], a3 = a[threadIdx.x * 4 + 3];
  uint32_t b0 = b[threadIdx.x * 2 + 0], b1 = b[threadIdx.x * 2 + 1];
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(x0), "+f"(x1), "+f"(x2), "+f"(x3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  d[threadIdx.x * 4 + 0] = x0;
#endif
}
