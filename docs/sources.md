# 规范与实现依据

- OCP Microscaling Formats (MX) Specification v1.0:
  https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- NVIDIA PTX ISA, Matrix Multiply-Accumulate Operation using `mma` instruction:
  https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-multiply-accumulate-operation-using-mma-instruction
- NVIDIA CUTLASS release v4.5.2:
  https://github.com/NVIDIA/cutlass/releases/tag/v4.5.2
- Pinned CUTLASS commit:
  https://github.com/NVIDIA/cutlass/commit/db1c288993354c88e551c40c19a8fb93a774a241
- CUDA Toolkit 12.8 Update 1 release notes:
  https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html
- PyTorch 2.7.1 CUDA 12.8 wheel installation matrix:
  https://pytorch.org/get-started/previous-versions/
- NVIDIA CUDA installation guide for Linux:
  https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-linux/index.html

OCP v1.0 section 6.3 defines the recommended block conversion: choose the largest power of two
not exceeding the block maximum, divided by the largest power of two representable by the element
type; E2M1 conversion supports roundTiesToEven and saturating overflow. PTX defines the SM120
m16n8k64 MXFP4 `scale_vec::2X` instruction and its scale-factor fragment layouts.
