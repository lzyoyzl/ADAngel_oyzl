#pragma once
#include <memory>
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

// Reuse O0's exact cuBLASLt HMMA/FP32/no-split-K selection policy. All plan,
// workspace and output allocation happens before the benchmark's timed region.
class AdangelFp16Runner {
 public:
  virtual ~AdangelFp16Runner() = default;
  virtual void run(cudaStream_t stream) = 0;
  virtual pybind11::dict metadata() const = 0;
};
std::unique_ptr<AdangelFp16Runner> adangel_make_fp16_runner(
    const at::Tensor& a, const at::Tensor& w, const at::Tensor& y);
