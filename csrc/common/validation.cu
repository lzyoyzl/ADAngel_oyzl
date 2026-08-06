#include <torch/extension.h>

#include "adangel/kernel_api.h"

bool adangel_validate_cuda_inputs(
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4,
    const at::Tensor& w_scale) {
  TORCH_CHECK(a_int8.is_cuda() && a_scale.is_cuda() && w_mxfp4.is_cuda() && w_scale.is_cuda(),
              "all inputs must be CUDA tensors");
  TORCH_CHECK(a_int8.scalar_type() == at::kChar, "A_int8 must be int8");
  TORCH_CHECK(a_scale.scalar_type() == at::kFloat, "A_scale must be fp32");
  TORCH_CHECK(w_mxfp4.scalar_type() == at::kByte, "W_mxfp4 must be uint8");
  TORCH_CHECK(w_scale.scalar_type() == at::kByte, "W_scale must be uint8");
  TORCH_CHECK(a_int8.is_contiguous() && a_scale.is_contiguous() &&
              w_mxfp4.is_contiguous() && w_scale.is_contiguous(), "all inputs must be contiguous");
  auto m = a_int8.size(0), k = a_int8.size(1), n = w_mxfp4.size(0);
  TORCH_CHECK(k % 64 == 0, "K must be divisible by 64");
  TORCH_CHECK(a_scale.sizes() == at::IntArrayRef({m}), "invalid A_scale shape");
  TORCH_CHECK(w_mxfp4.sizes() == at::IntArrayRef({n, k / 2}), "invalid W_mxfp4 shape");
  TORCH_CHECK(w_scale.sizes() == at::IntArrayRef({n, k / 32}), "invalid W_scale shape");
  return true;
}
