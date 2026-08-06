#pragma once

#include <torch/extension.h>
#include <utility>


// The Python ABI is bound in csrc/bindings.cpp. These symbols separate the validated single-warp
// ISA tests from the publication-performance CUTLASS adapter.
bool adangel_validate_cuda_inputs(
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4,
    const at::Tensor& w_scale);

std::pair<bool, float> adangel_verify_o2_layout_cuda();
bool adangel_o2_cutlass_is_implemented();
