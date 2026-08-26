#pragma once

#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <string>
#include <utility>

constexpr int kAdangelDefaultConversionTimingInnerRepeats = 100;

// The Python ABI is bound in csrc/bindings.cpp. These symbols separate the validated single-warp
// ISA tests from the publication-performance CUTLASS adapter.
bool adangel_validate_cuda_inputs(
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4,
    const at::Tensor& w_scale);

void adangel_launch_mxfp4_to_fp16(
    const at::Tensor& packed,
    const at::Tensor& scales,
    at::Tensor& output,
    cudaStream_t stream);

void adangel_launch_int8_to_fp16(
    const at::Tensor& input,
    const at::Tensor& scales,
    at::Tensor& output,
    cudaStream_t stream);

void adangel_launch_mxfp4_to_int8(
    const at::Tensor& packed,
    at::Tensor& output,
    cudaStream_t stream);

void adangel_launch_mxfp4_to_q4(
    const at::Tensor& packed,
    at::Tensor& output,
    cudaStream_t stream);

void adangel_launch_split_int8_to_int4(
    const at::Tensor& input,
    at::Tensor& output,
    cudaStream_t stream);

void adangel_launch_int8_bitplanes(
    const at::Tensor& input,
    at::Tensor& output,
    cudaStream_t stream);

void adangel_launch_q4_bitplanes(
    const at::Tensor& packed,
    at::Tensor& output,
    cudaStream_t stream);

bool adangel_o0_is_implemented();
pybind11::dict adangel_run_o0(
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4,
    const at::Tensor& w_scale,
    const std::string& mode);
pybind11::dict adangel_benchmark_o0(
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4,
    const at::Tensor& w_scale,
    const std::string& mode,
    int warmup,
    int repeats,
    int conversion_inner_repeats);

bool adangel_o1_is_implemented();
pybind11::dict adangel_run_o1(
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4,
    const at::Tensor& w_scale,
    const std::string& mode);
pybind11::dict adangel_benchmark_o1(
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4,
    const at::Tensor& w_scale,
    const std::string& mode,
    int warmup,
    int repeats,
    int conversion_inner_repeats);
pybind11::dict adangel_benchmark_o1_impl(
    const std::string& implementation,
    const std::string& mode,
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4,
    const at::Tensor& w_scale,
    int warmup,
    int repeats,
    int conversion_inner_repeats);

std::pair<bool, float> adangel_verify_o2_layout_cuda();
bool adangel_o2_cutlass_is_implemented();
pybind11::dict adangel_run_o2(
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4,
    const at::Tensor& w_scale,
    const std::string& mode);
pybind11::dict adangel_benchmark_o2(
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4,
    const at::Tensor& w_scale,
    const std::string& mode,
    int warmup,
    int repeats,
    int conversion_inner_repeats);

bool adangel_o3_is_implemented();
pybind11::dict adangel_run_o3(
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4_g128,
    const at::Tensor& w_scale_g128,
    const std::string& mode);
pybind11::dict adangel_benchmark_o3(
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4_g128,
    const at::Tensor& w_scale_g128,
    const std::string& mode,
    int warmup,
    int repeats,
    int conversion_inner_repeats);

bool adangel_o4_is_implemented();
pybind11::dict adangel_run_o4(
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4_g128,
    const at::Tensor& w_scale_g128,
    const std::string& mode);
pybind11::dict adangel_benchmark_o4(
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4_g128,
    const at::Tensor& w_scale_g128,
    const std::string& mode,
    int warmup,
    int repeats,
    int conversion_inner_repeats);
