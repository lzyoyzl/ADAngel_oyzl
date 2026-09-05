#include <torch/extension.h>

#include <string>

#include "adangel/kernel_api.h"

namespace py = pybind11;

namespace {

py::dict capabilities() {
  py::dict result;
  result["compiled_sm120a"] = true;
  result["o0_fp16_tc"] = adangel_o0_is_implemented();
  result["o1_int8_tc"] = adangel_o1_is_implemented();
  result["o2_mxf4_block_scale"] = adangel_o2_cutlass_is_implemented();
  result["o2_cutlass_tiled"] = adangel_o2_cutlass_is_implemented();
  result["o3_int4_tc"] = adangel_o3_is_implemented();
  result["o3_tma_warp_specialized"] = adangel_o3_is_implemented();
  result["o4_int1_tc"] = adangel_o4_is_implemented();
  result["o4_tma_warp_specialized"] = adangel_o4_is_implemented();
  result["status"] = "O0/O1/O2/O3/O4 production adapters are implemented";
  return result;
}

py::dict benchmark(
    const std::string& variant,
    const std::string& mode,
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4,
    const at::Tensor& w_scale,
    int warmup,
    int repeats,
    int conversion_inner_repeats) {
  if (variant == "o0") {
    return adangel_benchmark_o0(
        a_int8,
        a_scale,
        w_mxfp4,
        w_scale,
        mode,
        warmup,
        repeats,
        conversion_inner_repeats);
  }
  if (variant == "o1") {
    return adangel_benchmark_o1(
        a_int8,
        a_scale,
        w_mxfp4,
        w_scale,
        mode,
        warmup,
        repeats,
        conversion_inner_repeats);
  }
  if (variant == "o2") {
    return adangel_benchmark_o2(
        a_int8,
        a_scale,
        w_mxfp4,
        w_scale,
        mode,
        warmup,
        repeats,
        conversion_inner_repeats);
  }
  if (variant == "o3") {
    return adangel_benchmark_o3(
        a_int8,
        a_scale,
        w_mxfp4,
        w_scale,
        mode,
        warmup,
        repeats,
        conversion_inner_repeats);
  }
  if (variant == "o4") {
    return adangel_benchmark_o4(
        a_int8,
        a_scale,
        w_mxfp4,
        w_scale,
        mode,
        warmup,
        repeats,
        conversion_inner_repeats);
  }
  TORCH_CHECK(
      false,
      "The ",
      variant,
      " SM120 production benchmark is not implemented; reference code must not be used for "
      "formal timing.");
  return py::dict();
}

py::dict verify_layout() {
  auto [passed, max_abs_error] = adangel_verify_o2_layout_cuda();
  py::dict result;
  result["passed"] = passed;
  result["max_abs_error"] = max_abs_error;
  return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("capabilities", &capabilities);
  module.def("run_o0", &adangel_run_o0);
  module.def("run_o1", &adangel_run_o1);
  module.def("run_o2", &adangel_run_o2);
  module.def("run_o3", &adangel_run_o3);
  module.def("run_o4", &adangel_run_o4);
  module.def(
      "benchmark",
      &benchmark,
      py::arg("variant"),
      py::arg("mode"),
      py::arg("a_int8"),
      py::arg("a_scale"),
      py::arg("w_mxfp4"),
      py::arg("w_scale"),
      py::arg("warmup"),
      py::arg("repeats"),
      py::arg("conversion_inner_repeats") =
          kAdangelDefaultConversionTimingInnerRepeats);
  module.def(
      "_benchmark_o1_impl",
      &adangel_benchmark_o1_impl,
      py::arg("implementation"),
      py::arg("mode"),
      py::arg("a_int8"),
      py::arg("a_scale"),
      py::arg("w_mxfp4"),
      py::arg("w_scale"),
      py::arg("warmup"),
      py::arg("repeats"),
      py::arg("conversion_inner_repeats") =
          kAdangelDefaultConversionTimingInnerRepeats);
  module.def(
      "_benchmark_o3_impl",
      &adangel_benchmark_o3_impl,
      py::arg("implementation"),
      py::arg("mode"),
      py::arg("a_int8"),
      py::arg("a_scale"),
      py::arg("w_mxfp4"),
      py::arg("w_scale"),
      py::arg("warmup"),
      py::arg("repeats"),
      py::arg("conversion_inner_repeats") =
          kAdangelDefaultConversionTimingInnerRepeats);
  module.def(
      "_benchmark_o3_split_int8x2",
      &adangel_benchmark_o3_split_int8x2_diagnostic,
      py::arg("mode"),
      py::arg("a_int8"),
      py::arg("a_scale"),
      py::arg("w_mxfp4"),
      py::arg("w_scale"),
      py::arg("warmup"),
      py::arg("repeats"),
      py::arg("conversion_inner_repeats") =
          kAdangelDefaultConversionTimingInnerRepeats);
  module.def(
      "_benchmark_o4_impl",
      &adangel_benchmark_o4_impl,
      py::arg("implementation"),
      py::arg("mode"),
      py::arg("a_int8"),
      py::arg("a_scale"),
      py::arg("w_mxfp4"),
      py::arg("w_scale"),
      py::arg("warmup"),
      py::arg("repeats"),
      py::arg("conversion_inner_repeats") =
          kAdangelDefaultConversionTimingInnerRepeats);
  module.def("verify_layout", &verify_layout);
}
