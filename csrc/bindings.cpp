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
  result["o2_mxf4_block_scale"] = false;
  result["o2_cutlass_tiled"] = adangel_o2_cutlass_is_implemented();
  result["status"] = "O0/O1 production adapters are implemented; O2 is pending";
  return result;
}

py::object unavailable(py::args, py::kwargs) {
  TORCH_CHECK(false,
      "The SM120 production adapter is intentionally disabled until it is compiled, numerically "
      "validated, and instruction-audited on an RTX 5090. Reference code must not be used for "
      "formal timing.");
  return py::none();
}

py::dict benchmark(
    const std::string& variant,
    const std::string& mode,
    const at::Tensor& a_int8,
    const at::Tensor& a_scale,
    const at::Tensor& w_mxfp4,
    const at::Tensor& w_scale,
    int warmup,
    int repeats) {
  if (variant == "o0") {
    return adangel_benchmark_o0(
        a_int8, a_scale, w_mxfp4, w_scale, mode, warmup, repeats);
  }
  if (variant == "o1") {
    return adangel_benchmark_o1(
        a_int8, a_scale, w_mxfp4, w_scale, mode, warmup, repeats);
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
  module.def("run_o2", &unavailable);
  module.def("benchmark", &benchmark);
  module.def("verify_layout", &verify_layout);
}
