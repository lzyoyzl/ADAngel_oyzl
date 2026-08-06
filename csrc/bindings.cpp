#include <torch/extension.h>

#include "adangel/kernel_api.h"

namespace py = pybind11;

namespace {

py::dict capabilities() {
  py::dict result;
  result["compiled_sm120a"] = true;
  result["o0_fp16_tc"] = false;
  result["o1_int8_tc"] = false;
  result["o2_mxf4_block_scale"] = false;
  result["o2_cutlass_tiled"] = adangel_o2_cutlass_is_implemented();
  result["status"] = "ISA probes are present; production adapters must be validated on RTX 5090";
  return result;
}

py::object unavailable(py::args, py::kwargs) {
  TORCH_CHECK(false,
      "The SM120 production adapter is intentionally disabled until it is compiled, numerically "
      "validated, and instruction-audited on an RTX 5090. Reference code must not be used for "
      "formal timing.");
  return py::none();
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
  module.def("run_o0", &unavailable);
  module.def("run_o1", &unavailable);
  module.def("run_o2", &unavailable);
  module.def("benchmark", &unavailable);
  module.def("verify_layout", &verify_layout);
}
