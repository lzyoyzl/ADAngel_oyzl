// Host-only bridge: experimental device templates must not enter the legacy TU.
#pragma once
#include <cstddef>
#include <cstdint>

namespace adangel_sm80_experiment {
using Kernel = void (*)(const uint8_t*, const uint8_t*, const float*,
                       const uint8_t*, float*, int, int, int);
Kernel select_three_stage_kernel(bool dual, bool fast, int tune);
Kernel select_warp_reuse_kernel(bool dual, bool fast, int tune);
Kernel select_reuse_pipeline_kernel(bool dual, bool fast, int tune);
Kernel select_reduction_kernel(bool dual, bool fast, int tune);
Kernel select_reduction_budget_kernel(bool dual, bool fast, int tune);
size_t three_stage_shared_bytes(bool dual);
} // namespace adangel_sm80_experiment
