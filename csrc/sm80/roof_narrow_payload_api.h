// Host-only bridge for the isolated N64 / four-warp experiment.
#pragma once
#include "roof_pipeline_api.h"

namespace adangel_sm80_experiment {
Kernel select_narrow_payload_kernel(bool dual, bool fast, int tune);
size_t narrow_payload_shared_bytes(bool dual, int tune);
}
