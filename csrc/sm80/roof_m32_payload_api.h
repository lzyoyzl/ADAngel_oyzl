#pragma once
#include "roof_pipeline_api.h"
namespace adangel_sm80_experiment {
Kernel select_m32_payload_kernel(int tune);
size_t m32_payload_shared_bytes(int tune);
}
