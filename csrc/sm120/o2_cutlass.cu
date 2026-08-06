#include "adangel/kernel_api.h"

// Publication-performance O2 must be instantiated from a pinned CUTLASS SM120 block-scaled GEMM
// builder and validated on the target toolkit. This truthful capability bit prevents the PTX ISA
// probe above from ever being benchmarked as the formal 4096^3 kernel.
bool adangel_o2_cutlass_is_implemented() { return false; }
