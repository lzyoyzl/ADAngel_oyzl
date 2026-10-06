// v110 diagnostic: no public binding, quantizer or native-extension change.
#include "roof_o78_eight_chain_probe.cu"
namespace {
#include "o78_shell_generated.cuh"
}
extern "C" __global__ __launch_bounds__(128,3)
void adangel_capacity_register_shell(float* y,int groups,int seed) {
  o78_shell_capacity_experiment::body<0>(y,groups,seed);
}
extern "C" __global__ __launch_bounds__(128,3)
void adangel_capacity_shared_shell(float* y,int groups,int seed) {
  o78_shell_capacity_experiment::body<1>(y,groups,seed);
}
extern "C" __global__ __launch_bounds__(128,3)
void adangel_capacity_scaled_shared_shell(float* y,int groups,int seed) {
  o78_shell_capacity_experiment::body<2>(y,groups,seed);
}
