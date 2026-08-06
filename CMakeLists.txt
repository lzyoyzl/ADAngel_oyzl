cmake_minimum_required(VERSION 3.24)
project(adangel_sm120 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 120a)

find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)

if(NOT CUDAToolkit_VERSION VERSION_EQUAL "12.8")
  message(FATAL_ERROR "This experiment is pinned to CUDA Toolkit 12.8; found ${CUDAToolkit_VERSION}")
endif()
if(DEFINED ENV{ADANGEL_CUTLASS_ROOT})
  set(ADANGEL_CUTLASS_ROOT "$ENV{ADANGEL_CUTLASS_ROOT}")
else()
  set(ADANGEL_CUTLASS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass-src")
endif()

if(NOT EXISTS "${ADANGEL_CUTLASS_ROOT}/include/cutlass/cutlass.h")
  message(FATAL_ERROR "CUTLASS not found. Initialize third_party/cutlass or set ADANGEL_CUTLASS_ROOT.")
endif()

set(ADANGEL_EXPECTED_CUTLASS "db1c288993354c88e551c40c19a8fb93a774a241")
execute_process(
  COMMAND git -C "${ADANGEL_CUTLASS_ROOT}" rev-parse HEAD
  OUTPUT_VARIABLE ADANGEL_ACTUAL_CUTLASS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE ADANGEL_CUTLASS_GIT_RESULT)
if(NOT ADANGEL_CUTLASS_GIT_RESULT EQUAL 0)
  message(FATAL_ERROR "CUTLASS must be a git checkout so its revision can be verified")
endif()
if(NOT ADANGEL_ACTUAL_CUTLASS STREQUAL ADANGEL_EXPECTED_CUTLASS)
  message(FATAL_ERROR "CUTLASS revision mismatch: expected ${ADANGEL_EXPECTED_CUTLASS}, got ${ADANGEL_ACTUAL_CUTLASS}")
endif()

add_library(adangel_sm120 MODULE
  csrc/bindings.cpp
  csrc/common/validation.cu
  csrc/sm120/conversion.cu
  csrc/sm120/o0_gemm.cu
  csrc/sm120/o1_gemm.cu
  csrc/sm120/o2_microkernel.cu
  csrc/sm120/o2_cutlass.cu)

target_include_directories(adangel_sm120 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${ADANGEL_CUTLASS_ROOT}/include
  ${TORCH_INCLUDE_DIRS})
target_link_libraries(adangel_sm120 PRIVATE ${TORCH_LIBRARIES} CUDA::cudart CUDA::cublasLt)
target_compile_definitions(adangel_sm120 PRIVATE TORCH_EXTENSION_NAME=_sm120)
target_compile_options(adangel_sm120 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-lineinfo>)
set_target_properties(adangel_sm120 PROPERTIES PREFIX "" OUTPUT_NAME "_sm120")
