#!/usr/bin/env bash
# Source this file after `conda activate adangel-sm120`.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "error: source this script instead of executing it:" >&2
  echo "  source scripts/activate_server_env.sh" >&2
  exit 2
fi

_adangel_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_adangel_repo_root="$(cd "${_adangel_script_dir}/.." && pwd)"

if [[ "${CONDA_DEFAULT_ENV:-}" != "adangel-sm120" ]]; then
  echo "warning: expected conda environment 'adangel-sm120'; current: '${CONDA_DEFAULT_ENV:-none}'" >&2
fi

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export TORCH_CUDA_ARCH_LIST=12.0a
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ADANGEL_CUTLASS_ROOT="${_adangel_repo_root}/third_party/cutlass-src"

unset _adangel_script_dir
unset _adangel_repo_root

echo "ADAngel server environment loaded: ${CONDA_DEFAULT_ENV:-no-conda}, CUDA ${CUDA_HOME}, GPU ${CUDA_VISIBLE_DEVICES}"
