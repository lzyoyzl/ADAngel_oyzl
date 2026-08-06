#!/usr/bin/env bash
set -euo pipefail

build_dir=${1:?usage: audit_instructions.sh BUILD_DIR OUTPUT_DIR}
output_dir=${2:?usage: audit_instructions.sh BUILD_DIR OUTPUT_DIR}
mkdir -p "$output_dir"

binary=$(find "$build_dir" -type f \( -name '_sm120*.so' -o -name '_sm120*.pyd' \) -print -quit)
if [[ -z "$binary" ]]; then
  echo "native extension not found below $build_dir" >&2
  exit 1
fi
command -v cuobjdump >/dev/null || { echo 'cuobjdump is required' >&2; exit 1; }

cuobjdump --dump-ptx "$binary" > "$output_dir/extension.ptx"
cuobjdump --dump-sass "$binary" > "$output_dir/extension.sass"

grep -E 'mxf4.*block_scale|block_scale.*mxf4' "$output_dir/extension.ptx" >/dev/null || {
  echo 'O2 MXFP4 block-scaled MMA not found in PTX' >&2; exit 1;
}
grep -E 'mma.*s8.*s8.*s32' "$output_dir/extension.ptx" >/dev/null || {
  echo 'O1 INT8 MMA not found in PTX' >&2; exit 1;
}
grep -E 'mma.*f16.*f16.*f32|HMMA' "$output_dir/extension.ptx" "$output_dir/extension.sass" >/dev/null || {
  echo 'O0 FP16 Tensor Core instruction not found' >&2; exit 1;
}
grep -E 'MMA|HMMA|IMMA' "$output_dir/extension.sass" >/dev/null || {
  echo 'no Tensor Core instruction mnemonic found in SASS' >&2; exit 1;
}

{
  echo "binary=$binary"
  echo "status=PASS"
  echo "manual_review=Confirm the matched instructions belong to adangel_o0/o1/o2 kernel symbols."
} > "$output_dir/summary.txt"
echo "instruction audit passed; review $output_dir/summary.txt and extension.sass"
