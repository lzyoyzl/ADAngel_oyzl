#!/usr/bin/env bash
set -euo pipefail

build_dir=${1:?usage: audit_instructions.sh BUILD_DIR OUTPUT_DIR}
output_dir=${2:?usage: audit_instructions.sh BUILD_DIR OUTPUT_DIR}
mkdir -p "$output_dir"
summary_file="$output_dir/summary.txt"

{
  echo "status=RUNNING"
  echo "note=The audit did not finish."
} > "$summary_file"
audit_complete=0
trap 'if [[ "$audit_complete" != 1 ]]; then
  {
    echo "status=FAIL"
    echo "note=The audit stopped before all production-kernel checks passed."
  } > "$summary_file"
fi' EXIT

binary=$(find "$build_dir" -type f \( -name '_sm120*.so' -o -name '_sm120*.pyd' \) -print -quit)
if [[ -z "$binary" ]]; then
  echo "native extension not found below $build_dir" >&2
  exit 1
fi
command -v cuobjdump >/dev/null || { echo 'cuobjdump is required' >&2; exit 1; }

cuobjdump --dump-ptx "$binary" > "$output_dir/extension.ptx"
cuobjdump --dump-sass "$binary" > "$output_dir/extension.sass"
cuobjdump --dump-resource-usage "$binary" > "$output_dir/extension.resources.txt"

grep -E 'mma.*f16.*f16.*f32|HMMA' "$output_dir/extension.ptx" "$output_dir/extension.sass" >/dev/null || {
  echo 'O0 FP16 Tensor Core instruction not found' >&2; exit 1;
}

find_ptx_symbol() {
  local family=$1
  local needle=${2:-}
  awk -v family="$family" -v needle="$needle" '
    function reset_entry() {
      entry = ""
      symbol = ""
      depth = 0
      opened = 0
      has_tma = 0
      has_mma = 0
    }
    function finish_entry() {
      if (!opened) return
      if (family == "o1" &&
          index(symbol, needle) != 0 &&
          has_tma && has_mma) {
        print symbol
        found = 1
        exit 0
      }
      if (family == "o2" &&
          symbol !~ /o2_mxf4_layout_probe/ &&
          has_tma && has_mma) {
        print symbol
        found = 1
        exit 0
      }
      reset_entry()
    }
    BEGIN { reset_entry() }
    /^[[:space:]]*(\.visible[[:space:]]+)?\.entry[[:space:]]/ {
      reset_entry()
      entry = $0
      symbol = $0
      sub(/^.*entry[[:space:]]+/, "", symbol)
      sub(/\(.*/, "", symbol)
    }
    entry != "" {
      line = $0
      opens = gsub(/\{/, "{", line)
      closes = gsub(/\}/, "}", line)
      if (opens > 0) opened = 1
      depth += opens - closes
      if (/cp\.async\.bulk\.tensor/) has_tma = 1
      if (family == "o1" && /mma.*s8.*s8.*s32/) has_mma = 1
      if (family == "o2" &&
          (/mma.*kind::mxf4.*block_scale/ ||
           /mma.*block_scale.*kind::mxf4/)) has_mma = 1
      if (opened && depth == 0) finish_entry()
    }
    END { if (!found) exit 1 }
  ' "$output_dir/extension.ptx"
}

check_sass_symbol() {
  local symbol=$1
  local family=$2
  awk -v symbol="$symbol" -v family="$family" '
    /^[[:space:]]*Function[[:space:]]*:/ {
      inside = index($0, symbol) != 0
      next
    }
    inside && /UTMALDG|UTMA/ { has_tma = 1 }
    inside && family == "o1" && /IMMA/ { has_mma = 1 }
    inside && family == "o2" && /OMMA/ { has_mma = 1 }
    inside && /LDL|STL/ { has_local = 1 }
    END { exit (has_tma && has_mma) ? 0 : 1 }
  ' "$output_dir/extension.sass"
}

check_no_local_sass() {
  local symbol=$1
  awk -v symbol="$symbol" '
    /^[[:space:]]*Function[[:space:]]*:/ {
      inside = index($0, symbol) != 0
      next
    }
    inside && /LDL|STL/ { found = 1 }
    END { exit found ? 1 : 0 }
  ' "$output_dir/extension.sass"
}

source_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
production_impl=$(sed -n 's/.*kProductionO1Implementation.*"\([^"]*\)".*/\1/p' \
  "$source_root/csrc/sm120/o1_gemm.cu" | head -n 1)
case "$production_impl" in
  shared_partial) production_needle=adangel_o1_shared_partial_baseline ;;
  register_64x32) production_needle=adangel_o1_register_partial_64x32 ;;
  register_128x128) production_needle=adangel_o1_register_partial_128x128 ;;
  *) echo "unknown O1 production implementation: $production_impl" >&2; exit 1 ;;
esac

o1_symbol=$(find_ptx_symbol o1 "$production_needle") || {
  echo 'O1 production kernel must contain both TMA load and INT8 MMA in the same PTX entry' >&2
  exit 1
}
o1_register_64_symbol=$(find_ptx_symbol o1 adangel_o1_register_partial_64x32) || {
  echo 'O1 register_64x32 candidate must contain same-entry TMA and INT8 MMA' >&2
  exit 1
}
o1_register_128_symbol=$(find_ptx_symbol o1 adangel_o1_register_partial_128x128) || {
  echo 'O1 register_128x128 candidate must contain same-entry TMA and INT8 MMA' >&2
  exit 1
}
o2_symbol=$(find_ptx_symbol o2) || {
  echo 'O2 production kernel must contain both TMA load and MXFP4 block-scaled MMA in the same PTX entry' >&2
  exit 1
}

check_sass_symbol "$o1_symbol" o1 || {
  echo "O1 production SASS entry lacks TMA or IMMA: $o1_symbol" >&2
  exit 1
}
check_sass_symbol "$o1_register_64_symbol" o1 || {
  echo "O1 register_64x32 SASS lacks TMA or IMMA: $o1_register_64_symbol" >&2
  exit 1
}
check_sass_symbol "$o1_register_128_symbol" o1 || {
  echo "O1 register_128x128 SASS lacks TMA or IMMA: $o1_register_128_symbol" >&2
  exit 1
}
check_no_local_sass "$o1_register_64_symbol" || {
  echo "O1 register_64x32 contains local-memory load/store (possible spill)" >&2
  exit 1
}
check_no_local_sass "$o1_register_128_symbol" || {
  echo "O1 register_128x128 contains local-memory load/store (possible spill)" >&2
  exit 1
}
check_sass_symbol "$o2_symbol" o2 || {
  echo "O2 production SASS entry lacks TMA or OMMA: $o2_symbol" >&2
  exit 1
}

audit_complete=1
trap - EXIT
{
  echo "binary=$binary"
  echo "status=PASS"
  echo "o1_production_implementation=$production_impl"
  echo "o1_production_symbol=$o1_symbol"
  echo "o1_register_64_symbol=$o1_register_64_symbol"
  echo "o1_register_128_symbol=$o1_register_128_symbol"
  echo "o1_register_spill_check=PASS(no LDL/STL in candidate SASS)"
  echo "o2_production_symbol=$o2_symbol"
  echo "verified=O0 Tensor Core; O1 same-entry TMA+INT8 MMA; O2 same-entry TMA+MXFP4 block-scaled MMA"
} > "$summary_file"
echo "instruction audit passed; review $output_dir/summary.txt and extension.sass"
