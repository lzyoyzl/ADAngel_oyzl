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
      has_mma_second = 0
    }
    function finish_entry() {
      if (!opened) return
      if (family == "o1" &&
          index(symbol, needle "IN") != 0 &&
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
      if (family == "o3" &&
          index(symbol, needle) != 0 &&
          has_tma && has_mma && has_mma_second) {
        print symbol
        found = 1
        exit 0
      }
      if (family == "o4" &&
          index(symbol, needle) != 0 &&
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
      if (family == "o3" && /mma.*s32.*u4.*s4.*s32/) has_mma = 1
      if (family == "o3" && /mma.*s32.*s4.*s4.*s32/) has_mma_second = 1
      if (family == "o4" && /mma.*s32.*b1.*b1.*s32.*and\.popc/) has_mma = 1
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
    inside && family == "o3" && /IMMA/ { has_mma = 1 }
    inside && family == "o4" && /BMMA|IMMA/ { has_mma = 1 }
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

resource_usage_for_symbol() {
  local symbol=$1
  awk -v symbol="$symbol" '
    /^[[:space:]]*Function[[:space:]]+/ {
      found = index($0, symbol) != 0
      next
    }
    found && /^[[:space:]]*REG:/ {
      sub(/^[[:space:]]*/, "")
      print
      exit 0
    }
    END { if (!found) exit 1 }
  ' "$output_dir/extension.resources.txt"
}

check_zero_stack_local_resource() {
  local usage=$1
  [[ " $usage " == *" STACK:0 "* && " $usage " == *" LOCAL:0 "* ]]
}

source_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
production_impl=$(sed -n 's/.*kProductionO1Implementation.*"\([^"]*\)".*/\1/p' \
  "$source_root/csrc/sm120/o1_gemm.cu" | head -n 1)
case "$production_impl" in
  shared_partial) production_needle=adangel_o1_shared_partial_baseline ;;
  register_64x32) production_needle=adangel_o1_register_partial_64x32 ;;
  register_128x64_k64_scale_shared_row_dedup)
    production_needle=adangel_o1_register_partial_128x64_k64_scale_shared_row_dedup
    ;;
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
o3_symbol=$(find_ptx_symbol o3 adangel_o3_split_tma_ws) || {
  echo 'O3 production kernel must contain same-entry TMA, U4xS4 MMA, and S4xS4 MMA' >&2
  exit 1
}
o4_symbol=$(find_ptx_symbol o4 adangel_o4_bitwise_tma_ws) || {
  echo 'O4 production kernel must contain same-entry TMA and B1 AND-POPC MMA' >&2
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
o1_production_resource=$(resource_usage_for_symbol "$o1_symbol") || {
  echo "O1 production resource usage not found: $o1_symbol" >&2
  exit 1
}
o1_register_64_resource=$(resource_usage_for_symbol "$o1_register_64_symbol") || {
  echo "O1 register_64x32 resource usage not found: $o1_register_64_symbol" >&2
  exit 1
}
o1_register_128_resource=$(resource_usage_for_symbol "$o1_register_128_symbol") || {
  echo "O1 register_128x128 resource usage not found: $o1_register_128_symbol" >&2
  exit 1
}
check_no_local_sass "$o1_symbol" &&
  check_zero_stack_local_resource "$o1_production_resource" || {
    echo "O1 production kernel spills to local memory: $o1_symbol ($o1_production_resource)" >&2
    exit 1
  }
check_zero_stack_local_resource "$o1_register_64_resource" || {
  echo "O1 register_64x32 has non-zero stack/local usage: $o1_register_64_resource" >&2
  exit 1
}

o1_register_128_spill_check="PASS(no LDL/STL; STACK=0; LOCAL=0)"
if ! check_no_local_sass "$o1_register_128_symbol" ||
   ! check_zero_stack_local_resource "$o1_register_128_resource"; then
  o1_register_128_spill_check="DISQUALIFIED(local-memory spill)"
  if [[ "$production_impl" == "register_128x128" ]]; then
    echo "O1 register_128x128 production candidate spills: $o1_register_128_resource" >&2
    exit 1
  fi
fi
check_sass_symbol "$o2_symbol" o2 || {
  echo "O2 production SASS entry lacks TMA or OMMA: $o2_symbol" >&2
  exit 1
}
check_sass_symbol "$o3_symbol" o3 || {
  echo "O3 production SASS entry lacks TMA or INT4 IMMA: $o3_symbol" >&2
  exit 1
}
check_sass_symbol "$o4_symbol" o4 || {
  echo "O4 production SASS entry lacks TMA or binary MMA: $o4_symbol" >&2
  exit 1
}
o3_resource=$(resource_usage_for_symbol "$o3_symbol") || {
  echo "O3 production resource usage not found: $o3_symbol" >&2
  exit 1
}
o4_resource=$(resource_usage_for_symbol "$o4_symbol") || {
  echo "O4 production resource usage not found: $o4_symbol" >&2
  exit 1
}
check_no_local_sass "$o3_symbol" && check_zero_stack_local_resource "$o3_resource" || {
  echo "O3 production kernel spills to local memory: $o3_symbol ($o3_resource)" >&2
  exit 1
}
check_no_local_sass "$o4_symbol" && check_zero_stack_local_resource "$o4_resource" || {
  echo "O4 production kernel spills to local memory: $o4_symbol ($o4_resource)" >&2
  exit 1
}

audit_complete=1
trap - EXIT
{
  echo "binary=$binary"
  echo "status=PASS"
  echo "o1_production_implementation=$production_impl"
  echo "o1_production_symbol=$o1_symbol"
  echo "o1_production_resource=$o1_production_resource"
  echo "o1_register_64_symbol=$o1_register_64_symbol"
  echo "o1_register_64_resource=$o1_register_64_resource"
  echo "o1_register_64_spill_check=PASS(no LDL/STL; STACK=0; LOCAL=0)"
  echo "o1_register_128_symbol=$o1_register_128_symbol"
  echo "o1_register_128_resource=$o1_register_128_resource"
  echo "o1_register_128_spill_check=$o1_register_128_spill_check"
  echo "o2_production_symbol=$o2_symbol"
  echo "o3_production_symbol=$o3_symbol"
  echo "o3_production_resource=$o3_resource"
  echo "o4_production_symbol=$o4_symbol"
  echo "o4_production_resource=$o4_resource"
  echo "verified=O0 Tensor Core; O1 same-entry TMA+INT8 MMA; O2 same-entry TMA+MXFP4 block-scaled MMA; O3 same-entry TMA+U4/S4 INT4 MMA; O4 same-entry TMA+B1 AND-POPC MMA"
} > "$summary_file"
echo "instruction audit passed; review $output_dir/summary.txt and extension.sass"
