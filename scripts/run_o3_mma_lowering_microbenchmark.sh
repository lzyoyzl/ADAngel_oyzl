#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CUDA_ROOT=${CUDA_HOME:-/usr/local/cuda-12.8}
CUTLASS_ROOT=${ADANGEL_CUTLASS_ROOT:-"$ROOT/third_party/cutlass-src"}
OUTPUT_DIR=${1:-"$ROOT/reports/o3_mma_lowering"}
NVCC="$CUDA_ROOT/bin/nvcc"
SOURCE="$ROOT/benchmarks/o3_mma_lowering.cu"
BINARY="$OUTPUT_DIR/o3_mma_lowering"

if [[ ! -x "$NVCC" ]]; then
  echo "nvcc not found at $NVCC" >&2
  exit 1
fi
if [[ ! -f "$CUTLASS_ROOT/include/cute/arch/mma_sm80.hpp" ]]; then
  echo "CUTLASS headers not found below $CUTLASS_ROOT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

"$NVCC" \
  -O3 \
  -std=c++17 \
  -lineinfo \
  --expt-relaxed-constexpr \
  -gencode=arch=compute_120a,code='[sm_120a,compute_120a]' \
  -I"$CUTLASS_ROOT/include" \
  "$SOURCE" \
  -o "$BINARY"

"$CUDA_ROOT/bin/cuobjdump" --dump-ptx "$BINARY" > "$OUTPUT_DIR/microbench.ptx"
"$CUDA_ROOT/bin/cuobjdump" --dump-sass "$BINARY" > "$OUTPUT_DIR/microbench.sass"
"$CUDA_ROOT/bin/cuobjdump" --dump-resource-usage "$BINARY" \
  > "$OUTPUT_DIR/microbench.resources.txt" 2>&1

: > "$OUTPUT_DIR/results.jsonl"
for shape in m16n8k64 m16n8k32 m8n8k32; do
  for kind in u4s4 s4s4 s8s8; do
    for chains in 1 4; do
    "$BINARY" \
      --kind "$kind" \
      --shape "$shape" \
      --chains "$chains" \
      --blocks 512 \
      --warps 8 \
      --iterations 512 \
      --warmup 10 \
      --repeats 50 \
      | tee "$OUTPUT_DIR/${shape}_${kind}_c${chains}.json"
    python - "$OUTPUT_DIR/${shape}_${kind}_c${chains}.json" "$OUTPUT_DIR/results.jsonl" <<'PY'
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
record = json.loads(source.read_text())
with destination.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, sort_keys=True) + "\n")
PY
    done
  done
done

python - "$OUTPUT_DIR/results.jsonl" "$OUTPUT_DIR/summary.json" <<'PY'
import json
import pathlib
import sys

records = [json.loads(line) for line in pathlib.Path(sys.argv[1]).read_text().splitlines()]
by_key = {
    (record["shape"], record["kind"], record["chains"]): record
    for record in records
}
summary = {
    "records": records,
    "throughput_chain4": {
        shape: {
            kind: by_key[(shape, kind, 4)]["logical_tops"]
            for kind in ("u4s4", "s4s4", "s8s8")
        }
        for shape in ("m16n8k64", "m16n8k32", "m8n8k32")
    },
    "latency_chain1_median_ms": {
        shape: {
            kind: by_key[(shape, kind, 1)]["median_ms"]
            for kind in ("u4s4", "s4s4", "s8s8")
        }
        for shape in ("m16n8k64", "m16n8k32", "m8n8k32")
    },
    "logical_throughput_vs_s8_chain4": {
        shape: {
            kind: by_key[(shape, kind, 4)]["logical_tops"] /
            by_key[(shape, "s8s8", 4)]["logical_tops"]
            for kind in ("u4s4", "s4s4", "s8s8")
        }
        for shape in ("m16n8k64", "m16n8k32", "m8n8k32")
    },
}
pathlib.Path(sys.argv[2]).write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

python - "$OUTPUT_DIR/microbench.ptx" "$OUTPUT_DIR/microbench.sass" \
  "$OUTPUT_DIR/audit_summary.txt" <<'PY'
import pathlib
import re
import sys

ptx = pathlib.Path(sys.argv[1]).read_text(errors="replace")
sass = pathlib.Path(sys.argv[2]).read_text(errors="replace")
checks = {
    "ptx_u4s4": bool(re.search(r"mma\.sync[^\n]*\.u4\.s4\.s32", ptx)),
    "ptx_s4s4": bool(re.search(r"mma\.sync[^\n]*\.s4\.s4\.s32", ptx)),
    "ptx_s8s8": bool(re.search(r"mma\.sync[^\n]*\.s8\.s8\.s32", ptx)),
    "sass_u8s8_imma": bool(re.search(r"IMMA[^\n]*\.U8\.S8", sass)),
    "sass_s8s8_imma": bool(re.search(r"IMMA[^\n]*\.S8\.S8", sass)),
    "sass_bit_ops": bool(re.search(r"\b(?:LOP3|SHF|IMAD)\b", sass)),
    "sass_explicit_u4_or_s4": bool(re.search(r"IMMA[^\n]*\.(?:U4|S4)", sass)),
}
lines = [f"{name}={str(value).lower()}" for name, value in checks.items()]
pathlib.Path(sys.argv[3]).write_text("\n".join(lines) + "\n")
print("\n".join(lines))
if not all(checks[name] for name in ("ptx_u4s4", "ptx_s4s4", "ptx_s8s8")):
    raise SystemExit("required PTX MMA semantics were not found")
if not checks["sass_s8s8_imma"]:
    raise SystemExit("expected S8 IMMA was not found in SASS")
PY

python "$ROOT/scripts/analyze_o3_mma_lowering.py" "$OUTPUT_DIR" \
  > "$OUTPUT_DIR/static_instruction_attribution.stdout.json"

echo "O3 MMA lowering microbenchmark complete: $OUTPUT_DIR"
