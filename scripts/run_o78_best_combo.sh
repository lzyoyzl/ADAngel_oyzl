#!/usr/bin/env bash
# Reuse audited binaries; all generated files stay inside the repository.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
TASK_ROOT=$(pwd -P)
export PATH="/home/zlouyang/miniconda3/envs/adangel-a100/bin:/usr/local/cuda-12.8/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$TASK_ROOT/tmp"
RUN_ID=${1:-o378_v142}
[[ "$RUN_ID" =~ ^o378_v142[a-zA-Z0-9_]*$ ]] || exit 2
[[ ! -e "reports/${RUN_ID}_extension_before.txt" ]] || exit 2
sha256sum python/adangel/_sm80*.so > "reports/${RUN_ID}_extension_before.txt"
python -m pytest tests/unit/test_o78_best_combo.py tests/unit/test_o7_conversion_combo.py -q
for variant in o7 o8; do
    python scripts/benchmark_o78_best_combo.py --variant "$variant" --validate-only \
        --output "runs/${RUN_ID}_${variant}_preflight"
done
for variant in o7 o8; do
    python scripts/benchmark_o78_best_combo.py --variant "$variant" \
        --output "runs/${RUN_ID}_${variant}_full24"
    python scripts/analyze_o78_best_combo.py --input "runs/${RUN_ID}_${variant}_full24" \
        --output "reports/${RUN_ID}_${variant}_analysis.json"
done
# Small M/N, K4096 candidate-entry checks, not a full-model sanitizer claim.
for variant in o7 o8; do
    compute-sanitizer --tool memcheck --error-exitcode 9 \
        --kernel-name kne=adangel_roof_o78_output_streaming_candidate \
        python scripts/benchmark_o78_best_combo.py --variant "$variant" --validate-only \
        --output "runs/${RUN_ID}_${variant}_memcheck" \
        > "reports/${RUN_ID}_${variant}_memcheck.log" 2>&1
done
sha256sum python/adangel/_sm80*.so > "reports/${RUN_ID}_extension_after.txt"
cmp "reports/${RUN_ID}_extension_before.txt" "reports/${RUN_ID}_extension_after.txt"
printf 'BEST COMBINATION FULL24 FINISHED\n'
