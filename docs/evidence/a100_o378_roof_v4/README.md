# Guarded exact power-of-two activation scale: negative result

CUDA source `cdacddb`, binary SHA-256 `11fe95ea64d7705844c30ad19ba53626dda2c840723fdaecf67cc1027c951bb8`. Audit script advanced to `ce3ab62` without changing CUDA. O7 real-trace comparison uses 24 original FP16 samples ×3 cyclically balanced rounds, 50 warmup/200 measurements. No production default changed.

Candidate11 replaces `round_fp32(A_scale*W_scale)` by exact exponent adjustment only when A is a positive normal power of two, W positive normal, and all products normal. Other cases use candidate6's ordinary multiply. I2F and ordered G128 FMA are unchanged; this is not magic-bias.

| O7 compute-only | Median ms | Paired speedup vs original | CV≥3% /72 |
|---|---:|---:|---:|
| Actual original production | 0.594944 | 1.0000× | 31 |
| Candidate6: interleaved wider N fragment | 0.552960 | 1.0741× | 36 |
| Candidate11: direct exponent adjustment | 0.565248 | 1.0523× | 35 |

Candidate11 vs candidate6 paired speedup: **0.98016×**, descriptive sample-bootstrap 95% CI `[0.97642, 0.98529]`. It is slower and is not adopted. All 216 trace records bitwise equal production; O7 median MSE vs O0 `0.01190031796545981`, vs O5 `0.0055361724266658075`, unchanged. All raw timings/CV failures retained.

Correctness also includes 216 synthetic checks (four shapes, six patterns, three variants, three implementations), plus a 36-record one-sample four-mode smoke test. The latter is not full 24-sample four-mode acceptance. Audit confirms native U4/S4 IMMA and cp.async; special candidate11 removes FMUL but preserves I2F/FFMA. Some stack/local instructions remain and are reported, not mislabelled zero spill.

## NCU explains why fewer FMULs did not help

Same binary and O7 power-of-two activation-scale synthetic input; nine sections, 18 replay passes, no clock/cache control. No simultaneous performance test. A host CUDA build was active; NCU duration is diagnostic, not substituted for Event results.

| Metric | Candidate6 | Candidate11 |
|---|---:|---:|
| NCU duration, ms | 0.489888 | 0.500544 |
| Dynamic warp instructions | 127,238,144 | 135,364,608 |
| Dynamic FMUL | 16,777,216 | 0 |
| Dynamic IMMA / I2F / FFMA (each) | 16,777,216 | 16,777,216 |
| Eligible warps per scheduler | 0.886 | 0.978 |
| Issue active | 47.18% | 48.87% |
| L1TEX work/capacity at 1.41 GHz, ms | 0.28810 | 0.28083 |

Integer bias adjustments and compiler scheduling increased total instructions by 6.39%. Fewer local loads and better readiness did not offset that work. L1TEX service demand decreased, yet runtime increased: a single utilization metric or resource floor is not an attainable-time prediction. Preserve G128 semantics; the next candidate moves reusable exponent-bias subtraction to the per-row/per-group shared prefetch stage rather than repeating it per output.

Recompute counts (CPU only):

```bash
python scripts/analyze_roof_scale_ncu.py \
  --directory docs/evidence/a100_o378_roof_v4/reports/o378_roof_v4
```

`runs/*ncu*` timings are instrumented and excluded from performance summaries. Full `.ncu-rep` files remain under the A100 project's `reports/o378_roof_v4/`; this archive stores CSV/log exports and binary identities. No safety claim for candidate11 beyond correctness and ISA is inferred from earlier candidates' sanitizer runs.
