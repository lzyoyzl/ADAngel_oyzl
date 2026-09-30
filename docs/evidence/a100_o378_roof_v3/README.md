# A100 resource-roof candidates: balanced real-trace comparison

Source/binary: `eac859634f798459185c2a5593cb343cad623f6d`, SHA-256 `6c6a1d8c590c9b1edb2940a37b67d5114bcb38dd15d308311cd1baa954d9d902`.
Production defaults unchanged. This archive does not prove completion of the resource-roof goal.

## Evidence

- `reports/o378_roof_v3/audit/`: 27 instantiated candidate functions; native U4×S4 and S4×S4 IMMA plus cp.async pass. Local-memory/stack warnings remain for some candidates and are not waived ISA failures.
- `runs/o378_roof_v3_screen/`: 240 bitwise correctness checks, then balanced cyclic synthetic screen of -1/6/8/9/10, ten rounds. Candidate8 (512 threads), 9 (smaller N) and 10 (K128) are slower; do not adopt them. Candidate10 is spill-free, demonstrating that zero spill alone is insufficient.
- `runs/o378_roof_v3_trace24_balanced/`: 24 original-FP16 trace samples, 3 rounds, 50 warmup/200 repeats; 432 compute-only records comparing actual old production (-1) against candidate6. Per variant, 36 old-first and 36 candidate-first pairs. All outputs bitwise equal; FP64-reduced MSE unchanged.
- `reports/o378_roof_v3/racecheck.log`: candidate6, K768 and O3/O7/O8, zero hazards/errors/warnings. This is a bounded pipeline diagnostic, not exhaustive safety proof or performance evidence.

| Variant | Old median ms | Candidate6 median ms | Paired speedup median | Bootstrap 95% CI | CV≥3% records old/new (out of 72 each) |
|---|---:|---:|---:|---:|---:|
| O3 | 0.558592 | 0.532480 | 1.0497× | [1.0474, 1.0532] | 49 / 52 |
| O7 | 0.600064 | 0.557056 | 1.0745× | [1.0714, 1.0784] | 34 / 47 |
| O8 | 0.600064 | 0.559104 | 1.0726× | [1.0709, 1.0756] | 35 / 46 |

No outliers/failed-CV records removed. Confidence intervals are descriptive sample-level bootstrap, not proof of independent samples: projections share a trace. No clock locking; sparse GPU snapshots do not establish a definitive cause for each outlier. This is stronger order-balanced evidence than v1/v2, but **not an all-stages CV<3% acceptance** and not four-mode acceptance. Native original-FP16 provenance and all raw timings retained.

Median MSE vs O0: O3 `0.006653010285119311`, O7 `0.01190031796545981`, O8 `0.009440354830464444`. Paired FP16 references: O7 vs O5 `0.0055361724266658075`; O8 vs O6 `0.004411084948644645`. Candidate6 vs unchanged production MSE is zero.
