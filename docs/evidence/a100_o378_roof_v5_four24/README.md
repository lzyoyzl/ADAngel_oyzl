# Balanced four-mode confirmation of the interleaved MMA candidate

Source `30ddc4a`; binary SHA-256 `a2599ff739f94e527be5d722d075a6bb6b0e52c1fb10c3f6e956bb80ba89deef`. This is the completed run excluded from the earlier v5 partial archive. Production defaults remain unchanged.

24 original-FP16 trace samples, O3/O7/O8, original production symbol vs candidate6, all four modes: **576 records**. Each case uses 50 warmups, 200 event samples, conversion inner repeats 100. Each variant/mode has 12 AB and 12 BA sample pairs. All outputs are finite FP32 and bitwise equal production; MSE relative to O0 and paired O5/O6 references is unchanged. No records or outliers removed.

| Variant | Mode | Original median ms | Candidate6 median ms | Paired speedup median | Sample-bootstrap 95% CI |
|---|---|---:|---:|---:|---|
| O3 | conversion-only | 0.100908 | 0.100874 | 1.00112 | [0.99828, 1.00310] |
| O3 | compute-only | 0.581888 | 0.553984 | 1.04320 | [1.03380, 1.06679] |
| O3 | cold | 0.684544 | 0.658432 | 1.04072 | [1.02439, 1.04851] |
| O3 | steady-state | 0.637696 | 0.593920 | 1.04508 | [1.03066, 1.07254] |
| O7 | conversion-only | 0.143813 | 0.144015 | 0.99773 | [0.99397, 1.00360] |
| O7 | compute-only | 0.613888 | 0.580096 | 1.06954 | [1.04363, 1.10054] |
| O7 | cold | 0.771072 | 0.721920 | 1.07012 | [1.04816, 1.07801] |
| O7 | steady-state | 0.692224 | 0.654336 | 1.06367 | [1.05156, 1.06666] |
| O8 | conversion-only | 0.116756 | 0.116767 | 0.99923 | [0.99807, 0.99991] |
| O8 | compute-only | 0.611840 | 0.574976 | 1.06353 | [1.04956, 1.07372] |
| O8 | cold | 0.742400 | 0.701440 | 1.06021 | [1.03937, 1.07718] |
| O8 | steady-state | 0.690176 | 0.657408 | 1.06245 | [1.03756, 1.07705] |

Paired speedups are computed per sample before aggregation, not ratios of aggregate medians. Bootstrap intervals are descriptive: the 24 projections share the same input trace. One balanced round confirms direction, not independence from shared-GPU conditions. Conversion code is unchanged, so sub-percent conversion differences are not optimization claims.

| Any-stage CV >=3%, original / candidate6 (out of24 each) | Conversion | Compute | Cold | Steady |
|---|---:|---:|---:|---:|
| O3 | 0 / 0 | 10 / 13 | 21 / 21 | 20 / 22 |
| O7 | 0 / 0 | 10 / 19 | 1 / 8 | 1 / 5 |
| O8 | 0 / 0 | 8 / 13 | 0 / 13 | 2 / 14 |

No claim of passing the strict CV gate. GPU snapshots cannot establish the exact cause of each outlier. End-to-end remains single-execution direct timing; conversion stages are batch-amortized, and their medians are not additive with compute-only medians.

Median MSE vs O0: O3 `0.006653010285119311`, O7 `0.01190031796545981`, O8 `0.009440354830464444`. Paired FP16 MSE: O3 vs O0 same as above; O7 vs O5 `0.0055361724266658075`; O8 vs O6 `0.004411084948644645`. Candidate vs production MSE is zero in every record.

Raw records, environment/binary hashes, source-format provenance and GPU snapshots are retained under `runs/o378_roof_v5_four24_balanced/`. This supports a modest improvement, not completion of the resource-roof optimization objective.
