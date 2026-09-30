# O3 group-major UE8M0 scale candidate

Source `4690624`, binary SHA-256 `17500c40d0466c29e9bb686c4c376fcd40f2e93cca61a0f531ef6815a3d181b1`. Candidate13 retains candidate6's interleaved two-route INT4 core and changes only the physical W scale layout from natural `[N,G]` to `[G,N]`. The original U8 values and scale decoding are unchanged. Reordering is included in W conversion for cold/conversion-only; steady-state and compute-only cache the result.

Production remains unchanged. Records are unfiltered; NCU/instrumented timings are diagnostics, not CUDA Event results.

## Acceptance evidence

- 72 synthetic bitwise checks: 4 shapes ×6 patterns ×3 implementations, including K768 and nondefault CUDA stream. Reordered U8 scales also match the explicit transpose exactly.
- Same-function native ISA audit passes for 37 instantiations: cp.async/LDGSTS and both U4×S4/S4×S4 INT4 routes; no INT8 replacement. The fast O3 candidate uses 128 registers/thread, 24-byte stack and LDL/STL, so this is not a zero-spill result.
- Memcheck: 24 candidate13 checks plus K768 screen, 0 errors. K768 racecheck: 0 hazards/errors/warnings. These are bounded safety checks, not exhaustive coverage.
- Real trace: 24 samples ×3 balanced cyclic rounds ×3 implementations =216 prepared-compute records, all bitwise equal production. Median MSE vs O0 remains `0.006653010285119311`; candidate-vs-production MSE is zero.
- Four-mode **one-sample** smoke passed, 12 records. This is not full 24-sample end-to-end acceptance.

| O3 | Compute median ms | Paired speedup vs production | Sample-bootstrap 95% CI | CV>=3% /72 |
|---|---:|---:|---|---:|
| Production | 0.558080 | 1.00000 | [1,1] | 33 |
| Candidate6 | 0.531968 | 1.04672 | [1.04419,1.04990] | 31 |
| Candidate13 | 0.527616 | 1.05696 | [1.04981,1.05857] | 36 |

Candidate13 vs6 paired speedup is `1.00724`, CI `[1.00192,1.01167]`: a small gain, not a major advance toward the roof. Intervals are descriptive because samples share a trace; no CV-failed rows are dropped.

In the one-sample four-mode smoke, W reordering increases total conversion from0.100116ms (t6) to0.104253ms (t13). Cold is0.620544ms for both, steady0.551936/0.547840ms; compute0.502784/0.505344ms. This variability and conversion cost prevent claiming a universal end-to-end gain.

## Same-binary NCU diagnosis

Synthetic4096³, 9sections, kernel replay, clock/cache control none; two separate profiles, not paired formal timing. Binary identity is retained in each run environment.

| Metric | Candidate6 | Candidate13 |
|---|---:|---:|
| NCU duration ms | 0.466912 | 0.462752 |
| Dynamic warp instructions | 117,268,480 | 118,800,384 |
| IMMA / I2F / FFMA, each | 16,777,216 | 16,777,216 |
| Extra theoretical global sectors | 8,126,464 | 0 |
| Source shared wavefronts | 42,205,184 | 42,205,184 |
| Extra shared wavefronts | 0 | 0 |
| Long-scoreboard cycles/issue | 0.826152 | 0.495038 |
| Eligible warps/scheduler | 0.785622 | 0.858588 |
| Issue active % | 45.562929 | 46.506220 |
| L1TEX capacity work /1410MHz, ms | 0.274858 | 0.265723 |

The layout correction eliminates excessive scale sectors and reduces global-dependency waiting, but **does not reduce shared fragment work or MMA/I2F/FMA counts**. Total instruction count rises slightly. This explains why fixing this genuine inefficiency yields only a modest latency gain. Sector counts are not actual HBM bytes; stalls are not additive wall-time fractions. The last row is one resource's conditional service requirement, not a predicted attainable full-kernel time.

Recompute the diagnostic from preserved raw/source CSVs:

```bash
python scripts/analyze_roof_scale_ncu.py --variant o3 --tunes 6 13 \
  --directory docs/evidence/a100_o378_roof_v6/reports/o378_roof_v6
```

Full `.ncu-rep` remains in A100 `reports/o378_roof_v6/`; text evidence is archived here. No production-default change.
