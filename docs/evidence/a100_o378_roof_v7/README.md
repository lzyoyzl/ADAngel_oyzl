# O7/O8 asynchronous FP32 scale copy

Built source `bc5b2c6`, binary SHA-256 `cdd745ee7d08be833076e76c378e1848db759344dcf425b8fda9cd11f7ab6c14`. Later archive/parser-only commit `029d0e6` was merged before the24-sample run; CUDA code/binary remained unchanged. Candidate14 copies each G128 scale panel with16-byte cp.async, sharing the existing payload commit/wait/barrier. FP32 FMUL, ordinary I2F and ascending G128 FFMA are unchanged. No production-default changes.

144 synthetic bitwise checks,38 same-function ISA audit instances,48 memcheck and48 synccheck cases pass; K768 racecheck for O7/O8 reports0 hazards/errors/warnings. Candidate14 has128 registers/thread,40-byte stack, LDL/STL; spill is allowed, not absent. Four-mode smoke covers one real sample ×2 variants ×3 implementations ×4 modes =24 records. It is not full24-sample end-to-end acceptance.

24 original-FP16 samples ×3 balanced cyclic rounds ×2 variants ×3 implementations =432 prepared-compute records. All are finite FP32 and bitwise equal production. No filtering. Shared-GPU CV failures remain; bootstrap intervals are descriptive, with correlated same-trace samples.

| Variant / implementation | Median ms | Paired speedup vs production | 95% CI | CV>=3% /72 |
|---|---:|---:|---|---:|
| O7 production | 0.596992 | 1 | [1,1] | 22 |
| O7 candidate6 | 0.556544 | 1.07176 | [1.06938,1.07366] | 29 |
| O7 candidate14 | 0.553984 | 1.07922 | [1.07743,1.08318] | 33 |
| O8 production | 0.600064 | 1 | [1,1] | 25 |
| O8 candidate6 | 0.560128 | 1.07179 | [1.07091,1.07551] | 25 |
| O8 candidate14 | 0.557056 | 1.07707 | [1.07353,1.08318] | 27 |

Median MSE unchanged: O7 vs O0 `0.01190031796545981`, vs O5 `0.0055361724266658075`; O8 vs O0 `0.009440354830464444`, vs O6 `0.004411084948644645`. Candidate-vs-production MSE is0 in every record.

## NCU: a small supply improvement, not a roof-level speedup

Same-binary O7 synthetic4096³ profiles,9sections, kernel replay, clock/cache control none. Do not use profiler-instrumented Event times in `runs/*ncu*` as performance results.

| Metric | Candidate6 | Candidate14 |
|---|---:|---:|
| NCU duration ms | 0.489536 | 0.487456 |
| Dynamic warp instructions | 127,238,144 | 128,270,336 |
| IMMA/I2F/FMUL/FFMA, each | 16,777,216 | 16,777,216 |
| Long-scoreboard cycles/issue | 0.702981 | 0.498332 |
| Eligible warps/scheduler | 0.890600 | 0.915828 |
| Issue active % | 47.229259 | 47.632781 |
| L1TEX capacity work /1410MHz, ms | 0.288097 | 0.277193 |
| Source shared wavefronts | 43,384,832 | 44,855,295 |
| Extra shared wavefronts | 0 | 860,159 |
| Extra theoretical global sectors | 0 | 491,520 |

The extra source wavefront/sector counts are localized to two predicated16-byte A-scale LDGSTS sites (half a warp participates). These are diagnostic request-efficiency counters, not directly additive HBM traffic or saved milliseconds. Candidate15 will compare flattened full-warp scale-panel copies. Group math remains independent; no regrouping of FP32 accumulation is permitted.

Full `.ncu-rep` files remain on A100 under `reports/o378_roof_v7/`. Raw/source exports and derived `ncu_comparison.json` are archived here. Remaining gap to the model roof is substantial.
