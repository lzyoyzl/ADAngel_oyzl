# Prefetch-time activation exponent bias: no additional gain established

Built source `30ddc4a`, binary SHA-256 `a2599ff739f94e527be5d722d075a6bb6b0e52c1fb10c3f6e956bb80ba89deef`. Production defaults unchanged. Candidate12 prepares the guarded power-of-two A exponent offset once per row/group/CTA in shared prefetch; candidate11 did this subtraction in per-output work. It preserves exact scale products, ordinary I2F, two native INT4 routes and ascending G128 FP32 FMA. Non-power2/zero/subnormal/extreme cases use the normal multiply fallback.

- Synthetic screen: 288 bitwise checks (4 shapes ×6 patterns ×3 variants ×4 implementations), including A=0.5/1/2 for bit-only prebiased payloads, then 8 cyclically balanced performance rounds. No outliers removed.
- Native ISA audit: 35 candidate instantiations pass U4/S4 IMMA and cp.async checks. Candidate12 specialized function has no FMUL, retains I2F/FFMA, 128 registers/thread and 40-byte stack; not spill-free.
- Memcheck and synccheck: 72 checks each, 0 errors, including K256/512/768/4096 and a nondefault CUDA stream. These are bounded diagnostics, not exhaustive racecheck coverage. Instrumented timings must not be used as performance results.
- O7 original-FP16 real trace: 24 samples ×3 cyclic rounds ×3 implementations =216 records, 50 warmup/200 repeats, all outputs bitwise equal production.

| O7 implementation | Compute median ms | Paired speedup vs original | CV≥3% /72 |
|---|---:|---:|---:|
| Original production | 0.592896 | 1.0000× | 28 |
| Candidate6 | 0.553984 | 1.07182× | 32 |
| Candidate12 | 0.557056 | 1.06822× | 31 |

Candidate12 vs candidate6 paired speedup median is **0.99724×**, sample-bootstrap 95% CI `[0.99269, 1.00000]`; no reliable gain established, not adopted. These intervals are descriptive because samples share a trace. CV failures are retained and are not claimed as passing a strict 3% gate.

Median MSE unchanged: vs O0 `0.01190031796545981`, vs O5 `0.0055361724266658075`; candidate vs production MSE is zero. O3/O8 synthetic results do not imply that the O7-specific power2 shortcut applies to their general scale distributions.

The candidate6 24-sample four-mode balanced run was still live when this partial archive was assembled. It has since completed with all576 records bitwise correct and 12AB/12BA sample pairs per variant/mode; its separate [complete archive](../a100_o378_roof_v5_four24/README.md) records the timing gains and remaining CV failures. Next candidate13 addresses O3 scale layout rather than continuing ineffective scalar replacements.
