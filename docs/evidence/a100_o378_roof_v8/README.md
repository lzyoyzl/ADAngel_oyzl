# Full-warp scale-panel copy: validated, not selected

Built source `c971aff`, binary SHA-256 `9b1f03336170f5a7c76b5a464a7b6f9d8130142ee7ec0648ab47be8362e1263e`.
Candidate15 flattens two physical G128 scale panels across full copy warps. It does not merge their mathematical scales, change the two native INT4 routes, or alter FP32 group/FMA order. Production defaults remain unchanged.

192 synthetic bitwise checks (four shapes, six patterns, two variants, four implementations),39 same-function ISA audit instances,48 memcheck cases and48 synccheck cases pass. Bounded K768 racecheck reports0 hazards/errors/warnings. One original real sample ×2 variants ×4 implementations ×4 modes =32 smoke records; this is not full24-sample acceptance. Every checked output is bitwise equal to its old implementation; candidate-vs-old MSE is0.

## Prepared-compute screen

Synthetic4096³, eight cyclic rounds per implementation,50 warmups,200 samples/round. Numbers are medians of round medians, not final real-trace results. No outliers are removed.

| Variant | Old production ms | Candidate6 ms | Candidate14 ms | Candidate15 ms |
|---|---:|---:|---:|---:|
| O7 | 0.599552 | 0.558592 | 0.555520 | 0.564736 |
| O8 | 0.599808 | 0.559616 | 0.558080 | 0.563200 |

CV>=3% rounds out of8, old/6/14/15: O7 `4/4/2/3`; O8 `3/3/5/4`. Candidate15 has no observed benefit over14 in this screen, so it is not promoted and a full24-trace performance campaign is not justified yet.

## Same-binary NCU diagnostic

O7 synthetic4096³,9sections, kernel replay, clock/cache control none. These are profiler durations, not ordinary CUDA Event acceptance times. Instrumented Event timings in the NCU/sanitizer runs must never enter performance summaries.

| Metric | Candidate14 | Candidate15 |
|---|---:|---:|
| NCU duration ms | 0.488032 | 0.492960 |
| Dynamic warp instructions | 128,270,336 | 124,698,624 |
| Eligible warps/scheduler | 0.915042 | 0.872458 |
| Issue active % | 47.639453 | 46.011284 |
| Long-scoreboard cycles/issue | 0.494634 | 0.625587 |
| Extra shared wavefronts | 858,219 | 401,408 |
| Extra theoretical global sectors | 484,577 | 229,376 |
| Theoretical local sectors | 10,485,760 | 12,582,912 |
| L1TEX capacity work /1410MHz, ms | 0.277180 | 0.277440 |

Fewer instructions and fewer extra copy transactions did not reduce total supply demand or improve ready-warp availability. Increased local traffic and dependency waiting counteract the intended improvement; these counters identify a tradeoff, not an additive attribution of all elapsed time. IMMA/I2F/FMUL/FFMA remain16,777,216 warp instructions each.

Full `.ncu-rep`, PTX and SASS dumps stay in A100 `reports/o378_roof_v8/`; compact exports and raw checks/timings are archived here. Transfer archive SHA-256: `0ed8ab969e5884510be0e6300f320f1475d4e7241f3413e59e716ba9904d36e7`.

The next isolated test uses N128/K128, three cp.async stages and a three-CTA launch-bound budget, comparing streamed N32/N64 fragments. It targets latency hiding without changing G128 semantics; GPU acceptance is pending.
