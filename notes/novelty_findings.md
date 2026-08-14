# Novelty push — findings (error-budget Pareto + attribution + cost model)

Node: ft-node-2 (2-core Xeon @2.20GHz, AVX2, **no AVX512-VNNI** — runtime-probed).
Date: 2026-08-12/13. All measurements via evals/eval_v8.py (18-cell grid, f64 ref).

## Headline result: the latency-vs-error frontier is FLAT fp32

For budgets {1e-3, 1e-2} (max_abs_err vs f64), **fp32 wins every one of the 18
cells**. int8 (per-channel weights / per-row activations, u8-s8 with column-sum
correction, AVX2 vpmaddubsw + AVX512-VNNI kernels, weight-quant cache) and bf16
(round-to-nearest-even truncation, fp32 accumulate) are **dominated at every
cell**: slower AND less accurate.

| variant | max_abs_err tiny/default | vs champion (canonical cells) |
|---|---|---|
| c_fp32_e6 (champion) | 1.1e-6 / 1.2e-6 | 1.00x (35.1 / 262.6 us) |
| c_fp32_e4 (exp deg 4) | 9.2e-6 / 6.1e-6 | ~1.00x |
| c_fp32_e3 (exp deg 3) | 2.5e-4 / 1.7e-4 | 1.00-1.05x (wins 3 default cells) |
| c_int8_e6 | 1.5e-1 / 1.6e-1 | 0.5x (67.5 us tiny) — LOSES |
| c_int8_attn_e6 | 5.0e-2 / 6.0e-2 | ~0.8x — LOSES |
| c_bf16_e6 | 2.6e-2 / 2.5e-2 | 0.7x — LOSES |

- The only real dial: **fast-exp Chebyshev degree**. deg-4 is free (same speed,
  err 1e-5); deg-3 passes 1e-3 with err ~2e-4 and wins default_b1_s16/s32/s64,
  default_b16_s32 by 1-5% (attention-heavy cells where exp cost matters).
- 13/18 cells sub-ms at 1e-3 (up from 12/18 on the earlier v6 baseline; the
  default_b4_s32 cell crossed under 1ms on this node).

## Why quantization loses at this scale (the interesting part)

1. **Fixed overhead dominates small cells.** Cost model: latency_us = 23.2 +
   3.66e-5*FLOPs + 1.70e-3*B*S^2 (median rel err 3.9% over 54 cells). The 23 us
   intercept (ctypes, buffer prep, threading) is as large as the entire GEMM
   budget at B=1. int8 adds per-call activation quantization on top of it.
2. **No VNNI on this node class.** The 2-4x int8 win requires AVX512-VNNI
   (vpdpbusd); on AVX2, vpmaddubsw is only ~1-1.5x fp32 FMA and the
   quantization overhead eats that.
3. **vpmaddubsw saturates int16 at 32767**, capping u8xs8 activations at
   [0,127] (63 levels) — halving activation precision vs the textbook u8[0,255].
   (Full-range u8 pairs overflow the accumulator lanes.)
4. **exp amplification**: quantized attention scores are exponentiated in
   softmax, multiplying score error ~5-10x into the logits.
5. Effective GEMM rate 27.3 GFLOP/s = ~0.45 of node peak — consistent with the
   earlier v5 roofline finding (0.43).

## Attribution (FT_PROFILE, per-stage medians, log-overhead-inflated but proportional)

| stage | tiny (35 us) | default (261 us) |
|---|---|---|
| layernorm x3 | ~0.8 each | ~1.9 each |
| QKV matmul | 14% | 12% |
| attention (blocked, fused) | 25% | 12% |
| wo matmul | 6% | 7% |
| FFN matmuls | 19% | 18% |
| **out projection** | **31%** | **49%** |

**The single biggest lever is the output projection** — a thin-M (B*S rows) ×
wide-N (vocab) GEMM at ~half the machine's achievable rate. A future push would
target out-projection-specific tiling (e.g., pre-transposed out_w, wider N
tiles, better thread split) before touching anything else.

## Bugs found and fixed along the way (methodology notes)
- vpmaddubsw int16 saturation with u8[0,255] activations (fixed: u8[0,127]).
- AVX2 accumulator lane-mixing across columns (fixed: per-column 256-bit kernel).
- int8 weight-cache buffer reuse without size check -> heap corruption
  (fixed: realloc on shape change).
- Wrapper rebuilt weight blocks per (B,S) cell, churning the pointer-keyed
  quant cache (fixed: weight prep cached separately from workspace).

## What this means
For sub-millisecond CPU transformers at these shapes, fp32 remains the
frontier; quantization is a *speed* liability until AVX512-VNNI/AMX hardware.
The publishable bits: (a) the measured flat frontier + the five mechanisms
above; (b) the cost model with its 23 us overhead intercept; (c) the
attribution table naming the out-projection as the real target; (d) the
recursive-agent methodology that produced and fixed the kernel bugs via the
eval ladder.

## Follow-up: out-projection tuned dispatch (FT_OUT_TUNE) — NEGATIVE result
Hypothesis: attribution showed out-projection = 31%/49% of stage time ->
a shape-tuned dispatch (serial 4x16 for thin M, 4x32+OMP for mid-M wide-N,
from isolated hot-cache micro-benchmarks: up to 39% on the isolated GEMM)
should speed up the forward pass.
Result (v8 sweep, ft-node-1, same-node champion comparison): mostly a wash to
-8..-15% at most cells; wins only default_b1_s16 (+23%); loses default_b16_s16
by 2x (untested 4x32 regime). Hot-cache isolation does NOT transfer: in-situ
out_w (128KB) is L2-evicted between calls, so the GEMM is memory-side cold and
the champion's OMP threading hides that latency better than serial tiles.
Takeaway: the out GEMM was already at the machine's effective ~27 GF/s; the
attribution named where time goes, but it was not waste. Knob kept (off by
default) as a hardware probe. Champion unchanged.

## Follow-up 2: interleaved A/B of the serial-out dispatch — rule REJECTED
Controlled protocol (ft-node-1, 5 randomized rounds, in-situ full-model bench,
2000 iters/warmup 100): champion (omp-4x16) vs M<=32-serial vs always-serial:
- default_b1_s16: 283.7 / 317.0 / 319.3 us  -> serial LOSES (11-13%)
- default_b1_s32: 257.6 / 296.0 / 286.9 us  -> serial LOSES (11-15%)
- tiny_b1_s32:    34.6  / 34.4  / 34.1 us   -> tie (serial ~1% faster, noise)
The earlier +23% b1_s16 "win" was run-condition noise (non-interleaved sweep).
Conclusion: no evidence ANY out-dispatch beats the champion on this node class.
FT_OUT_TUNE stays off by default (hardware probe only). Champion unchanged.

## Final hardware verdict (both ft-node-1/2 probed)
Skylake-SP class: AVX2 only, no AVX512F/VNNI/VL, no AVX-VNNI, no AMX.
Effective GEMM 27.3 GF/s = ~78% of the 2-core AVX2 theoretical ceiling (~35 GF/s
= 2 cores x 8 FLOP/cycle x 2.2 GHz). fp32 is at the machine floor; int8/bf16/
tiling dispatch cannot beat it here. Remaining headroom is hardware-bound
(VNNI/AMX/more cores) — the v8 frontier + cost model quantify exactly where.

## Follow-up 3: wrapper overhead + attention variants (all NEGATIVE)
- Wrapper overhead measured (interleaved, 5000 iters): python+ctypes adds only
  0.5us (2%) on tiny, ~0 on default. The cost model's 23us intercept is C-side
  per-GEMM fixed cost (loop setup/tile init/writeback across 8 matmul calls),
  not Python. Nothing to squeeze there.
- attn_fused (flash-style, already causal): loses 12/18 cells (up to +40% at
  S=64) - per-query scalar loops can't match the blocked matmul + SIMD softmax.
- Triangular scores (skip masked triangle, exact math, -47% scores MACs,
  no qc/kt copies): still loses 14/18 cells (up to +28%) - the dense 4x16
  kernel beats scalar dot loops even at 2x the MACs.
- VERDICT: on this 2-core AVX2 Skylake, the v14 blocked kernel structure is
  optimal; arithmetic reductions that sacrifice kernel efficiency lose. The
  27.3 GF/s effective rate (~78% of the AVX2 ceiling) is the floor for this
  kernel class. All variants kept in-repo as hardware probes.
