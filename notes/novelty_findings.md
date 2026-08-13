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
