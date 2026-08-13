# Error-pareto: latency-vs-error frontier sweep design + analysis plan

Agent: `error-pareto` (v2). Repo: /Users/ace/projects/sonic/fast-transformer.
Status: **DESIGN LOCKED** (naming coordinated with int8-kernels; see table below).
Do not run `colab` — parent owns ft-node-1. Analysis consumes results/ as parent supplies them.

## Goal
The correctness bar (max_abs_err < 1e-3 vs f64 ref) is a **dial, not a gate**.
Produce the latency-vs-error **Pareto frontier**: for each of the 18 v6 cells and
each error budget, the fastest implementation that passes. This is the
"novel at the intersection" artifact for the writeup.

## Lever space (sweep grid)
| lever | values | knob | source |
|---|---|---|---|
| exp degree | {6, 4, 3} | `-DFT_EXP_DEG=n` (Chebyshev fast-exp degree; 6 = champion default) | int8-kernels |
| GEMM precision | {fp32, int8, int8_attn, bf16} | `-DFT_INT8=1` / `-DFT_INT8_ATTN=1` / `-DFT_BF16=1` | int8-kernels |
| cells | cfg{tiny,default} x B{1,4,16} x S{16,32,64} = 18 | same grid as eval_v6 | — |
| budgets | {1e-3 (strict), 1e-2 (Pareto-only)} | runner flag, not a build knob | — |

Full cross product = 4 precisions x 3 degrees = 12 impl variants (+ champion alias).
Build priority (first node run gets the high-value subset):
1. `fp32_e4`, `fp32_e3` — exp dial alone; costs ~nothing, likely still < 1e-3.
2. `int8_e6`, `int8_attn_e6` — the 2-3x GEMM lever.
3. `bf16_e6` — cheap truncation path; ~1e-2 quant error alone -> likely a 1e-2-budget-only point.
4. int8/bf16 x {e4,e3} cross terms — only if build time permits (they refine the
   frontier's knee, they do not create new extreme points).

## Shared naming table (LOCKED with int8-kernels, 2026-08-12)
Uniform scheme: `.so` = `libft_<prec>_e<deg>.so`, registry = `c_<prec>_e<deg>`.
`prec` in {fp32, int8, int8_attn, bf16}, `deg` in {6,4,3}. Champion alias first:

| registry (src/impls.py) | .so | build flags (beyond v14 champion set*) | notes |
|---|---|---|---|
| `c_fp32_e6` | `libft.so` (= libft_v14.so copy) | (none) | **the incumbent champion**; must stay bit-identical (err ~1.1e-6, tiny 35.1us / default 260.9us on ft-node-1) |
| `c_fp32_e4` | `libft_fp32_e4.so` | `-DFT_EXP_DEG=4` | = old brief name libft_exp4.so |
| `c_fp32_e3` | `libft_fp32_e3.so` | `-DFT_EXP_DEG=3` | = old brief name libft_exp3.so |
| `c_int8_e6` | `libft_int8_e6.so` | `-DFT_INT8=1` | int8 GEMM everywhere (QKV, attn score, attnV, wo, FFN, out) |
| `c_int8_e4` | `libft_int8_e4.so` | `-DFT_INT8=1 -DFT_EXP_DEG=4` | |
| `c_int8_e3` | `libft_int8_e3.so` | `-DFT_INT8=1 -DFT_EXP_DEG=3` | |
| `c_int8_attn_e6` | `libft_int8_attn_e6.so` | `-DFT_INT8_ATTN=1` | int8 only for attention-score + attnV matmuls; fp32 elsewhere |
| `c_int8_attn_e4` | `libft_int8_attn_e4.so` | `-DFT_INT8_ATTN=1 -DFT_EXP_DEG=4` | |
| `c_int8_attn_e3` | `libft_int8_attn_e3.so` | `-DFT_INT8_ATTN=1 -DFT_EXP_DEG=3` | |
| `c_bf16_e6` | `libft_bf16_e6.so` | `-DFT_BF16=1` | fp32->bf16 truncate, fp32 accumulate |
| `c_bf16_e4` | `libft_bf16_e4.so` | `-DFT_BF16=1 -DFT_EXP_DEG=4` | |
| `c_bf16_e3` | `libft_bf16_e3.so` | `-DFT_BF16=1 -DFT_EXP_DEG=3` | |

\* champion v14 flag set: `-O3 -march=native -ffast-math -funroll-loops -fopenmp-simd -flto -fPIC
-shared -fopenmp -DFT_KERNEL=3 -DFT_FUSED=1 -DFT_FASTEXP=1 -DFT_FFN_MM=1 -DFT_TILE16=1
-DFT_ATTN_BLOCK=1 -DFT_OPENMP=1`. New knobs must be **no-op at their champion-default
values** (`FT_INT8=0/FT_BF16=0/FT_EXP_DEG=6` compile to the byte-same fp32 path).

Out-of-sweep (registered but not on the frontier): existing `c_v1..c_v14`,
`c_ground_up`, `numpy_opt*` — used only as cross-checks in plots if present.

## Error model / expected outcomes
Predicted max_abs_err (tiny / default) — to be replaced by int8-kernels' numpy
replication table when it lands:

| impl | predicted err | expected pass |
|---|---|---|
| fp32_e6 (champion) | 1.1e-6..2.3e-6 across 18 cells (measured, ft-node-1) | 1e-3 AND 1e-2 |
| fp32_e4 | O(1e-6..1e-4) — Chebyshev deg-4; needs int8-kernels' prediction | likely both |
| fp32_e3 | O(1e-4..1e-3) — Chebyshev deg-3; borderline for 1e-3 | 1e-2 likely, 1e-3 maybe |
| int8_e6 | per-channel int8 ~1e-3..1e-2 territory | 1e-2 likely, 1e-3 uncertain |
| int8_attn_e6 | between fp32 and int8_e6 | both likely |
| bf16_e6 | ~1e-2 (quant error alone, per brief) | 1e-2 only |

Latency expectation: int8 GEMM (AVX2 vpmaddubsw) ~2-3x fp32 on compute-bound cells.
Cells that are attention-heavy (large B*S^2) gain most from int8_attn; FFN/QKV/out
matmuls gain from int8 everywhere. fp32_e4/e3 save softmax/exp cost only (small on
GEMM-bound cells, visible on tiny).

**What "closing v6 cells" requires** (measured on ft-node-1):
Champion baseline on ft-node-1 (results/evals_v6_ft-node-1_c_ground_up_a1.json):
max_abs_err 1.1e-6..2.3e-6 on all 18 cells (huge 1e-3 margin; the entire budget
headroom is available for trading). 12/18 cells sub-ms at 1e-3 (vs 13/18 on
bench-node-3; default_b4_s32 is 1086.8us here, just over).

| cell | med us | err | cell | med us | err |
|---|---|---|---|---|---|
| tiny_b1_s16 | 55.8 | 1.1e-6 | default_b1_s16 | 184.7 | 1.2e-6 |
| tiny_b1_s32 | 68.0 | 1.2e-6 | default_b1_s32 | 282.3 | 1.2e-6 |
| tiny_b1_s64 | 129.8 | 1.6e-6 | default_b1_s64 | 563.5 | 1.4e-6 |
| tiny_b4_s16 | 125.0 | 1.1e-6 | default_b4_s16 | 558.1 | 1.7e-6 |
| tiny_b4_s32 | 202.0 | 1.9e-6 | default_b4_s32 | 1086.8 | 1.5e-6 |
| tiny_b4_s64 | 424.7 | 1.7e-6 | default_b4_s64 | 2203.3 | 1.6e-6 |
| tiny_b16_s16 | 390.5 | 1.8e-6 | default_b16_s16 | 2179.6 | 1.7e-6 |
| tiny_b16_s32 | 781.0 | 2.3e-6 | default_b16_s32 | 4532.0 | 1.7e-6 |
| tiny_b16_s64 | 1714.9 | 2.3e-6 | default_b16_s64 | 9624.8 | 1.7e-6 |

Speedup needed vs champion to reach sub-ms at 1e-3:
- default_b4_s32 1086.8us -> >=1.09x (int8 almost certainly closes it)
- tiny_b16_s64 1714.9us -> >=1.72x (int8/e4 likely closes)
- default_b4_s64 2203.3us -> >=2.21x (attention-heavy: int8_attn is the plausible
  lever; needs near-ideal int8 attention speedup)
- default_b16_s16 2179.6us -> >=2.18x (int8 plausible)
- default_b16_s32 4532.0us -> >=4.53x and default_b16_s64 9624.8us -> >=9.62x:
  beyond int8's ~2-3x — NOT closable on this 2-core node at 1e-3. These become
  "how much of the gap does the frontier close" rows, not win conditions.
- Success metric for the writeup: per-cell frontier table + per-budget sub-ms count.
  Baseline: 12/18 at 1e-3 on ft-node-1. Target: 15/18 at 1e-3 (adds
  default_b4_s32, tiny_b16_s64, and one of default_b4_s64 / default_b16_s16),
  and 15-16/18 at 1e-2 (int8's extra 2-3x matters most at the lax budget).

## Runner contract (evals/eval_v8.py) — see file
- CLI mirrors eval_v6: `python evals/eval_v8.py --node ft-node-1 [--impl a,b]
  [--cfg tiny,default] [--batch 1,4,16] [--seq 16,32,64] [--budget 1e-3,1e-2]
  [--attempt N] [--smoke] [--out-dir results]`.
- Per attempt file: `results/evals_v8_<node>_<impl>_a<attempt>.json` with
  `{impl, node, generation:"v8", attempt, criteria, cells{<cell>: {max_abs_err,
  median_us, p99_us, iters, gflops, budget_pass{<budget>:bool}, sub_ms}}}`.
- Budget evaluation is post-hoc arithmetic on `max_abs_err` (no per-budget rerun):
  one measurement per (impl, cell), all budgets computed from it.
- Correctness ref = float64 numpy reference (same as eval_v6 `reference()`).
- Summary: `results/evals_v8_<node>.json` (same shape as v6 summary).
- Impls come from `src.impls.IMPLS`; missing c_* names fall back to a lazy
  `_CImpl(libft_<prec>_e<deg>.so)` registration inside eval_v8 (self-healing if
  int8-kernels hasn't landed its registry entries yet). If the int8 wrapper needs
  special weight prep, eval_v8 imports the helper from src/impls.py instead
  (coordinated: see int8-kernels question (b)).

## Analysis contract (evals/pareto_analysis.py) — see file
- Ingests `results/evals_v8_<node>_<impl>_a<N>.json` attempt files, builds the
  per-(cell, budget) frontier: min median_us among impls with max_abs_err < budget,
  tolerating missing impls (populate incrementally); min-median across attempts.
- Outputs: `results/pareto_<node>.md` (frontier tables per budget + per-cell
  winner + speedup vs champion), `results/pareto_<node>.json` (machine-readable
  frontier), `results/pareto_<node>_frontier_tiny.png` / `_frontier_default.png`
  (latency vs err per cell, log-log, frontier polyline highlighted),
  `results/pareto_<node>_summary.png` (sub-ms pass counts per budget, bars).
- Frontier definition: impl i dominates j if err_i <= err_j AND median_i < median_j;
  frontier = non-dominated set per cell (all budgets overlaid).

## Status log
- 2026-08-12 (design locked): naming table proposed to int8-kernels (message sent;
  their ack pending — treat as locked on my side). Deliverables DONE and
  smoke-tested: notes/error-pareto.md, evals/eval_v8.py, evals/pareto_analysis.py
  (synthetic-data smoke: frontier/winner/plots verified; no-data path OK).
- Runner smoke: numpy_opt2 on 4 tiny cells -> budget_pass JSON correct.
- Analysis smoke: 5 fake impls x 4 cells -> winners, frontier sets, sub-ms counts,
  md/json/png all produced correctly (min-median across attempts works).
- Waiting on: int8-kernels' predicted-error table + build flag landing; parent's
  first node run to produce evals_v8_* JSONs. Run order for parent:
  `python evals/eval_v8.py --node ft-node-1` (after build.sh has the new .so files),
  then `python evals/pareto_analysis.py --node ft-node-1`.
