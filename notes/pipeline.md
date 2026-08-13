# FT novelty push — pipeline state (parent-owned)

**Phase:** v7 attribution -> int8/bf16 kernels -> error-budget Pareto -> writeup.
**Node:** `ft-node-1` (2-core Xeon @2.2GHz, 13.6GB, numpy 2.0.2). Node commands:
`colab --auth adc <cmd> -s ft-node-1 ...` — **PARENT ONLY runs colab commands.**
**Node-2:** quota-blocked (TooManyAssignments). Retry `colab --auth adc new -s ft-node-2` periodically.

## Baseline on ft-node-1 (attempt 01, Aug 12)
| cfg | median_us | max_abs_err | pass |
|---|---|---|---|
| tiny (d32 L1 S16 V256) | 35.1 | 1.1e-6 | yes |
| default (d64 L2 S32 V512) | 260.9 | 1.2e-6 | yes |

Remote tree: `/content/ft_gu` (tarball /content/ft.tgz). Results: `/content/ft_gu/results/`.
Local results: `results/`. Remote->local: `colab --auth adc download -s ft-node-1 <remote> <local>` (no -f flag).

## Workers (recursive children, spawn their own grandchildren if useful)
- `v7-attribution`   -> brief: notes/brief_v7_attribution.md   (cycle attribution + cost model)
- `int8-kernels`     -> brief: notes/brief_int8_kernels.md     (int8/bf16 GEMM + exp-degree dial)
- `error-pareto`     -> brief: notes/brief_error_pareto.md     (sweep design + frontier analysis)

## Rules
- Children never run `colab`; they write code/scripts/notes and reply via agent_message.
- Parent (heartbeat turns) runs node tasks: upload tarball, build, run runner scripts, download results.
- Correctness bar: max_abs_err < 1e-3 vs f64 ref (strict); also track 1e-2 budget for Pareto.
- v6 cells not yet sub-ms: default b4s64 (1965us), all B=16. int8 is the lever to close them.
- Signed journal entries in notes/journal.md per convention.

## Heartbeat log
- HB#1 (14:48): workers all RUNNING (recon phase, no artifacts yet); ft-node-1 alive/IDLE; ft-node-2 still TooManyAssignments.
- HB#2 (15:00): workers progressing (v7 analyzing results formats; int8 reading driver patterns; error-pareto drafting naming table + will message int8 for coordination). ft-node-2 retry: still quota-blocked.
- HB#3 (15:12): no artifacts yet; nudged all 3 workers to checkpoint partial work + report 1-line status. error-pareto messaged int8-kernels re naming. ft-node-2 retry: still quota-blocked.
- HB#4 (15:25): v6 battery for c_ground_up launched DETACHED on ft-node-1 (pid 3005, out results/v6_stdout.log). error-pareto v1 died without artifacts -> deleted + re-spawned v2 with checkpoint rule. v7-attribution & int8-kernels still running. ft-node-2: still quota-blocked.
- HB#5 (15:37): v6 battery DONE on ft-node-1 -> results/evals_v6_ft-node-1_c_ground_up_a1.json. 12/18 sub-ms; fails: tiny_b16_s64(1715us), default_b4_s32(1087), b4_s64(2203), b16_s16(2180), b16_s32(4532), b16_s64(9625). int8 (~2.5-3x GEMM) can plausibly close first 4 -> 16/18; b16_s32/s64 remain roofline-impossible. Workers healthy (all streaming; false stuck-alarm from stale previews). ft-node-2: still quota-blocked.
- HB#5b (15:40): v7-attribution v1 died mid-recon (no artifacts) -> deleted, re-spawned v2 with inline-facts tight brief (artifacts-first). int8-kernels + error-pareto v2 still streaming. Lesson: children must write files before reading lots of repo context; embedding facts inline prevents recon-death.
- HB#6 (15:50): error-pareto v2 landed evals/eval_v8.py (verified: imports clean, v6 helpers present) + notes/error-pareto.md (LOCKED naming table: libft_<prec>_e<deg>.so / c_<prec>_e<deg>; knobs FT_INT8/FT_INT8_ATTN/FT_BF16/FT_EXP_DEG). int8-kernels v1 stalled (23 msgs/50min, zero artifacts) -> killed, re-spawned v2 with locked naming + tight brief. v7-attribution v2 active. pareto_analysis.py still in flight.
- HB#7 (16:02): error-pareto v2 landed evals/pareto_analysis.py (imports OK, incremental frontier, min-median convention). All 3 workers active (error-pareto ~done, int8-kernels v2 9msgs in, v7-attribution working). ft-node-2: still quota-blocked.
- HB#8 (16:14): workers all active, no new artifacts this tick (error-pareto finishing journal; int8-kernels studying ctypes wrapper for quantized init; v7-attribution in-flight). ft-node-2: still quota-blocked.
- HB#9 (16:26): error-pareto still active (58msgs); v7-attribution at 21msgs ~25min (last 2.3min, likely long code-write; watch next tick); int8-kernels at 15msgs active. No new artifacts. ft-node-2: quota-blocked.
