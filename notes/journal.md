# Fast Transformer Race - Journal

Goal: fastest transformer forward pass on a Google Colab CPU node (sub-millisecond).
Strategy: recursive multi-agent tree, each subtree owns a Colab node, evals escalate
when they saturate, all numbers land in results/, graphs on localhost:9023.

## Timeline
- setup: 3 colab CPU sessions (bench-node-1/2/3), ADC auth, gh=yoits9090, repo fast-transformer (private)
- baseline impls: numpy_naive, numpy_vec (src/impls.py)
- eval v1: correctness vs f64 ref (1e-3 abs) + latency percentiles (evals/)

## 2026-08-11 — ground-up-c: C implementation ready
- Wrote ground_up/ (ft.h, matmul.c, transformer.c, build.sh, node_run.py): zero-ML-dep
  transformer forward pass. Own matmul (naive + 8x8 register-blocked, FMA/auto-SIMD),
  own layernorm/softmax/attention/FFN. Fused QKV (one matmul over [Q|K|V] block),
  flash-style causal attention (no SxS materialized), fused row-wise ReLU FFN.
  int32->fp32 accumulation is N/A here (weights are fp32); we accumulate fp32 with
  FMA contraction — error ~1e-6 rel, far under the 1e-3 bar.
- Variants: v1 naive, v2 blocked matmul, v3 blocked+fused ops, v4 +OpenMP, v5 +static
  workspace/output reuse (ctypes wrapper in src/impls.py: c_v1..c_v5, c_ground_up).
- Scratch layout sized in ft_scratch_bytes(); wrapper caches weight blocks once per
  weight set (keyed by array address) so the hot path is a single ctypes call.
- Local syntax-only checks pass (clang, no -fopenmp on this Mac; node builds with gcc).
- No nodes up yet (provisioner attempt 10/30); waiting to upload + build + bench.

## 2026-08-11 — evals setup + first findings (agent: evals)
- Eval infra committed: evals/registry.py (v1/v2/v3 registry), evals/eval_v2.py (scaling: seq 8..128, batch 1/4/16, p90/p99/max tails), evals/eval_v3.py (5-seed stability, cold-cache ratio, tracemalloc alloc counts), evals/leaderboard.py (update + aggregate, 10% discrepancy flag), evals/node_driver.py (upload->exec->download for bench-node-3, results/manifest.json attempt log).
- ANOMALY: numpy_opt3/numpy_opt4 fail correctness vs f64 reference on TINY (err 3.55) and DEFAULT (3.97). Root cause: ReLU applied AFTER w1 up-projection (max(LN(h)@w1,0)@w2) instead of BEFORE (max(LN(h),0)@w1@w2). All pre-activations match to 7e-7 — only ReLU placement wrong. Reported to numpy-optimizer.
- NOTE: numpy_opt1/opt2 are batch-1-only (return (1,S,V)); v2 batch sweep will flag batch>1 (by design for the sub-ms race, but recorded).
- colab sessions still provisioning (provisioner attempt ~8 at time of writing); bench-node-3 reserved for evals.

## 2026-08-11 — opt3/opt4 fixed (agent: evals)
- numpy-optimizer fixed numpy_opt3/numpy_opt4 ReLU placement (now max(h-mu,0)*rstd before w1) AND made all opt* impls batch-general (B=1/4/16 all PASS vs f64 ref on DEFAULT).
- Local pre-flight state: 6 numpy impls (naive, vec, opt1..opt4) all pass TINY+DEFAULT, all batches. C impls (c_v1..c_ground_up) pending .so builds on nodes.
- Eval harness ready: v1 (run_all), v2 (scaling+tails), v3 (stability/cold/alloc); leaderboard+aggregate validated on synthetic data (10% discrepancy flag works).
- Awaiting colab sessions (provisioner ~attempt 20, 0/3 ready).
