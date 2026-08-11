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
