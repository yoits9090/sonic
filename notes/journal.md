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

## 2026-08-11T03:30:39 — numpy-optimizer
- Correctness bug found (with evals@bench-node-3): opt3/opt4 had ReLU AFTER up-projection
  (max(h@w1,0)@w2 instead of max(ln(h),0)@w1@w2). Fixed: FFN fused form is
  max(h-mu,0)*rstd @ w1 @ w2 (valid since rstd>0).
- opt1..opt5 all pass correctness (max_abs_err <= 1.8e-6 vs f64 ref, both cfgs) and B>1
  (fallback to numpy_vec). opt5 adds preallocated out= buffers.
- Key tricks in opt3/opt4/opt5: LN folded into downstream projections
  (ln(x)@P = rstd*(x@A - mu*c) + b@P with A=g*P, c=g@P precomputed), denom-factorized
  attention ctx = (e@V)/sum(e) (row-wise denom factors out of the t-contraction),
  mask as add (-1e9), no-max-subtraction softmax (safe: |att|<=~12 < 87 overflow bound),
  Q pre-scaled by 1/sqrt(dh), B=1 squeeze to 2D matmuls.
- Waiting on bench-node-1 (provisioner retrying, Colab API rejects session creation).

## research (agent: researcher, 2026-08-11)
- completed web research sprint: BLAS microkernels, llama.cpp kernel tricks, threading for 2 vCPUs, int8/VNNI, fused attention, XLA/oneDNN. Full findings in notes/research.md.
- key insight: TINY forward is 0.57 MFLOP with only 96.8 KB of weights (L2-resident) -> the 1ms budget is an OVERHEAD problem, not a FLOPs problem. numpy+OpenBLAS pays 30-150us dispatch + threading pathology (OpenBLAS #731) on tiny GEMMs.
- TOP-5 for the tree: (1) fused single-pass C/numba kernel, (2) numpy fast-path (threads=1 + weight folds + direct sgemm), (3) 2-thread row-sliced head GEMV only if 2 physical cores, (4) int16/int8 static weights with tolerance check (int8 likely violates 1e-3 on logits), (5) fused causal attention + fast exp/rsqrt.
- first numbers to beat: compute floor 35-140us at 4-16 GFLOP/s; output head = 46% of MACs (131K of 283K).
- sources: salykova.github.io/matmul-cpu, llama.cpp ggml (simd-gemm.h, vec.cpp, x86/quants.c), Intel oneMKL JIT small-matrix article, OpenBLAS #731, victorbona CPU-LLM compendium, arXiv 2406.02528 / 2409.16997 / 2509.25853, CMSIS-NN, XNNPACK, oneDNN BRGeMM.

- fix (pre-node): mm_blocked restructured for perfect nesting (collapse(2) validity); wrapper now
  builds separate wq/wk/wv blocks for the unfused v1/v2 path (was pointing at wo -> wrong weights).
  Local clang syntax checks pass for all 5 build configs. Waiting on provisioner for bench-node-2.

## FIRST NODE RUN (bench-node-2, attempt 01, gcc 11.4, 2 cores) — ground-up-c
- Build OK after fixing missing <stdint.h> in ft.h (gcc 11; clang local check missed it).
- CORRECTNESS: all C impls FAILED max_abs_err 3.5-5.3 (numpy_vec passes 7e-7). Root cause:
  attention kernels treated Q/K/V as contiguous (n,d) blocks but they are STRIDED views into
  the qkv buffer (fused: [Q|K|V] interleaved rows stride 3d; naive: 3 contiguous blocks stride d).
  FIXED: attn_fused/attn_naive now take base pointers + row stride.
- Also reworked ffn_fused w2 pass to outer-product form (w2 rows contiguous; old column loop
  had 256B-stride access — suspected cause of v3>v2 on default).
- Latency (still buggy build, 3000 iters/300 warmup): tiny median_us v1=151.1 v2=121.2 v3=113.5
  v4=104.0 v5=109.4 (numpy_vec=430.0). default: v1=2297 v2=1715 v3=2047 v4=1287 v5=1246
  (numpy_vec=1992). All C variants sub-ms on TINY.
- eval_v3 numpy_vec baseline: tiny steady 429.8us / default 1999.9us, allocs 19-20, peak 0.12-0.53MB.
- Node died after run (~30 min lifetime); results JSONs lost with VM. Re-running on next node.
