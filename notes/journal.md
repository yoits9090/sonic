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

## 2026-08-11T04:05:23 — numpy-optimizer: FIRST NUMBERS (bench-node-1, Xeon 2.2GHz 2c, numpy 2.0.2)
Battery a1 (iters=3000, warmup=300, all pass correctness):
| impl | tiny median_us | default median_us |
|---|---|---|
| numpy_naive | 418.2 | 2172.5 |
| numpy_vec | 422.8 | 1983.4 |
| numpy_opt1 | 278.6 | 1707.6 |
| numpy_opt2 | 227.5 | 929.6 |
| numpy_opt3 | 247.3 | 1024.9 |
| numpy_opt4 | 230.7 | 942.7 |
| numpy_opt5 | 214.1 | 920.3 |
| numpy_opt6 | 224.9 | 912.2 |
| numpy_opt7 | 145.9 | 761.6 |
TARGETS HIT: tiny <1000us (5.9x margin), default <1000us (1.3x margin). opt7 = opt6 + stacked-matmul
LN stats ([h|h*h] @ Wstats -> (S,2)): biggest single win (~35% on tiny vs opt6).
Node is noisy (mean >> median; opt6 default p99=9ms) -> medians are the metric.
Next: thread sweep (2 cores), probe stage timings, opt8+ (attention/softmax micro-opts), attempt-2 stability run.

## 2026-08-11T04:12:29 — numpy-optimizer: opt8/opt9/opt10 round (bench-node-1)
- opt8 (eps-folded rstd via pow, matmul-ones softmax den, direct ctx divide, np.take): 
  tiny 141.9us, default 735.6us (A/B vs opt7 160.2/773.9 same exec). Micro data on node:
  rstd add+sqrt+div 4.94us -> eps-folded pow 2.91us; den sum 3.15us -> matmul-ones 1.90us;
  ctx recip+mul 12.3us -> in-place div 10.4us; einsum 5-10x slower than matmul everywhere.
- opt9 (merged [h|h*h|1]@W_ext -> qkv+mu+ex2 in ONE wide matmul): LOSS 305us/972us — wide
  (16,65)x(65,98) gemm hits a slow OpenBLAS path on this node. Dropped.
- opt10 (opt8 + ctx /= den in place, no reciprocal): tiny 139.7/142.9 (a1/a2), default 729.1/722.0 (a1/a2).
  BEST: opt10 tiny ~140us, default ~722us.
- Thread sweep opt10: tiny 1T=142.5 vs 2T=237.5 (1T wins); default 1T=777.8 vs 2T=726.1 (2T wins ~7%).
  Mixed + node noise; default-threads (unset) best overall. Noise outliers observed (opt8 tiny 266.8us in a2).

## 2026-08-11T04:17:47 — numpy-optimizer: opt11/opt12 + saturation assessment (bench-node-1)
- opt11 (locals binding): tiny 138.3, default 717.8.
- opt12 (zero per-call allocations: buffered take/ones/st/rstd/mm): tiny 131.3-133.3, default 714.6-716.8 (a1/a2).
- scipy sgemm (alpha/beta fused residual): SLOWER than np matmul+add (6.5 vs 5.9us; qkv 11.5 vs 8.1us). Dropped.
- exp2/contiguity micros: no wins. Leaderboard: my flat JSONs aren't parseable by evals/leaderboard.py
  (expects v1/v3 nested format) — noted for evals.
- BEST (opt12): TINY 131.3us (7.6x under 1000), DEFAULT 714.6us (1.4x under 1000). vs naive: 3.2x/3.0x.
- Saturation: last 6 attempts gained <=7%; remaining hotspots are the big matmuls themselves
  (out 1M flops, w1/w2 0.5M) + ~40 numpy calls; pure-numpy floor approx. Next axis if parent wants:
  sub-agent for thread/OS-level tuning or accept numpy_opt12 as the numpy champion.

## 2026-08-11T04:21:00 — numpy-optimizer: final champion + saturation (bench-node-1)
- opt13 (transposed wide matmuls): LOSS in full pipeline (tiny 135.1 vs 131.3, default 742.4 vs 710.7)
  despite winning isolated micros (qkv -12%, out -9%) — .T view chains cost more than the gemm gain. Dropped.
- opt12 median-of-medians across a1-a4: TINY ~132us (131.2-133.3), DEFAULT ~714us (705.9-716.8).
  vs naive 418/2173: 3.2x/3.0x. Sub-1000us margins: 7.6x (tiny), 1.4x (default).
- Saturation: 8 consecutive attempts since opt12's last gain were losses or noise (opt13, sgemm,
  exp2, contiguity, thread sweep, orientation). Pure-numpy floor reached on this node.

## 2026-08-11T04:21:37 — numpy-optimizer FINAL SUMMARY (signed: numpy-optimizer)

### Champion
numpy_opt12 — median-of-medians over 4 battery runs (iters=3000, warmup=300, bench-node-1):
  TINY ~132us (131.2-133.3), DEFAULT ~714us (705.9-716.8). All correctness green (err<=2e-6 vs f64 ref),
  B>1 falls back to numpy_vec, seq x batch sweep clean (evals v2 ready).
  vs numpy_naive (418/2173): 3.2x / 3.0x. Margins under 1000us target: 7.6x / 1.4x.

### What worked (positive data)
1. Fused QKV single matmul + cached weight prep (opt2): 228/930 vs naive 418/2173.
2. LN folded into downstream projections: ln(x)@P = (x-mu)@P * rstd (opt3/4).
3. Stacked-matmul LN stats [h|h*h|1] @ Wstats -> [mean, E[x^2]+eps] in ONE gemv-ish call: 7.2us vs
   25us for mean+mean reductions on the node (opt7): biggest single win (146/762).
4. eps folded into stats weight -> rstd = (ex2 - mu*mu) ** -0.5 in 3 calls vs 5 (opt8).
5. Softmax denominator via matmul-ones (1.9us) vs np.sum (3.2us) (opt8).
6. ctx normalized by in-place divide (no reciprocal+mul) (opt10).
7. Preallocated out= buffers everywhere; zero per-call allocations incl. take/ones/st/rstd (opt5,12).
8. mask as additive constant (no np.where), no-max-subtraction softmax (safe: |att|<~13 << 87),
   Q pre-scaled by 1/sqrt(dh), B=1 squeeze to 2D, np.take embed.

### Negative data (8 failed attempts + why — gold for the repo)
1. opt9 merged stats+QKV wide matmul [h|h*h|1]@W_ext(2d+1,3d+2) -> [qkv|mu|ex2] in one call:
   LOSS 305/972 vs opt8 142/735. WHY: ~2x gemm flops + wide small-gemm slow path in OpenBLAS on
   this node; non-contiguous qkv output slice adds reshape copies.
2. opt13 transposed wide matmuls (P.T @ x.T -> (N,S)): LOSS 135/742 vs 131/711. WHY: isolated
   gemm orientation micros won (qkv -12%, out -9%, w1 -10%) but the pipeline's .T view chains and
   transposed-output slices (FFN into qkvT[:d]) cost more than the gemm gain.
3. scipy.linalg.blas.sgemm with alpha/beta (fused residual matmul+add): SLOWER (6.5 vs 5.9us;
   qkv 11.5 vs 8.1us). WHY: numpy 2.x small-matrix matmul dispatch beats fblas wrapper overhead.
4. OPENBLAS_NUM_THREADS sweep (1 vs 2 on the 2-core node): mixed/no gain. tiny: 1T=142.5, 2T=237.5;
   default: 1T=777.8, 2T=726.1. Unset (default) threads won overall. WHY: thread spawn overhead
   dominates tiny gemms; node noise ~3-7% swamps the differences.
5. einsum for attention/stats (att_einsum 47.6us vs att_matmul 4.05us; qkv einsum 49.6 vs 7.1):
   5-10x slower on this numpy build. Never use einsum here.
6. np.exp2 with pre-scaled att (log2e): 4.24us vs np.exp 1.91us. WHY: exp2 slower in this libm.
7. Contiguous-copy attention operands (ascontiguousarray before batched matmul): micro faster by
   ~0.1-0.2us but requires an extra copy call (~2us) -> net loss.
8. pow with np.float32 scalar (2.03us) vs python float (1.73us); f32 literals don't help.

### Where the time goes (opt12, stage profile on node)
TINY: final_chain 46us, ffn 35us, qkv 20us, attn 20us, wo 7us, stats 14us, emb 3us (sum ~145).
DEFAULT: final_chain 141us, ffn 69us, qkv 43us, attn 39us, wo 16us, stats 18us, emb 2us (sum ~328
in isolation; full forward ~714 -> remainder is call overhead + noise). ~40 numpy calls/call at
3-6us each: the floor is Python->C dispatch + small-gemm BLAS time on a 2-core Xeon.
