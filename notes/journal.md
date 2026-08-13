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

## 2026-08-11 — FIRST FULL BATTERY on bench-node-3 (agent: evals)
- Full battery v1+v2+v3+v4+v5 ran on bench-node-3 (Xeon 2.2GHz 2-core, numpy 2.0.2), 15 impls (naive, vec, opt1..opt13), attempt 4, snapshot-isolated per-attempt upload (fixed stale kernel module cache across execs).
- Correctness: ALL GREEN — v1 (TINY), v2 (seq 8..128 x batch 1/4/16 grid), v3 (5 weight/input seeds), v4 (edge seq 1/2/16/32/64/256, skewed/same/edges/bursty token dists). No failures anywhere.
- Leaderboard (best median): TINY numpy_opt12=131.2us (opt11 137.0, opt7 143.6, opt10 140.2, opt8 140.7). DEFAULT numpy_opt12=704.6us (opt11 716.0, opt10 716.9, opt8 731.8, opt7 758.9). All < 1ms target.
- Cross-node (aggregate.json): node-1 vs node-3 agree within ~1-5% for same impls (opt12: 131.17 vs 132.47; opt3: 210.9 vs 209.1; opt8: 140.7 vs 146.7). NO >10% discrepancies remain. Earlier FLAGs (opt8/opt12) were impl-version skew (node-1 a1 files measured older code), resolved after re-measure.
- v5 headroom: opt12 DEFAULT 6.93 GF/s vs ~17 GF/s pure-matmul ceiling (~2.4x overhead); opt7 8.9 GF/s @1.9x overhead — matmul ceiling is the machine limit for these shapes.
- Fixes in this round: run_all.py JSON parse (multi-line indent bug), eval_latency compact print, node_driver positional upload/download args, remote-dir pre-create, snapshot isolation, IPython echo suppression, leaderboard flat-format + v3 cfg-key support.

## 2026-08-11 — ground-up-c: CORRECT + FAST (attempts 04-06, bench-node-2)
- ROOT CAUSE FOUND (after many probes): this benchmark's FFN is h += relu(ln2(h)) @ W1 @ W2 —
  ReLU on the LAYERNORM OUTPUT, BEFORE W1 (matches numpy_vec + f64 ref). My C (and my diagnostic
  mirror) applied ReLU to the hidden layer (between W1 and W2). Fixed in both ffn paths.
  Debugging lessons: (1) the colab exec stdout channel interleaved stale kernel output — trust only
  downloaded JSON files, not exec stdout; (2) stage-diff against numpy_vec (pure arithmetic,
  done locally) pinpointed h_ffn as the diverging stage.
- CORRECTNESS: all c_v1..v6 + c_ground_up PASS tiny + default (max_abs_err 0.9-1.9e-6 vs 1e-3 bar),
  confirmed attempts 04/05/06, plus eval_v3 5-seed stability.
- LATENCY (3000 iters / 300 warmup, median_us, attempts 05+06):
    TINY:    c_v6 36.8  c_ground_up 37.4  c_v5 39.1  | numpy_vec 451  | numpy_opt12 target 131.3
    DEFAULT: c_v6 350-362  c_ground_up 352-354  c_v5 361-366  | numpy_vec 2052-2326 | target 714.6
  Both configs sub-ms. ~3.6x / ~2.0x faster than the numpy-optimizer targets.
- KEY OPTIMIZATION: specialized mm_blocked8 kernel (compile-time 8x8 tiles, pragma GCC unroll 8,
  all dims %8==0) — DEFAULT dropped 1148->366us (v4->v5). Plus dot16 two-chain attention dots,
  -funroll-loops -fopenmp-simd, OpenMP on tile loops (v6), wrapper-cached buffers.
- Variant progression (DEFAULT median): v1 naive 2354 -> v2 blocked 1645 -> v3 fused 1327-1763
  -> v4 +omp 1072-1148 -> v5 spec8x8 361-366 -> v6 +omp+cached 350-362.
- results/ JSONs: correctness+latency attempts 04,05,06 + evals_v3 a4/a5.
- c_v6 eval_v3: allocs 11, peak 0.00MB (numpy 19-20, 0.12-0.53MB); cold/steady ratio ~1.0-1.1.

## 2026-08-11 — C impls battery on bench-node-3 (agent: evals)
- Full battery (v1-v5) for c_v1..c_v6, c_ground_up on bench-node-3 (attempt 5, snapshot-isolated; libft .so built ON node via ground_up/build.sh, gcc 11.4, -march=native).
- CORRECTNESS: all C impls pass v1/v2/v4, stability seeds, adversarial (edge seq 1..256, skewed/same/edges/bursty). Allocs: 9-12/forward, peak <=0.15MB (v6/ground_up ~0).
- LATENCY (best median): TINY c_v6=35.8us (c_v5 38.5, c_ground_up ~36 fresh) vs numpy_opt12 131.2 → 3.7x. DEFAULT c_v6~350us (c_v5 352.1, c_ground_up ~342 fresh) vs numpy_opt12 704.6 → 2x.
- v5 headroom: c_v6 20.1 GF/s EXCEEDS numpy matmul-ceiling measurement (16.9) — fused C kernels are at the machine's practical limit (overhead 0.84x).
- ANOMALY INVESTIGATED: bench-node-3 shows intermittent 3-5s host-contention episodes (~1.5x slowdown) hitting fast C kernels (c_v5/c_v6/c_ground_up bimodal medians ~350 vs ~520us); zero steal, const nominal freq, affects numpy matmuls too (64mm 30→17 GF/s blip). Not an impl bug; leaderboard min-median is robust (matches node-2: c_v6 350/362). c_v4 TINY flag (112 vs 85) = noise window on node-2; DEFAULT consistent (961-1148 vs 1038-1084).
- Methodology: OMP_NUM_THREADS=2 pinned in node_driver runner.

## 2026-08-11 — ground-up-c: FINAL (attempts 07-11, bench-node-2)
- Final champion = c_ground_up (libft.so, v9 config): specialized 4x16 unrolled matmul +
  fused QKV/attention/FFN + minimax-6 fast-exp + OpenMP (only matmuls >=256k MACs) +
  cached workspace/output. Correctness 1.0-1.2e-6 (bar 1e-3), all variants pass tiny+default.
- FINAL LATENCY (median_us, 3000 iters/300 warmup, attempt 11):
    c_ground_up: TINY 36.6  DEFAULT 352.6   (sub-ms both)
    c_v9:        39.3/352.8   c_v8: 32.1/552.9*  c_v7: 35.2/385.2  (*v8 default noisy)
    numpy_vec:   ~447/2052-2326   (10-12x slower)
    numpy_opt12 targets: 131.3/714.6  ->  3.6x / 2.0x faster
- v3 gate (5 seeds): c_ground_up 47.1-47.7us tiny / 350-355us default, allocs 9, peak ~1.2KB.
- omp tuning lesson: per-thread scratch must be declared INSIDE the parallel loop (omp_get_thread_num
  before the region returns 0 for all threads -> race); matmul thread threshold 256k MACs (131k
  forked on tiny's out-matmul and hurt tiny). fast_exp: Taylor-5 was 1e-4 rel err -> Chebyshev-LS
  degree-6 gives 2.6e-9 rel err (error back to 1e-6).
- Variant progression (DEFAULT median): v1 2354 -> v2 1645 -> v3 1327 -> v4 1072 -> v5 361-366
  (specialized 8x8 kernel) -> v6 350-397 (+omp) -> v7 366-387 (fastexp+ffn-mm) -> v9 330-362
  (4x16+fastexp+ffn-mm+omp). All correctness-passing.
- results/ now holds attempts 04-11 JSONs (correctness+latency) + evals_v3 a4/a5/a10.

## research batch escalation (agent: researcher, 2026-08-11)
- B=4 DEFAULT escalation: computed budget = 13.6M MACs (27.3 MFLOP), 512KB weights, 2-core fp32 AVX2 peak ~80 GFLOP/s -> floor ~340us, realistic 450-700us -> sub-ms needs the GEMM playbook, not overhead tricks. Batch folds into M=128 rows; weights reused 128x -> compute-bound, weight-stationary wins.
- TOP-5 updated in notes/research.md with B=4 section: (1) register-blocked weight-stationary microkernel + weights pre-packed at load (llama.cpp repack/llamafile sgemm pattern), (2) fold batch into rows + 2-thread row-chunking (llama.cpp mul_mat does exactly this), (3) int16/int8 static weights (2-4x MACs/cycle; int8 tolerance check first), (4) numpy path: reshape (B,S,d)->(B*S,d) single 2-D sgemm, never per-item loops / 4-D strided matmul (OpenBLAS has no strided-batched API, #5447), (5) fused causal attention over flattened rows with per-item masking.
- new sources: OpenBLAS #5447, Intel batch GEMM article, llama.cpp repack.cpp + llamafile sgemm.cpp + mul_mat threading, DeepWiki batch pipeline.

# FINAL VERDICT — sub-ms race (2026-08-11, agent: evals)

## Champion
**c_ground_up v14 (libft.so = v14 kernels: spec 8x8 + fast-exp + ffn-via-matmul + 4x16 tiles + OpenMP + blocked attention)**
Measured on bench-node-3 (2-core Xeon 2.2GHz, numpy 2.0.2), cross-validated vs bench-node-2 (agreement 3-6%).

| domain | result |
|---|---|
| B=1 (race target) | sub-ms EVERYWHERE: tiny 48-109us, default 161-477us (default B1 S32 = 236us, best ever) |
| B=4 | tiny ALL sub-ms (102-360us); default S16 433us, **S32 883us SUB-MS (B=4 frontier closed)**, S64 2111us OVER |
| B=16 | tiny S16 332us / S32 656us sub-ms; default all OVER (1777-8612us) |
| v6 score | **13/18 cells sub-ms** (8/9 tiny, 5/9 default) |
| roofline | 0.43 of measured peak GEMM (66.7 GF/s) at canonical default B1 S32 |
| cold-start | 859us first call vs 226us steady (3.8x: ctypes struct build + buffer prep) |

## Remaining gaps + engineering rationale
1. **default B4 S64 = 2111us (~2.1x over)**: attention is O(B·S²) — S64 quadruples attention flops vs S32 while QKV/FF only double; the blocked-attention kernel (v14) reclaimed S32 but S64 needs 2.1x more effective throughput. On a 2-core Xeon the practical ceiling is the ~66 GF/s GEMM peak; the impl already runs at 43% of it, so the next 2x must come from bandwidth-side wins (tile cache reuse, avoiding the softmax round-trip) — not from more cores.
2. **B=16 default floor (1777-8612us)**: 16x the tokens of B=1; even the 477us B1 S64 rate implies ~7.6ms at B16 if perfectly linear — sub-ms B=16 is >1.8x beyond the machine's roofline at these shapes. Verdict: NOT worth pursuing on this hardware; the race target is single-forward B=1 where the champion has 2-8x margin.
3. **Cold start 3.8x**: one-time weight-struct/warmup cost; steady-state is the race metric; a serving impl could amortize via a persistent process (already the case in benchmarks).

## Eval ladder v1 -> v6 (closing summary)
- **v1** (correctness + latency percentiles): caught numpy_opt3/opt4 wrong ReLU placement (err 3.5 vs f64 ref) → fixed by numpy-optimizer.
- **v2** (seq 8-128 x batch 1/4/16 grid + p99/max tails): flagged opt1/opt2 batch>1 limitation; all impls pass on the grid.
- **v3** (5-seed stability, cold-cache, alloc counts): surfaced C-impl latency variance on shared nodes (host contention episodes ~1.5x, bimodal medians) → leaderboard hardened to min-median; numpy_opt8 cold ratio 11.5x flagged.
- **v4** (adversarial: edge seq 1/2/16/32/64/256, skewed/same/edges/bursty token dists): zero failures — no impl is brittle to input pathology.
- **v5** (headroom: per-cfg FLOPs, matmul ceiling, overhead x): C fused kernels EXCEED the numpy matmul ceiling (20.1 GF/s vs 16.9) — numpy best 8.9 GF/s @ 1.9x overhead; quantified the 2-core machine's real ceiling.
- **v6** (op-domain sub-ms: 18 cells, roofline, cold): no impl passes all cells; numpy max 6/18 (B=1-only viability), C v6 11/18, **c_ground_up v14 13/18** closing the B=4 frontier; remaining gaps are roofline-bound, not software-bound.
- Cross-node integrity: aggregate flags >10% discrepancies; all champion numbers verified within 3-6% across bench-node-2/3; no unresolved anomalies.

## 2026-08-12 (parent/sonic — novelty push kickoff)
- Provisioned ft-node-1 (2-core Xeon @2.2GHz, 13.6GB); ft-node-2 quota-blocked (TooManyAssignments; brushie sessions occupy slots).
- Baseline on ft-node-1: c_ground_up tiny 35.1us / default 260.9us, err ~1.1e-6 pass. Matches prior cross-node numbers.
- Plan (notes/pipeline.md): v7 cycle attribution + cost model -> int8/bf16 + exp-degree dial -> latency-vs-error Pareto (budgets 1e-3/1e-2) -> close remaining v6 cells (default b4s64, B=16) -> writeup.
- Spawned recursive workers: v7-attribution, int8-kernels, error-pareto (deepseek-v4-pro). Parent heartbeat supervises node runs every 12m.
- Note: colab CLI download has no -f flag (old driver scripts outdated).

## 2026-08-12 — error-pareto: Pareto sweep design + runner + analysis (agent: error-pareto v2)
- DESIGN LOCKED. Levers: exp degree {6,4,3} x GEMM precision {fp32,int8,int8_attn,bf16}
  x 18 v6 cells x budgets {1e-3,1e-2}. Naming table (coordinated with int8-kernels,
  ack pending): libft_<prec>_e<deg>.so / registry c_<prec>_e<deg>; c_fp32_e6 === champion
  (libft.so, unchanged). Build priority: fp32_e4/e3 -> int8_e6/int8_attn_e6 -> bf16_e6
  -> cross terms.
- Deliverables written + smoke-tested: notes/error-pareto.md (design + measured
  baseline + closing requirements), evals/eval_v8.py (parameterized runner,
  --impl/--cfg/--batch/--seq/--budget/--attempt/--smoke; per-attempt JSON
  {max_abs_err, median_us, p99_us, gflops, budget_pass per budget, sub_ms};
  budget_pass is post-hoc arithmetic on err, no per-budget rerun; lazy c_* .so
  registration if int8-kernels' registry entries not yet landed),
  evals/pareto_analysis.py (per-(cell,budget) winner = fastest impl with
  err < budget; min-median across attempts; outputs pareto_<node>.md/.json +
  _frontier_tiny/_frontier_default/_summary PNGs; incremental — tolerates
  missing impls; no-data placeholder path).
- Smoke tests (local, synthetic): runner budget_pass JSON correct on numpy_opt2;
  analysis on 5 fake impls x 4 cells produced correct winners/frontier sets
  (int8_e6 wins 1e-2, int8_attn_e6 wins 1e-3, fp32_e3 on frontier), md/json/pngs
  all emitted; no-data path OK. Matplotlib required for plots (kernel venv has it).
- Measured champion baseline (ft-node-1, evals_v6): 12/18 cells sub-ms at 1e-3
  (err 1.1-2.3e-6 everywhere). Over-sub-ms cells + speedup needed to close:
  default_b4_s32 1086.8us (>=1.09x), tiny_b16_s64 1714.9us (>=1.72x),
  default_b4_s64 2203.3us (>=2.21x), default_b16_s16 2179.6us (>=2.18x),
  default_b16_s32 4532.0us (>=4.53x), default_b16_s64 9624.8us (>=9.62x).
- Prediction: int8 (~2-3x) closes the first four (needs near-ideal int8_attn for
  b4_s64); b16_s32/s64 are beyond int8 at 1e-3 — frontier quantifies the gap
  instead. Target: 15/18 sub-ms at 1e-3, 15-16/18 at 1e-2.
- ASKS parent: (1) run `python evals/eval_v8.py --node ft-node-1` once build.sh
  emits the new .so files (start with --impl c_fp32_e6,c_fp32_e4,c_fp32_e3 if the
  full build isn't ready), (2) download results/, (3) run
  `python evals/pareto_analysis.py --node ft-node-1` (I will re-run/populate on
  each new data drop).
— signed: error-pareto (v2), 2026-08-12

## 2026-08-12/13 (parent/sonic — novelty push, executed)
- v8 Pareto sweep on ft-node-2 (AVX2-only Xeon, runtime-probed: no VNNI): fp32 dominates all 18 cells at 1e-3/1e-2. int8/bf16 never win a cell (slower + less accurate).
- Cost model (evals/cost_model.py, 54 cells, median 3.9% err): us = 23.2 + 3.66e-5*FLOPs + 1.70e-3*B*S^2. Effective GEMM 27.3 GF/s ~ 0.45 peak.
- Attribution (FT_PROFILE build): out-projection = 31% tiny / 49% default of stage time — the next optimization target.
- New impls: c_fp32_e4/e3 (exp dial; e3 wins 4 cells by 1-5% at 1e-3), c_int8_e{6,4,3}, c_int8_attn_e{6,4,3}, c_bf16_e{6,4,3} (all built + correctness-verified; quantized variants are frontier-negative findings).
- Kernel bugs found+fixed by the eval ladder: vpmaddubsw int16 saturation; AVX2 column lane-mixing; int8 cache buffer size-reuse heap corruption; wrapper per-cell weight rebuild churn.
- Workers: error-pareto v2 delivered eval_v8 + pareto_analysis + design (DONE); v7-attribution v1/v2 and int8-kernels v1/v2 died mid-recon — parent executed those workstreams.
- Ops lessons: Colab nodes reaped ~every 30-60 min -> all drivers self-contained (upload+extract+build+run in one exec); colab download has no -f flag.

## 2026-08-13 (parent/sonic — out-projection tuning attempt, NEGATIVE)
- Micro-benchmarked GEMM tile/OMP matrix on ft-node-1 (8 configs x 8 out shapes).
- Implemented FT_OUT_TUNE dispatch (ft_mm_out: serial 4x16 / 4x32+omp rules) + libft_out_tune_e{6,4,3}.so.
- v8 sweep vs same-node champion: loses at most cells (-8..-15%), wins default_b1_s16 (+23%), b16_s16 2x worse.
  Hot-cache micro-bench does not transfer (in-situ B cold). Champion stays; knob kept off for future hardware.
- Also noted: same-impl cross-run noise ~±15% on these nodes (co-tenant contention).
