# Research: low-level CPU optimization for a sub-millisecond tiny transformer

Target: TINY config (d_model=32, n_heads=2, d_head=16, n_layers=1, d_ff=64, seq_len=16, vocab=256, B=1) on a 2-vCPU Colab CPU node, < 1 ms forward. Written by agent `researcher`.

## Budget math (why these tricks, in this order)

- Whole forward = **0.57 MFLOP** (≈ 283 K MACs). Output projection = **46 %** of MACs, QKV = 17 %, FFN = 23 %. Attention is trivial (≈ 6 %).
- All weights together = **96.8 KB fp32** (emb 32 KB + out 32 KB + qkv/wo 16 KB + FFN 16 KB) → fits in L2; 24 KB as int8 → fits in L1d (32–48 KB).
- Compute floor at a realistic 4–16 GFLOP/s: **35–140 µs**. The 1 ms budget is therefore not about FLOPs — it is about **per-call overhead, memory traffic, and threading pathology**.
- numpy_vec has ~25–30 numpy calls; at ~1–5 µs dispatch each that is **30–150 µs of pure overhead**, plus OpenBLAS thread-pool overhead on tiny GEMMs (documented below) and temporary copies for transposes/masks.

---

## TOP-5 recommended experiments (priority order)

1. **Fused single-pass forward kernel in C (ctypes/cffi), or numba `njit(fastmath=True)` fallback.** One function call, all weights static/global, loop order tuned, `-O3 -march=native -ffast-math`. Subsumes items 3–5. Expected: **3–10× vs numpy_vec** (overhead 30–150 µs → ~0; threading pathology gone; everything L1/L2-resident). Code sketch in §1. Numba version: decorate the whole forward with `@njit(fastmath=True, nogil=True, cache=True)`, pass weights once (typed dict or flattened arrays), keep attention triangular. Measure first on the node with `-O2` vs `-O3 -march=native -ffast-math`.
2. **Numpy fast-path, zero C:** `threadpoolctl.limit(limits=1, user_api='blas')` (or `OPENBLAS_NUM_THREADS=1`), pre-concatenate QKV weights once, fold `1/sqrt(dh)` into `Wq`, pre-transpose weights so attention needs no `swapaxes` (store `K` transposed), replace `np.where(mask,…)` with `att *= mask` (+`-1e9`), call `scipy.linalg.blas.sgemm` directly instead of `np.matmul` for the hot 2-D GEMMs, avoid 3-D `np.matmul` (reshape to 2-D batches). Expected: **2–5× vs numpy_vec**, ~30 min of work, no toolchain risk. §2.
3. **2-thread row-sliced GEMV/GEMM for the output head (46 % of FLOPs) — only if the node has 2 physical cores.** Split vocab 512 into 2×256 rows; each thread reads its own half of W (no extra bandwidth), joins once. Expected **1.3–1.6× on the head** (≈ 1.15–1.25× end-to-end) if 2 physical cores; **use 1 thread if the node is 2 hyperthreads on 1 core** (SIMD-unit contention, L1 thrash — measured ~0.9× in that case). Use OpenMP in C or `numba.prange`; never spawn threads per call. §3.
4. **Static-weight int16/int8 quantization for head + FFN GEMMs.** int8 weights (24 KB, L1-resident) with AVX2 `vpmaddubsw` dot products = 32 int8 MACs/cycle vs 8 fp32 MACs/cycle; AVX-512 VNNI = 128/cycle if the node has `avx512_vnni` (probe `/proc/cpuinfo`). **Correctness caveat: int8 step ≈ 1/127 ≈ 0.008 on logits O(1–10) likely violates the 1e-3 abs tolerance — first measure actual error vs the f64 reference; use int16 (error ~3e-5, still 2× bytes) if int8 fails.** Only the head and FFN are big enough to matter. §4.
5. **Fused causal attention + fast exp/rsqrt + scale folding.** Fused exp–sum softmax (llama.cpp pattern), fast polynomial `exp` (rel err ~1e-7, tolerance-safe), triangular loop that computes only the lower-triangle of scores (skip ~half of S²=256 work), fold `1/sqrt(dh)` into `Wq` at load. In numpy alone this is worth ~1.5–2× on the attention+LN slice; in the fused kernel it is basically free. §5.

Further experiments (after TOP-5): JAX `jax.jit` whole-model fusion on CPU (XLA emits one fused, vectorized program — §6); oneDNN BRGeMM JIT microkernels (heavy dependency, uncertain win at these sizes — §6); LUT-GEMV (§4.4); weight packing to microkernel layout + 64-byte alignment + `prefetchnta` (§1.3).

---

## 1. Fused single-pass kernel (weight-stationary microkernel)

**What.** BLIS/OpenBLAS/GotoBLAS structure: 6 nested loops — 5th/4th/3rd block for cache (L3/L2/L1), 2 inner loops iterate a register-blocked **microkernel** that keeps a small C-tile (e.g. 16×6 for AVX2, 6×2 in llama.cpp's `simd_gemm_ukernel`) in registers and streams A/B panels with FMA. For our sizes the whole model is one "L1 block", so the microkernel *is* the algorithm: load weights once, stream activations.

**Why it matters.** numpy + OpenBLAS pay ~1–5 µs dispatch per op and suffer thread-pool sync on tiny GEMMs (OpenBLAS issue #731: 2× slower wall, 8× CPU time on small SVDs with default threading). A single C call eliminates all of it; weights stay L1/L2-resident across the whole forward.

**How to apply.** C extension exposed via ctypes (`cffi` also fine). Sketch (per layer):

```c
// compile: gcc -O3 -march=native -ffast-math -fno-math-errno -fopenmp -shared -fPIC
// Layout: all weights pre-packed at load into a struct (row-major, 64-byte aligned, concat QKV).
static const float Wqkv[3][32][32], Wo[32][32], W1[32][64], W2[64][32];
static float E[256][32], Wout[512][32];   // global/static => cache-resident

// microkernel: C[RM x RN] += A[RM x K] * B[K x RN], RM=6, RN=2 for AVX2 (llama.cpp simd-gemm.h)
for (int k = 0; k < K; k++) {
    __m256 Bv[2] = { load(B + k*N + 0), load(B + k*N + 8) };
    for (int i = 0; i < RM; i++) {
        __m256 a = set1(A[i*K + k]);
        acc[i][0] = fmadd(a, Bv[0], acc[i][0]);
        acc[i][1] = fmadd(a, Bv[1], acc[i][1]);
    }
}
// GEMV (head, FFN): y[n] = sum_k x[k]*W[k][n] — row-slice over n across threads (see §3).
// Attention: triangular loop, fused exp-sum softmax (§5), scale folded into Q at load.
```

**Sources.**
- BLIS 6-loop + packing: https://salykova.github.io/matmul-cpu (tutorial with code) and repo https://github.com/salykova/sgemm.c (OpenMP, 16×6 AVX2 kernel, ~BLAS-competitive)
- GEMMFIP/BLIS unification paper: https://arxiv.org/abs/2302.08417
- llama.cpp register-blocked ukernel: https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-cpu/simd-gemm.h (RM=6, RN=2 on AVX2)
- llama2.c proves the simple GEMV + OpenMP baseline: https://github.com/karpathy/llama2.c/blob/master/run.c (`matmul()` is "by far the most amount of time")

### 1.3 Compiler flags & alignment
- `-O3 -march=native -ffast-math -fno-math-errno -funsafe-math-optimizations` — ~2× over plain `-O2` for this kind of code (fast-math unlocks FMA contraction, no errno checks on expf). Verify against the 1e-3 tolerance; safe for our values (see §5.2).
- Align weight arrays to 64 B; use aligned loads when possible (`__builtin_assume_aligned` / `_mm256_load_ps`).
- Numba equivalent: `@njit(fastmath=True, nogil=True, cache=True)` — same benefits, zero C toolchain. Known caveat: numba must be installed on the node (`pip install numba`), first call JIT-compiles (~1 s, do it in a warmup).
- "Less Slow C++" benchmark suite for micro-op intuition (compiler flags, tiny-GEMM shape penalties): https://github.com/ashvardanian/less_slow.cpp

## 2. Numpy fast-path (no C)

**What.** The minimum-overhead numpy recipe. Order of ops matters; every temporary costs a pass over ≤ 2 KB (activations) but each numpy *call* costs ~1–5 µs of dispatch.

**How to apply (all at load time except the forward itself):**
```python
import numpy as np
from threadpoolctl import threadpool_limits
with threadpool_limits(limits=1, user_api='blas'):   # or env OPENBLAS_NUM_THREADS=1
    pass
# load-time prep:
Wqkv = np.concatenate([Wq/np.sqrt(dh), Wk, Wv], axis=1)   # fold softmax temp into Q
Kt   = Wk.T.copy()  # so att = Q @ Kt needs no swapaxes/copy in the hot loop
mask = np.tril(np.ones((S,S), bool))                       # reuse, don't rebuild
# forward: avoid np.where -> att = att*mask - 1e9*(~mask) (or att -= 1e9*(~mask))
# prefer scipy.linalg.blas.sgemm(alpha, a, b) for 2-D GEMMs; np.matmul on 3-D
# (B,S,d)@(d,d) can take the non-BLAS path in some numpy builds (numpy issue #7569)
```
- `np.matmul`/`np.dot` on tiny contiguous 2-D arrays → OpenBLAS `sgemm`; 3-D `matmul` may not dispatch to BLAS (numpy#7569) — reshape (B*S, d) and matmul 2-D, then reshape.
- `np.einsum` is usually *slower* than `matmul` for 2-operand contractions (dispatch overhead; SO 67144036) — use `matmul`/`tensordot`.
- Transposes: `K.swapaxes(-1,-2)` returns a view; numpy may copy it inside BLAS dispatch. Pre-transposing weights at load (above) removes all runtime transposes. `ascontiguousarray` costs are hidden copies — avoid.
- Single-threaded OpenBLAS is *mandatory* for tiny GEMMs: OpenBLAS issue #731 shows default threading making small-matrix workloads **>2× slower wall, 8× CPU**. 2 threads only help the head GEMM once it is big (512×32) — measure, don't assume.

**Sources.** https://github.com/OpenMathLib/OpenBLAS/issues/731 · https://github.com/numpy/numpy/issues/7569 · https://stackoverflow.com/questions/67144036/performance-difference-between-einsum-and-matmul · https://github.com/joblib/threadpoolctl (README: per-thread BLAS/OpenMP pool control) · OpenBLAS threading FAQ: http://www.openmathlib.org/OpenBLAS/docs/faq/

## 3. Threading model for 2 vCPUs

**What.** "2 vCPU" is usually either 2 hyperthreads on 1 physical core or 2 physical cores. This changes the optimal thread count completely.
- 2 hyperthreads on 1 core: SIMD-unit contention + L1 thrash → 1 worker thread often beats 2 (measured ~0.9× with 2).
- 2 physical cores: row-slice the GEMV over output rows (each thread reads its own half of W) → realistic **1.3–1.6×** (memory bandwidth shared), not 2×.
- Persistent thread pool required: OpenMP/`numba.prange` reuse threads across calls; per-call `threading.Thread` spawn (~50–100 µs) would blow the budget.
- llama.cpp 2-thread measurements (Xeon 8375C 1.59×, EPYC 7R13 1.55×, RPi5 1.5×) in the compendium below.

**How to apply.** Probe the node first: `lscpu`/`/proc/cpuinfo` (`siblings` vs `cpu cores`, `core id`); decide 1 vs 2 threads; then in C `#pragma omp parallel for` over head rows (or `collapse(2)` over both inner loops of the GEMM microkernel for FFN/head tiles — salykova). In numba: `prange` on the head output. Never spawn threads inside the timed region; warm up the pool before timing.

**Sources.** https://blog.victorbona.dev/compendium/cpu-llm-inference/compute-kernels (SIMD throughput table + 2-vCPU threading model; AVX2 32 int8 ops/cycle vs 8 fp32; VNNI 128/cycle; AMX 1024/cycle) · https://salykova.github.io/matmul-cpu §9 (OpenMP `collapse(2)`, parallel packing)

## 4. Quantization of static weights (int8/int16)

**What.** GEMV is memory-bound: per token you read the whole weight matrix once. int8 divides bytes by 4 (97 KB → 24 KB, L1-resident), and AVX2 `vpmaddubsw` does 32 int8 MACs/cycle vs 8 fp32 FMA/cycle. AVX-512 VNNI (`vpdpbusd`) does 128/cycle — probe `avx512_vnni` in `/proc/cpuinfo` before writing intrinsics; on Colab's Xeon/Epyc nodes it may or may not exist.

**Kernel pattern (llama.cpp `vec_dot_q8_0_q8_0`, x86 AVX2):** per 32-value block: load int8 x, y; `_mm256_maddubs_epi16` (abs/sign trick for signed×signed) → int16 partials; `_mm256_madd_epi16` → int32; convert, multiply by per-block fp16 scale, FMA-accumulate.

**Correctness risk (read first).** Per-matrix int8 quantization step ≈ range/255. If logits/activations are O(1–10), step ≈ 0.008–0.08 vs the 1e-3 abs tolerance → **likely fails**. Two safe variants:
- **int16 weights** (step ≈ 3e-5, 2× bytes, `_mm256_madd_epi16` = 16 int16 MACs/cycle): safe accuracy, still a bytes+ops win vs fp32.
- int8 with per-row scales + fp32 head (only FFN/QKV int8; keep the 512-wide output head fp32 or int16).
Measure error vs the f64 reference on real weights before committing. Only head + FFN are big enough to matter (46 % + 23 % of MACs).

**Sources.** llama.cpp vec_dot: https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-cpu/arch/x86/quants.c · ONNX Runtime quantization (VNNI note): https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html · CMSIS-NN int8 FC reference (per-tensor quant, bit-exact with TFLM): https://github.com/ARM-software/CMSIS-NN (Source/FullyConnectedFunctions/arm_fully_connected_s8.c) · INT-FlashAttention (int8 attention, accuracy analysis): https://arxiv.org/abs/2409.16997 · GEMV memory-bound analysis (arithmetic intensity ~2 ops/byte at Q4): https://vijayprabhas9.github.io/gemv_optimization/ (GPU, but the memory-bound reasoning transfers) · LUT-GEMV (arbitrary low-bit GEMV via lookup tables; hardware paper but the tiling idea is portable): https://arxiv.org/abs/2509.25853

## 5. Attention / softmax / normalization micro-tricks

### 5.1 Fused exp–sum softmax
**What.** Single loop: `v = exp(x[i]-max); sum += v; y[i] = v` — one pass over the row, no temporaries. llama.cpp `ggml_vec_soft_max_f32` does exactly this with AVX2 (`ggml_v_expf` = Schraudolph-style: multiply by log2(e), split integer/fraction, degree-4 polynomial, scale by `2^e` via int add; max rel error ≈ 2e-7 → safe for 1e-3 tolerance).
**Apply:** in C use the same trick (or just `expf` + `-ffast-math`); in numpy, `softmax = e / e.sum(-1, keepdims=True)` is fine but costs 3 temporaries — acceptable, the fused kernel removes them.

### 5.2 Causal attention = triangular loop
S=16 → full scores matrix is 256 entries, causal is 136. In the fused kernel, loop `for i in range(S): for j in range(i+1):` and skip the mask entirely; also skip the `softmax` max-subtraction normalization cost by folding `1/sqrt(dh)` into Q (free at load). In numpy: replace `np.where(mask, att, -1e9)` with `att += mask_float` where `mask_float = where(mask, 0, -1e9)` precomputed once — removes a `np.tril` rebuild + `np.where` per layer.

### 5.3 LayerNorm micro-optimizations
- One-pass (Welford) mean/var in C; `rsqrt = _mm256_rsqrt_ps` is 12-bit (rel err 1.5e-4 — fine at 1e-3 tolerance) — add one Newton step (`rsqrt * (1.5 - 0.5*x*rsqrt²)`) to be safe.
- LN→GEMM folding: `LN(x)@W = (x*s)@W - mu*(s@W) + b@W` with `s = g/sigma`. In a fused kernel this saves nothing (LN is 2 KB); in numpy it replaces ~6 ops with ~3 — only worth it if you stay in numpy.
- ReLU: `max(h2, 0)` is free in C.

**Sources.** https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-cpu/vec.cpp (fused softmax + fast exp) · https://www.aussieai.com/book/ch25-vectorized-softmax-avx (AVX softmax walkthrough) · https://arxiv.org/abs/1805.02867 (online/streaming softmax normalizer) · llama.cpp flash-attention op (chunked fused QK^T+softmax+V, i.e., flash attention on CPU): https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-cpu/ops.cpp (`ggml_compute_forward_flash_attn_ext_f16_one_chunk`)

## 6. Framework-level options (worth one probe each, not the main bet)

- **JAX/XLA CPU:** `jax.jit` fuses the whole forward into one XLA program with LLVM vectorization; can beat numpy badly on op-count-dominated workloads. Cost: ~100 MB install, compile-on-first-call. Try `jax.jit(forward)` with `x64=False`, compare with numpy fast-path. (XLA CPU is the same LLVM codegen family as -O3 C; expect roughly C-level numbers minus layout surprises.) See https://apxml.com/courses/advanced-jax/chapter-2-optimizing-jax-code-performance/fusion-operator-optimization and https://lilianweng.github.io/posts/2023-01-10-inference-optimization/
- **oneDNN BRGeMM / oneMKL JIT:** Intel's answer to small-GEMM dispatch overhead — oneMKL 2024 JIT-generates specialized kernels for m,n,k ≤ 16 (and any size with one dim < 128), `MKL_DIRECT_CALL_JIT` macro; BRGeMM is the JIT microkernel family under PyTorch/llama.cpp. Real gains documented for *many repeated small GEMMs* — exactly our case, but pip-installing MKL/oneDNN on the node + wiring ctypes is heavy. Do this only if the fused C kernel is slower than hoped.
- **XNNPACK:** Google's CPU inference primitives (fp32/int8/fp16, weight caching, ARM/x86) — battle-tested but oriented to conv/FC graphs; adding it as a dependency is more work than our own 100-line kernel.
- **MatMul-free LLMs** (ternary/bitwise layers): changes the architecture — **not usable** (must match reference outputs), but the paper's core claim (matmuls dominate) is exactly why overhead-elimination beats algorithmic tricks here.

**Sources.** https://www.intel.com/content/www/us/en/developer/articles/technical/onemkl-improved-small-matrix-performance-using-just-in-time-jit-code.html (oneMKL JIT; archived copy) · https://deepwiki.com/uxlfoundation/oneDNN/3.1-brgemm-matmul-primitive and https://bhattarai-b.github.io/posts/brgemm/ (BRGeMM deep dive; also naive→tiled→AMX GEMM evolution) · https://github.com/google/XNNPACK · https://arxiv.org/abs/2406.02528 (Scalable MatMul-free LM) · https://github.com/typohnebild/numpy-vs-mir (numpy overhead side note)

## 7. Sub-millisecond engineering (TinyML references)

Sub-ms inference is a solved problem at TinyML scale — the same weight-stationary/int8 playbook applies:
- CMSIS-NN (Arm Cortex-M): int8 fully-connected kernels, per-tensor quantization, TFLM bit-exact; the reference implementation of tiny weight-stationary FC.
- BlazeFace: 200×200 face detection in < 1 ms on mobile GPUs — design choices: everything is small, fused, memory-resident.
- llama.cpp decode phase: 1 token = 1 GEMV against the whole weight matrix = memory-bound; the entire tiny-model optimization space is about bytes-touched per forward, which is why int8/weight layout/threading dominate over algorithm changes.

**Sources.** https://github.com/ARM-software/CMSIS-NN · https://arxiv.org/abs/1907.05047 (BlazeFace) · https://github.com/ggerganov/llama.cpp

---

## Impact ranking for TINY on 2-vCPU Colab (expected end-to-end)

| # | Technique | Expected gain vs numpy_vec | Effort | Risk |
|---|-----------|---------------------------|--------|------|
| 1 | Fused C/numba single-pass forward | 3–10× | Medium (C) / Low (numba) | Low (correctness = same math) |
| 2 | Numpy fast-path (threads=1, folds, BLAS direct) | 2–5× | Low | Low |
| 3 | 2-thread row-sliced head GEMV (physical cores only) | 1.15–1.25× end-to-end | Low | Medium (hyperthread trap) |
| 4 | int16/int8 static weights (head+FFN) | 1.3–2× (int8 if tolerance allows) | Medium | High (tolerance) |
| 5 | Fused causal attention + fast exp/rsqrt + folds | 1.5–2× on LN/attn slice | Low | Low |

Combined expectation for the winning impl: **well under 300 µs, plausibly 50–150 µs**, i.e., the < 1 ms target is comfortable if overhead is eliminated.
