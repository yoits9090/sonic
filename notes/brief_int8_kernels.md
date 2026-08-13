# Brief: int8-kernels (int8/bf16 GEMM + fast-exp degree dial)

You are the `int8-kernels` agent. Repo: /Users/ace/projects/sonic/fast-transformer.
Read notes/pipeline.md first. NEVER run `colab` commands — parent owns the node.
Reply to parent via `await agent_message.send(..., receiver_role='parent')`.

## Goal
Break the fp32 compute floor. B=16 cells and default-b4s64 are roofline-bound in
fp32; int8 GEMM (AVX2 vpmaddubsw) is ~2-3x fp32 and is the lever. Also make the
Chebyshev fast-exp degree a compile-time dial for the error-budget Pareto work.

## Tasks
1. **Study.** ground_up/matmul.c (kernel variants via FT_KERNEL, 4x16 tiles,
   FT_TILE16, FT_ATTN_BLOCK), ground_up/transformer.c (fused QKV/attention/FFN,
   FT_FASTEXP Chebyshev), ground_up/ft.h, ground_up/build.sh flag matrix.
2. **int8 path.** Quantize weights per-output-channel (scales in fp32), activations
   per-row or global; int8 x int8 -> int32 accumulate -> dequant to fp32.
   Two implementations: (a) portable scalar C for correctness (compiles anywhere),
   (b) AVX2 vpmaddubsw + vpmaddwd path behind `#if defined(__x86_64__) && defined(__AVX2__)`
   with `-mavx2` on the node build. Wire via new build flags (e.g. -DFT_INT8=1)
   and new .so variants in build.sh: libft_int8.so (int8 GEMM everywhere),
   libft_int8_attn.so (int8 only for attention-score and V matmuls), etc.
3. **bf16 (optional but valued).** fp32->bf16 truncation path, accumulate fp32:
   -DFT_BF16=1, libft_bf16.so. Cheap to add; quant error ~1e-2 alone so likely a
   Pareto-only point.
4. **exp dial.** -DFT_EXP_DEG=n (3/4/6, default 6) controlling Chebyshev degree;
   build libft_exp3.so, libft_exp4.so.
5. **Error prediction locally.** In pure numpy (use src/impls.py + src/random_state.py
   for identical seeds), replicate each path's quantization error and predict
   max_abs_err on tiny+default; write the expected table vs budgets 1e-3 / 1e-2.
6. **Registration + runner.** Register new impls in src/impls.py (ctypes wrappers,
   names like c_int8); write colab/run_int8.sh (upload changed files, build on
   node, run correctness+latency for the new variants, download results). Parent
   executes it. Do NOT regress the fp32 champion (keep its path bit-identical
   when new flags are off).

## Deliverables
- modified ground_up/*.c + ft.h + build.sh, src/impls.py additions,
  colab/run_int8.sh, notes/int8-kernels.md (design + predicted-vs-measured error
  table), signed journal entry.
