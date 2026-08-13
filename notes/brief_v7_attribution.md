# Brief: v7-attribution (cycle-level attribution + tiny-roofline cost model)

You are the `v7-attribution` agent. Repo: /Users/ace/projects/sonic/fast-transformer.
Read notes/pipeline.md first. NEVER run `colab` commands — parent owns the Colab node.
You produce code/scripts; the parent runs them on ft-node-1. Reply to parent via
`await agent_message.send(..., receiver_role='parent')` when done or blocked.

## Context
Champion c_ground_up sits at ~0.43 of the node's GEMM roofline. The missing 0.57
(non-GEMM ops, memory movement, threading, ctypes boundary, softmax/LN) has never
been measured. Goal: attribute every microsecond of the forward pass, then fit a
predictive cost model over all 18 v6 cells.

## Tasks
1. **Profile build.** Read ground_up/ft.h, matmul.c, transformer.c. Add a profile
   mode (e.g. `-DFT_PROFILE=1`): time each op category (QKV matmul, attention
   scores matmul, softmax+exp, attention×V matmul, layernorm(s), FFN matmuls,
   ctypes/py overhead via python side, total) using rdtsc on x86-64
   (`__builtin_ia32_rdtsc()`, guard `#if defined(__x86_64__)`) with median over
   many iters. Print a per-op attribution table for tiny+default and B=1/4,
   S=16/32/64. Keep champion fp32 path unchanged when FT_PROFILE is off.
2. **eval_v7.** Write evals/eval_v7.py: runs the profile build (parent will exec
   it on the node), collects the attribution table + per-cell latency grid
   (cfg x B x S, the v6 cell set), writes JSON to results/. Also write
   colab/run_v7.sh (bash) that the parent executes on ft-node-1: uploads
   changed ground_up/ + evals/, rebuilds, runs eval_v7.py, downloads results.
   (Parent runs the script; you just write it correctly.)
3. **Cost model.** Fit latency = c0 + c1*FLOPs + c2*bytes + c3*(BS^2 attention
   term) + overhead(d,L,S,B,threads) on existing results/*.json (250 files, v1-v6)
   + any new v7 data. Use pandas/numpy locally. Validate on held-out cells;
   report coefficients, residuals, and which cells the model misfits.
4. **Write notes/v7-attribution.md**: methodology, attribution table, model
   fit, and what it predicts about int8 (i.e., where compute-bound cells gain
   2-3x vs where they don't).

## Deliverables (all in repo)
- ground_up/*.c profile hooks (FT_PROFILE), evals/eval_v7.py, colab/run_v7.sh,
  evals/cost_model.py, notes/v7-attribution.md, signed journal entry.
## Constraint
Compile safety on node gcc only matters for x86-64; keep any SIMD behind guards.
