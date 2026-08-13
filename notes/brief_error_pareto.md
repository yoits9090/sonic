# Brief: error-pareto (latency-vs-error frontier sweep design + analysis)

You are the `error-pareto` agent. Repo: /Users/ace/projects/sonic/fast-transformer.
Read notes/pipeline.md first. NEVER run `colab` commands — parent owns the node.
Reply to parent via `await agent_message.send(..., receiver_role='parent')`.

## Goal
The correctness bar (max_abs_err < 1e-3 vs f64) is a dial, not a gate. Produce the
latency-vs-error Pareto frontier: for each cell, the fastest implementation that
passes a given error budget. This is the "novel at the intersection" artifact.

## Tasks
1. **Sweep design.** Levers: exp degree {3,4,6} x precision {fp32, int8, bf16} x
   cells (cfg{tiny,default} x B{1,4,16} x S{16,32,64}) x budgets {1e-3, 1e-2}.
   Coordinate exact impl names/flag names with the int8-kernels agent's brief
   (notes/brief_int8_kernels.md); keep a shared naming table in your notes file.
2. **eval_v8.** Write evals/eval_v8.py: parameterized runner (--impl/--cfg/
   --batch/--seq/--budget) producing {latency_us, max_abs_err, budget_pass}
   JSON per cell; reuse eval_v6.py + registry.py patterns.
3. **Analysis.** evals/pareto_analysis.py: per-cell frontier table (fastest
   passing impl per budget), matplotlib Pareto plots (latency vs err, log axes),
   summary markdown. Populate incrementally from results/ as parent supplies data.
4. **Notes.** notes/error-pareto.md: design rationale, expected outcomes
   (from int8-kernels' predicted errors), what "closing v6 cells" requires
   (B=16 cells need >= ~2x GEMM speedup at <=1e-3 err).

## Deliverables
- evals/eval_v8.py, evals/pareto_analysis.py, notes/error-pareto.md,
  signed journal entry.
