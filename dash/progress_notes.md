# Progress Notes - Fast Transformer Race

Live dashboard: http://localhost:9023 (auto-refreshes every 5 s; raw data at /data.json).

## The race
Fastest transformer forward pass on a Google Colab CPU node. **Target: < 1 ms median**
(1000 us). Everything measured there, never on this laptop. All numbers land in
`results/*.json` (gitignored on purpose - benchmark outputs are local artifacts).

## Story so far
- **Setup**: 3 Colab CPU sessions (bench-node-1/2/3) with ADC auth; private repo
  yoits9090/fast-transformer; a background provisioner retries missing sessions.
- **Impls (v1)**: `numpy_naive` (baseline - per-head Python loops) and `numpy_vec`
  (vectorized attention via reshape/transpose + fused QKV matmul).
- **Eval v1**: correctness vs f64 reference (1e-3 abs) + latency percentiles
  (p50/p90/p99, min/max, throughput) over 3000 iters, per node.
- **Dashboard (this page)**: 3x2 graph grid - best latency per impl, latency
  progression over time, percentile spread of the leader, throughput, per-impl
  eval-generation coverage (v1..vN), and distance to the 1 ms target
  (1000 - median_us; above zero = sub-ms). Plus a raw latest-results table.

## What to watch
- `numpy_vec` should beat `numpy_naive` on every node; a win is any impl crossing
  the red 1 ms line.
- Generation count per impl shows how many eval rounds a subtree has pushed.
- Next lever: the C extension in `ground_up/` - expect a step change in the
  distance-to-target panel.

## How numbers flow in
`evals/run_all.py --node bench-node-N` (on a Colab node) -> `results/nodeN_*.json`
-> dashboard picks them up on the next refresh. Journal: `notes/journal.md`.


## Dashboard internals (2026-08-11 upgrade)
- 3x2 grid + raw latest-results table; 5 s auto-refresh; /data.json + /notes.md endpoints.
- Parses all result shapes: v1 (run_all/eval_latency), v2 (seq/batch scaling sweeps),
  v3 (per-config stability/cold-cache). Derived files (leaderboard/aggregate/manifest,
  evals_v[23]_<node>.json summaries) are skipped; malformed JSON is reported, never fatal.
- Plots use *canonical race points* per impl+generation (tiny config, or default at
  S=32/B=1) so sweep points don't distort the race story; the table shows all raw points.
— dashboard agent, 2026-08-11
