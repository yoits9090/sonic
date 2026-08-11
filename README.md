# fast-transformer
Race: fastest transformer forward pass on a Google Colab CPU node (target < 1 ms).

- `src/` — implementations (numpy, C extension in `ground_up/`)
- `evals/` — correctness + latency evals (escalating registry)
- `results/` — per-node JSON benchmark outputs
- `dash/server.py` — progress graphs on http://localhost:9023
- `notes/` — research + journal
- `colab/` — Colab CLI helpers (probe, session creation, run scripts)
