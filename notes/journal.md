# Fast Transformer Race - Journal

Goal: fastest transformer forward pass on a Google Colab CPU node (sub-millisecond).
Strategy: recursive multi-agent tree, each subtree owns a Colab node, evals escalate
when they saturate, all numbers land in results/, graphs on localhost:9023.

## Timeline
- setup: 3 colab CPU sessions (bench-node-1/2/3), ADC auth, gh=yoits9090, repo fast-transformer (private)
- baseline impls: numpy_naive, numpy_vec (src/impls.py)
- eval v1: correctness vs f64 ref (1e-3 abs) + latency percentiles (evals/)
