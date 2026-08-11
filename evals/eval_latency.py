"""Latency eval: warmup + N iters, report percentiles in microseconds.
Usage: python evals/eval_latency.py --impl numpy_vec --node bench-node-1 --out results/node1.json
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DEFAULT, TINY
from src.random_state import RNG, make_input
from src.impls import IMPLS

def bench(fn, W, x, cfg, iters=2000, warmup=200):
    fn(W, x, cfg)  # compile/alloc warmup
    for _ in range(warmup):
        fn(W, x, cfg)
    times = np.empty(iters)
    for i in range(iters):
        t0 = time.perf_counter_ns()
        fn(W, x, cfg)
        times[i] = (time.perf_counter_ns() - t0) / 1000.0  # us
    return times

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--cfg", default="tiny", choices=["tiny", "default"])
    ap.add_argument("--node", default="local")
    ap.add_argument("--out", default=None)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--extra-cfgs", action="store_true")
    a = ap.parse_args()
    cfg = TINY if a.cfg == "tiny" else DEFAULT
    W = RNG(0).weights(cfg)
    x = make_input(cfg)
    fn = IMPLS[a.impl]
    t = bench(fn, W, x, cfg, a.iters, a.warmup)
    res = {
        "impl": a.impl, "cfg": cfg.to_dict(), "node": a.node,
        "iters": a.iters, "numpy_version": np.__version__,
        "median_us": float(np.median(t)), "mean_us": float(t.mean()),
        "p50_us": float(np.percentile(t, 50)), "p90_us": float(np.percentile(t, 90)),
        "p99_us": float(np.percentile(t, 99)), "min_us": float(t.min()),
        "max_us": float(t.max()),
        "throughput_tok_s": float(cfg.seq_len / (np.median(t) / 1e6)),
        "sub_ms": bool(np.median(t) < 1000.0),
    }
    print(json.dumps(res))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
