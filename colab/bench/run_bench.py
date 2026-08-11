"""Node-side bench runner (numpy-optimizer agent).
Env config: IMPLS (comma-sep), CFGS (comma-sep tiny|default), ITERS, WARMUP, NODE.
Reads src/impls.py + src/config.py + src/random_state.py from ./src (uploaded).
Prints one line per (impl,cfg): "CORR {json}" then "LAT {json}".
"""
import os, sys, json, time
_HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, _HERE)
_d = _HERE
for _ in range(4):  # walk up to find repo root (dir containing src/impls.py)
    if os.path.isfile(os.path.join(_d, "src", "impls.py")):
        sys.path.insert(0, _d)
        break
    _d = os.path.dirname(_d)
import numpy as np
from src.config import DEFAULT, TINY
from src.random_state import RNG, make_input
from src.impls import IMPLS

def reference(W64, x, cfg):
    """float64 reference (mirror of evals/eval_correctness.py)."""
    B, S = x.shape
    h = W64["emb"][x]
    n_heads, dh = cfg.n_heads, cfg.d_head
    for i in range(cfg.n_layers):
        mu = h.mean(-1, keepdims=True); var = h.var(-1, keepdims=True)
        h2 = (h - mu) / np.sqrt(var + cfg.eps) * W64[f"ln1_g{i}"] + W64[f"ln1_b{i}"]
        Q = h2 @ W64[f"wq{i}"]; K = h2 @ W64[f"wk{i}"]; V = h2 @ W64[f"wv{i}"]
        ctx = np.zeros_like(h2)
        for hh in range(n_heads):
            q = Q[:, :, hh*dh:(hh+1)*dh]; k = K[:, :, hh*dh:(hh+1)*dh]; v = V[:, :, hh*dh:(hh+1)*dh]
            att = q @ k.swapaxes(-1, -2) / np.sqrt(dh)
            mask = np.tril(np.ones((S, S), dtype=bool))
            att = np.where(mask, att, -1e9)
            e = np.exp(att - att.max(-1, keepdims=True))
            att = e / e.sum(-1, keepdims=True)
            ctx[:, :, hh*dh:(hh+1)*dh] = att @ v
        h = h + ctx @ W64[f"wo{i}"]
        mu = h.mean(-1, keepdims=True); var = h.var(-1, keepdims=True)
        h2 = (h - mu) / np.sqrt(var + cfg.eps) * W64[f"ln2_g{i}"] + W64[f"ln2_b{i}"]
        h = h + (np.maximum(h2, 0) @ W64[f"w1{i}"]) @ W64[f"w2{i}"]
    mu = h.mean(-1, keepdims=True); var = h.var(-1, keepdims=True)
    h = (h - mu) / np.sqrt(var + cfg.eps) * W64["lnf_g"] + W64["lnf_b"]
    return h @ W64["out"]

def bench(fn, W, x, cfg, iters, warmup):
    fn(W, x, cfg)
    for _ in range(warmup):
        fn(W, x, cfg)
    times = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        t0 = time.perf_counter_ns()
        fn(W, x, cfg)
        times[i] = (time.perf_counter_ns() - t0) / 1000.0
    return times

def run_one(impl, cfgname, iters, warmup, node):
    cfg = TINY if cfgname == "tiny" else DEFAULT
    W64 = RNG(0).weights(cfg, dtype=np.float64)
    W32 = RNG(0).weights(cfg, dtype=np.float32)
    x = make_input(cfg)
    ref = reference(W64, x, cfg)
    out = IMPLS[impl](W32, x, cfg).astype(np.float64)
    abs_err = float(np.abs(out - ref).max())
    corr = {"impl": impl, "cfg": cfgname, "node": node, "max_abs_err": abs_err,
            "pass": bool(abs_err < 1e-3)}
    print("CORR " + json.dumps(corr), flush=True)
    t = bench(IMPLS[impl], W32, x, cfg, iters, warmup)
    res = {
        "impl": impl, "cfg": cfgname, "node": node,
        "iters": iters, "numpy_version": np.__version__,
        "median_us": float(np.median(t)), "mean_us": float(t.mean()),
        "p90_us": float(np.percentile(t, 90)), "p99_us": float(np.percentile(t, 99)),
        "min_us": float(t.min()),
        "throughput_tok_s": float(cfg.seq_len / (np.median(t) / 1e6)),
        "sub_ms": bool(np.median(t) < 1000.0),
    }
    print("LAT " + json.dumps(res), flush=True)

def main():
    impls = [s.strip() for s in os.environ.get("IMPLS", "numpy_opt4").split(",") if s.strip()]
    cfgs = [s.strip() for s in os.environ.get("CFGS", "tiny").split(",") if s.strip()]
    iters = int(os.environ.get("ITERS", "3000"))
    warmup = int(os.environ.get("WARMUP", "300"))
    node = os.environ.get("NODE", "bench-node-1")
    for impl in impls:
        for cfgname in cfgs:
            run_one(impl, cfgname, iters, warmup, node)

if __name__ == "__main__":
    main()
