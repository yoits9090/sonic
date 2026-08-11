"""Eval v4: adversarial inputs — edge-case seq_len, token-distribution extremes, multi-token bursts.
Writes per-impl attempt files results/evals_v4_<node>_<impl>_a<attempt>.json + summary results/evals_v4_<node>.json.
Usage: python evals/eval_v4.py --node bench-node-3 [--impl numpy_vec] [--attempt 1] [--smoke]
"""
import argparse, dataclasses, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DEFAULT, TINY
from src.random_state import RNG
from src.impls import IMPLS

ABS_TOL = 1e-3
EDGE_SEQ = [1, 2, 16, 32, 64, 256]   # incl. d_head=16, d_model=64, S>d_model


def reference(W64, x, cfg):
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


def make_adversarial(cfg, kind, seed, batch=1):
    """kinds: uniform | skewed | same | edges | bursty. Returns int64 (B,S) token ids."""
    g = np.random.default_rng(seed)
    V = cfg.vocab
    if kind == "uniform":
        return g.integers(0, V, (batch, cfg.seq_len), dtype=np.int64)
    if kind == "skewed":          # 90% of tokens from 10% of vocab
        small = g.integers(0, V // 10, (batch, cfg.seq_len), dtype=np.int64)
        mask = g.random((batch, cfg.seq_len)) < 0.1
        big = g.integers(V // 10, V, (batch, cfg.seq_len), dtype=np.int64)
        return np.where(mask, big, small)
    if kind == "same":            # every token identical
        t = g.integers(0, V, (1, 1), dtype=np.int64)[0, 0]
        return np.full((batch, cfg.seq_len), t, dtype=np.int64)
    if kind == "edges":           # only token 0 and V-1
        return np.where(g.random((batch, cfg.seq_len)) < 0.5, 0, V - 1).astype(np.int64)
    if kind == "bursty":          # runs of the same token (multi-token bursts)
        x = np.empty((batch, cfg.seq_len), dtype=np.int64)
        for b in range(batch):
            i = 0
            while i < cfg.seq_len:
                t = g.integers(0, V)
                ln = int(g.integers(1, 9))
                x[b, i:min(i + ln, cfg.seq_len)] = t
                i += ln
        return x
    raise ValueError(kind)


def bench(fn, W, x, cfg, iters=200, warmup=20):
    for _ in range(warmup):
        fn(W, x, cfg)
    times = np.empty(iters)
    for i in range(iters):
        t0 = time.perf_counter_ns()
        fn(W, x, cfg)
        times[i] = (time.perf_counter_ns() - t0) / 1000.0
    return times


def run_impl(name, node, attempt, out_dir, smoke=False):
    import warnings as _w
    _w.simplefilter("ignore", RuntimeWarning)
    fn = IMPLS[name]
    res = {"impl": name, "node": node, "generation": "v4", "attempt": attempt,
           "numpy_version": np.__version__, "criteria": {"abs_tol": ABS_TOL},
           "edge_seq": {}, "distributions": {}}
    iters = 8 if smoke else 200
    # 1) edge-case seq lengths (DEFAULT cfg shape, batch=1)
    for sl in EDGE_SEQ:
        cfg = dataclasses.replace(DEFAULT, seq_len=sl)
        W64 = RNG(0).weights(cfg, dtype=np.float64)
        W32 = RNG(0).weights(cfg, dtype=np.float32)
        x = make_adversarial(cfg, "uniform", seed=7)
        ref = reference(W64, x, cfg)
        out = fn(W32, x, cfg).astype(np.float64)
        err = float(np.abs(out - ref).max()) if out.shape == ref.shape else float("inf")
        t = bench(fn, W32, x, cfg, iters, 5 if smoke else 20)
        res["edge_seq"][str(sl)] = {"correct": bool(err < ABS_TOL), "max_abs_err": err,
                                    "median_us": float(np.median(t))}
        print(f"  seq={sl}: correct={err < ABS_TOL} err={err:.2e} med={np.median(t):.1f}us")
    # 2) token distributions on DEFAULT cfg
    cfg = DEFAULT
    W64 = RNG(0).weights(cfg, dtype=np.float64)
    W32 = RNG(0).weights(cfg, dtype=np.float32)
    for kind in ["uniform", "skewed", "same", "edges", "bursty"]:
        x = make_adversarial(cfg, kind, seed=7)
        ref = reference(W64, x, cfg)
        out = fn(W32, x, cfg).astype(np.float64)
        err = float(np.abs(out - ref).max()) if out.shape == ref.shape else float("inf")
        res["distributions"][kind] = {"correct": bool(err < ABS_TOL), "max_abs_err": err}
        print(f"  dist {kind}: correct={err < ABS_TOL} err={err:.2e}")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"evals_v4_{node}_{name}_a{attempt}.json")
    json.dump(res, open(path, "w"), indent=1)
    print(f"[v4 {name}] wrote {path}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", required=True)
    ap.add_argument("--impl", default=None)
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out_dir = a.out_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    names = [a.impl] if a.impl else list(IMPLS)
    summary = {"node": a.node, "generation": "v4", "attempt": a.attempt,
               "run_ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "impls": {}}
    for name in names:
        summary["impls"][name] = run_impl(name, a.node, a.attempt, out_dir, smoke=a.smoke)
    spath = os.path.join(out_dir, f"evals_v4_{a.node}.json")
    json.dump(summary, open(spath, "w"), indent=1)
    print(f"[v4] WROTE SUMMARY {spath}")


if __name__ == "__main__":
    main()
