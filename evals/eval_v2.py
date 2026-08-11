"""Eval v2: scaling curves (seq_len x batch) + p99/max tails on DEFAULT cfg.
Writes per-impl attempt files results/evals_v2_<node>_<impl>_a<attempt>.json
plus a combined summary results/evals_v2_<node>.json.
Usage: python evals/eval_v2.py --node bench-node-3 [--impl numpy_vec] [--attempt 1] [--out results/evals_v2_bench-node-3.json]
"""
import argparse, json, os, sys, time, dataclasses
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DEFAULT
from src.random_state import RNG, make_input
from src.impls import IMPLS

SEQ_SWEEP = [8, 16, 32, 64, 128]
BATCH_SWEEP = [1, 4, 16]
ABS_TOL = 1e-3


def reference(W64, x, cfg):
    """float64 reference (same architecture as impls)."""
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


def iters_for(B, S):
    n = B * S
    if n <= 32: return 2000
    if n <= 128: return 1000
    if n <= 512: return 500
    return 300


def bench(fn, W, x, cfg, iters, warmup=50):
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
    _w.simplefilter("ignore", RuntimeWarning)  # spurious BLAS matmul warnings on some platforms
    fn = IMPLS[name]
    impl_res = {"impl": name, "node": node, "generation": "v2", "attempt": attempt,
                "numpy_version": np.__version__, "seq_sweep": {}, "batch_sweep": {},
                "criteria": {"abs_tol": ABS_TOL}}
    # seq sweep at batch=1
    for sl in SEQ_SWEEP:
        cfg = dataclasses.replace(DEFAULT, seq_len=sl)
        W64 = RNG(0).weights(cfg, dtype=np.float64)
        x = make_input(cfg, batch=1, seed=1)
        ref = reference(W64, x, cfg)
        W32 = RNG(0).weights(cfg, dtype=np.float32)
        out = fn(W32, x, cfg).astype(np.float64)
        abs_err = float(np.abs(out - ref).max())
        it = 8 if smoke else iters_for(1, sl)
        t = bench(fn, W32, x, cfg, it, warmup=2 if smoke else 50)
        shape = f"d={cfg.d_model},h={cfg.n_heads},L={cfg.n_layers},ff={cfg.d_ff},S={sl},V={cfg.vocab},B=1"
        impl_res["seq_sweep"][str(sl)] = {
            "shape": shape, "correct": bool(abs_err < ABS_TOL), "max_abs_err": abs_err,
            "iters": it, "median_us": float(np.median(t)), "mean_us": float(t.mean()),
            "p90_us": float(np.percentile(t, 90)), "p99_us": float(np.percentile(t, 99)),
            "max_us": float(t.max()), "min_us": float(t.min()),
            "throughput_tok_s": float(sl / (np.median(t) / 1e6)),
        }
    # batch sweep at seq_len=32
    cfg = DEFAULT
    W64 = RNG(0).weights(cfg, dtype=np.float64)
    W32 = RNG(0).weights(cfg, dtype=np.float32)
    for b in BATCH_SWEEP:
        x = make_input(cfg, batch=b, seed=1)
        ref = reference(W64, x, cfg)
        out = fn(W32, x, cfg).astype(np.float64)
        abs_err = float(np.abs(out - ref).max())
        it = 8 if smoke else iters_for(b, cfg.seq_len)
        t = bench(fn, W32, x, cfg, it, warmup=2 if smoke else 50)
        shape = f"d={cfg.d_model},h={cfg.n_heads},L={cfg.n_layers},ff={cfg.d_ff},S={cfg.seq_len},V={cfg.vocab},B={b}"
        impl_res["batch_sweep"][str(b)] = {
            "shape": shape, "correct": bool(abs_err < ABS_TOL), "max_abs_err": abs_err,
            "iters": it, "median_us": float(np.median(t)), "mean_us": float(t.mean()),
            "p90_us": float(np.percentile(t, 90)), "p99_us": float(np.percentile(t, 99)),
            "max_us": float(t.max()), "min_us": float(t.min()),
            "throughput_tok_s": float(b * cfg.seq_len / (np.median(t) / 1e6)),
        }
        del x, ref, out
    # per-impl attempt file
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"evals_v2_{node}_{name}_a{attempt}.json")
    json.dump(impl_res, open(path, "w"), indent=1)
    print(f"[v2 {name}] wrote {path}")
    for sl, r in impl_res["seq_sweep"].items():
        print(f"  seq={sl}: correct={r['correct']} med={r['median_us']:.1f}us p99={r['p99_us']:.1f}us max={r['max_us']:.1f}us")
    for b, r in impl_res["batch_sweep"].items():
        print(f"  batch={b}: correct={r['correct']} med={r['median_us']:.1f}us p99={r['p99_us']:.1f}us max={r['max_us']:.1f}us")
    return impl_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", required=True)
    ap.add_argument("--impl", default=None)
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny iters, local CI only (no timing benchmark)")
    a = ap.parse_args()
    node = a.node
    out_dir = a.out_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    names = [x.strip() for x in a.impl.split(",")] if a.impl else list(IMPLS)
    summary = {"node": node, "generation": "v2", "attempt": a.attempt,
               "run_ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "impls": {}}
    for name in names:
        summary["impls"][name] = run_impl(name, node, a.attempt, out_dir, smoke=a.smoke)
    spath = os.path.join(out_dir, f"evals_v2_{node}.json")
    json.dump(summary, open(spath, "w"), indent=1)
    print(f"[v2] WROTE SUMMARY {spath}")


if __name__ == "__main__":
    main()
