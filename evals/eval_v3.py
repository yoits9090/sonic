"""Eval v3: stability across seeds/inputs, cold-cache latency, memory allocation counts.
Writes per-impl attempt files results/evals_v3_<node>_<impl>_a<attempt>.json
plus a combined summary results/evals_v3_<node>.json.
Usage: python evals/eval_v3.py --node bench-node-3 [--impl numpy_vec] [--attempt 1] [--out-dir results]
"""
import argparse, gc, json, os, sys, time, tracemalloc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DEFAULT, TINY
from src.random_state import RNG, make_input
from src.impls import IMPLS

N_SEEDS = 5
ABS_TOL = 1e-3


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


def bench(fn, W, x, cfg, iters, warmup=100):
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
    res = {"impl": name, "node": node, "generation": "v3", "attempt": attempt,
           "numpy_version": np.__version__, "n_seeds": N_SEEDS,
           "criteria": {"abs_tol": ABS_TOL}, "configs": {}}
    for cfg_name, cfg in (("tiny", TINY), ("default", DEFAULT)):
        c = {"correctness_seeds": [], "latency_seeds": [], "cold": None, "alloc": None}
        for s in range(N_SEEDS):
            W64 = RNG(s).weights(cfg, dtype=np.float64)
            x = make_input(cfg, seed=s)
            ref = reference(W64, x, cfg)
            W32 = RNG(s).weights(cfg, dtype=np.float32)
            out = fn(W32, x, cfg).astype(np.float64)
            abs_err = float(np.abs(out - ref).max())
            c["correctness_seeds"].append({"seed": s, "max_abs_err": abs_err,
                                           "pass": bool(abs_err < ABS_TOL)})
        # latency stability across input seeds (same weights)
        W32 = RNG(0).weights(cfg, dtype=np.float32)
        medians = []
        for s in range(N_SEEDS):
            x = make_input(cfg, seed=s)
            t = bench(fn, W32, x, cfg, 8 if smoke else 400, warmup=2 if smoke else 50)
            medians.append(float(np.median(t)))
        med = np.array(medians)
        c["latency_seeds"] = {"seed_medians_us": medians,
                              "median_of_medians_us": float(np.median(med)),
                              "min_median_us": float(med.min()), "max_median_us": float(med.max()),
                              "spread_pct": float((med.max() - med.min()) / med.min() * 100)}
        # cold-cache: first call right after weights+input materialization, GC'd
        gc.collect()
        x = make_input(cfg, seed=99)
        t0 = time.perf_counter_ns()
        fn(W32, x, cfg)
        cold_us = (time.perf_counter_ns() - t0) / 1000.0
        t = bench(fn, W32, x, cfg, 8 if smoke else 1500, warmup=2 if smoke else 100)
        steady = float(np.median(t))
        c["cold"] = {"cold_first_call_us": cold_us, "steady_median_us": steady,
                     "cold_ratio": cold_us / steady}
        # allocation counts via tracemalloc (single forward, no output kept)
        tracemalloc.start()
        gc.collect()
        fn(W32, x, cfg)
        cur, peak = tracemalloc.get_traced_memory()
        stats = tracemalloc.take_snapshot().statistics("filename")
        tracemalloc.stop()
        n_alloc = sum(st.count for st in stats)
        c["alloc"] = {"current_bytes": int(cur), "peak_bytes": int(peak),
                      "num_allocations": int(n_alloc)}
        res["configs"][cfg_name] = c
        print(f"[v3 {name} {cfg_name}] correctness all pass={all(r['pass'] for r in c['correctness_seeds'])}")
        print(f"  latency seed medians {[round(m,1) for m in medians]} spread={c['latency_seeds']['spread_pct']:.1f}%")
        print(f"  cold={cold_us:.0f}us steady={steady:.1f}us ratio={cold_us/steady:.1f}x  allocs={n_alloc} peak={peak/1e6:.2f}MB")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"evals_v3_{node}_{name}_a{attempt}.json")
    json.dump(res, open(path, "w"), indent=1)
    print(f"[v3 {name}] wrote {path}")
    return res


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
    summary = {"node": node, "generation": "v3", "attempt": a.attempt,
               "run_ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "impls": {}}
    for name in names:
        summary["impls"][name] = run_impl(name, node, a.attempt, out_dir, smoke=a.smoke)
    spath = os.path.join(out_dir, f"evals_v3_{node}.json")
    json.dump(summary, open(spath, "w"), indent=1)
    print(f"[v3] WROTE SUMMARY {spath}")


if __name__ == "__main__":
    main()
