"""Eval v6: sub-ms across the op-domain — batch{1,4,16} x seq{16,32,64} x cfg{tiny,default},
roofline efficiency (achieved GF/s vs measured peak GEMM GF/s), cold-start first-call latency.
CRITERION: an impl must hold median_us < 1000 on ALL v6 cells to pass.
Writes per-impl attempt files results/evals_v6_<node>_<impl>_a<attempt>.json + summary results/evals_v6_<node>.json.
Usage: python evals/eval_v6.py --node bench-node-3 [--impl a,b] [--attempt 1] [--smoke]
"""
import argparse, dataclasses, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DEFAULT, TINY
from src.random_state import RNG, make_input
from src.impls import IMPLS

ABS_TOL = 1e-3
BATCHES = [1, 4, 16]
SEQ_LENS = [16, 32, 64]
SUB_MS = 1000.0


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


def flops_of(cfg, B=1):
    S, d, dff, V = cfg.seq_len, cfg.d_model, cfg.d_ff, cfg.vocab
    H, dh, L = cfg.n_heads, cfg.d_head, cfg.n_layers
    fl = 0
    for _ in range(L):
        fl += 3 * 2 * B * S * d * d + 2 * 2 * B * H * S * S * dh + 2 * B * S * d * d
        fl += 2 * B * S * d * dff + 2 * B * S * dff * d
    fl += 2 * B * S * d * V
    return fl


def iters_for(B, S):
    n = B * S
    if n <= 128: return 2000
    if n <= 512: return 1000
    return 500


def peak_gemm_gflops(smoke=False):
    """Measured peak GEMM GF/s for fp32 matmuls on this node (max over sizes)."""
    g = np.random.default_rng(0)
    best = 0.0
    for n in ([64, 128, 256, 512] if not smoke else [64, 128]):
        reps = 8 if smoke else 30
        a = g.normal(0, 1, (n, n)).astype(np.float32)
        b = g.normal(0, 1, (n, n)).astype(np.float32)
        a @ b
        t0 = time.perf_counter()
        for _ in range(reps):
            a @ b
        gf = 2 * n**3 * reps / (time.perf_counter() - t0) / 1e9
        best = max(best, gf)
    return best


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
    _w.simplefilter("ignore", RuntimeWarning)
    fn = IMPLS[name]
    res = {"impl": name, "node": node, "generation": "v6", "attempt": attempt,
           "numpy_version": np.__version__, "criteria": {"abs_tol": ABS_TOL, "sub_ms_us": SUB_MS},
           "cells": {}, "configs": {}, "fail_cells": [], "b_general": True,
           "sub_ms_all_cells": True}
    for cfg_name, cfg in (("tiny", TINY), ("default", DEFAULT)):
        W64 = RNG(0).weights(cfg, dtype=np.float64)
        W32 = RNG(0).weights(cfg, dtype=np.float32)
        peak = peak_gemm_gflops(smoke=smoke)
        res["configs"][cfg_name] = {"peak_gemm_gflops": peak}
        for b in BATCHES:
            for sl in SEQ_LENS:
                c = dataclasses.replace(cfg, seq_len=sl)
                x = make_input(c, batch=b, seed=1)
                ref = reference(W64, x, c)
                out = fn(W32, x, c).astype(np.float64)
                ok_shape = out.shape == ref.shape
                abs_err = float(np.abs(out - ref).max()) if ok_shape else float("inf")
                correct = bool(ok_shape and abs_err < ABS_TOL)
                it = 8 if smoke else iters_for(b, sl)
                t = bench(fn, W32, x, c, it, warmup=2 if smoke else 50)
                med = float(np.median(t))
                key = f"{cfg_name}_b{b}_s{sl}"
                shape = f"d={c.d_model},h={c.n_heads},L={c.n_layers},ff={c.d_ff},S={sl},V={c.vocab},B={b}"
                cell = {"correct": correct, "max_abs_err": abs_err, "shape": shape,
                        "median_us": med, "p99_us": float(np.percentile(t, 99)),
                        "max_us": float(t.max()), "iters": it,
                        "gflops": float(flops_of(c, b) / (med / 1e6) / 1e9),
                        "sub_ms": bool(med < SUB_MS)}
                res["cells"][key] = cell
                if not correct:
                    res["fail_cells"].append(f"{key}:correct")
                    if b > 1:
                        res["b_general"] = False
                if not cell["sub_ms"]:
                    res["sub_ms_all_cells"] = False
                    res["fail_cells"].append(f"{key}:sub_ms")
                print(f"  [{cfg_name} B={b:2d} S={sl:3d}] correct={correct} med={med:9.1f}us "
                      f"{'SUB-MS' if med < SUB_MS else 'OVER  '}", flush=True)
        # cold-start: first call right after weights materialization (includes ctypes/prep/alloc)
        x = make_input(cfg, seed=7)
        gc.collect() if (gc := __import__("gc")) else None
        gc.collect()
        t0 = time.perf_counter_ns()
        fn(W32, x, cfg)
        cold_us = (time.perf_counter_ns() - t0) / 1000.0
        t = bench(fn, W32, x, cfg, 8 if smoke else 800, warmup=2 if smoke else 50)
        res["configs"][cfg_name]["cold"] = {"cold_first_call_us": cold_us,
                                            "steady_median_us": float(np.median(t)),
                                            "cold_ratio": cold_us / float(np.median(t))}
        # roofline at canonical cell (default B=1 S=32 / tiny B=1 S=16)
        canon_key = ("default_b1_s32" if cfg_name == "default" else "tiny_b1_s16")
        cell = res["cells"].get(canon_key)
        if cell:
            res["configs"][cfg_name]["roofline_efficiency"] = cell["gflops"] / peak
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"evals_v6_{node}_{name}_a{attempt}.json")
    json.dump(res, open(path, "w"), indent=1)
    verdict = "PASS" if res["sub_ms_all_cells"] and not res["fail_cells"] else "FAIL"
    print(f"[v6 {name}] {verdict} (cells={len(res['cells'])}, fails={len(res['fail_cells'])}, "
          f"b_general={res['b_general']}) -> {path}")
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
    names = [x.strip() for x in a.impl.split(",")] if a.impl else list(IMPLS)
    summary = {"node": a.node, "generation": "v6", "attempt": a.attempt,
               "run_ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "criteria": "median_us < 1000 on ALL batch{1,4,16}xseq{16,32,64}xcfg{tiny,default} cells",
               "impls": {}}
    for name in names:
        summary["impls"][name] = run_impl(name, a.node, a.attempt, out_dir, smoke=a.smoke)
    spath = os.path.join(out_dir, f"evals_v6_{a.node}.json")
    json.dump(summary, open(spath, "w"), indent=1)
    print(f"[v6] WROTE SUMMARY {spath}")


if __name__ == "__main__":
    main()
