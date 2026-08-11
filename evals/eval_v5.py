"""Eval v5: headroom analysis — per-stage share, achieved FLOPs/s vs theoretical, overhead vs pure-matmul ceiling.
Writes per-impl attempt files results/evals_v5_<node>_<impl>_a<attempt>.json + summary results/evals_v5_<node>.json.
Usage: python evals/eval_v5.py --node bench-node-3 [--impl numpy_vec] [--attempt 1] [--smoke]
"""
import argparse, dataclasses, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DEFAULT, TINY
from src.random_state import RNG, make_input
from src.impls import IMPLS


def flops_of(cfg, B=1):
    """Theoretical FLOPs of the forward pass (matmul FLOPs only, 2*m*n*k)."""
    S, d, dff, V = cfg.seq_len, cfg.d_model, cfg.d_ff, cfg.vocab
    H, dh, L = cfg.n_heads, cfg.d_head, cfg.n_layers
    fl = 0
    for _ in range(L):
        fl += 3 * 2 * B * S * d * d          # QKV
        fl += 2 * 2 * B * H * S * S * dh     # attn scores + ctx
        fl += 2 * B * S * d * d              # wo
        fl += 2 * B * S * d * dff            # w1
        fl += 2 * B * S * dff * d            # w2
    fl += 2 * B * S * d * V                 # out
    return fl


def bench(fn, W, x, cfg, iters=1500, warmup=100):
    for _ in range(warmup):
        fn(W, x, cfg)
    times = np.empty(iters)
    for i in range(iters):
        t0 = time.perf_counter_ns()
        fn(W, x, cfg)
        times[i] = (time.perf_counter_ns() - t0) / 1000.0
    return times


def matmul_ceiling(cfg, B=1, iters=800, warmup=50):
    """Pure-matmul workload with the exact matmul shapes of one forward pass
    (no LN/softmax/exp/mask/transposes beyond what matmul needs).
    Measures the machine's ceiling for these shapes -> overhead vs theoretical."""
    S, d, dff, V = cfg.seq_len, cfg.d_model, cfg.d_ff, cfg.vocab
    H, dh, L = cfg.n_heads, cfg.d_head, cfg.n_layers
    g = np.random.default_rng(0)
    x = g.normal(0, 0.125, (B, S, d)).astype(np.float32)
    mats = []
    for _ in range(L):
        mats.append(g.normal(0, 0.125, (d, 3 * d)).astype(np.float32))   # fused QKV
        mats.append(g.normal(0, 0.125, (d, d)).astype(np.float32))        # wo
        mats.append(g.normal(0, 0.125, (d, dff)).astype(np.float32))      # w1
        mats.append(g.normal(0, 0.125, (dff, d)).astype(np.float32))      # w2
    mats.append(g.normal(0, 0.125, (d, V)).astype(np.float32))            # out
    out = np.empty((B, S, V), dtype=np.float32)

    def run():
        h = x
        i = 0
        for _ in range(L):
            qkv = h @ mats[i]; i += 1                       # (B,S,3d)  [3d^2 flops]
            Q = qkv[:, :, :d].reshape(B, H, S, dh)
            K = qkv[:, :, d:2 * d].reshape(B, H, S, dh)
            Vv = qkv[:, :, 2 * d:3 * d].reshape(B, H, S, dh)
            att = Q @ K.swapaxes(-1, -2)                    # (B,H,S,S) [2*S^2*dh]
            ctx = att @ Vv                                  # (B,H,S,dh) [2*S^2*dh]
            h = h + ctx.reshape(B, S, d) @ mats[i]; i += 1  # wo
            ff = h @ mats[i]; i += 1                        # w1
            h = h + ff @ mats[i]; i += 1                    # w2
        out[:] = h @ mats[i]
        return out
    for _ in range(warmup):
        run()
    times = np.empty(iters)
    for j in range(iters):
        t0 = time.perf_counter_ns()
        run()
        times[j] = (time.perf_counter_ns() - t0) / 1000.0
    return times


def run_impl(name, node, attempt, out_dir, smoke=False):
    import warnings as _w
    _w.simplefilter("ignore", RuntimeWarning)
    fn = IMPLS[name]
    res = {"impl": name, "node": node, "generation": "v5", "attempt": attempt,
           "numpy_version": np.__version__, "configs": {}}
    for cfg_name, cfg in (("tiny", TINY), ("default", DEFAULT)):
        W32 = RNG(0).weights(cfg, dtype=np.float32)
        x = make_input(cfg, seed=1)
        fl = flops_of(cfg, B=1)
        it = 8 if smoke else 1500
        t = bench(fn, W32, x, cfg, it, 5 if smoke else 100)
        med = float(np.median(t))
        mc = matmul_ceiling(cfg, iters=8 if smoke else 800, warmup=5 if smoke else 50)
        mc_med = float(np.median(mc))
        res["configs"][cfg_name] = {
            "theoretical_flops": fl,
            "median_us": med,
            "achieved_gflops_s": float(fl / (med / 1e6) / 1e9),
            "matmul_ceiling_us": mc_med,
            "matmul_ceiling_gflops_s": float(fl / (mc_med / 1e6) / 1e9),
            "overhead_vs_ceiling_x": float(med / mc_med),
            "headroom_pct": float(max(0.0, 100.0 * (1.0 - med / mc_med))),
        }
        print(f"[v5 {name} {cfg_name}] flops={fl/1e6:.2f}M med={med:.1f}us "
              f"achieved={res['configs'][cfg_name]['achieved_gflops_s']:.2f} GF/s "
              f"ceiling={mc_med:.1f}us overhead={med/mc_med:.2f}x")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"evals_v5_{node}_{name}_a{attempt}.json")
    json.dump(res, open(path, "w"), indent=1)
    print(f"[v5 {name}] wrote {path}")
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
    summary = {"node": a.node, "generation": "v5", "attempt": a.attempt,
               "run_ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "impls": {}}
    for name in names:
        summary["impls"][name] = run_impl(name, a.node, a.attempt, out_dir, smoke=a.smoke)
    spath = os.path.join(out_dir, f"evals_v5_{a.node}.json")
    json.dump(summary, open(spath, "w"), indent=1)
    print(f"[v5] WROTE SUMMARY {spath}")


if __name__ == "__main__":
    main()
