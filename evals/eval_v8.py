"""Eval v8: latency-vs-error Pareto sweep runner (parameterized).

Mirrors eval_v6.py CLI/JSON style. One measurement per (impl, cell); every
error budget is evaluated post-hoc as arithmetic on max_abs_err — budgets are
dials on the analysis side, not re-runs.

Sweep grid: impls = precision {fp32,int8,int8_attn,bf16} x exp-degree {6,4,3}
(see notes/error-pareto.md naming table) x cfg{tiny,default} x B{1,4,16} x
S{16,32,64} x budgets {1e-3, 1e-2}.

Writes per-impl attempt files results/evals_v8_<node>_<impl>_a<attempt>.json
+ summary results/evals_v8_<node>.json.

Usage:
  python evals/eval_v8.py --node ft-node-1 [--impl c_int8_e6,c_fp32_e4]
      [--cfg tiny,default] [--batch 1,4,16] [--seq 16,32,64]
      [--budget 1e-3,1e-2] [--attempt 1] [--smoke] [--out-dir results]
"""
import argparse, dataclasses, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DEFAULT, TINY
from src.random_state import RNG, make_input
from src.impls import IMPLS, _CImpl, _register_c
from evals.eval_v6 import reference, flops_of, iters_for, bench

# Sweep impl registry-name -> .so basename (LOCKED naming table, notes/error-pareto.md).
SO_NAMES = {}
for _prec in ("fp32", "int8", "int8_attn", "bf16"):
    for _deg in (6, 4, 3):
        SO_NAMES[f"c_{_prec}_e{_deg}"] = f"libft_{_prec}_e{_deg}.so"
SO_NAMES["c_fp32_e6"] = "libft.so"  # champion alias: libft.so = libft_v14 copy
SO_NAMES["c_ground_up"] = "libft.so"

SWEEP_IMPLS = [
    "c_fp32_e6", "c_fp32_e4", "c_fp32_e3",
    "c_int8_e6", "c_int8_e4", "c_int8_e3",
    "c_int8_attn_e6", "c_int8_attn_e4", "c_int8_attn_e3",
    "c_bf16_e6", "c_bf16_e4", "c_bf16_e3",
]

ABS_TOL = 1e-3
SUB_MS = 1000.0
DEFAULT_BUDGETS = [1e-3, 1e-2]
DEFAULT_BATCHES = [1, 4, 16]
DEFAULT_SEQ_LENS = [16, 32, 64]
DEFAULT_CFGS = ("tiny", "default")


def _lazy_register(name):
    """Resolve a sweep impl name from src.impls.IMPLS, falling back to a lazy
    ctypes registration of libft_<prec>_e<deg>.so so the runner works even
    before int8-kernels lands its registry entries. Returns fn or None."""
    if name in IMPLS:
        return IMPLS[name]
    if name == "c_fp32_e6" and "c_ground_up" in IMPLS:
        IMPLS["c_fp32_e6"] = IMPLS["c_ground_up"]  # champion alias
        return IMPLS[name]
    so = SO_NAMES.get(name)
    if so and _register_c(name, so, rw=True, ro=True):
        return IMPLS[name]
    return None


def resolve_impls(names):
    out = {}
    missing = []
    for name in names:
        fn = _lazy_register(name)
        if fn is None:
            missing.append(name)
        else:
            out[name] = fn
    if missing:
        print(f"[v8] WARNING: impls unavailable (no IMPLS entry, no .so): {missing}", flush=True)
    return out


def run_impl(name, fn, node, attempt, out_dir, cfgs, batches, seqs, budgets, smoke=False):
    import warnings as _w
    _w.simplefilter("ignore", RuntimeWarning)
    res = {"impl": name, "node": node, "generation": "v8", "attempt": attempt,
           "numpy_version": np.__version__,
           "criteria": {"abs_tol": ABS_TOL, "sub_ms_us": SUB_MS, "budgets": [f"{b:g}" for b in budgets]},
           "cells": {}, "configs": {}, "fail_cells": []}
    for cfg_name, cfg in (("tiny", TINY), ("default", DEFAULT)):
        if cfg_name not in cfgs:
            continue
        W64 = RNG(0).weights(cfg, dtype=np.float64)
        W32 = RNG(0).weights(cfg, dtype=np.float32)
        res["configs"][cfg_name] = {}
        for b in batches:
            for sl in seqs:
                c = dataclasses.replace(cfg, seq_len=sl)
                x = make_input(c, batch=b, seed=1)
                ref = reference(W64, x, c)
                key = f"{cfg_name}_b{b}_s{sl}"
                shape = f"d={c.d_model},h={c.n_heads},L={c.n_layers},ff={c.d_ff},S={sl},V={c.vocab},B={b}"
                cell = {"shape": shape}
                try:
                    out = fn(W32, x, c).astype(np.float64)
                    ok_shape = out.shape == ref.shape
                    abs_err = float(np.abs(out - ref).max()) if ok_shape else float("inf")
                    it = 8 if smoke else iters_for(b, sl)
                    t = bench(fn, W32, x, c, it, warmup=2 if smoke else 50)
                    med = float(np.median(t))
                    cell.update({"max_abs_err": abs_err,
                                 "median_us": med,
                                 "p99_us": float(np.percentile(t, 99)),
                                 "max_us": float(t.max()),
                                 "iters": it,
                                 "gflops": float(flops_of(c, b) / (med / 1e6) / 1e9)})
                except Exception as e:  # noqa: BLE001 — record failure, keep runner alive
                    cell.update({"max_abs_err": float("inf"), "median_us": float("inf"),
                                 "p99_us": float("inf"), "max_us": float("inf"),
                                 "iters": 0, "gflops": 0.0, "error": repr(e)})
                budget_pass = {}
                for budget in budgets:
                    ok = bool(cell["max_abs_err"] < budget)
                    budget_pass[f"{budget:g}"] = ok
                    if not ok:
                        res["fail_cells"].append(f"{key}:budget_{budget:g}")
                cell["budget_pass"] = budget_pass
                cell["sub_ms"] = bool(cell["median_us"] < SUB_MS)
                if not cell["sub_ms"]:
                    res["fail_cells"].append(f"{key}:sub_ms")
                res["cells"][key] = cell
                err = cell.get("max_abs_err", float("inf"))
                med = cell.get("median_us", float("inf"))
                print(f"  [{cfg_name} B={b:2d} S={sl:3d}] err={err:.3e} med={med:9.1f}us "
                      f"pass={ {k: v for k, v in budget_pass.items()} }", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"evals_v8_{node}_{name}_a{attempt}.json")
    json.dump(res, open(path, "w"), indent=1)
    print(f"[v8 {name}] cells={len(res['cells'])} fails={len(res['fail_cells'])} -> {path}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", required=True)
    ap.add_argument("--impl", default=None,
                    help=f"comma list; default = all sweep impls available: {','.join(SWEEP_IMPLS)}")
    ap.add_argument("--cfg", default=",".join(DEFAULT_CFGS))
    ap.add_argument("--batch", default=",".join(map(str, DEFAULT_BATCHES)))
    ap.add_argument("--seq", default=",".join(map(str, DEFAULT_SEQ_LENS)))
    ap.add_argument("--budget", default=",".join(f"{b:g}" for b in DEFAULT_BUDGETS),
                    help="error budgets: max_abs_err < budget (comma list, e.g. 1e-3,1e-2)")
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = a.out_dir or os.path.join(root, "results")

    names = [x.strip() for x in a.impl.split(",")] if a.impl else SWEEP_IMPLS
    cfgs = [x.strip() for x in a.cfg.split(",")]
    batches = [int(x) for x in a.batch.split(",")]
    seqs = [int(x) for x in a.seq.split(",")]
    budgets = [float(x) for x in a.budget.split(",")]

    impls = resolve_impls(names)
    summary = {"node": a.node, "generation": "v8", "attempt": a.attempt,
               "run_ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "criteria": "budget_pass = max_abs_err < budget per (cell, budget); "
                           "budgets=" + ",".join(f"{b:g}" for b in budgets) +
                           f"; sub_ms: median_us < {SUB_MS:g}",
               "impls": {}}
    for name, fn in impls.items():
        summary["impls"][name] = run_impl(name, fn, a.node, a.attempt, out_dir,
                                           cfgs, batches, seqs, budgets, smoke=a.smoke)
    spath = os.path.join(out_dir, f"evals_v8_{a.node}.json")
    json.dump(summary, open(spath, "w"), indent=1)
    print(f"[v8] WROTE SUMMARY {spath} ({len(impls)} impls)", flush=True)


if __name__ == "__main__":
    main()
