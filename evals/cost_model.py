"""Tiny-roofline cost model: latency_us ~= c0 + c1*FLOPs + c2*B*S^2 (+ cfg-specific overhead).

Fits the linear model over all collected v8 (+v6) cells in results/ and prints
coefficients, effective GFLOP/s, residuals, and the worst-fit cells. This is the
v7-attribution workstream's model-fitting deliverable (parent-executed).

Usage: python evals/cost_model.py
"""
import glob, json, os, re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

DIMS = {"tiny": dict(d=32, dff=64, h=2, v=256, L=1),
        "default": dict(d=64, dff=128, h=4, v=512, L=2)}

def flops_of(cfg, b, s):
    p = DIMS[cfg]
    d, dff, h, v, L = p["d"], p["dff"], p["h"], p["v"], p["L"]
    dh = d // h
    n = b * s
    fl = L * (2 * n * 3 * d * d + 2 * 2 * b * h * s * s * dh + 2 * n * d * d
              + 2 * n * d * dff + 2 * n * dff * d)
    fl += 2 * n * d * v
    return fl

def collect(impl_prefixes=("c_fp32", "c_ground_up", "c_v1")):
    rows = []
    seen = set()
    for f in sorted(glob.glob(os.path.join(RESULTS, "evals_v8_*_a*.json"))):
        d = json.load(open(f))
        impl = d.get("impl", "")
        if not any(impl.startswith(p) for p in impl_prefixes):
            continue
        for cell, c in d.get("cells", {}).items():
            if not c.get("median_us") or not np.isfinite(c["median_us"]):
                continue
            m = re.match(r"(tiny|default)_b(\d+)_s(\d+)", cell)
            if not m:
                continue
            cfg, b, s = m.group(1), int(m.group(2)), int(m.group(3))
            key = (impl, cfg, b, s)
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(impl=impl, cfg=cfg, b=b, s=s, med=c["median_us"]))
    return rows

def fit(rows):
    X, y = [], []
    for r in rows:
        f = flops_of(r["cfg"], r["b"], r["s"])
        att = r["b"] * r["s"] ** 2
        X.append([1.0, f, att])
        y.append(r["med"])
    X, y = np.array(X), np.array(y)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    rel = np.abs(pred - y) / y
    return coef, pred, rel

def main():
    rows = collect()
    if not rows:
        print("no v8 fp32 cells found; skipping")
        return
    coef, pred, rel = fit(rows)
    print(f"cells: {len(rows)}")
    print(f"latency_us = {coef[0]:.1f} + {coef[1]:.3e}*FLOPs + {coef[2]:.3e}*B*S^2")
    print(f"effective rate: {1 / coef[1] / 1e3:.1f} GFLOP/s (compute term; FLOPs counted 2/MAC)")
    print(f"median rel err: {np.median(rel) * 100:.1f}%  max: {rel.max() * 100:.1f}%")
    for i in np.argsort(-rel)[:5]:
        r = rows[i]
        print(f"  worst: {r['impl']:12s} {r['cfg']:7s} b{r['b']:2d} s{r['s']:2d}: pred {pred[i]:7.0f} vs {r['med']:7.0f} ({rel[i] * 100:.0f}%)")
    out = os.path.join(RESULTS, "cost_model_ft-node-2.json")
    json.dump({"coef_us_per_flop": coef[1], "coef_us_per_bs2": coef[2],
               "intercept_us": coef[0], "eff_gflops": 1 / coef[1] / 1e3,
               "median_rel_err": float(np.median(rel)), "n_cells": len(rows),
               "cells": [dict(rows[i], pred_us=float(pred[i]), rel_err=float(rel[i]))
                         for i in range(len(rows))]}, open(out, "w"), indent=1)
    print("wrote", out)

if __name__ == "__main__":
    main()
