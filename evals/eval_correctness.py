"""Correctness eval: fp32 impls vs float64 reference. Usage:
python evals/eval_correctness.py --impl numpy_vec [--all]
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DEFAULT, TINY
from src.random_state import RNG, make_input
from src.impls import IMPLS

def reference(W64, x, cfg):
    """float64 reference - same architecture as impls, naive but exact."""
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default=None, help="single impl; default: all")
    ap.add_argument("--cfg", default="tiny", choices=["tiny", "default"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = TINY if a.cfg == "tiny" else DEFAULT
    W64 = RNG(0).weights(cfg, dtype=np.float64)
    x = make_input(cfg)
    ref = reference(W64, x, cfg)
    names = [a.impl] if a.impl else list(IMPLS)
    results = {}
    for name in names:
        W32 = RNG(0).weights(cfg, dtype=np.float32)
        out = IMPLS[name](W32, x, cfg).astype(np.float64)
        abs_err = np.abs(out - ref).max()
        rel_err = float(np.abs(out - ref).max() / (np.abs(ref).max() + 1e-9))
        results[name] = {"max_abs_err": float(abs_err), "max_rel_err": rel_err,
                         "pass": bool(abs_err < 1e-3)}
        print(f"{name}: max_abs_err={abs_err:.3e} max_rel_err={rel_err:.3e} pass={abs_err < 1e-3}")
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
