"""Manual stage timing for numpy_opt10's exact ops (TINY + DEFAULT)."""
import os, sys, json, time
try:
    _HERE = os.path.dirname(os.path.realpath(__file__))
except NameError:
    _HERE = os.getcwd()
sys.path.insert(0, _HERE)
for _d in ("/content", _HERE):
    if os.path.isfile(os.path.join(_d, "src", "impls.py")):
        sys.path.insert(0, _d)
        break
import importlib
import src.impls as _m
importlib.reload(_m)
from src.config import DEFAULT, TINY
from src.random_state import RNG, make_input
from src.impls import IMPLS, _prep_weights_v2, _stats_w3, _bufs, _maskadd
import numpy as np

res = {}
for cfgname, cfg in [("tiny", TINY), ("default", DEFAULT)]:
    W = RNG(0).weights(cfg, dtype=np.float32)
    x = make_input(cfg)
    S = cfg.seq_len; d = cfg.d_model; nh, dh = cfg.n_heads, cfg.d_head
    h = np.take(W["emb"], x[0], axis=0)
    p = _prep_weights_v2(W, cfg)
    maskadd = _maskadd(S)
    b = _bufs(cfg)
    Wstats = _stats_w3(d, cfg.eps)
    ones = np.ones((S, 1), np.float32)
    hc = b["hc3"]
    st = np.empty((S, 2), np.float32)

    def T(fn, n=2000):
        fn()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n * 1e6

    stages = {}
    stages["emb"] = T(lambda: np.take(W["emb"], x[0], axis=0))
    def s_stats():
        np.multiply(h, h, out=b["ff"])
        np.concatenate([h, b["ff"], ones], axis=-1, out=hc)
        np.matmul(hc, Wstats, out=st)
        mu = st[:, :1]
        return (st[:, 1:] - mu * mu) ** -0.5
    stages["stats3calls"] = T(s_stats)
    def s_stats_only():
        np.multiply(h, h, out=b["ff"])
        np.concatenate([h, b["ff"], ones], axis=-1, out=hc)
        np.matmul(hc, Wstats, out=st)
    stages["stats_norsd"] = T(s_stats_only)
    def s_rstd():
        mu = st[:, :1]
        return (st[:, 1:] - mu * mu) ** -0.5
    stages["rstd_only"] = T(s_rstd)
    def s_qkv():
        np.subtract(h, st[:, :1], out=b["ff"])
        np.matmul(b["ff"], p["qkv"][0], out=b["qkv"])
        b["qkv"] *= (st[:, 1:] - st[:, :1]*st[:, :1]) ** -0.5
    stages["sub_qkvmat_scal"] = T(s_qkv)
    def s_attn():
        q = b["qkv"][:, :d].reshape(S, nh, dh).transpose(1, 0, 2)
        k = b["qkv"][:, d:2*d].reshape(S, nh, dh).transpose(1, 0, 2)
        v = b["qkv"][:, 2*d:3*d].reshape(S, nh, dh).transpose(1, 0, 2)
        np.matmul(q, k.transpose(0, 2, 1), out=b["att"])
        b["att"] += maskadd
        np.exp(b["att"], out=b["e"])
        np.matmul(b["e"], ones, out=b["den"])
        np.matmul(b["e"], v, out=b["ctx"])
        b["ctx"] /= b["den"]
    stages["attn_chain"] = T(s_attn)
    def s_att_matmul():
        q = b["qkv"][:, :d].reshape(S, nh, dh).transpose(1, 0, 2)
        k = b["qkv"][:, d:2*d].reshape(S, nh, dh).transpose(1, 0, 2)
        np.matmul(q, k.transpose(0, 2, 1), out=b["att"])
    stages["att_matmul"] = T(s_att_matmul)
    def s_softmax():
        b["att"] += maskadd
        np.exp(b["att"], out=b["e"])
        np.matmul(b["e"], ones, out=b["den"])
    stages["mask_exp_den"] = T(s_softmax)
    def s_ctx():
        v = b["qkv"][:, 2*d:3*d].reshape(S, nh, dh).transpose(1, 0, 2)
        np.matmul(b["e"], v, out=b["ctx"])
        b["ctx"] /= b["den"]
    stages["ctx_mat_div"] = T(s_ctx)
    def s_wo():
        ctx = b["ctx"]
        np.matmul(ctx.transpose(1, 0, 2).reshape(S, d), p["wo"][0], out=b["proj"])
        np.add(h, b["proj"], out=h)
    stages["wo_add"] = T(s_wo)
    def s_ffn():
        np.multiply(h, h, out=b["ff"])
        np.concatenate([h, b["ff"], ones], axis=-1, out=hc)
        np.matmul(hc, Wstats, out=st)
        mu = st[:, :1]
        rstd = (st[:, 1:] - mu * mu) ** -0.5
        np.subtract(h, mu, out=b["ff"])
        np.maximum(b["ff"], 0, out=b["ff"])
        b["ff"] *= rstd
        np.matmul(b["ff"], p["w1"][0], out=b["up"])
        np.matmul(b["up"], p["w2"][0], out=b["ff"])
        np.add(h, b["ff"], out=h)
    stages["ffn_chain"] = T(s_ffn)
    def s_final():
        np.multiply(h, h, out=b["ff"])
        np.concatenate([h, b["ff"], ones], axis=-1, out=hc)
        np.matmul(hc, Wstats, out=st)
        mu = st[:, :1]
        rstd = (st[:, 1:] - mu * mu) ** -0.5
        np.subtract(h, mu, out=b["ff"])
        np.matmul(b["ff"], p["A3"], out=b["out"])
        b["out"] *= rstd
    stages["final_chain"] = T(s_final)
    res[cfgname] = {k: round(v, 2) for k, v in stages.items()}
print(json.dumps(res, indent=1))
