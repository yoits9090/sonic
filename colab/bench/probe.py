"""Node capability probe + per-stage timing profile (numpy-optimizer agent).
Prints JSON: cpu info, numpy/BLAS info, matmul micro-bench, per-stage us for
the opt2 pipeline, and thread-count experiments for a 512x512 gemm.
"""
import os, sys, json, time, importlib, platform
try:
    _HERE = os.path.dirname(os.path.realpath(__file__))
except NameError:  # executed via colab exec (notebook semantics, no __file__)
    _HERE = os.getcwd()
sys.path.insert(0, _HERE)
for _d in ("/content", _HERE):
    if os.path.isfile(os.path.join(_d, "src", "impls.py")):
        sys.path.insert(0, _d)
        break
import numpy as np
from src.config import DEFAULT, TINY
from src.random_state import RNG, make_input
import src.impls as _impls_mod
importlib.reload(_impls_mod)  # node kernels cache modules across execs
from src.impls import IMPLS

info = {"platform": platform.platform(), "python": sys.version.split()[0]}
try:
    with open("/proc/cpuinfo") as f:
        cpu = f.read()
    info["cpu"] = sorted(set(l.split(":")[1].strip() for l in cpu.splitlines() if l.startswith("model name")))
    info["cores"] = cpu.count("processor")
except Exception as e:
    info["cpu_err"] = str(e)
info["numpy"] = np.__version__
info["threads_env"] = {k: os.environ.get(k) for k in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"]}

# gemm microbench at several sizes, 200 iters each
def gemm_ms(m, n, k, iters=200):
    a = np.random.rand(m, k).astype(np.float32); b = np.random.rand(k, n).astype(np.float32)
    c = a @ b
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    return (time.perf_counter() - t0) / iters * 1e3

info["gemm_us"] = {}
for (m, n, k) in [(16,32,32),(16,48,32),(16,96,32),(16,16,16),(32,64,64),(64,128,64),(16,256,32),(16,32,64)]:
    info["gemm_us"][f"{m}x{n}x{k}"] = round(gemm_ms(m, n, k), 3)

# stage profile on TINY with opt2-like pipeline
cfg = TINY
S = cfg.seq_len; d = cfg.d_model; nh, dh = cfg.n_heads, cfg.d_head
W32 = RNG(0).weights(cfg, dtype=np.float32)
x = make_input(cfg)
prep = None
from src.impls import _prep_weights, _maskadd
prep = _prep_weights(W32, cfg)
maskadd = _maskadd(S)
inv = np.float32(1.0 / np.sqrt(dh))

def stage(name, fn, n=2000):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = (time.perf_counter() - t0) / n * 1e6
    return round(dt, 2)

h = W32["emb"][x[0]]
def s_emb(): return W32["emb"][x[0]]
def s_ln():
    mu = h.mean(-1, keepdims=True)
    var = (h * h).mean(-1, keepdims=True) - mu * mu
    rstd = 1.0 / np.sqrt(var + cfg.eps)
    return (h - mu) * (rstd * W32["ln1_g0"]) + W32["ln1_b0"]
h2 = s_ln()
def s_qkv(): return h2 @ prep["qkv"][0]
qkv = s_qkv()
def s_views():
    Q = qkv[:, :d].reshape(S, nh, dh).transpose(1, 0, 2)
    K = qkv[:, d:2*d].reshape(S, nh, dh).transpose(1, 0, 2)
    V = qkv[:, 2*d:3*d].reshape(S, nh, dh).transpose(1, 0, 2)
    return Q, K, V
Q, K, V = s_views()
def s_att():
    att = Q @ K.transpose(0, 2, 1)
    att += maskadd
    e = np.exp(att)
    e /= e.sum(-1, keepdims=True)
    return e
e = s_att()
def s_ctx(): return (e @ V).transpose(1, 0, 2).reshape(S, d)
def s_wo(): return h + s_ctx() @ W32["wo0"]
def s_ffn():
    hh = s_wo()
    mu = hh.mean(-1, keepdims=True)
    var = (hh * hh).mean(-1, keepdims=True) - mu * mu
    rstd = 1.0 / np.sqrt(var + cfg.eps)
    h2b = (hh - mu) * (rstd * W32["ln2_g0"]) + W32["ln2_b0"]
    return hh + (np.maximum(h2b, 0) @ W32["w10"]) @ W32["w20"]
def s_out():
    hh = s_ffn()
    mu = hh.mean(-1, keepdims=True)
    var = (hh * hh).mean(-1, keepdims=True) - mu * mu
    rstd = 1.0 / np.sqrt(var + cfg.eps)
    hf = (hh - mu) * (rstd * W32["lnf_g"]) + W32["lnf_b"]
    return (hf @ W32["out"])[None]

stages = {}
for name, fn in [("emb", s_emb), ("ln1", s_ln), ("qkv_matmul", s_qkv),
                 ("qkv_views", s_views), ("att_softmax", s_att), ("ctx_matmul", s_ctx),
                 ("wo_add", s_wo), ("ffn", s_ffn), ("final_ln_out", s_out)]:
    stages[name] = stage(name, fn)

fn = IMPLS["numpy_opt1"]
fn(W32, x, cfg)
t0 = time.perf_counter()
N = 2000
for _ in range(N):
    fn(W32, x, cfg)
info["opt1_full_us"] = round((time.perf_counter() - t0) / N * 1e6, 2)
fn2 = IMPLS["numpy_opt2"]
fn2(W32, x, cfg)
t0 = time.perf_counter()
for _ in range(N):
    fn2(W32, x, cfg)
info["opt2_full_us"] = round((time.perf_counter() - t0) / N * 1e6, 2)

info["stages_us"] = stages


# --- micro-alternatives on TINY shapes ---
def micro(name, fn, n=3000):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return round((time.perf_counter() - t0) / n * 1e6, 3)

micro_res = {}
S, d = 16, 32; nh, dh = 2, 16
rng = np.random.default_rng(0)
hh = rng.normal(size=(S, d)).astype(np.float32)
qq = rng.normal(size=(nh, S, dh)).astype(np.float32)
kk = rng.normal(size=(nh, S, dh)).astype(np.float32)
vv = rng.normal(size=(nh, S, dh)).astype(np.float32)
ww = rng.normal(size=(S, d, d)).astype(np.float32)  # dummy

# 1) layernorm stats: mean+mean vs stacked-matmul vs einsum
def stats_mm():
    mu = hh.mean(-1, keepdims=True)
    var = (hh * hh).mean(-1, keepdims=True) - mu * mu
    return mu, var
ones2 = np.ones((2*d, 2), np.float32) / d
hc = np.empty((S, 2*d), np.float32)
def stats_mat():
    np.concatenate([hh, hh*hh], axis=-1, out=hc)
    st = hc @ ones2
    return st[:, :1], st[:, 1:]
def stats_ein():
    mu = np.einsum("ij->i", hh, optimize=True) / d
    var = np.einsum("ij,ij->i", hh, hh, optimize=True) / d - mu * mu
    return mu, var
micro_res["ln_stats_mm_us"] = micro("mm", stats_mm)
micro_res["ln_stats_mat_us"] = micro("mat", stats_mat)
micro_res["ln_stats_ein_us"] = micro("ein", stats_ein)

# 2) attention: matmul w/ transpose vs einsum
def att_mm():
    return qq @ kk.transpose(0, 2, 1)
def att_ein():
    return np.einsum("hij,hkj->hik", qq, kk, optimize=True)
micro_res["att_matmul_us"] = micro("attmm", att_mm)
micro_res["att_einsum_us"] = micro("attein", att_ein)

# 3) softmax: exp+sum+div vs exp+sum+recip+mul
def soft_div():
    e = np.exp(att_mm())
    return e / e.sum(-1, keepdims=True)
def soft_mul():
    e = np.exp(att_mm())
    return e * (1.0 / e.sum(-1, keepdims=True))
micro_res["softmax_div_us"] = micro("softdiv", soft_div)
micro_res["softmax_mul_us"] = micro("softmul", soft_mul)

# 4) ctx: matmul vs einsum
def ctx_mm():
    return soft_mul() @ vv
def ctx_ein():
    return np.einsum("hij,hjk->hik", soft_mul(), vv, optimize=True)
micro_res["ctx_matmul_us"] = micro("ctxmm", ctx_mm)
micro_res["ctx_einsum_us"] = micro("ctxein", ctx_ein)

# 5) qkv matmul: 2D (S,d)@(d,3d) vs einsum
h2m = rng.normal(size=(S, d)).astype(np.float32)
P = rng.normal(size=(d, 3*d)).astype(np.float32)
def qkv_mm():
    return h2m @ P
def qkv_ein():
    return np.einsum("ij,jk->ik", h2m, P, optimize=True)
micro_res["qkv_matmul_us"] = micro("qkvmm", qkv_mm)
micro_res["qkv_einsum_us"] = micro("qkvein", qkv_ein)

info["micro_us"] = micro_res
print(json.dumps(info, indent=1))

