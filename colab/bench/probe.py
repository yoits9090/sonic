"""Node capability probe + per-stage timing profile (numpy-optimizer agent).
Prints JSON: cpu info, numpy/BLAS info, matmul micro-bench, per-stage us for
the opt2 pipeline, and thread-count experiments for a 512x512 gemm.
"""
import os, sys, json, time, platform
_HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, _HERE)
_d = _HERE
for _ in range(4):
    if os.path.isfile(os.path.join(_d, "src", "impls.py")):
        sys.path.insert(0, _d)
        break
    _d = os.path.dirname(_d)
import numpy as np
from src.config import DEFAULT, TINY
from src.random_state import RNG, make_input
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
print(json.dumps(info, indent=1))
