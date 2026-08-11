"""opt8 micro: pow vs sqrt+div, sum vs matmul-ones, exp+add vs exp+mul, recip+mul vs div."""
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
import numpy as np

rng = np.random.default_rng(1)
S, d, H = 16, 32, 2
h = rng.normal(size=(S, d)).astype(np.float32)
mu = h.mean(-1, keepdims=True).astype(np.float32)
ex2 = (h*h).mean(-1, keepdims=True).astype(np.float32)
att = rng.normal(size=(H, S, S)).astype(np.float32)
e = np.exp(att - att.max())
ones = np.ones((S, 1), np.float32)

def bench(fn, n=5000):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6

res = {}
# rstd variants (on (S,1))
def r1():  # current: add, sqrt, div
    v = ex2 - mu*mu
    return 1.0 / np.sqrt(v + 1e-5)
def r2():  # add, pow
    v = ex2 - mu*mu
    return (v + 1e-5) ** -0.5
def r3():  # eps-folded: sub, pow (ex2 already has +eps)
    return (ex2 - mu*mu) ** -0.5
def r4():  # eps-folded: sqrt, div
    v = ex2 - mu*mu
    return 1.0 / np.sqrt(v)
res["rstd_sqrt_div"] = bench(r1)
res["rstd_add_pow"] = bench(r2)
res["rstd_pow_epsfold"] = bench(r3)
res["rstd_sqrtdiv_epsfold"] = bench(r4)
# softmax denominator
def d1():
    return e.sum(-1, keepdims=True)
def d2():
    return np.matmul(e, ones, out=None)
def d3():
    return np.einsum("hij->hi", e)
res["den_sum"] = bench(d1)
res["den_matmul"] = bench(d2)
res["den_einsum"] = bench(d3)
# exp with mask: add-then-exp vs exp-then-mul
maskadd = np.where(np.tril(np.ones((S,S), bool)), 0.0, -1e9).astype(np.float32)
causal = np.tril(np.ones((S, S), np.float32))
def e1():
    return np.exp(att + maskadd)
def e2():
    return np.exp(att) * causal
res["exp_addmask"] = bench(e1)
res["exp_mulcausal"] = bench(e2)
# ctx normalize: mul by recip vs div
V = rng.normal(size=(H, S, 16)).astype(np.float32)
def c1():
    ctx = np.matmul(e, V)
    return ctx * (1.0 / e.sum(-1, keepdims=True))
def c2():
    ctx = np.matmul(e, V)
    return ctx / e.sum(-1, keepdims=True)
res["ctx_recip_mul"] = bench(c1)
res["ctx_div"] = bench(c2)
# qkv: (h-mu)@P vs h@P - mu*c
P = rng.normal(size=(d, 3*d)).astype(np.float32)
c = P.sum(0)
hm = np.empty((S, d), np.float32)
def q1():
    np.subtract(h, mu, out=hm)
    return hm @ P
def q2():
    return h @ P - mu * c
res["qkv_sub_mat"] = bench(q1)
res["qkv_mat_subc"] = bench(q2)
# gather
x = rng.integers(0, 256, (S,), dtype=np.int64)
emb = rng.normal(size=(256, d)).astype(np.float32)
def g1():
    return emb[x]
def g2():
    return np.take(emb, x, axis=0)
res["gather_fancy"] = bench(g1)
res["gather_take"] = bench(g2)
print(json.dumps(res, indent=1))
