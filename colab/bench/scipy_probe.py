"""scipy sgemm availability + speed vs np matmul+add (fixed signature)"""
import os, sys, json, time
try:
    _HERE = os.path.dirname(os.path.realpath(__file__))
except NameError:
    _HERE = os.getcwd()
sys.path.insert(0, _HERE)
import numpy as np
res = {"scipy": False}
try:
    import scipy
    from scipy.linalg.blas import sgemm
    res["scipy"] = scipy.__version__
except Exception as e:
    res["scipy_err"] = str(e)
    print(json.dumps(res))
    sys.exit(0)

rng = np.random.default_rng(0)
S, d = 16, 32
h = rng.normal(size=(S, d)).astype(np.float32)
h2 = h.copy()
ctx = rng.normal(size=(S, d)).astype(np.float32)
wo = rng.normal(size=(d, d)).astype(np.float32)

def bench(fn, n=5000):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6

def np_add():
    np.add(h, ctx @ wo, out=h)

def sg_fused():
    # C = alpha*A@B + beta*C, in place on C (h)
    sgemm(1.0, ctx, wo, 1.0, h, overwrite_c=1)

res["np_matmul_add"] = bench(np_add)
res["sgemm_fused"] = bench(sg_fused)
# verify same result
np.copyto(h2, h)
np.add(h2, ctx @ wo, out=h2)
sg_fused()
res["match"] = float(np.abs(h2 - h).max())
# pure gemm: sgemm vs matmul (qkv-style)
P = rng.normal(size=(d, 3*d)).astype(np.float32)
hm = rng.normal(size=(S, d)).astype(np.float32)
outbuf = np.empty((S, 3*d), np.float32)
def np_mm():
    np.matmul(hm, P, out=outbuf)
def sg_mm():
    sgemm(1.0, hm, P, 0.0, outbuf, overwrite_c=1)
res["np_qkv_matmul"] = bench(np_mm)
res["sgemm_qkv"] = bench(sg_mm)
print(json.dumps(res, indent=1))
