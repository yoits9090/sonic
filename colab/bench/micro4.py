"""matmul orientation: (S,d)@(d,N) vs (N,d)@(d,S) transposed-output form"""
import os, sys, json, time
try:
    _HERE = os.path.dirname(os.path.realpath(__file__))
except NameError:
    _HERE = os.getcwd()
sys.path.insert(0, _HERE)
import numpy as np
rng = np.random.default_rng(3)
res = {}
def bench(fn, n=3000):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6

for name, S, d, N in [("qkv", 16, 32, 96), ("wo", 16, 32, 32), ("out", 16, 32, 256),
                      ("qkv_d", 32, 64, 192), ("out_d", 32, 64, 512), ("w1_d", 32, 64, 128)]:
    a = rng.normal(size=(S, d)).astype(np.float32)
    w = rng.normal(size=(d, N)).astype(np.float32)
    out1 = np.empty((S, N), np.float32)
    out2 = np.empty((N, S), np.float32)
    def f1():
        np.matmul(a, w, out=out1)
    def f2():
        np.matmul(w.T, a.T, out=out2)   # (N,d)@(d,S) -> (N,S)
    res[f"{name}_normal"] = bench(f1)
    res[f"{name}_trans"] = bench(f2)
print(json.dumps(res, indent=1))
