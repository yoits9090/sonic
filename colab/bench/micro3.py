"""contiguity + scalar tweaks on node"""
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
rng = np.random.default_rng(2)
res = {}
def bench(fn, n=3000):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6

# attention matmul: strided views vs contiguous copies (tiny + default shapes)
for name, H, S, dh in [("tiny", 2, 16, 16), ("default", 4, 32, 16)]:
    q = rng.normal(size=(S, H, dh)).astype(np.float32)
    k = rng.normal(size=(S, H, dh)).astype(np.float32)
    Qv = q.transpose(1, 0, 2)          # (H,S,dh) strided view
    Kv = k.transpose(1, 0, 2)
    Kt = Kv.transpose(0, 2, 1)         # view
    Qc = np.ascontiguousarray(Qv)
    Kc = np.ascontiguousarray(Kv)
    Ktc = Kc.transpose(0, 2, 1)        # view of contiguous -> still transposed view
    att_buf = np.empty((H, S, S), np.float32)
    def f_view():
        np.matmul(Qv, Kt, out=att_buf)
    def f_copy():
        np.matmul(Qc, Ktc, out=att_buf)
    res[f"att_{name}_view"] = bench(f_view)
    res[f"att_{name}_copy"] = bench(f_copy)
    # ctx matmul strided vs contiguous
    e = rng.normal(size=(H, S, S)).astype(np.float32)
    Vv = q.transpose(1, 0, 2)
    Vc = np.ascontiguousarray(Vv)
    ctx_buf = np.empty((H, S, dh), np.float32)
    def c_view():
        np.matmul(e, Vv, out=ctx_buf)
    def c_copy():
        np.matmul(e, Vc, out=ctx_buf)
    res[f"ctx_{name}_view"] = bench(c_view)
    res[f"ctx_{name}_copy"] = bench(c_copy)

# exp on f32 vs exp2 with prescaled
att = rng.normal(size=(2, 16, 16)).astype(np.float32)
eb = np.empty_like(att)
log2e = np.float32(1.4426950408889634)
att_s = att * log2e
def fexp():
    np.exp(att, out=eb)
def fexp2():
    np.exp2(att_s, out=eb)
res["exp"] = bench(fexp)
res["exp2_prescaled"] = bench(fexp2)

# power f32 literal vs python float
x = np.abs(rng.normal(size=(32, 1))).astype(np.float32) + 1e-5
def p1():
    return x ** -0.5
def p2():
    return x ** np.float32(-0.5)
res["pow_pyfloat"] = bench(p1)
res["pow_f32"] = bench(p2)
print(json.dumps(res, indent=1))
