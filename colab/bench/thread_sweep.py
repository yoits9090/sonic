"""Thread sweep for numpy_opt7 on a node. Env: NODE. Runs bench with
OPENBLAS_NUM_THREADS in {1,2,4} + OMP_NUM_THREADS, prints JSON lines.
NOTE: env must be set before numpy import -> run via colab exec --env ... per variant.
This script just benches the CURRENT env config; orchestrate calls it 3x.
"""
import os, sys, json, time
_HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, _HERE)
for _d in ("/content", _HERE):
    if os.path.isfile(os.path.join(_d, "src", "impls.py")):
        sys.path.insert(0, _d)
        break
import numpy as np
from src.config import DEFAULT, TINY
from src.random_state import RNG, make_input
from src.impls import IMPLS

def bench(fn, W, x, cfg, iters, warmup):
    fn(W, x, cfg)
    for _ in range(warmup):
        fn(W, x, cfg)
    times = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        t0 = time.perf_counter_ns()
        fn(W, x, cfg)
        times[i] = (time.perf_counter_ns() - t0) / 1000.0
    return times

impl = os.environ.get("IMPL", "numpy_opt7")
cfgname = os.environ.get("CFG", "tiny")
iters = int(os.environ.get("ITERS", "3000"))
warmup = int(os.environ.get("WARMUP", "300"))
node = os.environ.get("NODE", "bench-node-1")
cfg = TINY if cfgname == "tiny" else DEFAULT
W32 = RNG(0).weights(cfg, dtype=np.float32)
x = make_input(cfg)
t = bench(IMPLS[impl], W32, x, cfg, iters, warmup)
print(json.dumps({
    "impl": impl, "cfg": cfgname, "node": node,
    "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    "median_us": float(np.median(t)), "mean_us": float(t.mean()),
    "p90_us": float(np.percentile(t, 90)),
}))
