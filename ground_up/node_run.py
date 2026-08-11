#!/usr/bin/env python3
"""Run on a Colab node: build libft variants, then correctness + latency evals.
Usage:  NODE=bench-node-2 python3 ground_up/node_run.py
Writes JSON results to results/ (names contain node + impl + attempt).
"""
import json, os, subprocess, sys, time

NODE = os.environ.get("NODE", "bench-node-2")
ATTEMPT = os.environ.get("ATTEMPT", "01")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

def sh(cmd, **kw):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)

t0 = time.time()
# 1. build
r = sh("bash ground_up/build.sh", timeout=900)
print(r.stdout[-4000:]); print(r.stderr[-2000:])
if r.returncode != 0:
    sys.exit(f"BUILD FAILED rc={r.returncode}")

def run_eval(script, args, timeout=900):
    cmd = f"python {script} {' '.join(args)}"
    r = sh(cmd, timeout=timeout)
    return r

c_impls = ["c_v1", "c_v2", "c_v3", "c_v4", "c_v5", "c_ground_up"]
summary = {}

# 2. correctness (both cfgs, all c impls + numpy reference impl)
for cfg in ("tiny", "default"):
    for impl in c_impls + ["numpy_vec"]:
        out = f"results/{NODE}_{impl}_{cfg}_correctness_{ATTEMPT}.json"
        r = run_eval("evals/eval_correctness.py",
                     ["--impl", impl, "--cfg", cfg, "--out", out])
        tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "NO OUTPUT"
        summary[f"corr {impl} {cfg}"] = tail
        print(f"[corr] {impl} {cfg}: rc={r.returncode} {tail}", flush=True)

# 3. latency (both cfgs, all c impls)
for cfg in ("tiny", "default"):
    for impl in c_impls:
        out = f"results/{NODE}_{impl}_{cfg}_latency_{ATTEMPT}.json"
        r = run_eval("evals/eval_latency.py",
                     ["--impl", impl, "--cfg", cfg, "--node", NODE, "--out", out])
        if r.returncode == 0 and r.stdout.strip():
            j = json.loads(r.stdout)
            summary[f"lat {impl} {cfg}"] = (j["median_us"], j["sub_ms"])
            print(f"[lat] {impl} {cfg}: median={j['median_us']:.1f}us sub_ms={j['sub_ms']}", flush=True)
        else:
            summary[f"lat {impl} {cfg}"] = f"rc={r.returncode} {r.stderr[-300:]}"
            print(f"[lat] {impl} {cfg}: FAILED rc={r.returncode}", flush=True)

# 4. numpy baseline for the journal
for cfg in ("tiny", "default"):
    out = f"results/{NODE}_numpy_vec_{cfg}_latency_{ATTEMPT}.json"
    r = run_eval("evals/eval_latency.py",
                 ["--impl", "numpy_vec", "--cfg", cfg, "--node", NODE, "--out", out])
    if r.returncode == 0 and r.stdout.strip():
        j = json.loads(r.stdout)
        summary[f"lat numpy_vec {cfg}"] = (j["median_us"], j["sub_ms"])
        print(f"[lat] numpy_vec {cfg}: median={j['median_us']:.1f}us", flush=True)


# 5. escalation: eval_v3 (5-seed correctness, latency stability, cold call, alloc counts)
import shutil
for impl in ["c_v3", "c_v5", "c_ground_up", "numpy_vec"]:
    r = run_eval("evals/eval_v3.py",
                 ["--node", NODE, "--impl", impl, "--attempt", ATTEMPT])
    tail = [l for l in r.stdout.strip().splitlines() if l.startswith("[v3")][-1] if r.stdout.strip() else "NO OUTPUT"
    print(f"[v3] {impl}: rc={r.returncode} {tail}", flush=True)

json.dump({"node": NODE, "attempt": ATTEMPT, "elapsed_s": round(time.time() - t0, 1),
           "summary": summary},
          open(f"results/{NODE}_summary_{ATTEMPT}.json", "w"), indent=1)
print("ALL DONE in", round(time.time() - t0, 1), "s")
