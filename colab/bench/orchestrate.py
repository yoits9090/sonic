#!/usr/bin/env python3
"""Local orchestrator for numpy-optimizer benches (runs on laptop; compute on node).
Usage: python3 colab/bench/orchestrate.py --node bench-node-1 [--impls ...] [--cfgs ...]
       [--iters 3000 --warmup 300] [--attempt 1] [--probe] [--env OPENBLAS_NUM_THREADS=1]
Steps: sync files -> exec runner on node -> parse CORR/LAT lines -> write results/*.json
"""
import argparse, json, os, subprocess, sys, re

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)

def sh(cmd, timeout=900):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        print("CMD FAIL:", cmd, r.stderr[-800:])
        sys.exit(1)
    return r.stdout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", default="bench-node-1")
    ap.add_argument("--impls", default="numpy_naive,numpy_vec,numpy_opt1,numpy_opt2,numpy_opt3,numpy_opt4,numpy_opt5,numpy_opt6")
    ap.add_argument("--cfgs", default="tiny,default")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--timeout", type=int, default=1200)
    a = ap.parse_args()

    node = a.node
    # 1) ensure dirs + sync src files (exec -f takes LOCAL paths and uploads itself)
    print(f"[sync] uploading src/ to {node}...")
    sh(f"colab --auth adc exec -s {node} -f colab/bench/mkdirs.py --timeout 120")
    sh(f"colab --auth adc upload -s {node} src/__init__.py /content/src/__init__.py")
    sh(f"colab --auth adc upload -s {node} src/config.py /content/src/config.py")
    sh(f"colab --auth adc upload -s {node} src/random_state.py /content/src/random_state.py")
    sh(f"colab --auth adc upload -s {node} src/impls.py /content/src/impls.py")

    if a.probe:
        print("[probe] running probe.py...")
        out = sh(f"colab --auth adc exec -s {node} -f colab/bench/probe.py --timeout 300")
        print(out)
        fn = f"results/{node}_probe.json"
        os.makedirs("results", exist_ok=True)
        # extract the json object from output
        m = re.search(r'\{[^{}]*"platform".*?\}', out, re.S)
        if m:
            json.dump(json.loads(m.group(0)), open(fn, "w"), indent=1)
            print("wrote", fn)
        return

    # 2) exec battery
    envs = " ".join(f"--env {e}" for e in a.env)
    cmd = (f"colab --auth adc exec -s {node} -f colab/bench/run_bench.py --timeout {a.timeout} "
           f"--env IMPLS={a.impls} --env CFGS={a.cfgs} --env ITERS={a.iters} "
           f"--env WARMUP={a.warmup} --env NODE={node} {envs}")
    print("[exec]", cmd)
    out = sh(cmd)
    print(out)

    # 3) parse and write results
    os.makedirs("results", exist_ok=True)
    n_ok = 0
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("CORR "):
            corr = json.loads(line[5:])
        elif line.startswith("LAT "):
            lat = json.loads(line[4:])
            impl, cfg = lat["impl"], lat["cfg"]
            fn = f"results/{node}_{impl}_{cfg}_a{a.attempt}.json"
            lat["correctness"] = corr
            json.dump(lat, open(fn, "w"), indent=1)
            print(f"wrote {fn}  median_us={lat['median_us']:.1f} pass={corr['pass']}")
            n_ok += 1
    print(f"[done] {n_ok} results written")

if __name__ == "__main__":
    main()
