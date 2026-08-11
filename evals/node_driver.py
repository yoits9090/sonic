"""Drive an eval generation on a colab node: upload code, exec, pull results.
Usage: python evals/node_driver.py --node bench-node-3 --gen v1 [--attempt N] [--impl numpy_vec] [--timeout 900]
Writes results/manifest.json attempt log. All compute happens on the node.
"""
import argparse, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REMOTE_BASE = "/content/ft_evals3"   # per-attempt subdir added at runtime: {REMOTE_BASE}/a{attempt}

UPLOAD_FILES = [
    "src/config.py", "src/random_state.py", "src/impls.py",
    "evals/run_all.py", "evals/eval_correctness.py", "evals/eval_latency.py",
    "evals/eval_v2.py", "evals/eval_v3.py", "evals/eval_v4.py", "evals/eval_v5.py", "evals/eval_v6.py", "evals/registry.py",
    "ground_up/build.sh", "ground_up/ft.h", "ground_up/matmul.c", "ground_up/transformer.c",
]

RUNNERS = {
    "v1": ("evals/run_all.py", ["--node", "{node}"]),
    "v2": ("evals/eval_v2.py", ["--node", "{node}", "--attempt", "{attempt}"]),
    "v3": ("evals/eval_v3.py", ["--node", "{node}", "--attempt", "{attempt}"]),
    "v4": ("evals/eval_v4.py", ["--node", "{node}", "--attempt", "{attempt}"]),
    "v5": ("evals/eval_v5.py", ["--node", "{node}", "--attempt", "{attempt}"]),
    "v6": ("evals/eval_v6.py", ["--node", "{node}", "--attempt", "{attempt}"]),
}
ALL_SEQ = ["v1", "v2", "v3", "v4", "v5", "v6"]
SUMMARY_FILES = {
    "v1": ["results/{node}_evals.json"],
    "v2": ["results/evals_v2_{node}.json", "results/evals_v2_{node}_{impl}_a{attempt}.json"],
    "v3": ["results/evals_v3_{node}.json", "results/evals_v3_{node}_{impl}_a{attempt}.json"],
    "v4": ["results/evals_v4_{node}.json", "results/evals_v4_{node}_{impl}_a{attempt}.json"],
    "v5": ["results/evals_v5_{node}.json", "results/evals_v5_{node}_{impl}_a{attempt}.json"],
    "v6": ["results/evals_v6_{node}.json", "results/evals_v6_{node}_{impl}_a{attempt}.json"],
}


def sh(args, timeout=300, input=None):
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, input=input)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed ({r.returncode}): {' '.join(args)}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout


def next_attempt(node, gen):
    man_path = os.path.join(ROOT, "results", "manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else {"runs": []}
    n = sum(1 for r in man["runs"] if r["node"] == node and r["gen"] == gen)
    return n + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", required=True)
    ap.add_argument("--gen", required=True, choices=sorted(RUNNERS) + ["all"])
    ap.add_argument("--attempt", type=int, default=None)
    ap.add_argument("--impl", default=None)
    ap.add_argument("--timeout", type=float, default=1200)
    ap.add_argument("--no-build", action="store_true", help="skip building libft .so on the node")
    a = ap.parse_args()
    node = a.node
    gens = ALL_SEQ if a.gen == "all" else [a.gen]
    attempt = a.attempt if a.attempt else max(next_attempt(node, g) for g in gens)
    rb = f"{REMOTE_BASE}/a{attempt}"  # snapshot-isolated run dir (avoids stale kernel module cache)
    runs = []
    for g in gens:
        script, tmpl_args = RUNNERS[g]
        args = [t.format(node=node, attempt=attempt, impl=(a.impl or "*").replace(",", "_")) for t in tmpl_args]
        if a.impl:
            args += ["--impl", a.impl]   # comma-separated list OK (eval scripts split)
        runs.append((script, args))

    # 0. ensure remote dirs
    mk = (f"import os\nfor d in ['src','evals','results','ground_up']: os.makedirs('{rb}/'+d, exist_ok=True)\n"
          f"print('dirs ok')\n")
    sh(["colab", "--auth", "adc", "exec", "-s", node, "--timeout", "60"], input=mk, timeout=180)
    # 1. upload (snapshot: unique per attempt so concurrent edits / kernel caches can't leak)
    for f in UPLOAD_FILES:
        remote = f"{rb}/{os.path.dirname(f)}/{os.path.basename(f)}"
        print(f"[driver] upload {f} -> {remote}")
        sh(["colab", "--auth", "adc", "upload", "-s", node, os.path.join(ROOT, f), remote], timeout=180)
    # 2. exec via stdin runpy (keeps __file__ based paths correct on the node)
    lines = ["import runpy, sys, os, subprocess",
             "for _m in list(sys.modules):",
             "    if _m == 'src' or _m.startswith('src.'): del sys.modules[_m]",
             f"sys.path = [p for p in sys.path if 'ft_evals3' not in p]",
             f"sys.path.insert(0, '{rb}')",
             f"os.environ['FT_GROUND_UP_DIR'] = '{rb}/ground_up'",
             "os.environ.setdefault('OMP_NUM_THREADS', '2')  # avoid OpenMP oversubscription on 2-core nodes"]
    if not a.no_build:
        lines.append(f"rc = subprocess.run(['bash', '{rb}/ground_up/build.sh']).returncode")
        lines.append(f"assert rc == 0, 'build failed rc=%d' % rc")
        lines.append("print('[driver] libft build OK')")
    for script, args in runs:
        lines.append(f"sys.argv = ['{script}'] + {args!r}")
        lines.append(f"runpy.run_path('{rb}/{script}', run_name='__main__')")
    lines.append("None")  # avoid IPython echoing runpy's returned globals dict
    runner = "\n".join(lines) + "\n"
    print(f"[driver] exec {gens} timeout={a.timeout}")
    r = subprocess.run(["colab", "--auth", "adc", "exec", "-s", node, "--timeout", str(a.timeout)],
                       input=runner, capture_output=True, text=True, timeout=a.timeout + 120)
    out = (r.stdout or "") + (r.stderr or "")
    print(out[-4000:])
    if r.returncode != 0 and "WROTE" not in out:
        raise RuntimeError(f"exec failed rc={r.returncode}")
    # 3. download
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    downloaded = []
    impltag = ("_" + a.impl.replace(",", "-")) if a.impl else ""
    for gen in gens:
        for pat in SUMMARY_FILES[gen]:
            fname = pat.format(node=node, attempt=attempt, impl=a.impl or "*")
            if "*" in fname:
                continue  # per-impl files handled below
            remote = f"{rb}/{fname}"
            # impl-subset runs get a tagged local copy so they don't clobber the all-impl summary
            base, ext = os.path.splitext(os.path.basename(fname))
            local = os.path.join(ROOT, "results", f"{base}{impltag}_a{attempt}{ext}" if impltag else f"{base}{ext}")
            try:
                sh(["colab", "--auth", "adc", "download", "-s", node, remote, local], timeout=180)
                downloaded.append(os.path.basename(local))
            except RuntimeError as e:
                print(f"[driver] download warn: {e}")
    for gen in gens:
        if gen not in ("v2", "v3", "v4", "v5", "v6"):
            continue
        sum_local = os.path.join(ROOT, "results", os.path.basename(SUMMARY_FILES[gen][0].format(node=node, attempt=attempt)))
        if os.path.exists(sum_local):
            data = json.load(open(sum_local))
            for impl in data.get("impls", {}):
                    fname = SUMMARY_FILES[gen][1].format(node=node, attempt=attempt, impl=impl)
                    remote = f"{rb}/{fname}"
                    local = os.path.join(ROOT, "results", os.path.basename(fname))
                    try:
                        sh(["colab", "--auth", "adc", "download", "-s", node, remote, local], timeout=180)
                        downloaded.append(os.path.basename(fname))
                    except RuntimeError as e:
                        print(f"[driver] download warn: {e}")
    # 4. manifest
    man_path = os.path.join(ROOT, "results", "manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else {"runs": []}
    for g in gens:
        man["runs"].append({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "node": node, "gen": g,
                            "attempt": attempt, "impl": a.impl, "files": downloaded})
    json.dump(man, open(man_path, "w"), indent=1)
    print(f"[driver] DONE attempt {attempt} files: {downloaded}")


if __name__ == "__main__":
    main()
