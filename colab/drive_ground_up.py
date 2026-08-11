#!/usr/bin/env python3
"""Local driver for the ground-up-C flow on bench-node-2.
Usage: python3 colab/drive_ground_up.py [--attempt 01] [--skip-upload] [--skip-run] [--skip-download]
Uploads ground_up/ + src/ + evals/ to /content/ft_gu, builds libft on the node,
runs correctness/latency/v3 evals, downloads results JSONs.
"""
import argparse, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = "bench-node-2"
REMOTE = "/content/ft_gu"

UPLOAD = [
    "ground_up/ft.h", "ground_up/matmul.c", "ground_up/transformer.c",
    "ground_up/build.sh", "ground_up/node_run.py",
    "src/__init__.py", "src/config.py", "src/random_state.py", "src/impls.py",
    "evals/__init__.py", "evals/eval_correctness.py", "evals/eval_latency.py",
    "evals/eval_v3.py",
]

def sh(args, timeout=300):
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed ({r.returncode}): {' '.join(args)}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return r.stdout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", default="01")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--skip-run", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()

    if not a.skip_upload:
        for f in UPLOAD:
            remote = f"{REMOTE}/{f}"
            print(f"[driver] upload {f}")
            sh(["colab", "--auth", "adc", "upload", "-s", NODE,
                os.path.join(ROOT, f), remote], timeout=180)

    if not a.skip_run:
        runner = (f"import runpy, sys, os\n"
                  f"os.environ['NODE'] = '{NODE}'\n"
                  f"os.environ['ATTEMPT'] = '{a.attempt}'\n"
                  f"sys.path.insert(0, '{REMOTE}')\n"
                  f"sys.argv = ['/content/ft_gu/ground_up/node_run.py']\n"
                  f"runpy.run_path('/content/ft_gu/ground_up/node_run.py', run_name='__main__')\n")
        r = subprocess.run(["colab", "--auth", "adc", "exec", "-s", NODE, "--timeout", "2400"],
                           input=runner, capture_output=True, text=True, timeout=2600)
        out = (r.stdout or "") + (r.stderr or "")
        print(out[-6000:])
        if r.returncode != 0 and "ALL DONE" not in out:
            raise RuntimeError(f"exec failed rc={r.returncode}")

    if not a.skip_download:
        os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
        for remote, local in DOWNLOAD_FILES(a.attempt):
            try:
                sh(["colab", "--auth", "adc", "download", "-s", NODE, "-f", remote, local], timeout=180)
                print(f"[driver] downloaded {os.path.basename(local)}")
            except RuntimeError as e:
                print(f"[driver] warn: {e}")

def DOWNLOAD_FILES(attempt):
    out = []
    for impl in ["c_v1", "c_v2", "c_v3", "c_v4", "c_v5", "c_ground_up", "numpy_vec"]:
        for cfg in ["tiny", "default"]:
            out.append((f"{REMOTE}/results/{NODE}_{impl}_{cfg}_correctness_{attempt}.json",
                        os.path.join(ROOT, "results", f"{NODE}_{impl}_{cfg}_correctness_{attempt}.json")))
            out.append((f"{REMOTE}/results/{NODE}_{impl}_{cfg}_latency_{attempt}.json",
                        os.path.join(ROOT, "results", f"{NODE}_{impl}_{cfg}_latency_{attempt}.json")))
        out.append((f"{REMOTE}/results/evals_v3_{NODE}_{impl}_a{attempt}.json",
                    os.path.join(ROOT, "results", f"evals_v3_{NODE}_{impl}_a{attempt}.json")))
    out.append((f"{REMOTE}/results/{NODE}_summary_{attempt}.json",
                os.path.join(ROOT, "results", f"{NODE}_summary_{attempt}.json")))
    out.append((f"{REMOTE}/results/evals_v3_{NODE}.json",
                os.path.join(ROOT, "results", f"evals_v3_{NODE}.json")))
    return out

if __name__ == "__main__":
    main()
