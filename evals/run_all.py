"""Run correctness + latency for all impls on the current node. Writes results/nodeX_evals.json
Usage: python evals/run_all.py --node bench-node-1
"""
import argparse, json, os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    node = a.node
    import importlib.util
    spec = importlib.util.spec_from_file_location("impls", os.path.join(ROOT, "src", "impls.py"))
    impls = importlib.util.module_from_spec(spec); spec.loader.exec_module(impls)
    out = {}
    for name in impls.IMPLS:
        r = subprocess.run([sys.executable, os.path.join(ROOT, "evals", "eval_correctness.py"),
                            "--impl", name], capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr[-500:])
        # correctness result parse
        last = r.stdout.strip().splitlines()[-1]
        out[name] = {"correctness": last}
        r2 = subprocess.run([sys.executable, os.path.join(ROOT, "evals", "eval_latency.py"),
                             "--impl", name, "--node", node, "--iters", "3000"],
                            capture_output=True, text=True)
        try:
            out[name]["latency"] = json.loads(r2.stdout.strip().splitlines()[-1])
        except Exception as e:
            out[name]["latency_err"] = (r2.stdout + r2.stderr)[-300:]
            print((r2.stdout + r2.stderr)[-300:])
    path = a.out or os.path.join(ROOT, "results", f"{node}_evals.json")
    json.dump(out, open(path, "w"), indent=1)
    print("WROTE", path)

if __name__ == "__main__":
    main()
