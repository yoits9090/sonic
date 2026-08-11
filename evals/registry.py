"""Eval registry: generation name -> script path -> summary writer + pass criteria.
An impl is 'saturated' at generation N when it passes everything and its vN
leaderboard is flat (no improvement) for 3+ consecutive rounds.
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

REGISTRY = {
    "v1": {
        "script": "evals/run_all.py",
        "args": ["--node", "{node}"],
        "summary": "results/{node}_evals.json",
        "criteria": {
            "correctness": "max_abs_err < 1e-3 vs float64 reference (tiny + default cfg)",
            "latency": "median/p90/p99/max us, throughput tok/s reported (3000 iters)",
        },
    },
    "v2": {
        "script": "evals/eval_v2.py",
        "args": ["--node", "{node}", "--attempt", "{attempt}"],
        "summary": "results/evals_v2_{node}.json",
        "per_impl": "results/evals_v2_{node}_{impl}_a{attempt}.json",
        "criteria": {
            "scaling": "seq_len in [8,16,32,64,128] at batch=1; batch in [1,4,16] at seq_len=32",
            "correctness": "all grid points abs_err < 1e-3 vs float64 reference",
            "tails": "p90/p99/max us per grid point",
        },
    },
    "v3": {
        "script": "evals/eval_v3.py",
        "args": ["--node", "{node}", "--attempt", "{attempt}"],
        "summary": "results/evals_v3_{node}.json",
        "per_impl": "results/evals_v3_{node}_{impl}_a{attempt}.json",
        "criteria": {
            "stability": "5 weight/input seeds all pass abs_err < 1e-3; latency seed-median spread reported",
            "cold": "cold (first-call) latency vs steady-state median, ratio reported",
            "alloc": "tracemalloc allocation count + peak bytes per forward pass",
        },
    },
}


def gen_names():
    return sorted(REGISTRY)


def summary_path(node, gen):
    return os.path.join(ROOT, REGISTRY[gen]["summary"].format(node=node))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        for g, info in REGISTRY.items():
            print(f"{g}: {info['script']} -> {info['summary']}")
            for k, v in info["criteria"].items():
                print(f"    {k}: {v}")
