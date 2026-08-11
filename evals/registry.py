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
    "v4": {
        "script": "evals/eval_v4.py",
        "args": ["--node", "{node}", "--attempt", "{attempt}"],
        "summary": "results/evals_v4_{node}.json",
        "per_impl": "results/evals_v4_{node}_{impl}_a{attempt}.json",
        "criteria": {
            "edge_seq": "seq_len in [1,2,16,32,64,256] (incl d_head/d_model edges, S>d_model) abs_err < 1e-3",
            "distributions": "uniform/skewed/same/edges/bursty token inputs abs_err < 1e-3",
        },
    },
    "v5": {
        "script": "evals/eval_v5.py",
        "args": ["--node", "{node}", "--attempt", "{attempt}"],
        "summary": "results/evals_v5_{node}.json",
        "per_impl": "results/evals_v5_{node}_{impl}_a{attempt}.json",
        "criteria": {
            "headroom": "theoretical FLOPs, achieved GF/s, pure-matmul ceiling, overhead x, headroom %",
        },
    },
    "v6": {
        "script": "evals/eval_v6.py",
        "args": ["--node", "{node}", "--attempt", "{attempt}"],
        "summary": "results/evals_v6_{node}.json",
        "per_impl": "results/evals_v6_{node}_{impl}_a{attempt}.json",
        "criteria": {
            "op-domain": "median_us < 1000 on ALL cells: batch{1,4,16} x seq{16,32,64} x cfg{tiny,default}",
            "roofline": "achieved GF/s vs measured peak GEMM GF/s per cfg",
            "cold": "first-call latency incl. ctypes/prep/alloc vs steady median",
            "b_general": "flag impls failing batch>1 cells (B=1-only) and assess B>1 worth",
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
