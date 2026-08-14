"""Benchmark figure: regular (numpy) transformer vs sonic, from results/leaderboard.json.

Two panels:
  A — canonical cells (TINY B1 S16, DEFAULT B1 S32), log scale, speedup labels.
  B — full batch x seq grid (18 cells), log scale, grouped by impl.
Writes docs/benchmarks.png.
"""
import json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LB = json.load(open(os.path.join(ROOT, "results", "leaderboard.json")))["impls"]

def shape_of(key):
    m = re.search(r"d=(\d+).*?S=(\d+).*?B=(\d+)", key)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None

def get(impl, d, s, b):
    for k, v in LB.get(impl, {}).items():
        sh = shape_of(k)
        if sh == (d, s, b):
            return v["median_us"]
    return None

IMPLS = ["numpy_naive", "numpy_vec", "numpy_opt12", "c_v1", "c_ground_up"]
LABELS = {"numpy_naive": "numpy (naive)", "numpy_vec": "numpy (reference)",
          "numpy_opt12": "numpy (opt12)", "c_v1": "C (naive)", "c_ground_up": "sonic"}
COLORS = {"numpy_naive": "#bdbdbd", "numpy_vec": "#9e9e9e", "numpy_opt12": "#616161",
          "c_v1": "#90caf9", "c_ground_up": "#1565c0"}
HATCH = {"numpy_naive": None, "numpy_vec": None, "numpy_opt12": None, "c_v1": None, "c_ground_up": "//"}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 2.2]})

# ---- Panel A: canonical cells ----
canon = [("TINY  (d=32, L=1, S=16, V=256)", 32, 16, 1), ("DEFAULT  (d=64, L=2, S=32, V=512)", 64, 32, 1)]
x = np.arange(len(canon) * len(IMPLS))
width = 0.8 / len(IMPLS)
for i, (label, d, s, b) in enumerate(canon):
    vals = [get(impl, d, s, b) for impl in IMPLS]
    for j, impl in enumerate(IMPLS):
        v = vals[j]
        axA.bar(x[i * len(IMPLS) + j], v, width=width * 0.9, color=COLORS[impl],
                edgecolor="white", linewidth=0.5, label=LABELS[impl] if i == 0 else None)
    ref = get("numpy_vec", d, s, b)
    sonic = get("c_ground_up", d, s, b)
    axA.text(i * len(IMPLS) + len(IMPLS) - 1, sonic * 1.15, f"{ref / sonic:.0f}x", ha="center", fontsize=11, fontweight="bold", color="#0d47a1")
axA.set_yscale("log")
axA.set_xticks(x)
axA.set_xticklabels([f"{LABELS[i]}" for i in IMPLS] * 2, rotation=30, ha="right", fontsize=7.5)
axA.set_title("Canonical forward pass (median, µs — log scale)", fontsize=10)
axA.set_ylabel("µs (lower is better)")
axA.grid(axis="y", alpha=0.3)

# ---- Panel B: 18-cell grid ----
cfgs = [(32, "tiny"), (64, "default")]
xtick = []
xx = []
for d, cfg in cfgs:
    for b in (1, 4, 16):
        for s in (16, 32, 64):
            xx.append((d, s, b, f"{cfg} B{b} S{s}"))
pos = np.arange(len(xx))
nb = len(["numpy_vec", "numpy_opt12", "c_ground_up"])
w = 0.8 / nb
for j, impl in enumerate(["numpy_vec", "numpy_opt12", "c_ground_up"]):
    vals = [get(impl, d, s, b) for d, s, b, _ in xx]
    axB.bar(pos + (j - 1) * w, vals, width=w * 0.9, color=COLORS[impl], edgecolor="white",
            linewidth=0.5, label=LABELS[impl])
axB.set_yscale("log")
axB.set_xticks(pos)
axB.set_xticklabels([t for _, _, _, t in xx], rotation=60, ha="right", fontsize=6.5)
axB.set_title("Batch × sequence grid (median, µs — log scale)", fontsize=10)
axB.axhline(1000, color="crimson", lw=0.8, ls="--")
axB.text(len(xx) - 0.5, 1000, " 1 ms", color="crimson", fontsize=8, va="bottom", ha="right")
axB.grid(axis="y", alpha=0.3)

handles, labels = axA.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=9, frameon=False)
fig.suptitle("", fontsize=0)
fig.tight_layout(rect=(0, 0.03, 1, 0.92))
out = os.path.join(ROOT, "docs", "benchmarks.png")
fig.savefig(out, dpi=150)
print("wrote", out)
