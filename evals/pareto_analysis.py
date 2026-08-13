"""Pareto frontier analysis for the eval_v8 sweep: latency-vs-error, per cell, per budget.

Ingests results/evals_v8_<node>_<impl>_a<attempt>.json (+ summaries) and produces:
  results/pareto_<node>.md              — frontier tables (fastest passing impl per budget)
  results/pareto_<node>.json            — machine-readable frontier
  results/pareto_<node>_frontier_tiny.png   — log-log latency-vs-err per tiny cell
  results/pareto_<node>_frontier_default.png
  results/pareto_<node>_summary.png     — sub-ms pass counts per budget + champion speedups

Incremental: tolerates missing impls/cells (frontier is computed from whatever
data is present). If multiple attempts exist per (impl, cell), the minimum
median_us wins (leaderboard min-median convention).

Usage: python evals/pareto_analysis.py [--node ft-node-1] [--budgets 1e-3,1e-2]
"""
import argparse, glob, json, os, re, sys, time
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:  # noqa: BLE001
    HAVE_MPL = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

CHAMPION = "c_fp32_e6"          # incumbent; fall back to c_ground_up entries
SUB_MS = 1000.0
DEFAULT_BUDGETS = [1e-3, 1e-2]
CFGS = ("tiny", "default")
BATCHES = (1, 4, 16)
SEQ_LENS = (16, 32, 64)
ATTEMPT_RE = re.compile(r"evals_v8_(.+)_a(\d+)\.json")


def discover(results_dir, node):
    """Returns (per, node): per = {impl: {cell: {"err", "med", "attempt", "src"}}}
    using the min-median across attempts. Filenames: evals_v8_<node>_<impl>_a<N>.json
    (impl names contain underscores, node names do not). When --node is omitted,
    the node with the most impl files wins."""
    per = {}
    files = sorted(glob.glob(os.path.join(results_dir, "evals_v8_*_a*.json")))
    for path in files:
        m = re.search(r"evals_v8_(.+)_a(\d+)\.json", os.path.basename(path))
        if not m:
            continue
        fnode, impl = m.group(1).split("_", 1)
        attempt = int(m.group(2))
        if node and fnode != node:
            continue
        d = json.load(open(path))
        for cell, c in d.get("cells", {}).items():
            if c.get("median_us") is None or c.get("max_abs_err") is None:
                continue
            med = float(c["median_us"])
            if not np.isfinite(med):
                continue
            e = per.setdefault((fnode, impl), {})
            prev = e.get(cell)
            if prev is None or med < prev["med"]:
                e[cell] = {"err": float(c["max_abs_err"]), "med": med,
                           "attempt": attempt, "src": os.path.basename(path)}
    if node:
        return {impl: cells for (n, impl), cells in per.items() if n == node}, node
    counts = {}
    for (n, _impl) in per:
        counts[n] = counts.get(n, 0) + 1
    if not counts:
        return {}, "pending"
    best = max(counts, key=counts.get)
    print(f"[pareto] no --node given; using node '{best}' ({counts[best]} impl files)")
    return {impl: cells for (n, impl), cells in per.items() if n == best}, best


def all_cells():
    return [f"{cfg}_b{b}_s{s}" for cfg in CFGS for b in BATCHES for s in SEQ_LENS]


def frontier_points(points):
    """points: list of (err, med, impl). Returns the non-dominated list, sorted by err.
    A point p dominates q iff p.err <= q.err and p.med < q.med."""
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    fr, best_med = [], float("inf")
    for err, med, impl in pts:
        if med < best_med:
            fr.append((err, med, impl))
            best_med = med
    return fr


def winner_for_budget(points, budget):
    """Fastest impl with err < budget, else None."""
    ok = [(med, impl) for err, med, impl in points if err < budget and np.isfinite(med)]
    return min(ok) if ok else (None, None)


def cell_frontier(per, cell):
    pts = []
    for impl, cells in per.items():
        c = cells.get(cell)
        if c and np.isfinite(c["med"]):
            pts.append((c["err"], c["med"], impl))
    return frontier_points(pts), pts


def champion_key(per):
    return CHAMPION if CHAMPION in per else ("c_ground_up" if "c_ground_up" in per else None)


def render_md(per, node, budgets, frontier_data, winners, sub_ms_counts):
    champ = champion_key(per)
    lines = [f"# Pareto frontier — latency vs error (node: {node})",
             "",
             f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} by evals/pareto_analysis.py",
             f"Champion impl: `{champ}` (fp32, exp-degree 6; err ~1.1e-6). "
             f"Budgets: max_abs_err < {', '.join(f'{b:g}' for b in budgets)} vs f64 ref.",
             f"Impls with data: {sorted(per)}",
             f"Convention: min median_us across attempts per (impl, cell).",
             ""]
    for budget in budgets:
        bkey = f"{budget:g}"
        lines.append(f"## Budget {bkey} — fastest passing impl per cell")
        lines.append("")
        lines.append("| cell | champion us | champion err | winner | winner us | winner err | speedup vs champ | sub-ms |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for cell in all_cells():
            w = winners[bkey][cell]
            wmed, wimpl = w["med"], w["impl"]
            c = per[champ].get(cell, {}) if champ else {}
            cmed, cerr = c.get("med", float("nan")), c.get("err", float("nan"))
            spd = f"{cmed / wmed:.2f}x" if wmed is not None and np.isfinite(cmed) else "-"
            sub = "-" if wmed is None else ("yes" if wmed < SUB_MS else "no")
            wimp = wimpl or "—"
            wmed_s = f"{wmed:.1f}" if wmed is not None else "—"
            werr = w["err"] if w["err"] is not None else float("nan")
            werr_s = f"{werr:.2e}" if np.isfinite(werr) else "—"
            cmed_s = f"{cmed:.1f}" if np.isfinite(cmed) else "—"
            cerr_s = f"{cerr:.2e}" if np.isfinite(cerr) else "—"
            lines.append(f"| {cell} | {cmed_s} | {cerr_s} | `{wimp}` | {wmed_s} | "
                         f"{werr_s} | {spd} | {sub} |")
        npass = sub_ms_counts[bkey]["n_pass"]
        lines.append("")
        lines.append(f"**Budget {bkey} score: {npass}/18 cells sub-ms with a passing impl.**")
        lines.append("")
    lines.append("## Frontier sets per cell (non-dominated points, all budgets overlaid)")
    lines.append("")
    lines.append("| cell | frontier (err -> impl @ us) |")
    lines.append("|---|---|")
    for cell in all_cells():
        fr = frontier_data[cell]
        items = ", ".join(f"{err:.1e} {impl}@{med:.0f}us" for err, med, impl in fr)
        lines.append(f"| {cell} | {items} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("- `c_fp32_e6` is the incumbent v14 champion (alias of `c_ground_up`/libft.so);")
    lines.append("  its row is the reference point for speedups.")
    lines.append("- Frontier excludes impls whose run errored (median_us non-finite).")
    lines.append("- Dominated points are omitted from frontier sets (strict: p.err<=q.err AND p.med<q.med).")
    return "\n".join(lines)


def make_plots(per, node, budgets, frontier_data, winners, out_dir):
    if not HAVE_MPL:
        print("[pareto] matplotlib unavailable — skipping plots")
        return
    for cfg in CFGS:
        fig, axes = plt.subplots(3, 3, figsize=(15, 12), squeeze=False)
        fig.suptitle(f"Pareto frontier — {cfg} cells (node: {node})", fontsize=13)
        for bi, b in enumerate(BATCHES):
            for si, s in enumerate(SEQ_LENS):
                ax = axes[bi][si]
                cell = f"{cfg}_b{b}_s{s}"
                fr = frontier_data[cell]
                for impl, cells in per.items():
                    c = cells.get(cell)
                    if c and np.isfinite(c["med"]) and c["med"] < np.inf and np.isfinite(c["err"]):
                        ax.scatter(c["med"], c["err"], s=28, alpha=0.7,
                                   label=None, color="tab:blue", zorder=2)
                        ax.annotate(impl.replace("c_", ""), (c["med"], c["err"]),
                                    fontsize=6, alpha=0.8, xytext=(3, 3),
                                    textcoords="offset points")
                if fr:
                    fmed = [p[1] for p in fr]; ferr = [p[0] for p in fr]
                    ax.plot(fmed, ferr, color="tab:red", lw=1.6, zorder=3,
                            label="frontier" if (bi == 0 and si == 0) else None)
                for budget in budgets:
                    ax.axhline(budget, color="gray", lw=0.7, ls=":", alpha=0.7)
                    ax.text(0.02, budget * 1.25, f"budget {budget:g}", fontsize=6,
                            color="gray", transform=ax.get_yaxis_transform())
                ax.set_xscale("log"); ax.set_yscale("log")
                ax.set_title(f"B={b} S={s}", fontsize=9)
                ax.set_xlabel("median us (log)"); ax.set_ylabel("max_abs_err (log)")
                ax.grid(True, which="both", alpha=0.25)
                ax.invert_yaxis()
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        path = os.path.join(out_dir, f"pareto_{node}_frontier_{cfg}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[pareto] wrote {path}")

    # summary figure: sub-ms pass counts per budget + speedup histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    bkeys = [f"{budget:g}" for budget in budgets]
    counts = [sum(1 for c in all_cells() if winners[bkey][c]["impl"] is not None
                  and winners[bkey][c]["med"] < SUB_MS) for bkey in bkeys]
    bars = ax1.bar(bkeys, counts, color=["tab:green", "tab:orange"][:len(bkeys)])
    ax1.set_ylim(0, 18); ax1.set_title("sub-ms cells with a passing impl")
    ax1.set_ylabel("# cells (of 18)"); ax1.axhline(13, color="gray", ls="--", lw=1)
    ax1.text(0, 13.2, "v6 champion = 13/18 @1e-3", fontsize=8, color="gray")
    for rect, c in zip(bars, counts):
        ax1.text(rect.get_x() + rect.get_width() / 2, c + 0.3, str(c),
                 ha="center", fontsize=10)
    spds, labels = [], []
    for bkey in bkeys:
        for cell in all_cells():
            w = winners[bkey][cell]
            champ = per.get(champion_key(per), {}).get(cell, {})
            if w["impl"] and w["impl"] != champion_key(per) and w["med"] and \
               np.isfinite(champ.get("med", np.nan)):
                spds.append(champ["med"] / w["med"])
                labels.append(f"{cell}@{bkey}")
    if spds:
        ax2.hist(spds, bins=20, color="tab:blue", alpha=0.7)
        ax2.axvline(1.0, color="gray", ls="--", lw=1)
        ax2.set_title("winner speedup vs champion")
        ax2.set_xlabel("speedup x"); ax2.set_ylabel("# (cell, budget) wins")
    else:
        ax2.text(0.5, 0.5, "no data yet", ha="center", va="center")
        ax2.set_title("winner speedup vs champion (pending data)")
    fig.tight_layout()
    path = os.path.join(out_dir, f"pareto_{node}_summary.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[pareto] wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", default=None,
                    help="node name prefix in evals_v8_<node>_* files (omit = use all v8 files)")
    ap.add_argument("--budgets", default=",".join(f"{b:g}" for b in DEFAULT_BUDGETS))
    ap.add_argument("--results-dir", default=RESULTS)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--out-md", default=None)
    a = ap.parse_args()
    budgets = [float(x) for x in a.budgets.split(",")]
    out_dir = a.out_dir or a.results_dir

    per, node = discover(a.results_dir, a.node)
    if not per:
        print("[pareto] no evals_v8_* data found — writing placeholder md and exiting")
        md = (f"# Pareto frontier — latency vs error (node: {node})\n\n"
              "No evals_v8 data yet. Waiting on parent node runs.\n")
        md_path = a.out_md or os.path.join(out_dir, f"pareto_{node}.md")
        open(md_path, "w").write(md)
        print(f"[pareto] wrote {md_path}")
        return

    frontier_data, winners = {}, {f"{b:g}": {} for b in budgets}
    for cell in all_cells():
        fr, pts = cell_frontier(per, cell)
        frontier_data[cell] = fr
        for budget in budgets:
            med, impl = winner_for_budget(pts, budget)
            err = next((p[0] for p in pts if p[2] == impl), None) if impl else None
            winners[f"{budget:g}"][cell] = {"med": med, "impl": impl, "err": err}

    sub_ms_counts = {}
    for bkey, w in winners.items():
        n_pass = sum(1 for c in all_cells() if w[c]["impl"] is not None and w[c]["med"] < SUB_MS)
        sub_ms_counts[bkey] = {"n_pass": n_pass}

    md = render_md(per, node, budgets, frontier_data, winners, sub_ms_counts)
    md_path = a.out_md or os.path.join(out_dir, f"pareto_{node}.md")
    open(md_path, "w").write(md)
    print(f"[pareto] wrote {md_path}")

    jd = {"node": node, "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
          "budgets": [f"{b:g}" for b in budgets],
          "champion": champion_key(per),
          "winners": winners, "frontier": {c: [list(p) for p in fr]
                                            for c, fr in frontier_data.items()},
          "sub_ms_counts": sub_ms_counts}
    jpath = os.path.join(out_dir, f"pareto_{node}.json")
    json.dump(jd, open(jpath, "w"), indent=1)
    print(f"[pareto] wrote {jpath}")

    make_plots(per, node, budgets, frontier_data, winners, out_dir)


if __name__ == "__main__":
    main()
