"""Leaderboard + aggregation over results/.
- update: results/leaderboard.json — best (lowest) median_us per impl per cfg key,
  with node/source/timestamp, updated after every eval round.
- aggregate: results/aggregate.json — merge node results per impl/cfg, flag
  discrepancies > 10% between nodes for the same impl/cfg.
Usage:
  python evals/leaderboard.py update
  python evals/leaderboard.py aggregate
"""
import glob, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DISC_THRESHOLD_PCT = 10.0


def iter_bench_files():
    """Yield (path, data) for every JSON file under results/ that contains impl/latency data."""
    for p in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        if os.path.basename(p) in ("leaderboard.json", "aggregate.json", "manifest.json"):
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        yield p, d


def extract_records(path, d):
    """Yield (impl, cfg_key, median_us, node, ts) records from a results file."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(os.path.getmtime(path)))
    # v1: {impl: {"correctness": ..., "latency": {...cfg:..., median_us...}}}
    if isinstance(d, dict) and "latency" in str(d.keys()) and any(isinstance(d.get(k), dict) and "latency" in d[k] for k in d):
        for impl, v in d.items():
            lat = v.get("latency")
            if not isinstance(lat, dict) or "median_us" not in lat:
                continue
            cfg = lat.get("cfg", {})
            key = cfg_key(cfg)
            yield impl, key, lat["median_us"], lat.get("node", "?"), ts
        return
    # v2/v3: {impls: {impl: {... seq_sweep/batch_sweep or configs}}}
    if isinstance(d, dict) and "impls" in d:
        node = d.get("node", "?")
        for impl, v in d["impls"].items():
            if isinstance(v, dict) and "seq_sweep" in v:  # v2
                base = {"seq_len": None, "batch": 1, "gen": "v2"}
                for sl, r in v["seq_sweep"].items():
                    yield impl, cfg_key({**base, "seq_len": int(sl)}), r["median_us"], node, ts
                for b, r in v["batch_sweep"].items():
                    yield impl, cfg_key({**base, "seq_len": 32, "batch": int(b)}), r["median_us"], node, ts
            elif isinstance(v, dict) and "configs" in v:  # v3 latency seeds
                for cname, r in v["configs"].items():
                    ls = r.get("latency_seeds")
                    if ls:
                        yield impl, cfg_key({"gen": "v3", "cfg": cname, "metric": "latency_seeds"}), ls["median_of_medians_us"], node, ts


def cfg_key(cfg):
    if not cfg:
        return "default?"
    parts = []
    for k in ("d_model", "n_heads", "n_layers", "d_ff", "seq_len", "batch", "vocab", "cfg", "gen", "metric"):
        if k in cfg and cfg[k] is not None:
            parts.append(f"{k}={cfg[k]}")
    return ",".join(parts) or json.dumps(cfg, sort_keys=True)


def update():
    best = {}  # (impl, cfgkey) -> record
    for path, d in iter_bench_files():
        for impl, key, med, node, ts in extract_records(path, d):
            r = best.setdefault((impl, key), {"median_us": 1e18})
            if med < r["median_us"]:
                best[(impl, key)] = {"median_us": med, "node": node, "source": os.path.basename(path), "ts": ts}
    lb = {"updated_ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
          "impls": {}}
    for (impl, key), r in sorted(best.items()):
        lb["impls"].setdefault(impl, {})[key] = r
    path = os.path.join(RESULTS, "leaderboard.json")
    json.dump(lb, open(path, "w"), indent=1)
    print("leaderboard ->", path)
    for impl in lb["impls"]:
        rows = lb["impls"][impl]
        best_any = min((r["median_us"] for r in rows.values()), default=None)
        print(f"  {impl}: {len(rows)} cfgs, best median {best_any:.1f}us")
    return lb


def aggregate():
    by = {}  # (impl, cfgkey) -> {node: median}
    for path, d in iter_bench_files():
        for impl, key, med, node, ts in extract_records(path, d):
            by.setdefault((impl, key), {}).setdefault(node, []).append(med)
    agg = {"updated_ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "threshold_pct": DISC_THRESHOLD_PCT, "impls": {}}
    for (impl, key), node_meds in sorted(by.items()):
        rec = {"nodes": {}, "flagged": False}
        for node, meds in node_meds.items():
            rec["nodes"][node] = {"median_us": min(meds), "n_runs": len(meds)}
        meds = [v["median_us"] for v in rec["nodes"].values()]
        if len(meds) > 1:
            lo, hi = min(meds), max(meds)
            rec["max_discrepancy_pct"] = float((hi - lo) / lo * 100)
            rec["flagged"] = rec["max_discrepancy_pct"] > DISC_THRESHOLD_PCT
        agg["impls"].setdefault(impl, {})[key] = rec
    path = os.path.join(RESULTS, "aggregate.json")
    json.dump(agg, open(path, "w"), indent=1)
    print("aggregate ->", path)
    for impl in agg["impls"]:
        for key, rec in agg["impls"][impl].items():
            flag = " FLAG" if rec.get("flagged") else ""
            print(f"  {impl} [{key}]{flag}: " + ", ".join(f"{n}={v['median_us']:.1f}us" for n, v in rec["nodes"].items()))
    return agg


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "update"
    if cmd == "update":
        update()
    elif cmd == "aggregate":
        aggregate()
    else:
        print("usage: leaderboard.py [update|aggregate]")
