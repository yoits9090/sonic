#!/usr/bin/env python3
"""Summarize results/*.json into a markdown table. Usage: python3 colab/bench/summarize.py [--node bench-node-1]"""
import argparse, glob, json, os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", default=None)
    a = ap.parse_args()
    files = sorted(glob.glob("results/*.json"))
    rows = []
    for fn in files:
        d = json.load(open(fn))
        if a.node and d.get("node") != a.node:
            continue
        if "median_us" not in d:
            continue
        rows.append((d.get("node","?"), d.get("impl","?"), d.get("cfg","?"),
                     d.get("median_us", float("nan")), d.get("correctness", {}).get("pass", "?"),
                     d.get("numpy_version","?"), fn.split("/")[-1]))
    rows.sort(key=lambda r: (r[2], r[3]))
    print(f"| node | impl | cfg | median_us | pass | numpy | file |")
    print(f"|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.1f} | {r[4]} | {r[5]} | {r[6]} |")
    print(f"\n{len(rows)} result files")

if __name__ == "__main__":
    main()
