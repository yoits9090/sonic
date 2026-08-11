"""Progress dashboard on http://localhost:9023
Serves an auto-refreshing HTML page + matplotlib graph rendered from results/*.json
Usage: python dash/server.py [--port 9023] [--root ../fast-transformer]
"""
import argparse, glob, json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

def load_results():
    rows = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        rows.append((os.path.basename(p), d))
    return rows

def make_plot():
    rows = load_results()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Fast Transformer Race - forward pass latency (lower=better)", fontsize=13)
    # 1: best median per impl
    best = {}
    for fname, d in rows:
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict) and "latency" in v and isinstance(v["latency"], dict):
                    impl = v["latency"].get("impl", k)
                    med = v["latency"].get("median_us")
                    if med is not None:
                        best.setdefault(impl, []).append(med)
                elif isinstance(v, dict) and "median_us" in v:
                    best.setdefault(v.get("impl", k), []).append(v["median_us"])
    ax = axes[0][0]
    if best:
        names = sorted(best, key=lambda n: min(best[n]))
        vals = [min(best[n]) for n in names]
        ax.barh(names, vals, color="steelblue")
        ax.axvline(1000, color="red", ls="--", lw=2, label="1 ms target")
        ax.set_xscale("log"); ax.set_xlabel("min median forward pass (us)"); ax.legend()
    else:
        ax.text(0.5, 0.5, "no results yet", ha="center", va="center"); ax.axis("off")
    ax.set_title("Best latency per impl")
    # 2: progression over time (per impl)
    ax = axes[0][1]
    series = {}
    for fname, d in rows:
        ts = os.path.getmtime(os.path.join(RESULTS, fname))
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict) and "latency" in v and isinstance(v["latency"], dict):
                    impl = v["latency"].get("impl", k)
                    series.setdefault(impl, []).append((ts, v["latency"]["median_us"]))
                elif isinstance(v, dict) and "median_us" in v:
                    series.setdefault(v.get("impl", k), []).append((ts, v["median_us"]))
    for impl, pts in sorted(series.items()):
        pts = sorted(pts)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=impl, lw=2)
    ax.axhline(1000, color="red", ls="--", lw=2)
    ax.set_yscale("log"); ax.set_ylabel("median us"); ax.set_xlabel("time")
    ax.legend(fontsize=8); ax.set_title("Latency progression")
    # 3: p50/p90/p99 spread for the current best
    ax = axes[1][0]
    if best:
        champ = min(best, key=lambda n: min(best[n]))
        pts = sorted(best[champ])
        ax.bar(["p50", "p90", "p99"], pts[:3], color=["#2e7d32", "#f9a825", "#c62828"])
        ax.set_title(f"Percentile spread: {champ}")
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center"); ax.axis("off")
    # 4: throughput (tokens/s)
    ax = axes[1][1]
    thr = {}
    for fname, d in rows:
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict) and "latency" in v and isinstance(v["latency"], dict):
                    impl = v["latency"].get("impl", k)
                    t = v["latency"].get("throughput_tok_s")
                    if t: thr.setdefault(impl, []).append(t)
    if thr:
        names = sorted(thr, key=lambda n: max(thr[n]))
        ax.barh(names, [max(thr[n]) for n in names], color="seagreen")
        ax.set_xlabel("max throughput (tok/s)")
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center"); ax.axis("off")
    ax.set_title("Throughput")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return buf.getvalue()

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            body = """<!doctype html><html><head><meta charset="utf-8">
<title>Fast Transformer Race - localhost:9023</title>
<style>body{font-family:monospace;background:#0d1117;color:#c9d1d9;padding:20px}
h1{color:#58a6ff} img{max-width:100%;border:1px solid #30363d;border-radius:6px}
.badge{color:#3fb950}</style></head><body>
<h1>&#9889; Fast Transformer Forward-Pass Race</h1>
<p class="badge">compute: google colab cpu nodes (bench-node-1/2/3) | target: &lt; 1 ms forward pass</p>
<p>updated: <span id="u">-</span></p>
<img src="/plot.png?t=%s" id="plot">
<script>
setInterval(function(){document.getElementById('plot').src='/plot.png?t='+Date.now();
fetch('/data.json').then(r=>r.json()).then(d=>{document.getElementById('u').textContent=new Date(d.generated*1000).toISOString();});},5000);
</script></body></html>""" % int(time.time())
            self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers()
            self.wfile.write(body.encode())
        elif self.path.startswith("/plot.png"):
            png = make_plot()
            self.send_response(200); self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "no-store"); self.end_headers()
            self.wfile.write(png)
        elif self.path.startswith("/data.json"):
            data = {"generated": time.time(), "results": load_results()}
            body = json.dumps(data).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()
    def log_message(self, *a):
        pass

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9023)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"dashboard on http://localhost:{a.port}")
    srv.serve_forever()
