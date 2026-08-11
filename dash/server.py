"""Progress dashboard on http://localhost:9023
Serves an auto-refreshing HTML page + matplotlib graph rendered from results/*.json.
Layout: 3x2 grid
  row1: best latency per impl | latency progression over time
  row2: percentile spread     | throughput
  row3: eval generation coverage per impl | distance to 1 ms target
Plus an HTML "latest results" raw-data table (client-rendered from /data.json).

Understands the repo's result file shapes:
  v1: {impl: {correctness, latency{...}}} or flat latency dicts        (run_all / eval_latency)
  v2: {impl, node, generation:"v2", attempt, seq_sweep{..}, batch_sweep{..}}  (eval_v2)
  v3: {impl, node, generation:"v3", attempt, configs{tiny|default{..}}}       (eval_v3)
Derived files (leaderboard/aggregate/manifest, evals_v[23]_<node>.json summaries)
are skipped. Malformed/partial JSON is skipped and reported, never fatal.

Usage: MPLBACKEND=Agg python dash/server.py [--port 9023]
"""
import argparse, glob, io, json, math, os, re, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
TARGET_US = 1000.0  # 1 ms
SKIP_FILES = {"leaderboard.json", "aggregate.json", "manifest.json"}

# ---------------------------------------------------------------- data layer

def _num(v):
    """float or None; rejects bools, strings, NaN/Inf."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if math.isfinite(f) else None

def _int_or_none(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    return None

def _lat_of(v):
    """Normalize one entry to a latency dict, or None."""
    if not isinstance(v, dict):
        return None
    if isinstance(v.get("latency"), dict):
        lat = v["latency"]
    elif any(k in v for k in ("median_us", "median", "p50_us")):
        lat = v
    else:
        return None
    return lat if isinstance(lat, dict) else None

def _median(lat):
    for k in ("median_us", "median", "p50_us"):
        m = _num(lat.get(k))
        if m is not None:
            return m
    return None

def _gen(lat, fname):
    """Generation int from explicit field, filename, or None (implicit later)."""
    for k in ("gen", "generation", "eval_gen", "version", "v"):
        v = lat.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s.startswith("v") and s[1:].isdigit():
                return int(s[1:])
            if s.isdigit():
                return int(s)
    m = re.search(r"[_.\-](?:v|gen|g)(\d+)", fname, re.I)
    if m:
        return int(m.group(1))
    return None

def _node(lat, fname):
    n = lat.get("node")
    if isinstance(n, str) and n.strip():
        return n.strip()
    m = re.search(r"(bench-node-\d+)", fname)
    return m.group(1) if m else "?"

def _cfg_label(lat):
    c = lat.get("cfg")
    if isinstance(c, dict):
        parts = []
        if c.get("seq_len") is not None:
            parts.append("S=%s" % c["seq_len"])
        if c.get("batch") is not None:
            parts.append("B=%s" % c["batch"])
        return ",".join(parts)
    if isinstance(c, str) and c:
        return c
    return ""

def _is_canon_point(point):
    """Race-relevant point: tiny config or default at batch 1."""
    toks = {t.strip() for t in point.split(",")}
    if "S=32" in toks and "B=1" in toks:
        return True
    return point.startswith("config=tiny") or point == ""

def _entry(fname, mtime, key, lat, point=None):
    """Build a normalized entry from a latency dict, or None if unusable."""
    med = _median(lat)
    if med is None:
        return None
    impl = lat.get("impl")
    if not isinstance(impl, str) or not impl:
        impl = key if isinstance(key, str) and key else fname
    point = point if point is not None else _cfg_label(lat)
    return {
        "file": fname, "mtime": mtime,
        "impl": impl, "node": _node(lat, fname),
        "gen": _gen(lat, fname),
        "attempt": _int_or_none(lat.get("attempt")),
        "point": point,
        "canon": _is_canon_point(point),
        "median_us": med,
        "p90_us": _num(lat.get("p90_us")), "p99_us": _num(lat.get("p99_us")),
        "min_us": _num(lat.get("min_us")), "max_us": _num(lat.get("max_us")),
        "throughput_tok_s": _num(lat.get("throughput_tok_s")),
        "sub_ms": bool(lat.get("sub_ms")) if isinstance(lat.get("sub_ms"), bool) else bool(med < TARGET_US),
        "iters": lat.get("iters"),
    }

def _flatten(fname, d, mtime):
    """Yield normalized entries from one parsed JSON doc."""
    if not isinstance(d, dict):
        return
    base = {"impl": d.get("impl"), "node": d.get("node"),
            "generation": d.get("generation"), "attempt": d.get("attempt")}
    # v2: seq/batch sweeps
    for skey, fmt in (("seq_sweep", "S=%s,B=1"), ("batch_sweep", "B=%s,S=32")):
        sw = d.get(skey)
        if isinstance(sw, dict):
            for k, v in sw.items():
                if not isinstance(v, dict):
                    continue
                lat = dict(v)
                for kk, vv in base.items():
                    if vv is not None:
                        lat.setdefault(kk, vv)
                e = _entry(fname, mtime, lat.get("impl") or fname, lat, point=fmt % k)
                if e:
                    yield e
    # v3: configs with cold/steady
    cfgs = d.get("configs")
    if isinstance(cfgs, dict):
        for k, v in cfgs.items():
            if not isinstance(v, dict):
                continue
            cold = v.get("cold")
            if not isinstance(cold, dict):
                continue
            steady = _num(cold.get("steady_median_us"))
            if steady is None:
                continue
            lat = {"impl": base["impl"], "node": base["node"],
                   "generation": base["generation"], "attempt": base["attempt"],
                   "median_us": steady, "throughput_tok_s": None}
            e = _entry(fname, mtime, lat.get("impl") or fname, lat, point="config=%s" % k)
            if e:
                yield e
    # generic shapes: flat latency dict, {key: latency}, {results: [...]}
    if "seq_sweep" not in d and "configs" not in d:
        if any(k in d for k in ("median_us", "median", "p50_us")):
            yield _entry(fname, mtime, d.get("impl") or fname, d)
            return
        items = []
        if isinstance(d.get("results"), list):
            items = [(fname, it) for it in d["results"]]
        else:
            items = [(k, v) for k, v in d.items()]
        for k, v in items:
            lat = _lat_of(v)
            if lat is None:
                continue
            e = _entry(fname, mtime, k, lat)
            if e:
                yield e

def load_results():
    """-> (entries, malformed)"""
    entries, malformed = [], []
    for p in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        fname = os.path.basename(p)
        if fname in SKIP_FILES or re.fullmatch(r"evals_v[23]_bench-node-\d+\.json", fname):
            continue
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception as e:
            malformed.append((fname, "parse: %s" % e))
            continue
        try:
            mtime = os.path.getmtime(p)
            got = 0
            for e in _flatten(fname, d, mtime):
                entries.append(e)
                got += 1
        except Exception as e:
            malformed.append((fname, "scan: %s" % e))
            continue
        if got == 0:
            malformed.append((fname, "no latency records found"))
    return entries, malformed

def canon_per_gen(entries):
    """One race-relevant entry per (impl, gen): prefer canonical point, then lower median."""
    best = {}
    for e in entries:
        if e["gen"] is None:
            continue
        key = (e["impl"], e["gen"])
        cur = best.get(key)
        if cur is None:
            best[key] = e
        else:
            ca, cb = e["canon"], cur["canon"]
            if (ca and not cb) or (ca == cb and e["median_us"] < cur["median_us"]):
                best[key] = e
    return list(best.values())

def assign_implicit_gens(entries):
    """Fill gen=None entries with 1-based order by mtime per impl."""
    by_impl = {}
    for i, e in enumerate(entries):
        by_impl.setdefault(e["impl"], []).append(i)
    for idxs in by_impl.values():
        idxs.sort(key=lambda i: entries[i]["mtime"])
        for n, i in enumerate(idxs, 1):
            if entries[i]["gen"] is None:
                entries[i]["gen"] = n

def gen_label(e):
    g = "v%d" % (e["gen"] or 0)
    if e.get("attempt") is not None:
        g += "-a%d" % e["attempt"]
    return g

def summary(entries, malformed):
    """Latest result file per impl (all its points) + status for /data.json and the table."""
    latest = {}
    for e in entries:
        cur = latest.get(e["impl"])
        if cur is None or e["mtime"] > cur["mtime"]:
            latest[e["impl"]] = e
    rows = [e for e in entries if e["impl"] in latest and e["mtime"] == latest[e["impl"]]["mtime"]]
    rows.sort(key=lambda e: (e["median_us"] is None, e["median_us"] or 0.0))
    canon = canon_per_gen(entries)
    champ = min(canon, key=lambda e: e["median_us"]) if canon else None
    return {
        "latest": rows,
        "champion": ({"impl": champ["impl"], "median_us": champ["median_us"],
                      "node": champ["node"], "gen": gen_label(champ), "sub_ms": champ["sub_ms"]}
                     if champ else None),
        "n_files": len({e["file"] for e in entries}),
        "n_entries": len(entries),
        "malformed": [{"file": f, "why": w} for f, w in malformed],
    }

# ---------------------------------------------------------------- plotting

def _no_data(ax, msg="no data yet"):
    ax.text(0.5, 0.5, msg, ha="center", va="center", color="#888")
    ax.axis("off")
    ax.set_title(msg)

def make_plot():
    try:
        return _make_plot()
    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "dashboard render error:\n%s" % e, ha="center", va="center", color="red")
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        return buf.getvalue()

def _make_plot():
    entries, malformed = load_results()
    assign_implicit_gens(entries)
    canon = canon_per_gen(entries)
    by_impl = {}
    for e in canon:
        by_impl.setdefault(e["impl"], []).append(e)
    for lst in by_impl.values():
        lst.sort(key=lambda e: (e["gen"] or 0))
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 13.5))
    fig.suptitle("Fast Transformer Race - forward pass latency (lower=better)", fontsize=14)

    # -- (0,0) best canonical median per impl --------------------------------
    ax = axes[0][0]
    if by_impl:
        names = sorted(by_impl, key=lambda n: min(e["median_us"] for e in by_impl[n]))
        vals = [min(e["median_us"] for e in by_impl[n]) for n in names]
        ax.barh(names, vals, color="steelblue")
        ax.axvline(TARGET_US, color="red", ls="--", lw=2, label="1 ms target")
        ax.set_xscale("log"); ax.set_xlabel("best median forward pass (us)"); ax.legend()
        ax.set_title("Best latency per impl")
    else:
        _no_data(ax, "no results yet")

    # -- (0,1) progression over time (all canonical points) -------------------
    ax = axes[0][1]
    if by_impl:
        for impl, lst in sorted(by_impl.items()):
            pts = sorted(lst, key=lambda e: e["mtime"])
            ax.plot([p["mtime"] for p in pts], [p["median_us"] for p in pts],
                    marker="o", label=impl, lw=2)
        ax.axhline(TARGET_US, color="red", ls="--", lw=2)
        ax.set_yscale("log"); ax.set_ylabel("median us"); ax.set_xlabel("time")
        ax.legend(fontsize=8); ax.set_title("Latency progression (canonical pts)")
    else:
        _no_data(ax)

    # -- (1,0) percentile spread of champion (latest canonical) ---------------
    ax = axes[1][0]
    champ = min(by_impl, key=lambda n: min(e["median_us"] for e in by_impl[n])) if by_impl else None
    if champ:
        e = max(by_impl[champ], key=lambda x: x["mtime"])
        p90 = e["p90_us"] if e["p90_us"] is not None else e["median_us"]
        p99 = e["p99_us"] if e["p99_us"] is not None else p90
        ax.bar(["p50", "p90", "p99"], [e["median_us"], p90, p99],
               color=["#2e7d32", "#f9a825", "#c62828"])
        ax.axhline(TARGET_US, color="red", ls="--", lw=1.5)
        for i, v in enumerate([e["median_us"], p90, p99]):
            ax.text(i, v, "%.0f" % v, ha="center", va="bottom", fontsize=8)
        ax.set_title("Percentile spread: %s (%s)" % (champ, gen_label(e)))
    else:
        _no_data(ax)

    # -- (1,1) throughput ------------------------------------------------------
    ax = axes[1][1]
    thr = {impl: max(e["throughput_tok_s"] for e in lst if e["throughput_tok_s"] is not None)
           for impl, lst in by_impl.items() if any(e["throughput_tok_s"] is not None for e in lst)}
    if thr:
        names = sorted(thr, key=lambda n: thr[n])
        ax.barh(names, [thr[n] for n in names], color="seagreen")
        ax.set_xlabel("max throughput (tok/s)")
        ax.set_title("Throughput")
    else:
        _no_data(ax)

    # -- (2,0) eval generation coverage -----------------------------------------
    ax = axes[2][0]
    if by_impl:
        for impl, lst in sorted(by_impl.items()):
            gens = [e["gen"] or 0 for e in lst]
            meds = [e["median_us"] for e in lst]
            label = impl if len(lst) < 2 else "%s (v%d..v%d)" % (impl, min(gens), max(gens))
            ax.plot(gens, meds, marker="o", label=label, lw=2)
        ax.axhline(TARGET_US, color="red", ls="--", lw=2, label="1 ms target")
        ax.set_xlabel("eval generation (v1..vN)")
        ax.set_ylabel("median us"); ax.set_yscale("log")
        ax.legend(fontsize=8); ax.set_title("Eval generation coverage per impl")
    else:
        _no_data(ax)

    # -- (2,1) distance to 1 ms target ------------------------------------------
    ax = axes[2][1]
    if by_impl:
        for impl, lst in sorted(by_impl.items()):
            gens = [e["gen"] or 0 for e in lst]
            dist = [TARGET_US - e["median_us"] for e in lst]
            ax.plot(gens, dist, marker="o", label=impl, lw=2)
        ax.axhline(0, color="black", lw=1)
        pos = max([TARGET_US - e["median_us"] for lst in by_impl.values() for e in lst] + [0.0])
        ax.axhspan(0, pos, color="green", alpha=0.08)
        ax.set_xlabel("eval generation (v1..vN)")
        ax.set_ylabel("1000 - median_us (us); >0 = under 1 ms")
        ax.legend(fontsize=8); ax.set_title("Distance to 1 ms target")
    else:
        _no_data(ax)

    if malformed:
        fig.text(0.01, 0.005,
                 "skipped %d unusable file(s): %s" % (len(malformed), ", ".join(f for f, _ in malformed[:6])),
                 color="#ff7b72", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return buf.getvalue()

# ---------------------------------------------------------------- http layer

HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>Fast Transformer Race - localhost:9023</title>
<style>
body{font-family:monospace;background:#0d1117;color:#c9d1d9;padding:20px;max-width:1400px;margin:auto}
h1{color:#58a6ff} img{max-width:100%;border:1px solid #30363d;border-radius:6px}
.badge{color:#3fb950} .warn{color:#ff7b72}
table{border-collapse:collapse;width:100%;font-size:12px;margin-top:8px}
th,td{border:1px solid #30363d;padding:4px 8px;text-align:right}
th{background:#161b22;color:#58a6ff} td:first-child,th:first-child{text-align:left}
tr.ok td{background:#12261a;color:#7ee787}
a{color:#58a6ff} .muted{color:#8b949e}
</style></head><body>
<h1>&#9889; Fast Transformer Forward-Pass Race</h1>
<p class="badge">compute: google colab cpu nodes (bench-node-1/2/3) | target: &lt; 1 ms forward pass</p>
<p>updated: <span id="u">-</span> &nbsp;|&nbsp; <span id="stats" class="muted">-</span></p>
<p id="warn" class="warn"></p>
<img src="/plot.png?t=__T__" id="plot">
<h2>Latest results <span class="muted">(raw, latest file per impl)</span></h2>
<div style="overflow-x:auto"><table id="latest">
<thead><tr><th>impl</th><th>point</th><th>node</th><th>gen</th><th>median us</th><th>p90 us</th><th>p99 us</th>
<th>min us</th><th>tok/s</th><th>&lt;1ms</th><th>file</th></tr></thead>
<tbody></tbody></table></div>
<p class="muted">links: <a href="/data.json">data.json</a> | <a href="/notes.md">progress notes</a></p>
<script>
function esc(s){return String(s).replace(/[&<>"]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}
function f(x,d){return (x==null||x==='')? (d||'-') : (+x).toFixed(1);}
function refresh(){
  document.getElementById('plot').src='/plot.png?t='+Date.now();
  fetch('/data.json').then(function(r){return r.json();}).then(function(d){
    document.getElementById('u').textContent=new Date(d.generated*1000).toISOString();
    var s=d.summary||{};
    var st=s.n_files+' file(s), '+s.n_entries+' record(s)';
    if(s.champion){st+=' | leader: '+esc(s.champion.impl)+' @ '+f(s.champion.median_us)+' us ('+esc(s.champion.gen)+')'+(s.champion.sub_ms?' SUB-MS':'');}
    document.getElementById('stats').textContent=st;
    var w=document.getElementById('warn');
    if(s.malformed&&s.malformed.length){w.textContent='skipped '+s.malformed.length+' unusable file(s): '+s.malformed.map(function(m){return m.file+' ('+m.why+')';}).join(', ');}
    else{w.textContent='';}
    var tb=document.querySelector('#latest tbody');tb.innerHTML='';
    (s.latest||[]).forEach(function(e){
      var tr=document.createElement('tr');if(e.sub_ms){tr.className='ok';}
      tr.innerHTML='<td>'+esc(e.impl)+'</td><td>'+esc(e.point||'-')+'</td><td>'+esc(e.node)+'</td><td>'+esc(e.gen_label)+'</td>'
        +'<td>'+f(e.median_us)+'</td><td>'+f(e.p90_us)+'</td><td>'+f(e.p99_us)+'</td>'
        +'<td>'+f(e.min_us)+'</td><td>'+f(e.throughput_tok_s)+'</td>'
        +'<td>'+(e.sub_ms?'YES':'')+'</td><td>'+esc(e.file)+'</td>';
      tb.appendChild(tr);
    });
  }).catch(function(){});
}
setInterval(refresh,5000);refresh();
</script></body></html>"""

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            body = HTML.replace("__T__", str(int(time.time())))
            self._send(200, "text/html; charset=utf-8", body.encode())
        elif path == "/plot.png":
            self._send(200, "image/png", make_plot(), cache="no-store")
        elif path == "/data.json":
            entries, malformed = load_results()
            assign_implicit_gens(entries)
            for e in entries:
                e["gen_label"] = gen_label(e)
            data = {"generated": time.time(), "results": entries,
                    "summary": summary(entries, malformed)}
            self._send(200, "application/json", json.dumps(data).encode(), cache="no-store")
        elif path == "/notes.md":
            p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "progress_notes.md")
            if os.path.exists(p):
                self._send(200, "text/markdown; charset=utf-8", open(p, "rb").read())
            else:
                self._send(404, "text/plain", b"no progress notes yet")
        else:
            self._send(404, "text/plain", b"not found")

    def _send(self, code, ctype, body, cache="no-cache"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9023)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print("dashboard on http://localhost:%d" % a.port)
    srv.serve_forever()
