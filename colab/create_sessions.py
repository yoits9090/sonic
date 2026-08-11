#!/usr/bin/env python3
import subprocess, time, sys, json, os
LOG = "/Users/ace/projects/sonic/fast-transformer/colab/session_retry.log"
READY = "/Users/ace/projects/sonic/fast-transformer/colab/sessions_ready.json"
def log(m):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {m}\n")
names = ["bench-node-1", "bench-node-2", "bench-node-3"]
attempt = 0
while attempt < 30:
    attempt += 1
    ok = []
    for n in names:
        r = subprocess.run(["colab", "--auth", "adc", "new", "-s", n],
                           capture_output=True, text=True, timeout=300)
        out = (r.stdout + r.stderr)[-300:]
        if r.returncode == 0:
            ok.append(n); log(f"CREATED {n}")
        else:
            log(f"fail {n}: {out.splitlines()[-1][:120] if out else 'no output'}")
    if len(ok) == 3:
        json.dump({"ready": True, "sessions": ok, "time": time.time()}, open(READY, "w"))
        log("ALL SESSIONS READY")
        sys.exit(0)
    log(f"attempt {attempt} -> {len(ok)}/3 ready; sleeping 45s")
    time.sleep(45)
json.dump({"ready": False, "sessions": ok, "time": time.time()}, open(READY, "w"))
log("GAVE UP after 30 attempts")
sys.exit(1)
