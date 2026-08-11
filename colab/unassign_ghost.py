#!/usr/bin/env python3
"""Release a ghost Colab assignment (lost session still holding an account slot).
Usage: python3 colab/unassign_ghost.py <endpoint-host>
  (find ghosts via `colab --auth adc sessions` -> entries prefixed [?])
Works around: TooManyAssignmentsError on `colab new` after a VM was reclaimed but its
assignment lingers server-side. Uses the same /tun/m/unassign endpoint as `colab stop`.
"""
import sys, json, subprocess, httpx

def main():
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(1)
    ep = sys.argv[1]
    tok = subprocess.run(["gcloud", "auth", "application-default", "print-access-token"],
                         capture_output=True, text=True).stdout.strip()
    h = {"Accept": "application/json", "Authorization": f"Bearer {tok}", "X-Colab-Client-Agent": "colab-cli"}
    url = f"https://colab.research.google.com/tun/m/unassign/{ep}?authuser=0"
    r = httpx.get(url, headers=h, timeout=60)
    j = json.loads(r.text.lstrip(")]}'\n"))
    r2 = httpx.post(url, headers={**h, "X-Goog-Colab-Token": j["token"]}, timeout=60)
    print("unassign:", r2.status_code, ep)
    if r2.status_code == 204:
        print("slot released; you can `colab --auth adc new -s <name>` again")

if __name__ == "__main__":
    main()
