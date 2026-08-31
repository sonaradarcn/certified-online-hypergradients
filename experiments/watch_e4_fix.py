"""Detached watcher: regenerate results/e4_fix/COMPARE.md as Phase-1 lands.

Runs compare_e4_fix.py every time a new results/e4_fix/*.json appears, so a
partial comparison exists as soon as the first seed finishes and the final
three-seed COMPARE.md is written the moment Phase 1 completes -- without
anyone having to be watching.  Exits once all three seeds are in (or after
--max-hours).
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FIX = os.path.join(ROOT, "results", "e4_fix")
CMP = os.path.join(HERE, "compare_e4_fix.py")
TARGET = 3


def n_json():
    return len(glob.glob(os.path.join(FIX, "*.json")))


def regen(why):
    p = subprocess.run([sys.executable, CMP], capture_output=True, text=True,
                       cwd=HERE)
    print("[e4fix-watch] %s -> COMPARE.md regenerated (rc=%d, %s)"
          % (why, p.returncode,
             "all seeds agree" if p.returncode == 0 else "partial/differs"),
          flush=True)
    tail = [l for l in p.stdout.splitlines() if l.strip()][-6:]
    for l in tail:
        print("    " + l, flush=True)
    if p.stderr.strip():
        print("[e4fix-watch] stderr: %s" % p.stderr[-800:], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll-s", type=int, default=300)
    ap.add_argument("--max-hours", type=float, default=36.0)
    a = ap.parse_args()
    t0 = time.time()
    seen = -1
    print("[e4fix-watch] armed, pid %d, poll %ds, target %d json @ %s"
          % (os.getpid(), a.poll_s, TARGET, time.ctime()), flush=True)
    while True:
        n = n_json()
        if n != seen:
            seen = n
            regen("%d/%d seeds present" % (n, TARGET))
            if n >= TARGET:
                print("[e4fix-watch] Phase 1 complete; exiting @ %s"
                      % time.ctime(), flush=True)
                return
        if time.time() - t0 > a.max_hours * 3600:
            print("[e4fix-watch] max-hours reached with %d/%d seeds; exiting"
                  % (n, TARGET), flush=True)
            return
        time.sleep(a.poll_s)


if __name__ == "__main__":
    main()
