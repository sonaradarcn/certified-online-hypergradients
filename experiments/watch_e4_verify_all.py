"""Detached watcher: regenerate results/e4_verify_all/COMPARE_ALL.md as the
full-scope P4 reruns land.

Polls results/e4_verify_all for new *.json (ignoring COMPARE_ALL.md and the
.claim / .tmpdir scratch of launch_e4_verify_all.py) and re-runs
compare_e4_verify_all.py every time the set changes, so a partial but always
truthful comparison sits on disk from the first rerun onward and the final
14-pair table is written the moment the queue drains -- with nobody watching.

Exits when all TARGET reruns are in (or after --max-hours).
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
NEW = os.path.join(ROOT, "results", "e4_verify_all")
CMP = os.path.join(HERE, "compare_e4_verify_all.py")

# reruns queued by launch_e4_verify_all.py (the e4_fix trio is folded in by the
# comparison script from results/e4_fix and is not counted here)
TARGET = 11


def landed():
    return sorted(os.path.basename(p) for p in
                  glob.glob(os.path.join(NEW, "*.json")))


def regen(why):
    p = subprocess.run([sys.executable, CMP, "--quiet"],
                       capture_output=True, text=True, cwd=HERE)
    print("[vfy-watch] %s -> COMPARE_ALL.md regenerated (rc=%d, %s)"
          % (why, p.returncode,
             "all pairs agree, scope complete" if p.returncode == 0
             else "partial or a pair differs"), flush=True)
    for line in [x for x in p.stdout.splitlines() if x.strip()][-4:]:
        print("    " + line, flush=True)
    if p.stderr.strip():
        print("[vfy-watch] stderr: %s" % p.stderr[-1000:], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll-s", type=int, default=300)
    ap.add_argument("--max-hours", type=float, default=72.0)
    a = ap.parse_args()
    os.makedirs(NEW, exist_ok=True)
    t0 = time.time()
    seen = None
    print("[vfy-watch] armed, pid %d, poll %ds, target %d reruns in %s @ %s"
          % (os.getpid(), a.poll_s, TARGET, NEW, time.ctime()), flush=True)
    while True:
        cur = landed()
        if cur != seen:
            new = [f for f in cur if seen is None or f not in seen]
            seen = cur
            regen("%d/%d reruns present%s"
                  % (len(cur), TARGET,
                     (" (new: %s)" % ", ".join(new)) if new else ""))
            if len(cur) >= TARGET:
                print("[vfy-watch] full scope complete; exiting @ %s"
                      % time.ctime(), flush=True)
                return
        if time.time() - t0 > a.max_hours * 3600:
            print("[vfy-watch] max-hours reached with %d/%d reruns; exiting"
                  % (len(cur), TARGET), flush=True)
            return
        time.sleep(a.poll_s)


if __name__ == "__main__":
    main()
