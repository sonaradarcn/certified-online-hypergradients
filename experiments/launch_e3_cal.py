"""E3-domain meta-LR calibration for the raw-scale baselines (fairness,
findings F19: the E2-frozen ml=200 collapses on E3). Calibration seeds
{100, 101} disjoint from evaluation, well-set (lr, ewc) = (0.05, 10).
Amended selection rule (F11): lowest final avg_acc... i.e. HIGHEST avg_acc
among zero-event configs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "e3_cal")
PY = sys.executable

GRID = [("hd", ml) for ml in [0.002, 0.02, 0.2, 2.0]] + \
       [("tfmd", ml) for ml in [0.002, 0.02, 0.2, 2.0]]
CAL_SEEDS = [100, 101]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=int, default=0)
    ap.add_argument("--n-slices", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cfgs = [(m, ml, s) for (m, ml) in GRID for s in CAL_SEEDS]
    cfgs = [c for i, c in enumerate(cfgs) if i % args.n_slices == args.slice]
    for meth, ml, seed in cfgs:
        tag = f"cifar100_{meth}_ml{ml:g}_s{seed}"
        final = os.path.join(OUT, tag + ".json")
        if os.path.exists(final):
            continue
        t0 = time.time()
        r = subprocess.run(
            [PY, os.path.join(HERE, "e3_continual.py"), "--method", meth,
             "--seed", str(seed), "--lr", "0.05", "--ewc0", "10",
             "--meta-lr", str(ml), "--device", args.device, "--out-dir", OUT],
            capture_output=True, text=True, cwd=HERE)
        src = os.path.join(OUT, f"cifar100_{meth}_lr0.05_ewc10_s{seed}.json")
        if os.path.exists(src):
            os.replace(src, final)
            print(f"done {tag} ({time.time() - t0:.0f}s)", flush=True)
        else:
            print(f"FAIL {tag}\n{r.stderr[-300:]}", flush=True)


if __name__ == "__main__":
    main()
