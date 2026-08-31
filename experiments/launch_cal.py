"""Pre-registered meta-LR calibration sweep (separate seeds from evaluation).

Methods with raw-scale hypergradients get a small meta-LR grid; the best
per method (by mean NMSE over both lr0 scenarios) is then FROZEN for the
E2 v3 evaluation runs. Calibration seeds {100, 101} never appear in eval.

Runs its slice sequentially on one GPU. Split across servers by --slice i/n.
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# round 2 (2026-07-03): best sat at grid boundary -> extend once
# (pre-registered rule); also re-attempts round-1 runs lost to the
# eigvalsh crash (now fixed), since existing results are skipped.
# round 3: bracket the remaining boundary optima (fmd already bracketed by
# divergence at 20). Terminal round by protocol: interior optimum or
# divergence bracket, whichever comes first.
GRID = []
for meth, mls in [("hd", [0.02, 0.2, 2.0, 20.0, 200.0]),
                  ("tfmd", [0.02, 0.2, 2.0, 20.0, 200.0]),
                  ("fmd", [0.02, 0.2, 2.0, 20.0]),
                  ("cohg", [0.05, 0.1, 0.2, 0.4, 0.8]),
                  ("cohg_nogate", [0.05, 0.1, 0.2, 0.4, 0.8])]:
    for ml in mls:
        GRID.append((meth, ml))

CAL_SEEDS = [100, 101]
LR0S = [0.003, 0.03]
DATASET = "mackey"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=int, default=0)
    ap.add_argument("--n-slices", type=int, default=1)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out_dir = os.path.join(HERE, "..", "results", "e2_cal")
    os.makedirs(out_dir, exist_ok=True)
    cfgs = [(m, ml, lr0, s) for (m, ml), lr0, s in
            itertools.product(GRID, LR0S, CAL_SEEDS)]
    cfgs = [c for i, c in enumerate(cfgs) if i % args.n_slices == args.slice]
    print(f"slice {args.slice}/{args.n_slices}: {len(cfgs)} runs", flush=True)
    for meth, ml, lr0, seed in cfgs:
        tag = f"{DATASET}_{meth}_ml{ml:g}_lr{lr0:g}_s{seed}"
        if os.path.exists(os.path.join(out_dir, f"{tag}.json")):
            continue
        t0 = time.time()
        r = subprocess.run(
            [PY, os.path.join(HERE, "e2_timeseries.py"), "--method", meth,
             "--dataset", DATASET, "--seed", str(seed), "--lr", str(lr0),
             "--meta-lr", str(ml), "--steps", str(args.steps),
             "--device", args.device, "--out-dir", out_dir],
            capture_output=True, text=True, cwd=HERE)
        src = os.path.join(out_dir, f"{DATASET}_{meth}_lr{lr0:g}_s{seed}.json")
        dst = os.path.join(out_dir, f"{tag}.json")
        if os.path.exists(src):
            os.replace(src, dst)
            print(f"done {tag} ({time.time() - t0:.0f}s)", flush=True)
        else:
            print(f"FAIL {tag}\n{r.stderr[-300:]}", flush=True)


if __name__ == "__main__":
    main()
