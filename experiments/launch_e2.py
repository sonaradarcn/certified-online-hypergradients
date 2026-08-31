"""E2 launcher: two GPU workers draining a config queue (tracker R020-R023).

Run matrix:
- fixed lambda grid: lr in {0.003, 0.01, 0.03, 0.1, 0.3}  (fixed family + oracle-fixed)
- adaptive methods from THREE initial LRs {0.003, 0.03, 0.3}: well-set and
  deliberately mis-set (10x low / 10x high) — gate-opening scenarios (F6)
- methods: hd, hd_scalar, hdm, tfmd, fmd, cohg, cohg_r0, cohg_nogate
- datasets: mackey, lorenz, sunspot, santafe
- seeds: 0..9

Skips configs whose result JSON already exists (resumable).

Usage: python launch_e2.py [--seeds 10] [--dry-run] [--steps 20000]
"""

from __future__ import annotations

import argparse
import itertools
import os
import queue
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "e2")
PY = sys.executable

# v3 matrix (2026-07-03): LR grid extended to bracket the optimum and reach
# the instability region (0.6+); mis-set-high moved to 1.0 (genuinely
# unstable -> safety arena); nonstationary drift datasets added.
FIXED_LRS = [0.003, 0.01, 0.03, 0.1, 0.3, 0.6, 1.0]
ADAPTIVE_WELL = ["hd", "hd_scalar", "hdm", "tfmd", "fmd",
                 "cohg", "cohg_r0", "cohg_nogate"]
ADAPTIVE_MISSET = ["hd", "cohg", "cohg_nogate"]
WELL_LR = 0.03
MISSET_LRS = [0.003, 1.0]
DATASETS = ["mackey", "lorenz", "sunspot", "santafe",
            "mackey_drift", "lorenz_drift"]
# adaptive arms only where adaptation is the question (drift + synthetic
# stationary pair); sunspot/santafe keep the fixed family as oracle anchors
# (F6/F8: adaptation gains live in drift/mis-set scenarios) — cuts E2 v3
# wall-clock by ~1/3
ADAPTIVE_DATASETS = ["mackey", "lorenz", "mackey_drift", "lorenz_drift"]
# meta-LR per method: FROZEN from the pre-registered calibration sweep
# (results/e2_cal, seeds 100-101 disjoint from evaluation; amended rule:
# lowest NMSE among zero-event configs; nogate/r0 inherit cohg's value as
# ablations; findings F10/F11)
META_LR = {"hd": 200.0, "hd_scalar": 200.0, "hdm": 0.1, "tfmd": 200.0,
           "fmd": 2.0, "cohg": 0.4, "cohg_r0": 0.4, "cohg_nogate": 0.4}


def make_configs(seeds: int, seed_start: int = 0, only_fixed: bool = False):
    cfgs = []
    for ds, seed in itertools.product(DATASETS, range(seed_start, seeds)):
        for lr in FIXED_LRS:
            cfgs.append(("fixed", ds, seed, lr))
        if only_fixed or ds not in ADAPTIVE_DATASETS:
            continue
        for meth in ADAPTIVE_WELL:
            cfgs.append((meth, ds, seed, WELL_LR))
        for meth, lr in itertools.product(ADAPTIVE_MISSET, MISSET_LRS):
            cfgs.append((meth, ds, seed, lr))
    return cfgs


def result_path(meth, ds, seed, lr):
    return os.path.join(OUT, f"{ds}_{meth}_lr{lr:g}_s{seed}.json")


def worker(gpu: int, q: "queue.Queue", steps: int, failures: list):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    while True:
        try:
            meth, ds, seed, lr = q.get_nowait()
        except queue.Empty:
            return
        cmd = [PY, os.path.join(HERE, "e2_timeseries.py"),
               "--method", meth, "--dataset", ds, "--seed", str(seed),
               "--lr", str(lr), "--steps", str(steps), "--device", "cuda:0",
               "--gamma", "0.9", "--kw-eps", "0.1", "--probe-every", "20"]
        ml = META_LR.get(meth)
        if ml is not None:
            cmd += ["--meta-lr", str(ml)]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                              cwd=HERE)
        tag = f"{ds}/{meth}/lr{lr:g}/s{seed}"
        if proc.returncode != 0:
            failures.append(tag)
            print(f"[gpu{gpu}] FAIL {tag}\n{proc.stderr[-500:]}", flush=True)
        else:
            print(f"[gpu{gpu}] done {tag} ({time.time() - t0:.0f}s) "
                  f"| left ~{q.qsize()}", flush=True)
        q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--only-fixed", action="store_true",
                    help="run only the fixed-lambda arms (calibration-free)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    cfgs = [c for c in make_configs(args.seeds, args.seed_start,
                                    args.only_fixed)
            if not os.path.exists(result_path(*c))]
    print(f"{len(cfgs)} configs to run on GPUs {args.gpus}")
    if args.dry_run:
        for c in cfgs[:20]:
            print("  ", c)
        return

    q: "queue.Queue" = queue.Queue()
    for c in cfgs:
        q.put(c)
    failures: list = []
    threads = [threading.Thread(target=worker,
                                args=(g, q, args.steps, failures))
               for g in args.gpus]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print(f"ALL DONE | failures: {len(failures)}")
    for f in failures:
        print("  FAIL", f)


if __name__ == "__main__":
    main()
