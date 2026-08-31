"""E3 launcher: continual-learning run matrix (tracker R040).

Matrix:
- fixed (lr, ewc) grid: center (0.05, 10) + one-factor variations
  {(0.02,10), (0.1,10), (0.05,1), (0.05,100)}  -> 5 fixed arms
- adaptive arms at (lr0, ewc0) = (0.05, 10): hd, tfmd, cohg, cohg_r0,
  cohg_nogate; plus mis-set (0.01, 0.1): hd, cohg, cohg_nogate
- seeds 0..9 (class order and stream reshuffle per seed)

Usage: python launch_e3.py --seeds 3 [--only-fixed] [--gpus 0 1] [--dry-run]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import queue
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "e3")
PY = sys.executable

# fixed grid + probed edges (results/e3_probe: lr 0.4 -> -5pts;
# ewc 3000 -> collapse 0.10/1164 events)
FIXED = [(0.05, 10.0), (0.02, 10.0), (0.1, 10.0), (0.05, 1.0), (0.05, 100.0),
         (0.4, 10.0), (0.05, 1000.0)]
ADAPT_WELL = ["hd", "tfmd", "cohg", "cohg_r0", "cohg_nogate"]
ADAPT_MIS = ["hd", "cohg", "cohg_nogate"]
WELL = (0.05, 10.0)
MIS_LIST = [(0.05, 1000.0),   # EWC near-collapse: adaptation must pull back
            (0.4, 10.0)]      # LR too high

META_LR_PATH = os.path.join(HERE, "..", "results", "e2_cal",
                            "FROZEN_META_LR.json")
# E3-domain calibration overrides for the raw-scale baselines (F19: the
# E2-frozen values are 4 orders of magnitude off on this domain)
META_LR_E3_PATH = os.path.join(HERE, "..", "results", "e3_cal",
                               "FROZEN_META_LR_E3.json")


def meta_lr_for(meth):
    frozen = {}
    try:
        frozen.update(json.load(open(META_LR_PATH)))
    except FileNotFoundError:
        pass
    try:
        frozen.update(json.load(open(META_LR_E3_PATH)))
    except FileNotFoundError:
        pass
    base = {"cohg": "cohg", "cohg_r0": "cohg", "cohg_nogate": "cohg_nogate",
            "hd": "hd", "tfmd": "tfmd"}.get(meth)
    return frozen.get(base)


def make_configs(seeds, seed_start=0, only_fixed=False):
    cfgs = []
    for seed in range(seed_start, seeds):
        for lr, ewc in FIXED:
            cfgs.append(("fixed", seed, lr, ewc))
        if only_fixed:
            continue
        for meth in ADAPT_WELL:
            cfgs.append((meth, seed, *WELL))
        for meth in ADAPT_MIS:
            for mis in MIS_LIST:
                cfgs.append((meth, seed, *mis))
    return cfgs


def result_path(meth, seed, lr, ewc):
    return os.path.join(OUT, f"cifar100_{meth}_lr{lr:g}_ewc{ewc:g}_s{seed}.json")


def worker(gpu, q, failures):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    while True:
        try:
            meth, seed, lr, ewc = q.get_nowait()
        except queue.Empty:
            return
        cmd = [PY, os.path.join(HERE, "e3_continual.py"), "--method", meth,
               "--seed", str(seed), "--lr", str(lr), "--ewc0", str(ewc),
               "--device", "cuda:0"]
        ml = meta_lr_for(meth)
        if ml is not None:
            cmd += ["--meta-lr", str(ml)]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                              cwd=HERE)
        tag = f"{meth}/lr{lr:g}/ewc{ewc:g}/s{seed}"
        if proc.returncode != 0:
            failures.append(tag)
            print(f"[gpu{gpu}] FAIL {tag}\n{proc.stderr[-400:]}", flush=True)
        else:
            print(f"[gpu{gpu}] done {tag} ({time.time() - t0:.0f}s) "
                  f"| left ~{q.qsize()}", flush=True)
        q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--only-fixed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    cfgs = [c for c in make_configs(args.seeds, args.seed_start,
                                    args.only_fixed)
            if not os.path.exists(result_path(*c))]
    print(f"{len(cfgs)} E3 configs on GPUs {args.gpus}")
    if args.dry_run:
        for c in cfgs[:12]:
            print("  ", c)
        return
    q = queue.Queue()
    for c in cfgs:
        q.put(c)
    failures = []
    threads = [threading.Thread(target=worker, args=(g, q, failures))
               for g in args.gpus]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print(f"E3 DONE | failures: {len(failures)}")
    for f in failures:
        print("  FAIL", f)


if __name__ == "__main__":
    main()
