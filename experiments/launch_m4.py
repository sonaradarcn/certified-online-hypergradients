"""M4 decisive-ablation launcher (tracker R030-R034).

Axes (datasets: mackey_drift + lorenz; lr0 = 0.03; 5 seeds; v3+DriftHold v2):
  rscan : cohg r in {0,1,2,4,8,16}        (K=10)         -- novelty isolation A3
  kscan : cohg K in {1,5,10,20,50}        (r=4)          -- laziness sweet spot
  cscan : cohg gate_factor in {1,1.5,2,3} (r=4, K=10)    -- gate threshold
  ctrl  : cohg_ogd vs cohg (sign)         (r=4, K=10)    -- T3-OGD vs practical
  gscan : gamma in {0.8, 0.9, 0.95}       (r=4, K=10)    -- discount ladder

Each config writes to results/m4/ with a tag encoding the axis, so nothing
collides with the main E2 files. Slice with --slice/--n-slices for the fleet.
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "m4")
PY = sys.executable

DATASETS = ["mackey_drift", "lorenz"]
SEEDS = range(5)
LR0 = 0.03


def make_configs():
    cfgs = []  # (label, method, extra_args, tag)
    for r in [0, 1, 2, 4, 8, 16]:
        cfgs.append(("rscan", "cohg", ["--rank", str(r)], f"_r{r}"))
    for K in [1, 5, 10, 20, 50]:
        cfgs.append(("kscan", "cohg", ["--K", str(K)], f"_K{K}"))
    for c in [1.0, 1.5, 2.0, 3.0]:
        cfgs.append(("cscan", "cohg", ["--gate-factor", str(c)], f"_c{c:g}"))
    cfgs.append(("ctrl", "cohg_ogd", [], "_ogd"))
    for g in [0.8, 0.9, 0.95]:
        cfgs.append(("gscan", "cohg", ["--gamma", str(g)], f"_g{g:g}"))
    full = []
    for ds in DATASETS:
        for seed in SEEDS:
            for label, meth, extra, tag in cfgs:
                full.append((ds, seed, label, meth, extra, tag))
    return full


def result_path(ds, meth, tag, seed):
    return os.path.join(OUT, f"{ds}_{meth}_lr{LR0:g}{tag}_s{seed}.json")


def worker(gpu, q, failures):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    while True:
        try:
            ds, seed, label, meth, extra, tag = q.get_nowait()
        except queue.Empty:
            return
        cmd = [PY, os.path.join(HERE, "e2_timeseries.py"), "--method", meth,
               "--dataset", ds, "--seed", str(seed), "--lr", str(LR0),
               "--steps", "12000", "--device", "cuda:0",
               "--gamma", "0.9", "--kw-eps", "0.1", "--probe-every", "20",
               "--meta-lr", "0.4", "--tag", tag, "--out-dir", OUT] + extra
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                              cwd=HERE)
        name = f"{label}/{ds}{tag}/s{seed}"
        if proc.returncode != 0:
            failures.append(name)
            print(f"[gpu{gpu}] FAIL {name}\n{proc.stderr[-300:]}", flush=True)
        else:
            print(f"[gpu{gpu}] done {name} ({time.time() - t0:.0f}s) "
                  f"| left ~{q.qsize()}", flush=True)
        q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=int, default=0)
    ap.add_argument("--n-slices", type=int, default=1)
    ap.add_argument("--gpus", type=int, nargs="+", default=[0])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    cfgs = [c for i, c in enumerate(make_configs())
            if i % args.n_slices == args.slice
            and not os.path.exists(result_path(c[0], c[3], c[5], c[1]))]
    print(f"M4 slice {args.slice}/{args.n_slices}: {len(cfgs)} configs")
    if args.dry_run:
        for c in cfgs[:10]:
            print("  ", c[2], c[0], c[5], "s" + str(c[1]))
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
    print(f"M4 slice DONE | failures: {len(failures)}")
    for f in failures:
        print("  FAIL", f)


if __name__ == "__main__":
    main()
