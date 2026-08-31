"""Fix-1 corrective campaign orchestrator (GPU1 by default).

Phase (a) — Task B: E2 flagship-table missing arms, run sequentially on one
GPU with e2_timeseries.py using the exact conventions of the existing
results/e2 runs (steps 12000, gamma 0.9, kw-eps 0.1, probe-every 20,
default rank/K/gate-factor/M-H):

  1. mackey_drift @ mis-set lr0=0.003, seeds 0-9, methods hdm (ml 0.1),
     tfmd (ml 200), fmd (ml 2) per results/e2_cal/FROZEN_META_LR.json /
     launch_e2.py META_LR, and cohg_ogd (ml 0.4 — ablation arms inherit
     the parent cohg meta-step per the documented amendment; matches
     meta_lr in results/e2/mackey_drift_cohg_lr0.003_s0.json).
  2. cohg (sign, ml 0.4 frozen, default rank/K/gamma) on sunspot and
     santafe at lr {0.003, 0.03}, seeds 0-9 — the two LR values existing
     baseline arms cover there (hd has 0.003/0.03/0.3; all other methods
     0.03; capped at 2 LRs: mis-set 0.003 + well-set 0.03).
  3. Seed top-up 4 -> 10 (existing runs are s5-8; enumerate s0-9 and skip
     existing) for the sunspot/santafe arms already present: hd @
     {0.003, 0.03, 0.3}, hd_scalar/hdm/tfmd/fmd @ 0.03 — all with
     meta-lr 0.02, matching the existing s5-8 runs exactly so seeds pool.

Any config whose results/e2 JSON already exists is skipped (resumable).

Phase (b): when the Task-B grid drains, this process joins the E4 v2 queue
(launch_e4v2.worker) as a second worker on the same GPU; .claim files
prevent duplication with the launch_e4v2.py process running on GPU0.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import launch_e4v2 as L4

E2OUT = os.path.join(HERE, "..", "results", "e2")
PY = sys.executable

E2_COMMON = ["--steps", "12000", "--gamma", "0.9", "--kw-eps", "0.1",
             "--probe-every", "20", "--device", "cuda:0"]

FROZEN_ML = {"hdm": 0.1, "tfmd": 200.0, "fmd": 2.0,
             "cohg_ogd": 0.4, "cohg": 0.4}
ANCHOR_ML = 0.02  # every existing sunspot/santafe adaptive arm used 0.02

TOPUP_ARMS = [("hd", 0.003), ("hd", 0.03), ("hd", 0.3),
              ("hd_scalar", 0.03), ("hdm", 0.03), ("tfmd", 0.03),
              ("fmd", 0.03)]


def make_b_configs():
    """(dataset, method, lr, meta_lr, seed), flagship arms first."""
    cfgs = []
    for meth in ["hdm", "tfmd", "fmd", "cohg_ogd"]:
        for s in range(10):
            cfgs.append(("mackey_drift", meth, 0.003, FROZEN_ML[meth], s))
    for ds in ["sunspot", "santafe"]:
        for lr in [0.003, 0.03]:
            for s in range(10):
                cfgs.append((ds, "cohg", lr, FROZEN_ML["cohg"], s))
    for ds in ["sunspot", "santafe"]:
        for meth, lr in TOPUP_ARMS:
            for s in range(10):
                cfgs.append((ds, meth, lr, ANCHOR_ML, s))
    return cfgs


def e2_result(ds, meth, lr, seed):
    return os.path.join(E2OUT, f"{ds}_{meth}_lr{lr:g}_s{seed}.json")


def pending():
    return [c for c in make_b_configs()
            if not os.path.exists(e2_result(c[0], c[1], c[2], c[4]))]


def run_task_b(gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    todo = pending()
    print(f"[fix1] Task B: {len(todo)} configs on GPU{gpu}", flush=True)
    failures = []
    for i, (ds, meth, lr, ml, seed) in enumerate(todo):
        res = e2_result(ds, meth, lr, seed)
        if os.path.exists(res):
            continue
        cmd = [PY, os.path.join(HERE, "e2_timeseries.py"),
               "--method", meth, "--dataset", ds, "--seed", str(seed),
               "--lr", str(lr), "--meta-lr", str(ml)] + E2_COMMON
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              env=env, cwd=HERE)
        tag = f"{ds}/{meth}/lr{lr:g}/s{seed}"
        if proc.returncode != 0 or not os.path.exists(res):
            failures.append(tag)
            print(f"[fix1] FAIL {tag}\n{proc.stderr[-400:]}", flush=True)
        else:
            print(f"[fix1] done {tag} ({time.time() - t0:.0f}s) "
                  f"| {i + 1}/{len(todo)}", flush=True)
    print(f"[fix1] Task B done | failures: {len(failures)}", flush=True)
    for f in failures:
        print("  FAIL", f, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    todo = pending()
    e4_todo = [c for c in L4.make_configs()
               if not os.path.exists(L4.result_path(c[0], c[1], c[2], c[4]))]
    print(f"[fix1] Task B pending: {len(todo)} | "
          f"E4v2 pending (pre-claim): {len(e4_todo)}", flush=True)
    if args.dry_run:
        for c in todo:
            print("   B:", c)
        for c in e4_todo:
            print("  E4:", c)
        return
    os.makedirs(E2OUT, exist_ok=True)
    run_task_b(args.gpu)
    print(f"[fix1] phase b: joining E4 v2 queue on GPU{args.gpu}",
          flush=True)
    L4.worker(args.gpu)
    print("[fix1] campaign complete", flush=True)


if __name__ == "__main__":
    main()
