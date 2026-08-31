"""Round-3 CPU campaign launcher (review replies P6 / P10 / P1 / P2).

All E2 arms share the frozen E2 mis-set-init config: mackey_drift, lr0=0.003,
12000 steps, kw-eps 0.1, probe-every 20, K 10, rank 4, gate factor c=2,
alpha (--meta-lr) 0.4, seeds 0..9, device CPU -- identical to
launch_e2_controls.py, so every number is directly comparable to
results/e2_controls/SUMMARY.md.

Parts
  verify   COHG reference arm re-run on the PATCHED script (seeds 0,1) to
           prove the additive edits left the legacy path bit-identical
  b1       madgate: calibration-free |ghat_j| > c * MAD_t gate     10 runs
  b2       ogd_doubling: prospective projected-OGD step size       10 runs
  b3       cohg + --validate-full at gamma in {0.8,0.9,0.95,0.99}  40 runs
  b4       E1 teacher/kw_drift M_H misspecification x fail-closed  11 procs
  all      everything above in one pool (longest first)
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
CTL = os.path.join(RES, "e2_controls")
GAM = os.path.join(RES, "e2_gamma")
MIS = os.path.join(RES, "e1_misspec")
VER = os.path.join(RES, "e2_verify")
PY = sys.executable

DATASET = "mackey_drift"
LR = 0.003
ALPHA = 0.4
STEPS = 12_000
BASE = ["--dataset", DATASET, "--lr", str(LR), "--kw-eps", "0.1",
        "--probe-every", "20", "--device", "cpu"]
SEEDS = list(range(10))

# E1 (B4)
E1_SEEDS = 5
E1_CAL_SEEDS = 3
E1_CAL_OFFSET = 100
MH_FACTORS = [1.0, 0.3, 0.1, 0.03, 0.01]


def e2_job(method, tag, seed, extra, out_dir, gamma="0.9", steps=STEPS):
    cmd = ([PY, os.path.join(HERE, "e2_timeseries.py"),
            "--method", method, "--seed", str(seed), "--steps", str(steps),
            "--meta-lr", str(ALPHA), "--tag", tag, "--out-dir", out_dir,
            "--gamma", gamma] + BASE + extra)
    path = os.path.join(out_dir, f"{DATASET}_{method}_lr{LR:g}{tag}_s{seed}.json")
    return (f"{method}{tag}/s{seed}", path, cmd)


def e1_job(tag, M_H, fail_closed, seeds, offset, out_dir):
    cmd = [PY, os.path.join(HERE, "e1_certificate.py"),
           "--problem", "teacher", "--tier", "kw_drift",
           "--seeds", str(seeds), "--seed-offset", str(offset),
           "--rs", "4", "--Ks", "10", "--gamma", "0.9", "--kw-eps", "0.05",
           "--M-H", repr(M_H), "--out-dir", out_dir, "--tag", tag,
           "--trace-config", "none"]
    if fail_closed:
        cmd.append("--fail-closed")
    return (f"e1{tag}", os.path.join(out_dir, f"teacher_kw_drift{tag}.json"),
            cmd)


def e1_mh_star() -> float:
    """Calibrated M_H* for the E1 teacher/kw_drift stream: the largest
    probe-to-probe observed drift rate M_obs on the CALIBRATION seeds
    (disjoint from the evaluation seeds).  M_obs is measured with the
    fail-closed monitor at an effectively infinite prior, so the probe
    schedule is the nominal one and the measurement is undistorted."""
    p = os.path.join(MIS, "teacher_kw_drift_cal.json")
    rows = json.load(open(p))
    allm = [x for r in rows for x in (r.get("m_obs") or [])]
    if not allm:
        raise RuntimeError("no m_obs in the E1 calibration run")
    allm.sort()
    return allm[-1]


def make_jobs(part, gammas=None, seed_list=None):
    jobs = []
    if part in ("b3", "all", "b3only"):
        os.makedirs(GAM, exist_ok=True)
        for g in (gammas or [0.8, 0.9, 0.95, 0.99]):
            for s in (seed_list or SEEDS):
                jobs.append(e2_job("cohg", f"_g{g:g}", s,
                                   ["--M-H", "5.0", "--validate-cert",
                                    "--validate-full"], GAM, gamma=repr(g)))
    if part in ("verify", "all", "tail"):
        os.makedirs(VER, exist_ok=True)
        for s in [0, 1]:
            jobs.append(e2_job("cohg", "_mh5_fc0", s,
                               ["--M-H", "5.0", "--validate-cert"], VER))
    if part in ("b2", "all", "tail"):
        os.makedirs(CTL, exist_ok=True)
        for s in SEEDS:
            jobs.append(e2_job("ogd_doubling", "", s, ["--M-H", "5.0"], CTL))
    if part in ("b2ref", "all"):
        # same-device (CPU) re-run of the fixed-alpha projected-OGD baseline:
        # results/e2 has it on GPU, and B5 shows the gate decisions are not
        # bit-stable across devices, so the doubling comparison needs its own
        # in-family reference.
        os.makedirs(CTL, exist_ok=True)
        for s in SEEDS:
            jobs.append(e2_job("cohg_ogd", "", s, ["--M-H", "5.0"], CTL))
    if part in ("b1", "all", "tail"):
        os.makedirs(CTL, exist_ok=True)
        for s in SEEDS:
            jobs.append(e2_job("madgate", f"_a{ALPHA:g}", s,
                               ["--madgate-window", "200",
                                "--madgate-warmup", "50"], CTL))
    if part in ("b4cal",):
        os.makedirs(MIS, exist_ok=True)
        jobs.append(e1_job("_cal", 1e12, True, E1_CAL_SEEDS, E1_CAL_OFFSET,
                           MIS))
    if part in ("b4", "all", "tail"):
        os.makedirs(MIS, exist_ok=True)
        mh_star = e1_mh_star()
        print(f"E1 calibrated M_H* = {mh_star:.6g}", flush=True)
        for f in MH_FACTORS:
            for fc in [0, 1]:
                jobs.append(e1_job(f"_x{f:g}_fc{fc}", mh_star * f, bool(fc),
                                   E1_SEEDS, 0, MIS))
    return jobs


def worker(q, failures, lock):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="")
    while True:
        try:
            name, path, cmd = q.get_nowait()
        except queue.Empty:
            return
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                              cwd=HERE)
        with lock:
            if proc.returncode != 0:
                failures.append(name)
                print(f"FAIL {name}\n{proc.stderr[-2000:]}", flush=True)
            else:
                print(f"done {name} ({time.time() - t0:.0f}s) | left "
                      f"~{q.qsize()} | {proc.stdout.strip()[-150:]}",
                      flush=True)
        q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["all", "tail", "b3only", "verify", "b1", "b2",
                             "b2ref", "b3", "b4", "b4cal"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--gammas", type=float, nargs="+", default=None)
    ap.add_argument("--seed-list", type=int, nargs="+", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = [j for j in make_jobs(args.part, args.gammas, args.seed_list)
            if not os.path.exists(j[1])]
    print(f"{len(jobs)} jobs on {args.workers} CPU workers", flush=True)
    if args.dry_run:
        for j in jobs:
            print("   ", " ".join(j[2][1:]))
        return
    q: "queue.Queue" = queue.Queue()
    for j in jobs:
        q.put(j)
    failures, lock = [], threading.Lock()
    ths = [threading.Thread(target=worker, args=(q, failures, lock))
           for _ in range(args.workers)]
    t0 = time.time()
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    print(f"ALL DONE in {time.time() - t0:.0f}s | failures: {len(failures)}",
          flush=True)
    for f in failures:
        print("  FAIL", f, flush=True)


if __name__ == "__main__":
    main()
