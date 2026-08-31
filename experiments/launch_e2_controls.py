"""E2 CONTROLS launcher: "certification vs generic conservatism" (review reply).

All arms share the E2 mis-set-init config: mackey_drift, lr0 = 0.003 (10x too
low), 12000 steps, gamma 0.9, kw-eps 0.1, probe-every 20, K 10, rank 4,
alpha (= --meta-lr) 0.4 unless swept, seeds 0..9 -> results/e2_controls/.

Arms
  A sign_nogate_sweep  cohg_nogate (already a PURE SIGN step of size alpha)
                       at alpha in {0.1, 0.2, 0.4, 0.8}          4 x 10
  B absgate            |ghat_j| > const threshold, rate-matched to COHG  10
  C randgate           i.i.d. open w.p. p* = COHG coord_open_frac        10
  D periodicgate       all coords open every round(1/p*) steps           10
  E t6clip             COHG + Theorem-6 step condition, rho_f in {1, 10} 2 x 10
  F failclosed/M_H     COHG, M_H in {5, 0.5, 0.05} x {no FC, FC},
                       all with --validate-cert                     6 x 10

Device: CPU. The model is a 13k-parameter GRU whose HVPs are kernel-launch
bound, so one CPU core is ~2x FASTER than a 3080 per run (41s vs 88s for a
200-step COHG probe run) and the box has 80 logical cores: 24 concurrent CPU
workers deliver ~17x the throughput of the 2-GPU queue. CPU and CUDA agree to
1.3e-7 relative NMSE with identical gate decisions and HVP counts on the same
config, and EVERY arm compared here (including the COHG reference arm
F/M_H=5/no-FC) runs on the same device, so the comparison is internally exact.

Usage:
    python launch_e2_controls.py --part calib          # B threshold fit
    python launch_e2_controls.py --part main --absgate-threshold <thr>
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import queue
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.abspath(os.path.join(ROOT, "results", "e2_controls"))
E2 = os.path.abspath(os.path.join(ROOT, "results", "e2"))
LOGS = os.path.abspath(os.path.join(ROOT, "results", "logs"))
PY = sys.executable

DATASET = "mackey_drift"
LR = 0.003
ALPHA = 0.4                     # frozen COHG meta-lr (results/e2_cal)
CALIB_SEEDS = [100, 101]        # disjoint from the 0..9 evaluation seeds

BASE = ["--dataset", DATASET, "--lr", str(LR), "--gamma", "0.9",
        "--kw-eps", "0.1", "--probe-every", "20", "--device", "cpu"]


def cohg_open_rate() -> float:
    """COHG's measured per-coordinate gate-open fraction on this config."""
    vals = []
    for f in sorted(glob.glob(os.path.join(
            E2, f"{DATASET}_cohg_lr{LR:g}_s*.json"))):
        d = json.load(open(f))
        if d["coord_open_frac"] is not None:
            vals.append(d["coord_open_frac"])
    if not vals:
        raise RuntimeError("no COHG reference runs in results/e2")
    return sum(vals) / len(vals)


def result_path(method, tag, seed):
    return os.path.join(OUT, f"{DATASET}_{method}_lr{LR:g}{tag}_s{seed}.json")


def job(method, tag, seed, extra, steps):
    cmd = ([PY, os.path.join(HERE, "e2_timeseries.py"),
            "--method", method, "--seed", str(seed), "--steps", str(steps),
            "--meta-lr", str(ALPHA), "--tag", tag, "--out-dir", OUT]
           + BASE + extra)
    return (method, tag, seed, cmd)


def make_jobs(args, p_star):
    """Longest arms first so the short ones backfill the tail."""
    jobs = []
    seeds = list(range(args.seeds))
    if args.part == "calib":
        for s in CALIB_SEEDS:
            jobs.append(job("absgate", "_calib", s, [], args.steps))
        if args.absgate_threshold is not None:
            for s in CALIB_SEEDS:
                jobs.append(job("absgate", "_verify", s,
                                ["--absgate-threshold",
                                 repr(args.absgate_threshold)], args.steps))
        return jobs
    if args.part == "helper":
        # Runs the TAIL of the `main` job list on otherwise-idle cores while
        # `main` works from the front. `main`'s skip-existing check happens
        # once at startup, so it will re-run whatever the helper finished --
        # kill the `main` launcher once its remaining queue is covered here.
        for mh, fc in itertools.product([0.05], [1, 0]):
            for s in seeds:
                extra = ["--M-H", str(mh), "--validate-cert"]
                if fc:
                    extra.append("--fail-closed")
                jobs.append(job("cohg", f"_mh{mh:g}_fc{fc}", s, extra,
                                args.steps))
        for a in [0.1, 0.2, 0.4, 0.8]:
            for s in seeds:
                m, t, sd, cmd = job("cohg_nogate", f"_a{a:g}", s, [],
                                    args.steps)
                cmd[cmd.index("--meta-lr") + 1] = str(a)
                jobs.append((m, t, sd, cmd))
        for s in seeds:
            jobs.append(job("randgate", f"_a{ALPHA:g}", s,
                            ["--randgate-p", repr(p_star)], args.steps))
        period = int(round(1.0 / p_star))
        for s in seeds:
            jobs.append(job("periodicgate", f"_a{ALPHA:g}", s,
                            ["--periodic-every", str(period)], args.steps))
        return jobs
    if args.part == "absgate":
        # B only: launched after the threshold is calibrated + frozen, so it
        # rides alongside the already-running `main` queue
        for s in seeds:
            jobs.append(job("absgate", f"_a{ALPHA:g}", s,
                            ["--absgate-threshold",
                             repr(args.absgate_threshold)], args.steps))
        return jobs
    # F: COHG x M_H x fail-closed, all certificate-validated (slowest)
    for mh, fc in itertools.product([5.0, 0.5, 0.05], [0, 1]):
        for s in seeds:
            extra = ["--M-H", str(mh), "--validate-cert"]
            if fc:
                extra.append("--fail-closed")
            jobs.append(job("cohg", f"_mh{mh:g}_fc{fc}", s, extra, args.steps))
    # E: Theorem-6 enforced step condition
    for rf in [1.0, 10.0]:
        for s in seeds:
            jobs.append(job("t6clip", f"_rf{rf:g}", s,
                            ["--rho-f", str(rf)], args.steps))
    # A: ungated pure-sign sweep
    for a in [0.1, 0.2, 0.4, 0.8]:
        for s in seeds:
            m, t, sd, cmd = job("cohg_nogate", f"_a{a:g}", s, [], args.steps)
            cmd[cmd.index("--meta-lr") + 1] = str(a)
            jobs.append((m, t, sd, cmd))
    # B/C/D: matched-conservatism gates
    if args.absgate_threshold is not None:
        for s in seeds:
            jobs.append(job("absgate", f"_a{ALPHA:g}", s,
                            ["--absgate-threshold",
                             repr(args.absgate_threshold)], args.steps))
    for s in seeds:
        jobs.append(job("randgate", f"_a{ALPHA:g}", s,
                        ["--randgate-p", repr(p_star)], args.steps))
    period = int(round(1.0 / p_star))
    for s in seeds:
        jobs.append(job("periodicgate", f"_a{ALPHA:g}", s,
                        ["--periodic-every", str(period)], args.steps))
    return jobs


def worker(q, failures, lock):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="")
    while True:
        try:
            method, tag, seed, cmd = q.get_nowait()
        except queue.Empty:
            return
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                              cwd=HERE)
        name = f"{method}{tag}/s{seed}"
        with lock:
            if proc.returncode != 0:
                failures.append(name)
                print(f"FAIL {name}\n{proc.stderr[-1500:]}", flush=True)
            else:
                print(f"done {name} ({time.time() - t0:.0f}s) "
                      f"| left ~{q.qsize()} | {proc.stdout.strip()[-160:]}",
                      flush=True)
        q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["calib", "main", "absgate", "helper"],
                    default="main")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=12_000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--absgate-threshold", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    p_star = cohg_open_rate()
    print(f"COHG measured coord_open_frac p* = {p_star:.8f} "
          f"-> periodic period {int(round(1.0 / p_star))}", flush=True)
    jobs = [j for j in make_jobs(args, p_star)
            if not os.path.exists(result_path(j[0], j[1], j[2]))]
    print(f"{len(jobs)} jobs to run on {args.workers} CPU workers", flush=True)
    if args.dry_run:
        for j in jobs[:8] + jobs[-8:]:
            print("   ", " ".join(j[3][1:]))
        return

    q: "queue.Queue" = queue.Queue()
    for j in jobs:
        q.put(j)
    failures: list = []
    lock = threading.Lock()
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
