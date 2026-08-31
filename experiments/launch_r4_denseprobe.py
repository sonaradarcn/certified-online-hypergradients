"""Round-4 follow-up: does an EARLY MEASUREMENT rescue certified adaptation?
(`--probe-dense-until`, remedy (b) of round4_warmup.md section 9)

round4_adaptmh.md / round4_warmup.md established that on E2 `mackey_drift`
every COHG gate opening happens at steps 1-15, while the first probe-to-probe
drift observation `M_obs` cannot exist before step `2*probe_every` = 20.
Holding the gate until an observation exists therefore removes ALL adaptation.

Remedy (b): make an observation EXIST before the adaptation window by probing
at EVERY step for t <= T, then reverting to the --probe-every cadence.  With
T = 20 the first `M_obs` lands at step 1.

Arms (mackey_drift, lr0 0.003, alpha 0.4, c=2, K 10, rank 4, gamma 0.9,
probe-every 20, 12000 steps, seeds 0-9, CPU, `--validate-cert`):

  i    cohg --adaptive-mh 1 --probe-dense-until 20             _amh1_dp20
  ii   ... + --gate-warmup first-obs                           _amh1_wfo_dp20
  iii  cohg --M-H 5 --fail-closed --probe-dense-until 20       _mh5_fc1_dp20

plus the DEFAULT-PATH regression check (`--probe-dense-until 0`, seeds 0-1,
lam-every 50) against results/e2_verify4, which must be bit-identical.

Outputs: results/e2_denseprobe/ (regression runs in .../verify/).
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "results", "e2_denseprobe")
VER = os.path.join(OUT, "verify")
PY = sys.executable

DATASET = "mackey_drift"
LR = 0.003
ALPHA = 0.4
STEPS = 12_000
SEEDS = list(range(10))
MH_FLOOR = 5.0
DENSE_T = 20


def job(tag, seed, extra, out_dir, lam_every=10):
    cmd = ([PY, os.path.join(HERE, "e2_timeseries.py"),
            "--method", "cohg", "--dataset", DATASET, "--seed", str(seed),
            "--steps", str(STEPS), "--lr", repr(LR),
            "--meta-lr", repr(ALPHA), "--gamma", "0.9", "--kw-eps", "0.1",
            "--probe-every", "20", "--K", "10", "--rank", "4",
            "--gate-factor", "2.0", "--device", "cpu",
            "--lam-every", str(lam_every),
            "--tag", tag, "--out-dir", out_dir] + extra)
    path = os.path.join(out_dir,
                        f"{DATASET}_cohg_lr{LR:g}{tag}_s{seed}.json")
    return (f"cohg{tag}/s{seed}", path, cmd)


def make_jobs(part="all"):
    jobs = []
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(VER, exist_ok=True)
    if part in ("all", "verify"):
        # default path regression vs results/e2_verify4 (must be identical)
        for s in (0, 1):
            jobs.append(job("_mh5_fc0", s,
                            ["--M-H", repr(MH_FLOOR), "--validate-cert"],
                            VER, lam_every=50))
    if part in ("all", "dense"):
        for s in SEEDS:
            jobs.append(job("_amh1_dp20", s,
                            ["--M-H", repr(MH_FLOOR), "--adaptive-mh", "1.0",
                             "--probe-dense-until", str(DENSE_T),
                             "--validate-cert"], OUT))
        for s in SEEDS:
            jobs.append(job("_amh1_wfo_dp20", s,
                            ["--M-H", repr(MH_FLOOR), "--adaptive-mh", "1.0",
                             "--gate-warmup", "first-obs",
                             "--probe-dense-until", str(DENSE_T),
                             "--validate-cert"], OUT))
        for s in SEEDS:
            jobs.append(job("_mh5_fc1_dp20", s,
                            ["--M-H", repr(MH_FLOOR), "--fail-closed",
                             "--probe-dense-until", str(DENSE_T),
                             "--validate-cert"], OUT))
    return jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["all", "dense", "verify"])
    a = ap.parse_args()
    js = make_jobs(a.part)
    print(f"{len(js)} jobs; "
          f"{sum(1 for _n, p, _c in js if os.path.exists(p))} already landed")
    for n, p, c in js:
        print("   ", n)
