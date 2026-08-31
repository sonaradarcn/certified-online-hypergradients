"""Round-4 follow-up: does a VERIFIED drift envelope keep any certified
adaptation?  (`--gate-warmup`)

round4_adaptmh.md showed that on E2 `mackey_drift` every COHG gate opening
happens at steps 1-15, i.e. BEFORE the first probe-to-probe drift observation
`M_obs` exists (the first probe pair completes at step 2*probe_every = 20), so
every opening is certified under the unverified floor.  This study holds the
gate shut until the envelope is backed by a measurement:

  first-obs   no coordinate may open before the first `M_obs` is recorded
  stable-env  ... and additionally the most recent probe must not have RAISED
              the envelope

Arms (mackey_drift, lr0 0.003, alpha 0.4, c=2, K 10, rank 4, gamma 0.9,
probe-every 20, 12000 steps, seeds 0-9, CPU, `--validate-cert`):

  A  cohg --adaptive-mh 1 --gate-warmup first-obs      _amh1_wfo
  B  cohg --adaptive-mh 1 --gate-warmup stable-env     _amh1_wse
  C  cohg --M-H 5 --fail-closed --gate-warmup first-obs  _mh5_fc1_wfo

plus the DEFAULT-PATH regression check (`--gate-warmup off`, seeds 0-1,
lam-every 50) against results/e2_verify4, which must be bit-identical.

Outputs: results/e2_warmup/ (regression runs in results/e2_warmup/verify/).
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "results", "e2_warmup")
VER = os.path.join(OUT, "verify")
PY = sys.executable

DATASET = "mackey_drift"
LR = 0.003
ALPHA = 0.4
STEPS = 12_000
SEEDS = list(range(10))
MH_FLOOR = 5.0


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
    if part in ("all", "warmup"):
        for s in SEEDS:
            jobs.append(job("_amh1_wfo", s,
                            ["--M-H", repr(MH_FLOOR), "--adaptive-mh", "1.0",
                             "--gate-warmup", "first-obs", "--validate-cert"],
                            OUT))
        for s in SEEDS:
            jobs.append(job("_amh1_wse", s,
                            ["--M-H", repr(MH_FLOOR), "--adaptive-mh", "1.0",
                             "--gate-warmup", "stable-env", "--validate-cert"],
                            OUT))
        for s in SEEDS:
            jobs.append(job("_mh5_fc1_wfo", s,
                            ["--M-H", repr(MH_FLOOR), "--fail-closed",
                             "--gate-warmup", "first-obs", "--validate-cert"],
                            OUT))
    return jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["all", "warmup", "verify"])
    a = ap.parse_args()
    js = make_jobs(a.part)
    print(f"{len(js)} jobs; "
          f"{sum(1 for _n, p, _c in js if os.path.exists(p))} already landed")
    for n, p, c in js:
        print("   ", n)
