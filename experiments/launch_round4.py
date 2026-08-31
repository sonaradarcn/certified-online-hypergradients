"""Round-4 CPU campaign launcher (reviewer objection: "early calibration then
refusal", and "a static offline threshold matches COHG").

Two studies, both on the frozen E2 mis-set-init config (mackey_drift,
lr0 = 0.003, 12000 steps, gamma 0.9, kw-eps 0.1, probe-every 20, K 10, rank 4,
gate factor c = 2, alpha (--meta-lr) 0.4, seeds 0..9, device CPU) so every
number is directly comparable to results/e2_controls/SUMMARY.md and
results/reanalysis/round3_cpu.md.

Parts
  verify   COHG reference arm re-run on the PATCHED script (seeds 0,1) ->
           results/e2_verify4, diffed against results/e2_verify
  shift    E1 (round-4): LATE AMPLITUDE SHIFT `--scale-shift F`.  The middle
           third of the series (inputs AND targets) is multiplied by F, so the
           right learning rate genuinely changes at stream step 4004 and
           changes back at 8007 -- the same boundaries as the tau switch.
           Arms per F: fixed x 5 LRs, hd, cohg (fc0, cert-audited), cohg
           (fail-closed), cohg_nogate, absgate with the TRANSFERRED constant
           threshold 0.05806520209 (no recalibration).
  adaptmh  E2 (round-4): ONLINE-ENFORCEABLE drift envelope `--adaptive-mh K`.
           M_H,t = max(M_H_floor, KAPPA * max_{s<=t} M_obs,s) with the
           fail-closed monitor on; floor = the deployed prior (5 for E2,
           2.27609 for E1) so the arm is never LESS conservative than the
           paper's.  E2 mackey_drift seeds 0-9 and E1 teacher/kw_drift
           seeds 0-4, KAPPA in {1, 2}.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
SHIFT = os.path.join(RES, "e2_shift")
E2ADAPT = os.path.join(RES, "e2_adaptmh")
E1ADAPT = os.path.join(RES, "e1_adaptmh")
VER = os.path.join(RES, "e2_verify4")
PY = sys.executable

DATASET = "mackey_drift"
LR = 0.003
ALPHA = 0.4
STEPS = 12_000
SEEDS = list(range(10))
BASE = ["--dataset", DATASET, "--lr", str(LR), "--kw-eps", "0.1",
        "--probe-every", "20", "--device", "cpu", "--gamma", "0.9"]

# frozen in results/e2_controls/SUMMARY.md, calibrated on the UNSHIFTED
# stream; transferred here with NO recalibration (that is the point).
ABSGATE_THR = 0.05806520209
HD_META_LR = 200.0              # the E2 convention (results/e2 hd arms)
FIXED_LRS = [0.003, 0.01, 0.03, 0.1, 0.3]

# chosen from the seed-0 pilot (see results/reanalysis/round4_shift.md)
DEFAULT_F = [10.0, 0.2]

# E1 deployed prior = the calibrated M_H* of the teacher/kw_drift stream
E1_MH_FLOOR = 2.2760914236726824
E2_MH_FLOOR = 5.0
KAPPAS = [1.0, 2.0]


def e2_job(method, tag, seed, extra, out_dir, lr=LR, meta_lr=ALPHA,
           steps=STEPS, lam_every=10):
    cmd = ([PY, os.path.join(HERE, "e2_timeseries.py"),
            "--method", method, "--dataset", DATASET, "--seed", str(seed),
            "--steps", str(steps), "--lr", repr(lr),
            "--meta-lr", repr(meta_lr), "--gamma", "0.9", "--kw-eps", "0.1",
            "--probe-every", "20", "--device", "cpu",
            "--lam-every", str(lam_every),
            "--tag", tag, "--out-dir", out_dir] + extra)
    path = os.path.join(out_dir, f"{DATASET}_{method}_lr{lr:g}{tag}_s{seed}.json")
    # the name must be UNIQUE per job: the fixed-LR family shares (method, tag)
    # across five learning rates, so the LR has to be in the key
    return (f"{method}{tag}_lr{lr:g}/s{seed}", path, cmd)


def e1_job(tag, M_H, fail_closed, adaptive_mh, seeds, offset, out_dir):
    cmd = [PY, os.path.join(HERE, "e1_certificate.py"),
           "--problem", "teacher", "--tier", "kw_drift",
           "--seeds", str(seeds), "--seed-offset", str(offset),
           "--rs", "4", "--Ks", "10", "--gamma", "0.9", "--kw-eps", "0.05",
           "--M-H", repr(M_H), "--out-dir", out_dir, "--tag", tag,
           "--trace-config", "none"]
    if adaptive_mh is not None:
        cmd += ["--adaptive-mh", repr(adaptive_mh)]
    elif fail_closed:
        cmd.append("--fail-closed")
    return (f"e1{tag}", os.path.join(out_dir, f"teacher_kw_drift{tag}.json"),
            cmd)


def shift_jobs(f_list):
    """Longest arms first so the cheap ones backfill the tail."""
    jobs = []
    os.makedirs(SHIFT, exist_ok=True)
    for F in f_list:
        ft = f"_F{F:g}"
        sh = ["--scale-shift", repr(F)]
        # 1. COHG, certificate-audited (slowest)
        for s in SEEDS:
            jobs.append(e2_job("cohg", ft + "_mh5_fc0", s,
                               sh + ["--M-H", "5.0", "--validate-cert"],
                               SHIFT))
        # 2. COHG + fail-closed monitor
        for s in SEEDS:
            jobs.append(e2_job("cohg", ft + "_mh5_fc1", s,
                               sh + ["--M-H", "5.0", "--fail-closed"], SHIFT))
        # 3. ungated pure-sign ablation
        for s in SEEDS:
            jobs.append(e2_job("cohg_nogate", ft + f"_a{ALPHA:g}", s, sh,
                               SHIFT))
        # 4. absgate with the TRANSFERRED constant threshold
        for s in SEEDS:
            jobs.append(e2_job("absgate", ft + f"_a{ALPHA:g}", s,
                               sh + ["--absgate-threshold",
                                     repr(ABSGATE_THR)], SHIFT))
        # 5. hypergradient descent baseline
        for s in SEEDS:
            jobs.append(e2_job("hd", ft, s, sh, SHIFT, meta_lr=HD_META_LR))
        # 6. fixed-LR family (gives the per-segment post-hoc-best oracle)
        for lr in FIXED_LRS:
            for s in SEEDS:
                jobs.append(e2_job("fixed", ft, s, sh, SHIFT, lr=lr))
    return jobs


def adaptmh_jobs():
    jobs = []
    os.makedirs(E2ADAPT, exist_ok=True)
    os.makedirs(E1ADAPT, exist_ok=True)
    # E1 first: fp64 teacher/student, ~150 s per process
    for k in KAPPAS:
        jobs.append(e1_job(f"_amh{k:g}", E1_MH_FLOOR, True, k, 5, 0, E1ADAPT))
    # fixed-prior E1 reference at the same floor (fail-closed and not), so the
    # "gate decisions vs the fixed-prior arm" comparison has a same-directory
    # partner; identical to results/e1_misspec/teacher_kw_drift_x1_fc{0,1}
    for fc in (0, 1):
        jobs.append(e1_job(f"_fixed_fc{fc}", E1_MH_FLOOR, bool(fc), None, 5, 0,
                           E1ADAPT))
    for k in KAPPAS:
        for s in SEEDS:
            jobs.append(e2_job("cohg", f"_amh{k:g}", s,
                               ["--M-H", repr(E2_MH_FLOOR),
                                "--adaptive-mh", repr(k),
                                "--validate-cert"], E2ADAPT))
    return jobs


def verify_jobs():
    os.makedirs(VER, exist_ok=True)
    return [e2_job("cohg", "_mh5_fc0", s, ["--M-H", "5.0", "--validate-cert"],
                   VER, lam_every=50) for s in (0, 1)]


def make_jobs(part="all", f_list=None):
    """Dispatch order: the adaptive-envelope study FIRST (E1 fp64 is quick,
    E2 is 10 seeds x 2 KAPPA), then the scale-shift grid."""
    f_list = f_list or DEFAULT_F
    jobs = []
    if part in ("all", "adaptmh"):
        jobs += adaptmh_jobs()
    if part in ("all", "shift"):
        jobs += shift_jobs(f_list)
    if part in ("verify",):
        jobs += verify_jobs()
    return jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["all", "shift", "adaptmh", "verify"])
    ap.add_argument("--F", type=float, nargs="+", default=None)
    args = ap.parse_args()
    js = make_jobs(args.part, args.F)
    print(f"{len(js)} jobs; {sum(1 for _n, p, _c in js if os.path.exists(p))}"
          f" already landed")
    for n, p, c in js[:4] + js[-4:]:
        print("   ", n, "|", " ".join(c[1:]))
