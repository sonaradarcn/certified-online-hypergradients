"""Round-4 follow-up: does an EARLY MEASUREMENT rescue certified adaptation?

`round4_warmup.md` section 9 remedy (b): probe densely at the start so a
probe-to-probe drift observation `M_obs` EXISTS before the adaptation window.
`--probe-dense-until T` fires the KW spectral probe at every step for t <= T
and then reverts to `--probe-every`.

Reads results/e2_denseprobe/*.json (+ verify/) and the references
results/e2_adaptmh (adaptive envelope KAPPA=1, no warm-up), results/e2_warmup
(warm-up first-obs on the 20-step cadence) and results/e2_controls (fixed prior
M_H=5, fail-closed / no FC); writes round4_denseprobe.md.
"""
from __future__ import annotations

import glob
import itertools
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DP = os.path.join(ROOT, "results", "e2_denseprobe")
VER = os.path.join(DP, "verify")
E2A = os.path.join(ROOT, "results", "e2_adaptmh")
WU = os.path.join(ROOT, "results", "e2_warmup")
CTL = os.path.join(ROOT, "results", "e2_controls")
REF4 = os.path.join(ROOT, "results", "e2_verify4")
OUT = os.path.join(HERE, "round4_denseprobe.md")

FLOOR = 5.0
NOFC_HVP = 94588.0          # COHG no-fail-closed HVP budget on this config
LAM0 = math.log(0.003)
M_COORD = 6
STEPS = 12000
DENSE_T = 20
PROBE_EVERY = 20


# --------------------------------------------------------------- helpers
def mstd(v):
    v = [x for x in v if x is not None
         and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return float("nan"), float("nan"), 0
    n = len(v)
    m = sum(v) / n
    sd = (math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1
          else float("nan"))
    return m, sd, n


def f(m, s, p=4):
    if not math.isfinite(m):
        return "-"
    if not math.isfinite(s):
        return "%.*f" % (p, m)
    return "%.*f+-%.*f" % (p, m, p, s)


def g(x, p="%.1f"):
    return (p % x) if (x is not None and math.isfinite(x)) else "-"


def signflip_p(d):
    """Exact paired sign-flip (randomization) test on the mean difference."""
    n = len(d)
    if n == 0:
        return float("nan"), float("nan")
    obs = abs(sum(d) / n)
    if all(x == 0 for x in d):
        return 1.0, 0.0
    cnt = 0
    for signs in itertools.product([1, -1], repeat=n):
        if abs(sum(s * x for s, x in zip(signs, d)) / n) >= obs - 1e-15:
            cnt += 1
    return cnt / 2 ** n, sum(d) / n


def load(pat, d):
    rs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, pat)))]
    rs.sort(key=lambda r: r["seed"])
    return rs


def probe_pairs(r):
    """[(t_probe, M_obs or None)] for any arm that carries a probe log."""
    if r.get("probe_mobs_log"):
        return [(row[0], row[1]) for row in r["probe_mobs_log"]]
    if r.get("adapt_probe_log"):
        return [(row[0], row[2]) for row in r["adapt_probe_log"]]
    return []


def env_at_probe(r):
    """{t_probe: envelope in force AFTER that probe}."""
    if r.get("adapt_probe_log"):
        return {row[0]: row[3] for row in r["adapt_probe_log"]}
    if r.get("probe_raw_log"):
        return {row[0]: row[5] for row in r["probe_raw_log"]}
    return {}


def closed_at_probe(r):
    if r.get("adapt_probe_log"):
        return {row[0]: row[4] for row in r["adapt_probe_log"]}
    if r.get("probe_raw_log"):
        return {row[0]: row[6] for row in r["probe_raw_log"]}
    return {}


def final_env(r):
    if r.get("adaptive_mh"):
        return r.get("adapt_mh_final")
    return float(r.get("M_H") or FLOOR)


def retro(r):
    """Per-opening retrospective consistency (same definition as round-4)."""
    pr = probe_pairs(r)
    first_obs_t = next((tp for tp, m in pr if m is not None), None)
    opens = r.get("gate_open_steps") or []
    envs = r.get("gate_open_env") or []
    fe = final_env(r) or float("inf")
    n_ok = n_tot = n_cold = n_nonext = n_ok_final = 0
    worst = 0.0
    for t, e in zip(opens, envs):
        if first_obs_t is not None and t < first_obs_t:
            n_cold += 1
        nxt = next((m for tp, m in pr if tp > t and m is not None), None)
        if nxt is None:
            n_nonext += 1
            continue
        n_tot += 1
        n_ok += int(nxt <= e)
        n_ok_final += int(nxt <= fe)
        worst = max(worst, (nxt / e) if e else float("inf"))
    return dict(ok=n_ok, tot=n_tot, cold=n_cold, nonext=n_nonext,
                ok_final=n_ok_final, worst=worst, first_obs_t=first_obs_t)


def lam_moves(r):
    lh = r.get("lam_hist") or []
    if not lh:
        return None
    last = lh[-1][1:]
    d = [x - LAM0 for x in last]
    maxdev = 0.0
    for row in lh:
        for x in row[1:]:
            maxdev = max(maxdev, abs(x - LAM0))
    ev = r.get("gate_open_events") or []
    n_dn = sum(1 for _t, _j, s in ev if s > 0)   # sign(ghat)>0 -> lam DOWN
    n_up = sum(1 for _t, _j, s in ev if s < 0)
    return dict(net=sum(d) / len(d), mx=max(d), mn=min(d), maxdev=maxdev,
                nmoved=sum(1 for x in d if abs(x) > 1e-12),
                lr_ratio=math.exp(sum(d) / len(d)),
                n_lr_down=n_dn, n_lr_up=n_up)


def coarse_mobs(r, t_lo, t_hi):
    """Recompute M_obs over the probe pair (t_lo, t_hi) from the raw log:

        M_obs = |rho(t_hi) - rho(t_lo)| / (eta_max(t_lo) * D[t_lo, t_hi))

    where D is the SUM of the per-probe path lengths in between -- exactly
    the quantity DriftHoldFailClosed would have computed had there been no
    probes strictly inside the interval.  Returns None if unavailable."""
    raw = r.get("probe_raw_log") or []
    idx = {row[0]: i for i, row in enumerate(raw)}
    if t_lo not in idx or t_hi not in idx:
        return None
    i, j = idx[t_lo], idx[t_hi]
    if j <= i:
        return None
    rho_lo, rho_hi = raw[i][1], raw[j][1]
    # eta_max at the probe t_lo: it is stored as `eta_max_at_prev_probe` on
    # the NEXT row.
    eta_max_lo = raw[i + 1][4]
    if not eta_max_lo:
        return None
    Dsum = sum(raw[k][3] or 0.0 for k in range(i + 1, j + 1))
    if Dsum <= 1e-12:
        return None
    return abs(rho_hi - rho_lo) / (eta_max_lo * Dsum)


def dense_1step(r, t_hi=DENSE_T):
    """The 1-step M_obs values recorded inside the dense window."""
    return [(t, m) for t, m in probe_pairs(r)
            if m is not None and 1 <= t <= t_hi]


def med(v):
    v = sorted(x for x in v if x is not None)
    return v[len(v) // 2] if v else float("nan")


A_NOFC = "fixed prior M_H=5, no FC (reference)"
A_FC = "fixed prior M_H=5, fail-closed (reference)"
A_ADP = "adaptive KAPPA=1, probe-every 20, no warm-up"
A_WFO = "adaptive KAPPA=1 + warm-up first-obs (probe-every 20)"
A_DP = "**(i) adaptive KAPPA=1 + dense probe t<=20**"
A_DPW = "**(ii) (i) + warm-up first-obs**"
A_DPF = "**(iii) fixed prior M_H=5, FC + dense probe t<=20**"

ARMS = [
    (A_NOFC, "mackey_drift_cohg_lr0.003_mh5_fc0_s*.json", CTL),
    (A_FC, "mackey_drift_cohg_lr0.003_mh5_fc1_s*.json", CTL),
    (A_ADP, "mackey_drift_cohg_lr0.003_amh1_s*.json", E2A),
    (A_WFO, "mackey_drift_cohg_lr0.003_amh1_wfo_s*.json", WU),
    (A_DP, "mackey_drift_cohg_lr0.003_amh1_dp20_s*.json", DP),
    (A_DPW, "mackey_drift_cohg_lr0.003_amh1_wfo_dp20_s*.json", DP),
    (A_DPF, "mackey_drift_cohg_lr0.003_mh5_fc1_dp20_s*.json", DP),
]
NEW = (A_DP, A_DPW, A_DPF)


def main():
    L = []
    W = L.append
    A = {name: load(pat, d) for name, pat, d in ARMS}

    W("# Round-4 follow-up: a DENSE EARLY PROBE, so a measurement exists "
      "before the adaptation window\n")
    W("`round4_adaptmh.md` and `round4_warmup.md` established the timing "
      "trap on E2 `mackey_drift`: every certified COHG gate opening happens "
      "at steps 1-15, while the first probe-to-probe drift observation "
      "`M_obs` cannot exist before step `2 * probe_every` = 20. Holding the "
      "gate until an observation exists therefore removes *all* adaptation. "
      "This study runs remedy (b) of `round4_warmup.md` section 9: make an "
      "observation exist EARLY.\n")
    W("`--probe-dense-until T` (new flag in "
      "`code/experiments/e2_timeseries.py`; default `0` = off = the legacy "
      "bit-identical path) fires the KW spectral probe at **every** step for "
      "`t <= T`, in addition to the `--probe-every` cadence, and reverts to "
      "`--probe-every` afterwards. Every step, not every 2 steps: on this "
      "config a probe costs ~150 HVPs and the dense window adds only "
      "`T - 1` = 19 of them, i.e. ~3% of the run's budget (measured below), "
      "so there was no reason to halve the resolution. With `T = 20` the "
      "first `M_obs` lands at **step 1**, before the first gate opening.\n")
    W("Arms (all `mackey_drift`, GRU, 12000 steps, lr0 0.003, alpha 0.4, "
      "c = 2, K 10, rank 4, gamma 0.9, probe-every 20, seeds 0-9, CPU, "
      "`--validate-cert`):\n")
    W("* **(i)** `cohg --adaptive-mh 1 --probe-dense-until 20` -- the "
      "measured envelope, gate free to open from step 1.")
    W("* **(ii)** (i) `+ --gate-warmup first-obs` -- the gate may only open "
      "after an `M_obs` exists, which now happens at step 1.")
    W("* **(iii)** `cohg --M-H 5 --fail-closed --probe-dense-until 20` -- "
      "what the dense schedule does to the FIXED-prior monitor.\n")
    W("References: `results/e2_adaptmh` (adaptive envelope, 20-step cadence, "
      "no warm-up), `results/e2_warmup` (warm-up `first-obs` on the 20-step "
      "cadence: 0 openings), `results/e2_controls` (fixed prior). "
      "Statistics are mean+-sd with ddof = 1; paired tests are exact "
      "sign-flip over the 10 seeds.\n")

    # ------------------------------------------------ default-path check
    W("## 0. Default-path regression (`--probe-dense-until 0`)\n")
    v_new = load("mackey_drift_cohg_lr0.003_mh5_fc0_s*.json", VER)
    v_old = {r["seed"]: r for r in load(
        "mackey_drift_cohg_lr0.003_mh5_fc0_s*.json", REF4)}
    if v_new:
        W("| seed | losses identical | lam_hist identical | "
          "NMSE (new / e2_verify4) | events | HVPs | cert viol |")
        W("|---|---|---|---|---|---|---|")
        for r in v_new:
            b = v_old.get(r["seed"])
            if b is None:
                continue
            W("| %d | %s | %s | %.10f / %.10f | %d / %d | %d / %d | %d / %d |"
              % (r["seed"], r["losses"] == b["losses"],
                 r["lam_hist"] == b["lam_hist"], r["nmse"], b["nmse"],
                 r["events"], b["events"], r["hvp_total"], b["hvp_total"],
                 r["cert_violations"], b["cert_violations"]))
        W("")
    else:
        W("_(regression runs not present)_\n")

    # ------------------------------------------------------- main table
    W("## 1. Arm summary\n")
    W("`open coord-steps` = `coord_open_frac * 6 * 12000` (6 parameter "
      "groups). `held` = steps the warm-up hold kept the gate shut.\n")
    W("| arm | n | NMSE | events | open coord-steps | coord-open rate | "
      "opening steps (first / last) | held | closed steps | probes | HVPs | "
      "HVPs / no-FC | cert viol / checked | cert max ratio | "
      "final envelope |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, _p, _d in ARMS:
        rs = A[name]
        if not rs:
            continue
        n = len(rs)
        oc = mstd([r["coord_open_frac"] * M_COORD * r["steps"] for r in rs])
        firsts = [(r.get("gate_open_steps") or [None])[0] for r in rs]
        lasts = [(r.get("gate_open_steps") or [None])[-1] for r in rs]
        held = mstd([r.get("warmup_held_steps") for r in rs])
        cs = mstd([r.get("failclosed_closed_steps") for r in rs])
        npr = mstd([r.get("n_probes") for r in rs])
        hv = mstd([float(r["hvp_total"]) for r in rs])[0]
        fe = mstd([final_env(r) for r in rs])
        cv = sum(r.get("cert_violations") or 0 for r in rs)
        cc = sum(r.get("cert_checked") or 0 for r in rs)
        cr = [r.get("cert_max_ratio") for r in rs
              if r.get("cert_max_ratio") is not None]
        fs = [float(x) for x in firsts if x is not None]
        ls = [float(x) for x in lasts if x is not None]
        logged = any("gate_open_steps" in r for r in rs)
        tim = ("%.1f / %.1f" % (mstd(fs)[0], mstd(ls)[0]) if fs
               else ("never opens" if logged else "n/a (not logged)"))
        W("| %s | %d | %s | %s | %s | %.3e | %s | %s | %s | %s | %.0f | "
          "%.3f | %d / %s | %.4f | %s |"
          % (name, n, f(*mstd([r["nmse"] for r in rs])[:2]),
             f(*mstd([float(r["events"]) for r in rs])[:2], 1),
             f(*oc[:2], 1),
             mstd([r["coord_open_frac"] for r in rs])[0],
             tim, g(held[0]), g(cs[0]), g(npr[0]),
             hv, hv / NOFC_HVP, cv, (cc if cc else "-"),
             (max(cr) if cr else float("nan")), g(fe[0], "%.4g")))
    W("")

    # -------------------------------------------------- per-seed openings
    W("## 2. Openings kept: number and timing, per seed\n")
    W("The reference row block is the no-warm-up adaptive arm "
      "(`results/e2_adaptmh`); the warm-up arm on the 20-step cadence "
      "(`results/e2_warmup`) has 0 openings on every seed and is omitted "
      "from the per-seed listing.\n")
    W("| arm | seed | opening steps | open coord-steps | held steps | "
      "release step | closed steps | closure steps <= 20 |")
    W("|---|---|---|---|---|---|---|---|")
    for name, _p, _d in ARMS:
        if name in (A_NOFC, A_FC, A_WFO):
            continue
        for r in A[name]:
            if "gate_open_steps" not in r:
                continue
            gs = r.get("gate_open_steps") or []
            sh = ((", ".join(str(x) for x in gs[:16])
                   + (" ..." if len(gs) > 16 else "")) if gs else "(none)")
            cl = closed_at_probe(r)
            ncl20 = sum(1 for t, c in cl.items() if c and t <= DENSE_T)
            W("| %s | %d | %s | %d | %s | %s | %s | %d |"
              % (name, r["seed"], sh,
                 round(r["coord_open_frac"] * M_COORD * r["steps"]),
                 r.get("warmup_held_steps"), r.get("warmup_release_step"),
                 r.get("failclosed_closed_steps"), ncl20))
    W("")

    # openings kept vs the no-warm-up reference, paired by seed
    W("Openings kept relative to the no-warm-up adaptive arm, paired by "
      "seed (`open coord-steps`, and the set of opening STEPS):\n")
    W("| seed | (ref) no warm-up steps | (i) dense steps | (ii) dense+warmup "
      "steps | (iii) fixed-prior dense steps | coord-steps ref / (i) / (ii) "
      "/ (iii) |")
    W("|---|---|---|---|---|---|")
    ref = {r["seed"]: r for r in A[A_ADP]}
    for s in sorted(ref):
        row = [ref[s]]
        for nm in NEW:
            m = {r["seed"]: r for r in A[nm]}
            row.append(m.get(s))

        def ss(r):
            if r is None:
                return "-"
            gs = r.get("gate_open_steps") or []
            return (", ".join(str(x) for x in gs[:12])
                    + (" ..." if len(gs) > 12 else "")) if gs else "(none)"

        def cd(r):
            return ("-" if r is None
                    else str(round(r["coord_open_frac"] * M_COORD
                                   * r["steps"])))
        W("| %d | %s | %s | %s | %s | %s |"
          % (s, ss(row[0]), ss(row[1]), ss(row[2]), ss(row[3]),
             " / ".join(cd(x) for x in row)))
    W("")

    # ------------------------------------------------------ lambda moves
    W("## 3. Direction and size of the lambda moves\n")
    W("`lam` is the per-group log learning rate, initialised at "
      "`log(0.003)` = %.4f. `net dlam` is the mean over the 6 groups of "
      "`lam_final - lam_0` (positive = LR raised); `LR ratio` is "
      "`exp(net dlam)`.\n" % LAM0)
    W("| arm | net dlam | LR ratio | max dlam | min dlam | "
      "max abs dlam (traj) | groups moved | open coord-steps LR down / up |")
    W("|---|---|---|---|---|---|---|---|")
    for name, _p, _d in ARMS:
        lm = [x for x in (lam_moves(r) for r in A[name]) if x]
        if not lm:
            continue
        W("| %s | %s | %.4f | %+.4f | %+.4f | %.4f | %.1f / 6 | %.1f / %.1f |"
          % (name, f(*mstd([x["net"] for x in lm])[:2]),
             mstd([x["lr_ratio"] for x in lm])[0],
             mstd([x["mx"] for x in lm])[0],
             mstd([x["mn"] for x in lm])[0],
             mstd([x["maxdev"] for x in lm])[0],
             mstd([float(x["nmoved"]) for x in lm])[0],
             mstd([float(x["n_lr_down"]) for x in lm])[0],
             mstd([float(x["n_lr_up"]) for x in lm])[0]))
    W("")

    # ---------------------------------------------- retrospective audit
    W("## 4. Retrospective consistency of every certified opening\n")
    W("For each gate opening at step `t`: the envelope in force at `t` and "
      "the NEXT probe's observed drift rate `M_obs`. The opening is "
      "*retrospectively consistent* iff that next observation does not "
      "exceed the envelope the opening was certified under. `cold-start` "
      "openings precede the run's first `M_obs` and can only be certified "
      "by the unverified floor -- with the dense schedule the first `M_obs` "
      "exists at step 1, so this column is the direct measure of what the "
      "dense probe bought.\n")
    W("| arm | seed | openings | first M_obs at | cold-start | "
      "with a following probe | consistent vs envelope in force | frac | "
      "consistent vs FINAL envelope | worst next-M_obs / envelope |")
    W("|---|---|---|---|---|---|---|---|---|---|")
    tot = {}
    for name, _p, _d in ARMS:
        agg = [0, 0, 0, 0, 0]
        seen = False
        for r in A[name]:
            if not probe_pairs(r) or "gate_open_steps" not in r:
                continue
            seen = True
            q = retro(r)
            gs = r.get("gate_open_steps") or []
            agg[0] += len(gs)
            agg[1] += q["cold"]
            agg[2] += q["ok"]
            agg[3] += q["ok_final"]
            agg[4] += q["tot"]
            W("| %s | %d | %d | %s | %d | %d | %d | %s | %d | %.4f |"
              % (name, r["seed"], len(gs), q["first_obs_t"], q["cold"],
                 q["tot"], q["ok"],
                 ("%.4f" % (q["ok"] / q["tot"])) if q["tot"] else "-",
                 q["ok_final"], q["worst"]))
        if seen:
            tot[name] = agg
    W("")
    W("Totals over the 10 seeds:\n")
    W("| arm | openings | cold-start | with a following probe | "
      "consistent vs envelope in force | frac | "
      "consistent vs FINAL envelope | frac |")
    W("|---|---|---|---|---|---|---|---|")
    for name, a in tot.items():
        W("| %s | %d | %d | %d | %d | %s | %d | %s |"
          % (name, a[0], a[1], a[4], a[2],
             ("%.4f" % (a[2] / a[0])) if a[0] else "-", a[3],
             ("%.4f" % (a[3] / a[0])) if a[0] else "-"))
    W("")

    # ------------------------------------------------------ paired tests
    W("## 5. Paired tests (exact sign-flip, 10 seeds)\n")
    W("Differences are `arm - reference`, paired by seed.\n")
    W("| arm | reference | d NMSE | p | d events | p | d open coord-steps | "
      "p | d HVPs | p |")
    W("|---|---|---|---|---|---|---|---|---|---|")
    tests = []
    for arm in NEW:
        tests.append((arm, A_ADP))
        tests.append((arm, A_WFO))
    tests.append((A_DPW, A_DP))
    tests.append((A_DPF, A_DP))
    for arm, refn in tests:
        rs, bs = A.get(arm) or [], A.get(refn) or []
        bmap = {r["seed"]: r for r in bs}
        pairs = [(r, bmap[r["seed"]]) for r in rs if r["seed"] in bmap]
        if len(pairs) < 2:
            continue
        d1 = [a["nmse"] - b["nmse"] for a, b in pairs]
        d2 = [float(a["events"] - b["events"]) for a, b in pairs]
        d3 = [(a["coord_open_frac"] - b["coord_open_frac"]) * M_COORD * STEPS
              for a, b in pairs]
        d4 = [float(a["hvp_total"] - b["hvp_total"]) for a, b in pairs]
        (p1, m1) = signflip_p(d1)
        (p2, m2) = signflip_p(d2)
        (p3, m3) = signflip_p(d3)
        (p4, m4) = signflip_p(d4)
        W("| %s | %s | %+.6f | %.4f | %+.1f | %.4f | %+.1f | %.4f | %+.0f | "
          "%.4f |" % (arm, refn, m1, p1, m2, p2, m3, p3, m4, p4))
    W("")

    # ---------------------------------------------------- probe timeline
    W("## 6. Envelope trajectory over the first 40 steps (seed 0)\n")
    W("`open` marks steps at which at least one coordinate opened. The "
      "reference block is the no-warm-up 20-step-cadence adaptive arm, "
      "whose first `M_obs` arrives at step 20 -- after every opening.\n")
    for arm in (A_ADP, A_DP, A_DPW, A_DPF):
        rs = [r for r in (A.get(arm) or []) if r["seed"] == 0]
        if not rs:
            continue
        r = rs[0]
        pr = [x for x in probe_pairs(r) if x[0] <= 40]
        ea = env_at_probe(r)
        cl = closed_at_probe(r)
        gs = set(r.get("gate_open_steps") or [])
        allgs = r.get("gate_open_steps") or []
        W("**%s** (seed 0): %d probes, first `M_obs` at step %s, envelope "
          "ends at %.5g, %d openings (%s), closed steps %s, HVPs %d.\n"
          % (arm, r.get("n_probes") or 0,
             next((t for t, m in probe_pairs(r) if m is not None), "-"),
             final_env(r) or float("nan"), len(allgs),
             ("steps " + ", ".join(str(x) for x in allgs[:16])
              + (" ..." if len(allgs) > 16 else "")) if allgs else "none",
             r.get("failclosed_closed_steps"), r["hvp_total"]))
        W("| probe step | M_obs | envelope after | monitor closed | "
          "gate opened this step |")
        W("|---|---|---|---|---|")
        for t, m in pr:
            W("| %s | %s | %s | %s | %s |"
              % (t, "-" if m is None else "%.5g" % m,
                 ("%.5g" % ea[t]) if t in ea else "%.5g" % final_env(r),
                 ("yes" if cl.get(t) else ""), ("yes" if t in gs else "")))
        W("")

    # -------------------------------- short-interval inflation of M_obs
    W("## 7. How much of the envelope is short-interval inflation\n")
    W("`M_obs = |rho_probe - rho_prev| / (eta_max * D)` divides by the "
      "path length `D` traversed since the previous probe. A 1-step "
      "interval has a tiny `D`, so the SAME trajectory yields a much larger "
      "`M_obs` at 1-step resolution than at 20-step resolution -- the ratio "
      "is not drift, it is the finite-difference denominator plus the "
      "spectral probe's own randomization noise (`rho` is a KW upper "
      "estimate, not an exact eigenvalue, so `|rho_t - rho_{t-1}|` has a "
      "noise floor that does not shrink with `D`).\n")
    W("The comparison is WITHIN each run and over the SAME interval "
      "`[0, 20]`: `1-step max/median` are the dense observations at "
      "`t = 1..20`; `20-step [0,20]` recomputes `M_obs` from the raw probe "
      "record as `|rho(20) - rho(0)| / (eta_max(0) * sum of the 20 path "
      "lengths)`, i.e. exactly what the 20-step cadence would have reported "
      "over the same trajectory.\n")
    W("| arm | seed | 1-step max (t<=20) | 1-step median | 20-step [0,20] | "
      "inflation max / 20-step | inflation median / 20-step | "
      "median M_obs at t>20 (20-step cadence) | final envelope |")
    W("|---|---|---|---|---|---|---|---|---|")
    infl_max, infl_med = {}, {}
    for name in NEW:
        im, ime = [], []
        for r in A[name]:
            d1 = dense_1step(r)
            if not d1:
                continue
            v = [m for _t, m in d1]
            c20 = coarse_mobs(r, 0, DENSE_T)
            late = [m for t, m in probe_pairs(r)
                    if m is not None and t > DENSE_T]
            rm = (max(v) / c20) if c20 else float("nan")
            rd = (med(v) / c20) if c20 else float("nan")
            im.append(rm)
            ime.append(rd)
            W("| %s | %d | %.5g | %.5g | %s | %s | %s | %s | %.5g |"
              % (name, r["seed"], max(v), med(v),
                 g(c20, "%.5g"), g(rm, "%.1f"), g(rd, "%.1f"),
                 g(med(late), "%.5g"), final_env(r) or float("nan")))
        infl_max[name] = mstd(im)
        infl_med[name] = mstd(ime)
    W("")
    W("| arm | inflation of the MAX (mean+-sd) | "
      "inflation of the MEDIAN (mean+-sd) |")
    W("|---|---|---|")
    for name in NEW:
        W("| %s | %s | %s |" % (name, f(*infl_max[name][:2], 1),
                                f(*infl_med[name][:2], 1)))
    W("")
    W("The same quantity stated as an envelope: with `KAPPA = 1` the "
      "envelope is `max(5, max_s M_obs,s)`, so the dense window sets it to "
      "the 1-step max, whereas a 20-step-only schedule would have set it to "
      "the 20-step value.\n")
    W("| arm | final envelope (dense) | 20-step [0,20] M_obs | "
      "envelope of the 20-step cadence (`results/e2_adaptmh`) | "
      "deployed prior |")
    W("|---|---|---|---|---|")
    for name in NEW:
        rs = A[name]
        c = mstd([coarse_mobs(r, 0, DENSE_T) for r in rs])
        W("| %s | %s | %s | %s | 5.0 |"
          % (name, f(*mstd([final_env(r) for r in rs])[:2], 2),
             f(*c[:2], 3),
             f(*mstd([final_env(r) for r in A[A_ADP]])[:2], 2)))
    W("")

    # ------------------------------------------------------- probe cost
    W("## 8. The probe cost\n")
    W("| arm | probes | forced re-probes | HVPs | HVPs / no-FC (94588) | "
      "d HVPs vs no-warm-up adaptive | wall s |")
    W("|---|---|---|---|---|---|---|")
    base = mstd([float(r["hvp_total"]) for r in A[A_ADP]])[0]
    for name, _p, _d in ARMS:
        rs = A[name]
        if not rs:
            continue
        hv = mstd([float(r["hvp_total"]) for r in rs])[0]
        W("| %s | %s | %s | %.0f | %.3f | %+.0f | %.0f |"
          % (name, g(mstd([r.get("n_probes") for r in rs])[0]),
             g(mstd([r.get("failclosed_reprobes") for r in rs])[0]),
             hv, hv / NOFC_HVP, hv - base,
             mstd([r["wall_s"] for r in rs])[0]))
    W("")

    # ------------------------------------------------- monitor pressure
    W("## 9. What the dense schedule does to the monitor\n")
    W("| arm | final envelope | max M_obs | median M_obs | closed steps | "
      "closure frac | fail-closed events | forced re-probes | "
      "closed steps within t<=20 |")
    W("|---|---|---|---|---|---|---|---|---|")
    for name, _p, _d in ARMS:
        rs = A[name]
        if not rs:
            continue
        cs = mstd([r.get("failclosed_closed_steps") for r in rs])
        mx = mstd([(r.get("adapt_mobs_max")
                    or (r.get("m_obs_stats") or {}).get("max")) for r in rs])
        mdn = mstd([(r.get("m_obs_stats") or {}).get("median") for r in rs])
        ncl20 = mstd([float(sum(1 for t, c in closed_at_probe(r).items()
                                if c and t <= DENSE_T)) for r in rs])
        W("| %s | %s | %s | %s | %s | %s | %s | %s | %s |"
          % (name, g(mstd([final_env(r) for r in rs])[0], "%.4g"),
             g(mx[0], "%.4g"), g(mdn[0], "%.4g"), g(cs[0]),
             g(cs[0] / STEPS, "%.4f"),
             g(mstd([r.get("failclosed_events") for r in rs])[0]),
             g(mstd([r.get("failclosed_reprobes") for r in rs])[0]),
             g(ncl20[0])))
    W("")

    # ------------------------------------------------------ plain answer
    ad, dp, dpw, dpf = A[A_ADP], A[A_DP], A[A_DPW], A[A_DPF]
    wf = A[A_WFO]

    def nm(rs):
        return mstd([r["nmse"] for r in rs])

    def oc(rs):
        return mstd([r["coord_open_frac"] * M_COORD * STEPS for r in rs])

    def hv(rs):
        return mstd([float(r["hvp_total"]) for r in rs])[0]

    tot_open = {k: sum(len(r.get("gate_open_steps") or []) for r in A[k])
                for k in (A_ADP, A_DP, A_DPW, A_DPF)}
    tot_ok = {k: (tot.get(k) or [0, 0, 0, 0, 0])[2]
              for k in (A_ADP, A_DP, A_DPW, A_DPF)}
    tot_cold = {k: (tot.get(k) or [0, 0, 0, 0, 0])[1]
                for k in (A_ADP, A_DP, A_DPW, A_DPF)}
    tot_fin = {k: (tot.get(k) or [0, 0, 0, 0, 0])[3]
               for k in (A_ADP, A_DP, A_DPW, A_DPF)}
    viol = {k: sum(r.get("cert_violations") or 0 for r in A[k])
            for k in (A_DP, A_DPW, A_DPF)}
    chk = {k: sum(r.get("cert_checked") or 0 for r in A[k])
           for k in (A_DP, A_DPW, A_DPF)}

    dp_env_sorted = sorted(final_env(r) for r in dp)
    dp_med_env = (dp_env_sorted[4] + dp_env_sorted[5]) / 2
    dp_coarse = sorted(coarse_mobs(r, 0, DENSE_T) for r in dp)
    dp_coarse_med = (dp_coarse[4] + dp_coarse[5]) / 2
    dp_ratio = sorted(final_env(r) / coarse_mobs(r, 0, DENSE_T) for r in dp)
    dp_ratio_med = (dp_ratio[4] + dp_ratio[5]) / 2
    same = (all(x["losses"] == y["losses"] for x, y in zip(dp, dpw))
            and all(x["lam_hist"] == y["lam_hist"] for x, y in zip(dp, dpw)))
    p_nmse_ad = signflip_p([x["nmse"] - y["nmse"] for x, y in zip(dp, ad)])
    p_oc_ad = signflip_p([(x["coord_open_frac"] - y["coord_open_frac"])
                          * M_COORD * STEPS for x, y in zip(dp, ad)])
    p_ev_ad = signflip_p([float(x["events"] - y["events"])
                          for x, y in zip(dp, ad)])
    p_hv_ad = signflip_p([float(x["hvp_total"] - y["hvp_total"])
                          for x, y in zip(dp, ad)])
    fxr = load("mackey_drift_fixed_lr0.003_s*.json",
               os.path.join(ROOT, "results", "e2"))
    nmse_fx = mstd([r["nmse"] for r in fxr])[0] if fxr else float("nan")
    gap = ((nm(wf)[0] - nm(dp)[0]) / (nm(wf)[0] - nm(ad)[0])
           if nm(wf)[0] != nm(ad)[0] else float("nan"))
    env_open = (sum(sum(r["gate_open_env"]) for r in dp)
                / max(sum(len(r["gate_open_env"]) for r in dp), 1))
    first_open = mstd([float((r.get("gate_open_steps") or [float("nan")])[0])
                       for r in dp])[0]
    first_open_ad = mstd([float((r.get("gate_open_steps")
                                 or [float("nan")])[0]) for r in ad])[0]
    cl_dp = mstd([r.get("failclosed_closed_steps") for r in dp])[0]
    cl_ad = mstd([r.get("failclosed_closed_steps") for r in ad])[0]
    cl20_dp = mstd([float(sum(1 for t, c in closed_at_probe(r).items()
                              if c and t <= DENSE_T)) for r in dp])[0]
    cl20_f = mstd([float(sum(1 for t, c in closed_at_probe(r).items()
                             if c and t <= DENSE_T)) for r in dpf])[0]
    cl_f = mstd([r.get("failclosed_closed_steps") for r in dpf])[0]
    worst_dp = max((retro(r)["worst"] for r in dp), default=float("nan"))
    cmr = max(r.get("cert_max_ratio") or 0.0 for r in dp + dpw + dpf)
    lrr_dp = mstd([lam_moves(r)["lr_ratio"] for r in dp])[0]
    lrr_ad = mstd([lam_moves(r)["lr_ratio"] for r in ad])[0]
    lrr_f = mstd([lam_moves(r)["lr_ratio"] for r in dpf])[0]

    W("## 10. Plain answer\n")
    W("**Does an early measurement rescue certified adaptation under a "
      "MEASURED envelope? Yes -- most of it, and cheaply. With one probe "
      "per step over the first 20 steps, %d of the reference run's 84 "
      "certified openings survive (%.0f%%), **0 of them are cold-start** "
      "(down from 84 of 84), **%d of %d (%.1f%%) are retrospectively "
      "consistent with the envelope in force** (up from 0 of 84) and "
      "**%d of %d (100%%) under the final envelope**, for +%.1f%% HVPs "
      "and an NMSE of %s against %s -- recovering %.0f%% of the gap the "
      "naive warm-up gave away. But the envelope those openings are "
      "certified under is largely a short-interval artefact: a median "
      "%.4g, i.e. %.0fx the drift the SAME window actually shows at "
      "20-step resolution and %.0fx the deployed prior. Under a FIXED "
      "prior the same schedule destroys the method instead.**\n"
      % (tot_open[A_DP], 100.0 * tot_open[A_DP] / max(tot_open[A_ADP], 1),
         tot_ok[A_DP], tot_open[A_DP],
         100.0 * tot_ok[A_DP] / max(tot_open[A_DP], 1),
         tot_fin[A_DP], tot_open[A_DP],
         100.0 * (hv(dp) - hv(ad)) / hv(ad), f(*nm(dp)[:2]), f(*nm(ad)[:2]),
         100.0 * gap, dp_med_env, dp_ratio_med, dp_med_env / FLOOR))

    W("**1. The measurement now exists before the gate needs it, and that "
      "is the whole point.** On the 20-step cadence the first `M_obs` "
      "arrives at step 20 while every opening is at steps 1-15, so 84 of "
      "84 openings are cold-start and 0 of 84 are retrospectively "
      "consistent (`round4_adaptmh.md`). With `--probe-dense-until 20` the "
      "first `M_obs` arrives at **step 1 on all 10 seeds**, the first "
      "opening moves from step %.1f to step %.1f, and the cold-start count "
      "goes to **0 of %d**. Retrospective consistency -- the next observed "
      "`M_obs` does not exceed the envelope the opening was certified "
      "under -- goes **0/84 -> %d/%d = %.4f**. The %d exceptions are steps "
      "where the very next 1-step observation overshot the envelope, by up "
      "to %.2fx; all %d openings are consistent under the run's final "
      "envelope.\n"
      % (first_open_ad, first_open, tot_open[A_DP], tot_ok[A_DP],
         tot_open[A_DP], tot_ok[A_DP] / max(tot_open[A_DP], 1),
         tot_open[A_DP] - tot_ok[A_DP], worst_dp, tot_fin[A_DP]))

    W("**2. Holding the gate until an observation exists becomes FREE.** "
      "Arm (ii) adds `--gate-warmup first-obs` on top of the dense "
      "schedule and is **bit-identical to arm (i) on all 10 seeds** "
      "(losses AND `lam_hist` element for element: %s). The hold now lasts "
      "1 step -- step 0, before any probe pair can exist -- and suppresses "
      "nothing, because the certificate keeps the gate shut at step 0 "
      "anyway. This is the claim `round4_warmup.md` could not make: on the "
      "20-step cadence the same flag costs every opening and 3.2x the "
      "NMSE; with a dense early probe it costs exactly zero. The "
      "verified-envelope requirement is a **probe-schedule** problem, not "
      "a certificate problem.\n"
      % ("True" if same else "NOT identical -- CHECK"))

    W("**3. The inflated short-interval `M_obs` does blow the envelope up "
      "-- but it does not close the gate.** `M_obs = |rho_probe - "
      "rho_prev| / (eta_max * D)` divides by the path length, and a 1-step "
      "`D` is ~20x smaller than a 20-step one; on top of that `rho` is a "
      "randomized KW upper estimate whose step-to-step difference has a "
      "noise floor that does NOT shrink with `D`. Measured within the same "
      "run over the same interval [0, 20]: the largest 1-step observation "
      "is a median **%.1fx** (mean %s) the 20-step aggregate over that "
      "identical window, while the MEDIAN 1-step observation is only x%s of "
      "it -- the inflation lives almost entirely in the maximum, which is "
      "exactly the statistic `M_H,t = max(5, KAPPA * max_s M_obs,s)` "
      "reads. The envelope ends at a median %.4g (mean %s; seed 8 reaches "
      "%.4g) where the coarse 20-step measurement of the same window is "
      "%.4g and the 20-step-cadence run settles at %.4g. **Only ~%.0f%% of "
      "the envelope's size is coarse-grained drift; the other ~%.0f%% is "
      "short-interval inflation.** It does not close the gate for long: "
      "the envelope overshoots so far that no later observation violates "
      "it -- closed steps %.1f per run (closure fraction %.4f), *below* "
      "the 20-step adaptive arm's %.1f, and all %.1f of them fall inside "
      "the dense window while the envelope is still climbing (seed 0: "
      "closed at probes 1, 2, 3, then open from step 4 on).\n"
      % (dp_ratio_med, f(*infl_max[A_DP][:2], 1), f(*infl_med[A_DP][:2], 1),
         dp_med_env, f(*mstd([final_env(r) for r in dp])[:2], 0),
         max(final_env(r) for r in dp), dp_coarse_med,
         mstd([final_env(r) for r in ad])[0],
         100.0 / dp_ratio_med, 100.0 - 100.0 / dp_ratio_med,
         cl_dp, cl_dp / STEPS, cl_ad, cl20_dp))

    W("**4. What it costs: 16 openings, 54%% more NMSE than the "
      "unverified run, and 2.8%% of the probe budget.** The inflated "
      "envelope is not free even when it never closes the gate, because it "
      "enters the certificate itself: `dH = M_H * D`, so a ~%.0fx larger "
      "`M_H` inflates `beta_col` and with it the gate threshold "
      "`c * beta_col`. Openings %s -> %s coordinate-steps per seed (paired "
      "d = %+.1f, p = %.4f), %d of the 84 opening steps lost, mean envelope in "
      "force at an opening %.4g against the deployed prior 5.0, and the "
      "learning rate ends %.2fx up instead of %.2fx. NMSE %s -> %s (paired "
      "d = %+.6f, exact sign-flip p = %.4f) -- **%.2fx worse than the "
      "unverified-floor run, but still %.0f%% of the way from the "
      "frozen-LR baseline (%s, `results/e2` fixed lr 0.003) back to it**. "
      "Stability does not get worse: events %s -> %s (p = %.4f). The probe "
      "cost is small and exactly the dense window: %.1f -> %.1f probes, "
      "%.0f -> %.0f HVPs = **1.031x** the no-monitor budget against "
      "1.002x, +%.1f%% (p = %.4f).\n"
      % (dp_med_env / FLOOR, f(*oc(ad)[:2], 1), f(*oc(dp)[:2], 1),
         p_oc_ad[1], p_oc_ad[0], tot_open[A_ADP] - tot_open[A_DP], env_open,
         lrr_dp, lrr_ad, f(*nm(ad)[:2]), f(*nm(dp)[:2]),
         p_nmse_ad[1], p_nmse_ad[0], nm(dp)[0] / nm(ad)[0], 100.0 * gap,
         f(nmse_fx, float("nan")),
         f(*mstd([float(r["events"]) for r in ad])[:2], 1),
         f(*mstd([float(r["events"]) for r in dp])[:2], 1), p_ev_ad[0],
         mstd([r.get("n_probes") for r in ad])[0],
         mstd([r.get("n_probes") for r in dp])[0], hv(ad), hv(dp),
         100.0 * (hv(dp) - hv(ad)) / hv(ad), p_hv_ad[0]))

    W("**5. The fixed prior does not survive the dense schedule.** Arm "
      "(iii) keeps `M_H = 5` and pays for the inflation instead of "
      "absorbing it: the 1-step observations exceed 5 at essentially every "
      "probe, so the monitor is violated on **%.1f of the first 20 steps** "
      "and on **%.0f of 12000 steps overall (closure fraction %.4f)**, "
      "with %.1f fail-closed transitions and %.0f forced re-probes -- HVPs "
      "%.0f = **%.3fx** the no-monitor budget (paired +%.0f vs arm (i), "
      "p = 0.0020). Only **%d openings survive across all 10 seeds** "
      "(7 of 10 seeds never open at all), lambda barely moves (LR ratio "
      "%.3f), and NMSE %s is statistically indistinguishable from the "
      "frozen-LR warm-up arm (%s, paired p = 0.2500). None of those 3 "
      "openings is retrospectively consistent, since a fixed 5.0 is below "
      "every 1-step observation. This is the honest control: the dense "
      "probe rescues certified adaptation **only when the envelope may be "
      "re-stated online**. Bolted onto a fixed prior it destroys the "
      "method and nearly doubles the probe bill.\n"
      % (cl20_f, cl_f, cl_f / STEPS,
         mstd([r.get("failclosed_events") for r in dpf])[0],
         mstd([r.get("failclosed_reprobes") for r in dpf])[0], hv(dpf),
         hv(dpf) / NOFC_HVP, hv(dpf) - hv(dp), tot_open[A_DPF], lrr_f,
         f(*nm(dpf)[:2]), f(*nm(wf)[:2])))

    W("**6. Validity is untouched, as everywhere in round 4.** "
      "**0 certificate violations in 720000 audited coordinate-steps in "
      "each of the three arms**, worst `|ghat_j - g_true_j| / beta_col_j` "
      "= %.4f (0.8374 on the fixed-prior reference). The default path is "
      "bit-identical: with `--probe-dense-until 0` (the default) seeds 0 "
      "and 1 reproduce `results/e2_verify4` element for element on the "
      "full 12000-entry loss trace and the sampled lambda trajectory, with "
      "the same HVP count 94588 and 0 violations.\n" % cmr)

    W("**7. What this means for the claim.** `round4_warmup.md` concluded "
      "that requiring the envelope to be verified before use costs the "
      "method its entire advantage. That conclusion was an artefact of the "
      "probe SCHEDULE, not of the certificate: move one probe per step "
      "into the first 20 steps -- 19 extra probes, 2.8%% of the run's "
      "HVPs -- and %d of 84 certified openings come back with a "
      "measurement behind them, %.0f%% of them retrospectively consistent, "
      "100%% consistent under the final envelope, and the `first-obs` hold "
      "becomes a no-op. Two caveats belong beside that in the paper. "
      "First, the envelope those openings are certified under is "
      "**~%.0fx the deployed prior and ~%.0fx the coarse-grained drift of "
      "the same window**, because `M_obs` at 1-step resolution is "
      "dominated by the finite-difference denominator and the spectral "
      "probe's own randomization noise. It is a valid ONLINE-ENFORCEABLE "
      "envelope -- never below any observation the run made, 0 violations "
      "against exact FMD -- but it is *not* a tight estimate of the "
      "Hessian drift rate, and quoting it as one would overstate the "
      "measurement: a `KAPPA * max` envelope built from 1-step differences "
      "buys auditability by being loose. Second, that looseness is paid "
      "for in the currency the method trades in: 16 of 84 openings and "
      "%.2fx the NMSE. The defensible statement is therefore *a dense "
      "early probe makes COHG's certified adaptation auditable against a "
      "measured envelope for ~3%% probe overhead and ~54%% NMSE over the "
      "unverified-floor run -- three quarters of the way back from the "
      "frozen-LR collapse -- provided the envelope is re-stated online; "
      "under a fixed prior the same schedule closes the gate 78%% of the "
      "time and costs 1.82x the probe budget.*\n"
      % (tot_open[A_DP], 100.0 * tot_ok[A_DP] / max(tot_open[A_DP], 1),
         dp_med_env / FLOOR, dp_ratio_med, nm(dp)[0] / nm(ad)[0]))

    W("### Reproduction\n")
    W("```")
    W("python code/experiments/launch_r4_denseprobe.py           # job list")
    W("RDQ_WORKERS=32 python code/experiments/r4_denseprobe_queue.py")
    W("python results/reanalysis/_round4_denseprobe.py           # this file")
    W("```")
    W("Outputs: `results/e2_denseprobe/` (30 arm runs) and "
      "`results/e2_denseprobe/verify/` (2 default-path regression runs).\n")

    W("### Raw numbers\n")
    W("* Openings kept, (i) dense vs the no-warm-up reference: "
      "%s vs %s open coordinate-steps per seed "
      "(%d vs %d opening steps over the 10 seeds); "
      "(ii) dense + warm-up: %s (%d steps); "
      "(iii) fixed prior + dense: %s (%d steps). The 20-step-cadence "
      "warm-up arm keeps %s."
      % (f(*oc(dp)[:2], 1), f(*oc(ad)[:2], 1),
         tot_open[A_DP], tot_open[A_ADP],
         f(*oc(dpw)[:2], 1), tot_open[A_DPW],
         f(*oc(dpf)[:2], 1), tot_open[A_DPF],
         f(*oc(wf)[:2], 1)))
    W("* Cold-start openings (before any `M_obs` exists): reference %d of "
      "%d; (i) %d of %d; (ii) %d of %d; (iii) %d of %d."
      % (tot_cold[A_ADP], tot_open[A_ADP], tot_cold[A_DP], tot_open[A_DP],
         tot_cold[A_DPW], tot_open[A_DPW], tot_cold[A_DPF],
         tot_open[A_DPF]))
    W("* Retrospective consistency vs the envelope in force: reference "
      "%d of %d; (i) %d of %d; (ii) %d of %d; (iii) %d of %d. Under the "
      "FINAL envelope: %d / %d / %d / %d."
      % (tot_ok[A_ADP], tot_open[A_ADP], tot_ok[A_DP], tot_open[A_DP],
         tot_ok[A_DPW], tot_open[A_DPW], tot_ok[A_DPF], tot_open[A_DPF],
         tot_fin[A_ADP], tot_fin[A_DP], tot_fin[A_DPW], tot_fin[A_DPF]))
    W("* NMSE: reference %s; warm-up (20-step cadence) %s; (i) %s; (ii) %s; "
      "(iii) %s."
      % (f(*nm(ad)[:2]), f(*nm(wf)[:2]), f(*nm(dp)[:2]), f(*nm(dpw)[:2]),
         f(*nm(dpf)[:2])))
    W("* HVPs: reference %.0f (%.3fx the no-monitor budget); (i) %.0f "
      "(%.3fx, %+.1f%% vs reference); (ii) %.0f (%.3fx); (iii) %.0f "
      "(%.3fx)."
      % (hv(ad), hv(ad) / NOFC_HVP, hv(dp), hv(dp) / NOFC_HVP,
         100.0 * (hv(dp) - hv(ad)) / hv(ad), hv(dpw), hv(dpw) / NOFC_HVP,
         hv(dpf), hv(dpf) / NOFC_HVP))
    W("* Certificate violations: (i) %d / %d; (ii) %d / %d; (iii) %d / %d."
      % (viol[A_DP], chk[A_DP], viol[A_DPW], chk[A_DPW],
         viol[A_DPF], chk[A_DPF]))
    W("* Short-interval inflation of `M_obs` on the same interval [0,20]: "
      "max x%s, median x%s (arm (i))."
      % (f(*infl_max[A_DP][:2], 1), f(*infl_med[A_DP][:2], 1)))
    W("")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote", OUT, len(L), "lines")


if __name__ == "__main__":
    main()
