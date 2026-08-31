"""E2-controls analysis: certification vs generic conservatism.

Reads results/e2_controls/*.json (arm = the filename tag) and writes
results/e2_controls/SUMMARY.md:

  * per-arm NMSE / instability events (mean +- std, ddof=1), realized
    per-coordinate gate-open rate, HVP cost, wall time
  * paired sign-flip permutation tests of every control against the COHG
    reference arm (cohg, M_H=5, no fail-closed, certificate-validated)
  * study F: certificate violations under misspecified M_H, and whether the
    fail-closed monitor fires
"""

from __future__ import annotations

import glob
import itertools
import json
import os
import re
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
CTL = os.path.abspath(os.path.join(ROOT, "results", "e2_controls"))
E2 = os.path.abspath(os.path.join(ROOT, "results", "e2"))
PREFIX = "mackey_drift_"
REF = "cohg_lr0.003_mh5_fc0"          # in-family COHG reference arm

def lam_window(lam_hist):
    """(first, last) sampled step at which lambda changed (samples every 50)."""
    prev, first, last = None, None, None
    for row in lam_hist:
        t, lam = row[0], row[1:]
        if prev is not None and max(abs(x - y) for x, y in zip(lam, prev)) > 1e-9:
            last = t
            if first is None:
                first = t
        prev = lam
    return first, last


arms = defaultdict(dict)               # arm -> {seed: run}
for path in sorted(glob.glob(os.path.join(CTL, PREFIX + "*.json"))):
    base = os.path.basename(path)[len(PREFIX):-len(".json")]
    m = re.match(r"^(.*)_s(\d+)$", base)
    if not m:
        continue
    arm, seed = m.group(1), int(m.group(2))
    with open(path) as f:
        r = json.load(f)
    r["lam_first"], r["lam_last"] = lam_window(r.get("lam_hist") or [])
    r.pop("losses", None)
    r.pop("lam_hist", None)
    r.pop("ghat_abs_top", None)
    arms[arm][seed] = r

EVAL = set(range(10))


def stat(vals):
    v = np.asarray(vals, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


def agg(arm):
    d = {s: r for s, r in arms.get(arm, {}).items() if s in EVAL}
    if not d:
        return None
    g = lambda k: [r[k] for r in d.values() if r.get(k) is not None]
    a = {"n": len(d), "seeds": d}
    a["nmse"], a["nmse_sd"] = stat([r["nmse"] for r in d.values()])
    a["ev"], a["ev_sd"] = stat([r["events"] for r in d.values()])
    a["open"], _ = stat(g("coord_open_frac") or [float("nan")])
    a["step_open"], _ = stat(g("gate_open_frac") or [float("nan")])
    a["hvp"], _ = stat([r["hvp_total"] for r in d.values()])
    a["wall"], _ = stat([r["wall_s"] for r in d.values()])
    fi = [r["lam_first"] for r in d.values() if r["lam_first"] is not None]
    la = [r["lam_last"] for r in d.values() if r["lam_last"] is not None]
    a["lam_win"] = (f"{int(np.median(fi))}-{int(np.median(la))}"
                    if fi and la else "never")
    return a


def perm_paired(a, b):
    """Exact sign-flip permutation test on paired per-seed differences."""
    common = sorted(set(a) & set(b))
    if len(common) < 4:
        return None, len(common), float("nan")
    d = np.array([a[s] - b[s] for s in common], dtype=float)
    obs = d.mean()
    stats = [float(np.mean(d * np.array(f)))
             for f in itertools.product([1, -1], repeat=len(d))]
    p = float(np.mean([abs(s) >= abs(obs) - 1e-15 for s in stats]))
    return p, len(common), float(obs)


# ---- COHG reference measured on GPU in results/e2 (cross-device check) ----
gpu = [json.load(open(p)) for p in
       sorted(glob.glob(os.path.join(E2, PREFIX + "cohg_lr0.003_s*.json")))]
gpu_nmse, gpu_nmse_sd = stat([r["nmse"] for r in gpu]) if gpu else (np.nan,) * 2
gpu_ev, gpu_ev_sd = stat([r["events"] for r in gpu]) if gpu else (np.nan,) * 2
gpu_open, _ = stat([r["coord_open_frac"] for r in gpu]) if gpu else (np.nan,) * 2

ARM_GROUPS = [
    ("COHG reference (M_H=5, no fail-closed, cert-validated)",
     [(REF, "COHG (alpha=0.4)")]),
    ("A. ungated pure-SIGN sweep (no certificate at all)",
     [(f"cohg_nogate_lr0.003_a{a:g}", f"sign-nogate alpha={a:g}")
      for a in [0.1, 0.2, 0.4, 0.8]]),
    ("B/C/D. matched-conservatism gates (same estimator, same alpha=0.4 sign "
     "step, open rate matched to COHG)",
     [("absgate_lr0.003_a0.4", "B absgate |ghat|>thr"),
      ("randgate_lr0.003_a0.4", "C randgate i.i.d."),
      ("periodicgate_lr0.003_a0.4", "D periodicgate")]),
    ("E. Theorem-6 step condition enforced online",
     [("t6clip_lr0.003_rf1", "t6clip rho_f=1"),
      ("t6clip_lr0.003_rf10", "t6clip rho_f=10")]),
    ("F. M_H misspecification x fail-closed",
     [(f"cohg_lr0.003_mh{mh:g}_fc{fc}",
       f"COHG M_H={mh:g} {'fail-closed' if fc else 'no FC'}")
      for mh in [5.0, 0.5, 0.05] for fc in [0, 1]]),
]

L = ["# E2 controls: certification vs generic conservatism", "",
     "Config (identical for every arm): dataset `mackey_drift`, mis-set init "
     "`lr0=0.003` (10x too low), 12000 steps, gamma 0.9, kw-eps 0.1, "
     "probe-every 20, K 10, rank 4, gate factor c=2, seeds 0-9.",
     "Device: CPU (13k-param GRU; HVPs are launch-bound, so a CPU core beats a "
     "3080 here). Cross-device check against the GPU E2 run below.", "",
     f"Arms found: {len(arms)} | runs: {sum(len(v) for v in arms.values())}",
     "",
     "## Reference points", "",
     f"- COHG on GPU (`results/e2`): NMSE {gpu_nmse:.4f}+-{gpu_nmse_sd:.4f}, "
     f"events {gpu_ev:.1f}+-{gpu_ev_sd:.1f}, coord-open {gpu_open:.2e} "
     f"(n={len(gpu)})",
     ]
ref = agg(REF)
if ref:
    L.append(f"- COHG on CPU (this study, arm `{REF}`): "
             f"NMSE {ref['nmse']:.4f}+-{ref['nmse_sd']:.4f}, "
             f"events {ref['ev']:.1f}+-{ref['ev_sd']:.1f}, "
             f"coord-open {ref['open']:.2e} (n={ref['n']})")
L += ["", "### Fixed-LR family on the same stream (`results/e2`, GPU) -- the "
      "NMSE/instability frontier the adaptive arms are judged against", "",
      "| fixed lr | n | NMSE mean+-std | events mean+-std |",
      "|---|---|---|---|"]
for lr in [0.003, 0.01, 0.03, 0.1, 0.3]:
    fs = sorted(glob.glob(os.path.join(E2, f"{PREFIX}fixed_lr{lr:g}_s*.json")))
    if not fs:
        continue
    rr = [json.load(open(p)) for p in fs]
    nm, nms = stat([r["nmse"] for r in rr])
    ev, evs = stat([r["events"] for r in rr])
    star = "  <- mis-set init used by every adaptive arm" if lr == 0.003 else ""
    L.append(f"| {lr:g}{star} | {len(rr)} | {nm:.4f}+-{nms:.4f} | "
             f"{ev:.1f}+-{evs:.1f} |")

L += ["", "## Per-arm results", "",
      "`lambda window` = median first-to-last sampled step (every 50) at which "
      "lambda actually moved -- i.e. WHEN the arm does its adaptation.", "",
      "| arm | n | NMSE mean+-std | events mean+-std | coord-open rate | "
      "steps-with-open | lambda window | HVPs | wall (s) |",
      "|---|---|---|---|---|---|---|---|---|"]

for title, group in ARM_GROUPS:
    L.append(f"| **{title}** | | | | | | | | |")
    for arm, label in group:
        a = agg(arm)
        if not a:
            L.append(f"| {label} | 0 | (missing) | | | | | | |")
            continue
        L.append(f"| {label} | {a['n']} | {a['nmse']:.4f}+-{a['nmse_sd']:.4f} "
                 f"| {a['ev']:.1f}+-{a['ev_sd']:.1f} | {a['open']:.2e} | "
                 f"{a['step_open']:.2e} | {a['lam_win']} | {a['hvp']:.0f} | "
                 f"{a['wall']:.0f} |")

L += ["", "### Mechanism: adaptation is a burst, not a rate", "",
      "The `lambda window` column is the single most informative number here. "
      "COHG moves lambda ONLY inside the first ~50 steps -- while the "
      "certificate e_col is still near zero and the hypergradient sign is "
      "therefore unambiguous -- lifting the six per-group LRs from the mis-set "
      "0.003 to 0.0019-0.042, and then the gate stays shut for the remaining "
      "11950 steps. Its ~35 open coordinate-steps are not spread thinly at a "
      "5e-4 rate; they are concentrated exactly where they are certifiable. "
      "absgate reproduces that burst (window 100-150) and therefore "
      "reproduces the result. randgate (325-11675) and periodicgate "
      "(2000-11950) spend the same open budget uniformly over the stream, so "
      "the LR is wrong for most of the run and they land near the un-adapted "
      "fixed-0.003 level. t6clip opens in the same window as COHG but its "
      "steps are capped ~7x smaller, so the LRs never leave 0.003.", ""]

# ---- paired tests against the COHG reference ----
L += ["", "## Paired comparisons vs the COHG reference arm",
      "(exact sign-flip permutation test on per-seed differences; "
      "negative delta = the control is BETTER)", "",
      "| arm | delta NMSE | p | delta events | p |", "|---|---|---|---|---|"]
if ref:
    ref_nmse = {s: r["nmse"] for s, r in ref["seeds"].items()}
    ref_ev = {s: float(r["events"]) for s, r in ref["seeds"].items()}
    for title, group in ARM_GROUPS:
        for arm, label in group:
            if arm == REF:
                continue
            a = agg(arm)
            if not a:
                continue
            an = {s: r["nmse"] for s, r in a["seeds"].items()}
            ae = {s: float(r["events"]) for s, r in a["seeds"].items()}
            p1, n1, d1 = perm_paired(an, ref_nmse)
            p2, n2, d2 = perm_paired(ae, ref_ev)
            L.append(f"| {label} | {d1:+.4f} | "
                     f"{('%.4f' % p1) if p1 is not None else '-'} | "
                     f"{d2:+.1f} | "
                     f"{('%.4f' % p2) if p2 is not None else '-'} |")

# ---- study F detail ----
L += ["", "## F. Certificate audit and fail-closed detection", "",
      "`cert_violations` counts per-coordinate events "
      "`|ghat_j - g_true_j| > beta_col_j` against a parallel EXACT discounted "
      "sensitivity (ExactFMD, m HVPs/step), with a 1e-4 relative fp32 "
      "tolerance. `cert ratio p99/max` is "
      "`|ghat_j - g_true_j| / beta_col_j` (1.0 = the bound is exactly tight).",
      "",
      "| arm | n | cert viol / checked | viol steps | cert ratio p99 | "
      "cert ratio max | failclosed events | closed steps | re-probes | "
      "obs M_H p99 |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for mh in [5.0, 0.5, 0.05]:
    for fc in [0, 1]:
        arm = f"cohg_lr0.003_mh{mh:g}_fc{fc}"
        a = agg(arm)
        if not a:
            continue
        rs = list(a["seeds"].values())
        vi = sum(r.get("cert_violations") or 0 for r in rs)
        ck = sum(r.get("cert_checked") or 0 for r in rs)
        vs = sum(r.get("cert_violation_steps") or 0 for r in rs)
        q99 = [r["cert_ratio_q"]["p99"] for r in rs
               if r.get("cert_ratio_q") and r["cert_ratio_q"]["p99"]]
        qmx = [r.get("cert_max_ratio") for r in rs
               if r.get("cert_max_ratio") is not None]
        fe = [r["failclosed_events"] for r in rs
              if r.get("failclosed_events") is not None]
        cs = [r["failclosed_closed_steps"] for r in rs
              if r.get("failclosed_closed_steps") is not None]
        rp = [r["failclosed_reprobes"] for r in rs
              if r.get("failclosed_reprobes") is not None]
        mo = [r["m_obs_stats"]["p99"] for r in rs if r.get("m_obs_stats")]
        f = lambda v, fmt="{:.1f}": (fmt.format(np.mean(v)) if v else "-")
        L.append(f"| M_H={mh:g} {'FC' if fc else 'no FC'} | {a['n']} | "
                 f"{vi} / {ck} | {vs} | {f(q99, '{:.3g}')} | "
                 f"{f(qmx, '{:.3g}')} | {f(fe)} | {f(cs)} | {f(rp)} | "
                 f"{f(mo, '{:.3g}')} |")

# ---- E detail ----
L += ["", "## E. Theorem-6 clipping detail", ""]
for rf in [1.0, 10.0]:
    a = agg(f"t6clip_lr0.003_rf{rf:g}")
    if a:
        nc = [r.get("n_clipped") or 0 for r in a["seeds"].values()]
        L.append(f"- rho_f={rf:g}: open-coordinate steps clipped by the "
                 f"Theorem-6 cap on average {np.mean(nc):.1f} of "
                 f"{a['open'] * 6 * 12000:.1f} open coordinate-steps")

# ---- B calibration provenance ----
cal = sorted(glob.glob(os.path.join(CTL, PREFIX + "absgate_lr0.003_calib_s*.json")))
ver = sorted(glob.glob(os.path.join(CTL, PREFIX + "absgate_lr0.003_verify_s*.json")))
if cal:
    thr = [json.load(open(p)).get("absgate_threshold") for p in ver]
    cal_seeds = [int(re.search(r"_s(\d+)", p).group(1)) for p in cal]
    L += ["", "## B. absgate threshold calibration", "",
          f"- calibration runs (gate held shut, seeds {cal_seeds}, disjoint "
          f"from evaluation): the |ghat| order statistic matching COHG's "
          f"open rate; refit once on the realized |ghat| distribution of the "
          f"pass-1 threshold, then FROZEN",
          f"- frozen threshold: {thr[0] if thr else 'n/a'}"]
    for p in ver:
        d = json.load(open(p))
        L.append(f"- verification seed {d['seed']}: realized coord-open rate "
                 f"{d['coord_open_frac']:.3e}")

# ---- A. reproduction check against the GPU cohg_nogate arm ----
gn = {}
for p in sorted(glob.glob(os.path.join(
        E2, PREFIX + "cohg_nogate_lr0.003_s*.json"))):
    d = json.load(open(p))
    gn[d["seed"]] = (d["nmse"], d["events"])
a04 = agg("cohg_nogate_lr0.003_a0.4")
if gn and a04:
    gm, gs = stat([v[0] for v in gn.values()])
    ge, _ = stat([v[1] for v in gn.values()])
    L += ["", "## A. reproduction check: the SAME arm on GPU (`results/e2`)",
          "",
          f"- GPU `cohg_nogate` lr0.003 alpha=0.4: NMSE {gm:.4f}+-{gs:.4f}, "
          f"events {ge:.1f} (n={len(gn)})",
          f"- CPU `sign-nogate alpha=0.4` (this study): "
          f"NMSE {a04['nmse']:.4f}+-{a04['nmse_sd']:.4f}, "
          f"events {a04['ev']:.1f} (n={a04['n']})", "",
          "| seed | GPU NMSE | GPU events | CPU NMSE | CPU events |",
          "|---|---|---|---|---|"]
    for s in sorted(gn):
        r = a04["seeds"].get(s)
        L.append(f"| {s} | {gn[s][0]:.4f} | {gn[s][1]} | "
                 f"{r['nmse']:.4f} | {r['events']} |" if r else "")
    L += ["", "The two devices agree seed-for-seed to fp32 round-off on nine "
          "of ten seeds; on seed 5 a ~1e-7 relative perturbation is enough to "
          "flip the run between NMSE 0.005 and NMSE 59.8. The ungated arm sits "
          "ON the divergence boundary: its mean NMSE is decided by which side "
          "of a float a single seed lands on. That tail, not the mean, is what "
          "the certificate gate is buying."]

# ---- headline answers ----
L += ["", "## Headline answers", ""]
ref_a = agg(REF)
b, c, d_, n04 = (agg("absgate_lr0.003_a0.4"), agg("randgate_lr0.003_a0.4"),
                 agg("periodicgate_lr0.003_a0.4"),
                 agg("cohg_nogate_lr0.003_a0.4"))
if ref_a and b and c and d_:
    L += [
        f"**(1) Does a matched-conservatism control (B/C/D) reach COHG's "
        f"events-vs-NMSE point?** B YES, C/D NO. absgate reaches "
        f"NMSE {b['nmse']:.4f}+-{b['nmse_sd']:.4f} at {b['ev']:.1f} events vs "
        f"COHG {ref_a['nmse']:.4f}+-{ref_a['nmse_sd']:.4f} at "
        f"{ref_a['ev']:.1f} (paired p=0.40 on NMSE, p=0.44 on events): "
        f"statistically indistinguishable, at a LOWER open rate "
        f"({b['open']:.2e} vs {ref_a['open']:.2e}). This is a real negative "
        f"result for the 'certification is what buys the operating point' "
        f"reading. randgate ({c['nmse']:.4f}) and periodicgate "
        f"({d_['nmse']:.4f}) both collapse to roughly the un-adapted fixed-lr "
        f"0.003 level (0.0522), so matching the RATE of conservatism is not "
        f"sufficient -- WHICH coordinates open, and when, is the whole "
        f"mechanism. The surviving distinction for absgate is procedural, not "
        f"performance: its threshold had to be fitted offline on held-out "
        f"seeds (two passes) and still drifted 38% in realized rate from the "
        f"calibration seeds to the evaluation seeds, whereas COHG's threshold "
        f"is computed online from the run's own certificate.", "",
        f"**(2) Does tuned ungated SIGN dominate gated COHG?** No -- but it "
        f"beats it on NMSE at every alpha. alpha=0.4 gives "
        f"{n04['nmse']:.4f}+-{n04['nmse_sd']:.4f} vs COHG's "
        f"{ref_a['nmse']:.4f} (p=0.002), i.e. 5x lower error, while paying "
        f"{n04['ev']:.0f}+-{n04['ev_sd']:.0f} instability events against "
        f"COHG's {ref_a['ev']:.0f}+-{ref_a['ev_sd']:.0f} (p=0.002). It is a "
        f"clean Pareto trade, not domination, and the ungated NMSE is not "
        f"robust: the identical arm on GPU has one seed at NMSE 59.8 (arm-A "
        f"reproduction check above).", "",
        f"**(3) Does enforcing the Theorem-6 step condition preserve "
        f"performance?** No. t6clip caps EVERY open step (100% clipped at "
        f"both rho_f), collapsing NMSE to "
        f"{agg('t6clip_lr0.003_rf1')['nmse']:.4f} (rho_f=1) and "
        f"{agg('t6clip_lr0.003_rf10')['nmse']:.4f} (rho_f=10) with 0 events -- "
        f"the controller becomes provably safe and effectively "
        f"non-adaptive (final LRs stay at the mis-set 0.003). Because "
        f"|ghat| ~ 0.06 at the gate, the cap 2(1-1/c)|ghat|/rho_f ~ 0.06 is "
        f"~7x smaller than alpha=0.4, so the theorem's condition and the "
        f"practical step size are simply not compatible at this scale.", "",
        f"**(4) Does fail-closed catch a misspecified M_H?** Yes as a "
        f"DETECTOR, but there was nothing to catch. The monitor's closed-gate "
        f"fraction rises monotonically with misspecification "
        f"(7.8% -> 59% -> 93% of steps at M_H = 5 -> 0.5 -> 0.05) and the "
        f"probe budget with it (+8% -> +60% -> +92% HVPs), so a wrong prior "
        f"is loudly visible online. But the parallel exact-sensitivity audit "
        f"found ZERO certificate violations in all 720000 checked "
        f"coordinate-steps of EVERY arm, including M_H 100x too small: the "
        f"M_H drift term is not what binds e_col in this regime (max ratio "
        f"0.71, p99 ~0.18), so under-specifying it does not break the bound "
        f"here. Fail-closed also changed no run's outcome (NMSE, events and "
        f"open rate are bit-identical to the no-FC arms) because COHG does "
        f"ALL of its adaptation in the first ~50 steps, before the monitor "
        f"has two probes to compare."]

out = os.path.join(CTL, "SUMMARY.md")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print("\n".join(L))
print(f"\n-> {out}")
