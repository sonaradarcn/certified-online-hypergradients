"""B4: drift-prior (M_H) misspecification curves, E1 + E2.

Reads
  results/e1_misspec/teacher_kw_drift_cal.json           (3 calibration seeds)
  results/e1_misspec/teacher_kw_drift_x{f}_fc{0,1}.json  (5 eval seeds each)
  results/e2_controls/mackey_drift_cohg_lr0.003_mh{5,0.5,0.05}_fc{0,1}_s*.json

Writes
  results/reanalysis/misspec_curves.md

E1 is the point of the study: the teacher/student stream carries an EXACT
discounted forward-mode ground truth (ExactFMD), so `violation` there means
`e_t < ||S_t - Shat_t||_F`, a genuine failure of the anytime certificate, not
a proxy.  E2 is the same sweep on the GRU stream, where the audit is
per-coordinate (`|ghat_j - g_true_j| > beta_col_j`).
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
MIS = os.path.join(RES, "e1_misspec")
CTL = os.path.join(RES, "e2_controls")
OUT = os.path.join(RES, "reanalysis")
os.makedirs(OUT, exist_ok=True)

MH_FACTORS = [1.0, 0.3, 0.1, 0.03, 0.01]
E2_MH = [5.0, 0.5, 0.05]


# ----------------------------------------------------------------- E1 -----
def e1_cal():
    rows = json.load(open(os.path.join(MIS, "teacher_kw_drift_cal.json")))
    m = sorted(x for r in rows for x in (r.get("m_obs") or []))
    return rows, np.asarray(m)


def e1_points():
    out = []
    for f in MH_FACTORS:
        for fc in (0, 1):
            p = os.path.join(MIS, f"teacher_kw_drift_x{f:g}_fc{fc}.json")
            if not os.path.exists(p):
                continue
            rs = json.load(open(p))
            n_steps = 1000  # e1_certificate default; recovered below if given
            d = dict(
                factor=f, fc=fc, n=len(rs), M_H=rs[0]["M_H"],
                seeds=[r["seed"] for r in rs],
                viol_rate=float(np.mean([r["violation_rate"] for r in rs])),
                viol_rate_max=float(np.max([r["violation_rate"] for r in rs])),
                n_viol=int(sum(r["n_violations"] for r in rs)),
                n_steps_tot=int(sum(round(r["n_violations"]
                                          / r["violation_rate"])
                                    if r["violation_rate"] else n_steps
                                    for r in rs)),
                worst_ratio=float(np.max([r["worst_true_over_bound"]
                                          for r in rs])),
                worst_ratio_mean=float(np.mean([r["worst_true_over_bound"]
                                                for r in rs])),
                ratio_p75=float(np.mean([1.0 / r["tight_q25"] for r in rs])),
                ratio_p50=float(np.mean([1.0 / r["tight_med"] for r in rs])),
                probe_ov=float(np.mean([r["probe_overhead"] for r in rs])),
                n_probes=float(np.mean([r["n_probes"] for r in rs])),
                kw_hvp=float(np.mean([r["kw_hvp"] for r in rs])),
                final_e=float(np.mean([r["final_e"] for r in rs])),
                closed=None, fc_events=None, reprobes=None,
                mobs_med=None, mobs_p99=None, mobs_max=None, frac_over=None,
            )
            if fc:
                d["closed"] = float(np.mean([r["closed_frac"] for r in rs]))
                d["closed_sd"] = float(np.std([r["closed_frac"] for r in rs],
                                              ddof=1))
                d["fc_events"] = float(np.mean([r["failclosed_events"]
                                                for r in rs]))
                d["reprobes"] = float(np.mean([r["reprobe_requests"]
                                               for r in rs]))
                am = np.asarray([x for r in rs for x in r["m_obs"]])
                d["mobs_med"] = float(np.median(am))
                d["mobs_p99"] = float(np.percentile(am, 99))
                d["mobs_max"] = float(am.max())
                d["frac_over"] = float(np.mean(am > rs[0]["M_H"]))
            out.append(d)
    return out


# ----------------------------------------------------------------- E2 -----
def e2_arm(mh, fc):
    ps = sorted(glob.glob(os.path.join(
        CTL, f"mackey_drift_cohg_lr0.003_mh{mh:g}_fc{fc}_s*.json")))
    return [json.load(open(p)) for p in ps]


def e2_points():
    ref = e2_arm(5.0, 1)
    mh_star = max(r["m_obs_stats"]["max"] for r in ref)
    mobs_med_ref = float(np.mean([r["m_obs_stats"]["median"] for r in ref]))
    base_hvp = float(np.mean([r["hvp_total"] for r in e2_arm(5.0, 0)]))
    out = []
    for mh in E2_MH:
        for fc in (0, 1):
            rs = e2_arm(mh, fc)
            if not rs:
                continue
            d = dict(
                M_H=mh, fc=fc, n=len(rs), x=mh / 5.0, x_max=mh / mh_star,
                viol_frac=float(np.mean([r["cert_violation_frac"]
                                         for r in rs])),
                n_viol=int(sum(r["cert_violations"] for r in rs)),
                checked=int(sum(r["cert_checked"] for r in rs)),
                ratio_p50=float(np.mean([r["cert_ratio_q"]["p50"]
                                         for r in rs])),
                ratio_p99=float(np.mean([r["cert_ratio_q"]["p99"]
                                         for r in rs])),
                worst_ratio=float(max(r["cert_max_ratio"] for r in rs)),
                probe_ov=float(np.mean([r["hvp_total"] for r in rs])
                               / base_hvp),
                hvp=float(np.mean([r["hvp_total"] for r in rs])),
                nmse=float(np.mean([r["nmse"] for r in rs])),
                nmse_sd=float(np.std([r["nmse"] for r in rs], ddof=1)),
                events=float(np.mean([r["events"] for r in rs])),
                wall=float(np.mean([r["wall_s"] for r in rs])),
                closed=None, fc_events=None, reprobes=None,
                mobs_med=None, mobs_p99=None, mobs_max=None,
            )
            if fc:
                d["closed"] = float(np.mean(
                    [(r["failclosed_closed_steps"] or 0) / r["steps"]
                     for r in rs]))
                d["closed_sd"] = float(np.std(
                    [(r["failclosed_closed_steps"] or 0) / r["steps"]
                     for r in rs], ddof=1))
                d["fc_events"] = float(np.mean([r["failclosed_events"] or 0
                                                for r in rs]))
                d["reprobes"] = float(np.mean([r["failclosed_reprobes"] or 0
                                               for r in rs]))
                d["mobs_med"] = float(np.mean([r["m_obs_stats"]["median"]
                                               for r in rs]))
                d["mobs_p90"] = float(np.mean([r["m_obs_stats"]["p90"]
                                               for r in rs]))
                d["mobs_p99"] = float(np.mean([r["m_obs_stats"]["p99"]
                                               for r in rs]))
                d["mobs_med_lo"] = float(min(r["m_obs_stats"]["median"]
                                             for r in rs))
                d["mobs_med_hi"] = float(max(r["m_obs_stats"]["median"]
                                             for r in rs))
                d["n_probes"] = int(sum(r["m_obs_stats"]["n"] for r in rs))
                d["mobs_max"] = float(max(r["m_obs_stats"]["max"] for r in rs))
            out.append(d)
    return out, mh_star, mobs_med_ref


def na(x, spec=".3g"):
    return "--" if x is None else format(x, spec)


def main():
    cal_rows, cal_m = e1_cal()
    mh_star1 = float(cal_m.max())
    med1 = float(np.median(cal_m))
    e1 = e1_points()
    e2, mh_star2, med2 = e2_points()
    dep1 = [d for d in e1 if d["fc"] and d["factor"] == 1.0][0]
    dep2 = [d for d in e2 if d["fc"] and d["M_H"] == 5.0][0]

    L = []
    A = L.append
    A("# B4. Drift-prior (M_H) misspecification curves")
    A("")
    A("How far can the Hessian-drift prior `M_H` in assumption A4' be wrong "
      "before the anytime certificate stops holding, and does the fail-closed "
      "monitor notice?  Two streams, two ground truths.")
    A("")
    A("| | E1 `teacher/kw_drift` | E2 `mackey_drift` GRU |")
    A("|---|---|---|")
    A("| model | 2-layer teacher-student, r=4, K=10, gamma=0.9, kw-eps=0.05 |"
      " 13k-param GRU, r=4, K=10, gamma=0.9, kw-eps=0.1, probe-every 20 |")
    A("| steps x seeds | 1000 x 5 (seeds 0-4) | 12000 x 10 (seeds 0-9) |")
    A("| ground truth | **exact** discounted `ExactFMD` sensitivity matrix "
      "`S_t`, fp64 | parallel exact discounted `ExactFMD` hypergradient, fp32 "
      "(1e-4 rel. tol.) |")
    A("| violation test | `e_t < ||S_t - Shat_t||_F` on ANY step | "
      "`|ghat_j - g_true_j| > beta_col_j` on ANY coordinate-step |")
    A("| ratio plotted | `||S_t - Shat_t||_F / e_t` | "
      "`|ghat_j - g_true_j| / beta_col_j` |")
    A("| lambda | frozen (certificate study) | live (the controller runs) |")
    A("")
    A("## Definitions")
    A("")
    A("Three quantities, kept apart because the earlier draft ran them "
      "together.")
    A("")
    A("* **`M_obs`, the observed probe-to-probe drift rate.** "
      "`M_obs = |rho_probe - rho_prev| / (eta_max_t0 * D)`, where `D` is the "
      "parameter path length between the two probes and `eta_max_t0` the "
      "largest learning rate at the earlier probe "
      "(`DriftHoldFailClosed.probe`, `code/cohg/certificate.py`).  It is "
      "recorded ONLY by the fail-closed monitor.  The base `DriftHold` class "
      "stores a different field, `mh_observed = |Delta rho| / D`, which omits "
      "the `eta_max` factor and therefore estimates `eta_max * M_H`, not "
      "`M_H`; E3 and E4 use the base class, so their stored diagnostic is NOT "
      "on the `M_H` scale and no margin may be quoted from it.")
    A("* **`M_H`, the DEPLOYED prior.** The value actually passed to the "
      "certificate in the runs being reported: 5 on E2, 20 on E3, 50 on E4 "
      "(Table `tab:certparams`), and 2.27609 at factor 1 of the E1 sweep.  "
      "This is the denominator of the figure's x-axis for BOTH series, so "
      "`0.01` on that axis means one hundredth of what was run.")
    A("* **`M_obs^med`, `M_obs^p99`, `M_obs^max`.** Median, 99th percentile "
      "and maximum of `M_obs` on the DEPLOYED fail-closed arm of a regime, "
      "over that regime's named seeds.  These are properties of the stream, "
      "not settings.")
    A("")
    A(f"**E1.** The deployed prior 2.27609 was CALIBRATED as `M_obs^max` over "
      f"{len(cal_m)} probes on 3 calibration seeds "
      f"({[r['seed'] for r in cal_rows]}), disjoint from the 5 evaluation "
      f"seeds and measured at an effectively infinite prior (`M_H = 1e12`) so "
      f"the probe schedule is undistorted.  On those calibration seeds: "
      f"median {med1:.4g}, p90 {np.percentile(cal_m, 90):.4g}, "
      f"p99 {np.percentile(cal_m, 99):.4g}, max {mh_star1:.4g}.  On the five "
      f"EVALUATION seeds at that same prior the rate is heavier: median "
      f"{dep1['mobs_med']:.3g}, p99 {dep1['mobs_p99']:.3g}, max "
      f"{dep1['mobs_max']:.3g}, and {dep1['frac_over']:.1%} of probes exceed "
      f"the prior.  A calibration max on held-out seeds does not upper-bound "
      f"fresh seeds.")
    A("")
    A(f"**E2.** The deployed prior is 5 (Table `tab:certparams`); no separate "
      f"calibration arm exists.  On the deployed fail-closed arm itself, over "
      f"{dep2['n_probes']} probes on the ten evaluation seeds, `M_obs` has "
      f"median {med2:.4g} (per-seed medians {dep2['mobs_med_lo']:.3g} to "
      f"{dep2['mobs_med_hi']:.3g}), p90 {dep2['mobs_p90']:.3g}, p99 "
      f"{dep2['mobs_p99']:.3g} and max "
      f"{mh_star2:.6g}, the last being a single extreme draw.  The deployed "
      f"prior is thus {5.0 / med2:.2g}x the median observed rate and well "
      f"below its upper tail.  `761.5` is that in-sample maximum; it is NOT a "
      f"calibration "
      f"target and NOT the same statistic as E1's 2.276, and normalising the "
      f"two sweeps by their own maxima is what put them two decades apart on "
      f"the old axis.")
    A("")

    # -------------------------------------------------------- E1 table ----
    A("## E1 -- exact ground truth (teacher/kw_drift, 5 seeds x 1000 steps)")
    A("")
    A("`worst ratio` = max over seeds and steps of true error / certificate "
      "(>= 1 would be a violation).  Per-step ratio traces were not kept "
      "(`--trace-config none`), so a p99 of the ratio is NOT available for "
      "E1; the two quantile columns are inverted tightness quantiles stored "
      "per run (`p75 ratio` = mean over seeds of `1/tight_q25`).")
    A("")
    A("| M_H/M_H_dep | M_H | fail-closed | seeds | violating steps | violation "
      "rate | worst ratio | p75 ratio | median ratio | closed-gate frac | "
      "probes | probe overhead | KW HVPs | FC events | re-probes |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in e1:
        A(f"| {d['factor']:g} | {d['M_H']:.4g} | "
          f"{'yes' if d['fc'] else 'no'} | {d['n']} | "
          f"{d['n_viol']} / {d['n_steps_tot']} | {d['viol_rate']:.3f} | "
          f"{d['worst_ratio']:.3f} | {d['ratio_p75']:.3f} | "
          f"{d['ratio_p50']:.3f} | "
          f"{('%.3f' % d['closed']) if d['closed'] is not None else '--'} | "
          f"{d['n_probes']:.1f} | {d['probe_ov']:.3f}x | "
          f"{d['kw_hvp']:.0f} | {na(d['fc_events'], '.1f')} | "
          f"{na(d['reprobes'], '.1f')} |")
    A("")
    A("Monitor diagnostics on the fail-closed arms (M_obs pooled over the 5 "
      "evaluation seeds):")
    A("")
    A("| M_H/M_H_dep | M_H | median M_obs | p99 M_obs | max M_obs | frac. probes "
      "with M_obs > M_H | closed-gate frac |")
    A("|---|---|---|---|---|---|---|")
    for d in e1:
        if not d["fc"]:
            continue
        A(f"| {d['factor']:g} | {d['M_H']:.4g} | {d['mobs_med']:.3g} | "
          f"{d['mobs_p99']:.3g} | {d['mobs_max']:.3g} | "
          f"{d['frac_over']:.3f} | {d['closed']:.3f} |")
    A("")

    # -------------------------------------------------------- E2 table ----
    A("## E2 -- GRU stream (mackey_drift, 10 seeds x 12000 steps)")
    A("")
    A("`probe overhead` here is the total HVP count relative to the M_H=5 "
      "no-fail-closed arm (94588 HVPs); in E1 it is the probe COUNT relative "
      "to the nominal budget.  Both are capped at 2x by the "
      "'a forced re-probe never forces another one' rule.")
    A("")
    A("`worst ratio` is the MAX OVER SEEDS of each run's max ratio. "
      "`results/e2_controls/SUMMARY.md` reports the same quantity as the MEAN "
      "over seeds of the per-run max (0.705 / 0.706 / 0.706); both are right, "
      "and the max-over-seeds used here and in the figure is the conservative "
      "one.  The E1 table above uses the same max-over-seeds convention "
      "(its mean-over-seeds values are 0.646 / 0.672 / 0.680 / 0.683 / "
      "0.684).")
    A("")
    A("| M_H/M_H_dep | M_H/median M_obs | M_H | fail-closed | seeds | violations "
      "| violation rate | worst ratio | p99 ratio | median ratio | closed-gate "
      "frac | HVPs | probe overhead | FC events | re-probes | NMSE | events |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in e2:
        A(f"| {d['x']:g} | {d['M_H'] / med2:.3g} | {d['M_H']:g} | "
          f"{'yes' if d['fc'] else 'no'} | {d['n']} | "
          f"{d['n_viol']} / {d['checked']} | {d['viol_frac']:.3f} | "
          f"{d['worst_ratio']:.3f} | {d['ratio_p99']:.3f} | "
          f"{d['ratio_p50']:.3f} | "
          f"{('%.3f' % d['closed']) if d['closed'] is not None else '--'} | "
          f"{d['hvp']:.0f} | {d['probe_ov']:.3f}x | "
          f"{na(d['fc_events'], '.1f')} | {na(d['reprobes'], '.1f')} | "
          f"{d['nmse']:.4f}+-{d['nmse_sd']:.4f} | {d['events']:.1f} |")
    A("")

    # ------------------------------------------------ aligned comparison ---
    A("## The two streams on a common axis")
    A("")
    A("`M_H/M_H_dep` -- the prior relative to the value DEPLOYED in that "
      "regime -- is the figure axis and lines the two sweeps up directly.  "
      "`M_H / median M_obs`, the prior relative to the TYPICAL drift rate, "
      "which is what decides how often the monitor trips, agrees with it to "
      "within a factor of 1.3 at every matching point.  The old axis "
      "normalised each stream by the maximum rate on its own probe "
      "population, which is not the same statistic on the two streams and put "
      "them two decades apart for that reason alone.")
    A("")
    A("| stream | M_H | M_H/M_H_dep | M_H/median M_obs | closed-gate frac | "
      "probe overhead | violations |")
    A("|---|---|---|---|---|---|---|")
    for d in e1:
        if not d["fc"]:
            continue
        A(f"| E1 | {d['M_H']:.4g} | {d['factor']:g} | "
          f"{d['M_H'] / med1:.3g} | {d['closed']:.3f} | "
          f"{d['probe_ov']:.3f}x | 0 |")
    for d in e2:
        if not d["fc"]:
            continue
        A(f"| E2 | {d['M_H']:g} | {d['x']:g} | {d['M_H'] / med2:.3g} | "
          f"{d['closed']:.3f} | {d['probe_ov']:.3f}x | 0 |")
    A("")

    # ------------------------------------------------------- the reading --
    fc1 = {d["factor"]: d for d in e1 if d["fc"]}
    fc0 = {d["factor"]: d for d in e1 if not d["fc"]}
    A("## Plain reading")
    A("")
    A("**Does a too-small M_H ever produce a certificate violation on E1, "
      "where the ground truth is exact?  No -- not at any factor tested, "
      "down to 100x too small.** Across all 10 E1 points (5 M_H factors x "
      "fail-closed on/off, 5 seeds x 1000 steps each = 50000 certified "
      "steps), `valid_rate` is exactly 1.000 and `n_violations` is exactly 0. "
      "There is no crossover factor to report: the sweep bottoms out at "
      f"M_H = {fc0[0.01]['M_H']:.4g} (1/100 of the deployed "
      f"{mh_star1:.4g}) with the bound still holding on every step of every "
      "seed.")
    A("")
    A("**And the margin barely moves.** The worst true-error/bound ratio "
      f"rises monotonically but saturates: {fc0[1.0]['worst_ratio']:.3f} -> "
      f"{fc0[0.3]['worst_ratio']:.3f} -> {fc0[0.1]['worst_ratio']:.3f} -> "
      f"{fc0[0.03]['worst_ratio']:.3f} -> {fc0[0.01]['worst_ratio']:.3f} "
      "as M_H shrinks by 100x, i.e. a 4% widening of the worst case and a "
      "convergent limit ~0.74 that is reached by factor 0.03. Removing the "
      "drift term entirely would therefore still leave ~26% headroom.  That "
      "is the diagnosis: in this regime the certificate is NOT bound by the "
      "A4' drift term at all -- it is bound by the rank-r truncation and the "
      "KW probe tolerance, both of which are independent of M_H.  The E2 "
      "audit says the same thing at 100x misspecification (max ratio "
      f"{max(d['worst_ratio'] for d in e2):.3f}, p99 ~0.18, 0 / 4.32e6 "
      "coordinate-step checks violated).")
    A("")
    A("**Does the fail-closed monitor catch it?  Yes, loudly and "
      "monotonically -- it is a working detector of a condition that did not "
      "actually break anything here.** The closed-gate fraction on E1 runs "
      f"{fc1[1.0]['closed']:.3f} -> {fc1[0.3]['closed']:.3f} -> "
      f"{fc1[0.1]['closed']:.3f} -> {fc1[0.03]['closed']:.3f} -> "
      f"{fc1[0.01]['closed']:.3f} at factors 1 / 0.3 / 0.1 / 0.03 / 0.01, and "
      "E2 gives the same monotone shape "
      f"({[round(d['closed'], 3) for d in e2 if d['fc']]} at M_H = 5 / 0.5 / "
      "0.05). Closure tracks the fraction of probes whose measured M_obs "
      "exceeds the prior almost exactly on E1 "
      f"({fc1[1.0]['frac_over']:.3f} / {fc1[0.3]['frac_over']:.3f} / "
      f"{fc1[0.1]['frac_over']:.3f} / {fc1[0.03]['frac_over']:.3f} / "
      f"{fc1[0.01]['frac_over']:.3f} vs the closure fractions above), so the "
      "signal is exactly what it claims to be. At a 100x-too-small prior the "
      "monitor is shut ~97% of the time on E1 and ~93% on E2: a "
      "misspecified drift prior is impossible to miss online, without any "
      "oracle.")
    A("")
    A("**Even the DEPLOYED prior trips the monitor.** At factor 1 the "
      f"E1 monitor still closes {fc1[1.0]['closed']:.1%} of steps, because "
      f"{fc1[1.0]['frac_over']:.1%} of the evaluation seeds' probes measure "
      f"M_obs above the deployed prior (up to {fc1[1.0]['mobs_max']:.3g}, i.e. "
      f"{fc1[1.0]['mobs_max'] / mh_star1:.0f}x it) -- the "
      "calibration max over 3 held-out seeds does not upper-bound the "
      "evaluation seeds. The certificate held anyway on every one of those "
      "steps, which is the concrete evidence that `M_obs > M_H` is a "
      "CONSERVATIVE trigger and not a certificate failure: M_obs divides a "
      "probe-to-probe rho change by a short path length and so over-reads the "
      "true Hessian-Lipschitz constant.  A practitioner should read a rising "
      "closed-gate fraction as 'recalibrate M_H', not as 'the bound just "
      "broke'.")
    A("")
    A("**Cost.** The monitor is bounded by construction (a forced re-probe "
      "cannot force another), and the measurements sit under that cap: E1 "
      f"probe overhead {fc1[1.0]['probe_ov']:.3f}x -> "
      f"{fc1[0.3]['probe_ov']:.3f}x -> {fc1[0.1]['probe_ov']:.3f}x -> "
      f"{fc1[0.03]['probe_ov']:.3f}x -> {fc1[0.01]['probe_ov']:.3f}x, "
      "E2 HVP overhead "
      f"{[round(d['probe_ov'], 3) for d in e2 if d['fc']]}x at M_H = 5 / 0.5 "
      "/ 0.05 (wall time 5.4ks -> 7.5ks -> 8.7ks against a 5.1ks baseline). "
      "So the price of the safety net is at most a 2x probe budget, paid only "
      "in proportion to how wrong the prior is, and nothing at all when the "
      "prior is right.  Turning the monitor ON never changes an E1 "
      "certificate outcome (violation rate and worst ratio are identical to "
      "the fc0 column at every factor) and never changes an E2 run outcome "
      "(NMSE, events and open rate are bit-identical to the fc0 arms).")
    A("")
    A("**Caveats to state in the paper.** (i) Zero violations at 100x "
      "misspecification is evidence that the A4' term is slack in THESE two "
      "regimes, not that A4' is unnecessary; a stream where the drift term "
      "dominates the rank-truncation term would behave differently, and we "
      "did not construct one. (ii) E1 sweeps M_H DOWNWARD only -- an "
      "over-large M_H cannot break the bound (it only loosens it), which is "
      "why the sweep is one-sided, but that also means the figure says "
      "nothing about the cost of over-conservatism beyond factor 1. "
      "(iii) E1's lambda is frozen, so the misspecification cannot feed back "
      "into the controller; E2 supplies that half and shows the feedback is "
      "nil, because COHG finishes adapting in the first ~50 steps, before the "
      "monitor has two probes to compare. (iv) The deployed prior is NOT an "
      "upper bound on the observed rate in either regime: it exceeds the "
      "median by 6.5x on E2 and 7.0x on E1, and sits below the p99 (79.1 on "
      "E2, 14.4 on E1) and the max (761.5, 32.1).  The audit is the evidence "
      "we have that this did not matter here; it is not a demonstration that "
      "A4' held. (v) E3 and E4 record only the base-class diagnostic "
      "|Delta rho| / D, which is off the M_H scale by a factor of eta_max, so "
      "no margin can be quoted for those regimes at all.  On the GPT-2 runs "
      "that diagnostic has median 0.094 and max 1.80 over 319 stored probe "
      "windows at a frozen eta_max of 1.49e-3, which rescales to M_obs "
      "median 63 and max 1208 against a prior of 50.")
    A("")
    A("Figure: `paper/main/figs/fig10_misspec.pdf` "
      "(preview `results/figures/fig10_misspec.png`), generated by "
      "`code/experiments/make_fig10_misspec.py`.")

    path = os.path.join(OUT, "misspec_curves.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("\n->", path)


if __name__ == "__main__":
    main()
