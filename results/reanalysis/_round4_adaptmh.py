"""Round-4 experiment 2 analysis: ONLINE-ENFORCEABLE drift envelope.

`--adaptive-mh KAPPA` re-states, at every spectral probe,

    M_H,t = max(M_H_floor, KAPPA * max_{s<=t} M_obs,s)

with the fail-closed monitor on and M_H_floor = the deployed prior (5.0 for E2,
2.2760914236726824 for E1), so the envelope in force is never below the paper's
prior and never below any diagnostic the run has observed.

Reads results/e2_adaptmh/*.json, results/e1_adaptmh/*.json and the fixed-prior
references (results/e2_controls, results/e1_misspec) and writes
results/reanalysis/round4_adaptmh.md.
"""
from __future__ import annotations

import glob
import itertools
import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
E2A = os.path.join(ROOT, "results", "e2_adaptmh")
E1A = os.path.join(ROOT, "results", "e1_adaptmh")
CTL = os.path.join(ROOT, "results", "e2_controls")
MIS = os.path.join(ROOT, "results", "e1_misspec")
OUT = os.path.join(HERE, "round4_adaptmh.md")

E2_FLOOR = 5.0
E1_FLOOR = 2.2760914236726824
NOFC_HVP = 94588        # COHG no-fail-closed HVP budget on this config


def mstd(v):
    v = [x for x in v if x is not None and not (isinstance(x, float)
                                                and math.isnan(x))]
    if not v:
        return float("nan"), float("nan"), 0
    n = len(v)
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else float("nan")
    return m, sd, n


def f(m, s, p=4):
    if not math.isfinite(m):
        return "inf"
    if not math.isfinite(s):
        return f"{m:.{p}f}"
    return f"{m:.{p}f}+-{s:.{p}f}"


def signflip_p(d):
    n = len(d)
    obs = abs(sum(d) / n)
    cnt = 0
    for signs in itertools.product([1, -1], repeat=n):
        if abs(sum(s * x for s, x in zip(signs, d)) / n) >= obs - 1e-15:
            cnt += 1
    return cnt / 2 ** n, sum(d) / n


def retro_consistency(d):
    """For every gate OPENING at step t: the envelope in force at t, and the
    NEXT probe's observed drift rate M_obs.  The opening is retrospectively
    consistent iff that next observation does not exceed the envelope the
    opening was certified under."""
    log = d.get("adapt_probe_log") or []
    probes = [(r[0], r[2]) for r in log]           # (t_probe, M_obs or None)
    first_obs_t = next((tp for tp, m in probes if m is not None), None)
    opens = d.get("gate_open_steps") or []
    envs = d.get("gate_open_env") or []
    n_ok = n_tot = n_nonext = n_cold = 0
    n_ok_final = 0
    final_env = d.get("adapt_mh_final") or float("inf")
    worst = 0.0
    for t, e in zip(opens, envs):
        if first_obs_t is not None and t < first_obs_t:
            n_cold += 1
        nxt = next((m for (tp, m) in probes if tp > t and m is not None), None)
        if nxt is None:
            n_nonext += 1
            continue
        n_tot += 1
        n_ok += int(nxt <= e)
        n_ok_final += int(nxt <= final_env)
        worst = max(worst, nxt / e if e else float("inf"))
    return n_ok, n_tot, n_nonext, worst, n_cold, n_ok_final


def load_e2(pat, d=E2A):
    out = []
    for p in sorted(glob.glob(os.path.join(d, pat))):
        out.append(json.load(open(p)))
    out.sort(key=lambda r: r["seed"])
    return out


def main():
    L = []
    W = L.append
    W("# Round-4 experiment 2: an online-enforceable conservative drift "
      "envelope\n")
    W("Envelope in force at probe `t`: `M_H,t = max(M_H_floor, KAPPA * "
      "max_{s<=t} M_obs,s)`, re-stated at every probe, fail-closed monitor "
      "on. `M_H_floor` is the **deployed prior** (5.0 for E2, 2.2760914 for "
      "E1 -- the calibrated `M_H*` of the teacher/kw_drift stream), so the "
      "arm is never LESS conservative than the shipped certificate. The "
      "monitor is evaluated against the envelope in force over the interval "
      "just traversed (before the new observation is folded in): raising the "
      "envelope afterwards cannot retroactively excuse that interval.\n")

    # =================================================================== E2
    W("## E2 -- GRU / `mackey_drift`, seeds 0-9, per-coordinate certificate "
      "audit (`--validate-cert`)\n")
    ref0 = load_e2("mackey_drift_cohg_lr0.003_mh5_fc0_s*.json", CTL)
    ref1 = load_e2("mackey_drift_cohg_lr0.003_mh5_fc1_s*.json", CTL)
    arms = {"fixed prior M_H=5, no FC": ref0,
            "fixed prior M_H=5, fail-closed": ref1}
    for k in (1.0, 2.0):
        rs = load_e2(f"mackey_drift_cohg_lr0.003_amh{k:g}_s*.json")
        if rs:
            arms[f"**adaptive envelope KAPPA={k:g}**"] = rs

    W("| arm | n | NMSE | events | coord-open rate | cert viol / checked | "
      "cert max ratio | closed steps | closure frac | HVPs | HVPs / no-FC | "
      "final envelope | envelope raises | max M_obs |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, rs in arms.items():
        if not rs:
            continue
        n = len(rs)
        cv = sum(r.get("cert_violations") or 0 for r in rs)
        cc = sum(r.get("cert_checked") or 0 for r in rs)
        cr = [r.get("cert_max_ratio") for r in rs
              if r.get("cert_max_ratio") is not None]
        cs = [r.get("failclosed_closed_steps") for r in rs
              if r.get("failclosed_closed_steps") is not None]
        hv = mstd([float(r["hvp_total"]) for r in rs])[0]
        fe = mstd([r.get("adapt_mh_final") for r in rs])
        nr = mstd([r.get("adapt_mh_raises") for r in rs])
        mo = mstd([r.get("adapt_mobs_max") for r in rs])
        W(f"| {name} | {n} | {f(*mstd([r['nmse'] for r in rs])[:2])} | "
          f"{f(*mstd([float(r['events']) for r in rs])[:2], 1)} | "
          f"{mstd([r['coord_open_frac'] for r in rs])[0]:.3e} | "
          f"{cv} / {cc if cc else '-'} | "
          f"{(max(cr) if cr else float('nan')):.4f} | "
          f"{(mstd(list(map(float, cs)))[0] if cs else float('nan')):.1f} | "
          f"{(mstd([c / 12000 for c in map(float, cs)])[0] if cs else float('nan')):.4f} | "
          f"{hv:.0f} | {hv / NOFC_HVP:.3f} | "
          f"{('-' if not math.isfinite(fe[0]) else f'{fe[0]:.4g}')} | "
          f"{('-' if not math.isfinite(nr[0]) else f'{nr[0]:.1f}')} | "
          f"{('-' if not math.isfinite(mo[0]) else f'{mo[0]:.4g}')} |")
    W("")

    # ---- gate decisions vs the fixed-prior arm ---------------------------
    W("### Gate decisions vs the fixed-prior arm (per seed)\n")
    W("`losses identical` / `lam identical` compare the FULL 12000-entry loss "
      "trace and the sampled lambda trajectory element for element against "
      "the fixed-prior fail-closed reference (`M_H = 5`, "
      "`results/e2_controls`). `open coord-steps` is "
      "`coord_open_frac * 6 * 12000`.\n")
    W("| KAPPA | seed | open coord-steps (adaptive) | (fixed prior) | "
      "openings lost | NMSE (adaptive / fixed) | events | losses identical | "
      "closed steps (adaptive / fixed) |")
    W("|---|---|---|---|---|---|---|---|---|")
    lost_tot = defaultdict(int)
    for k in (1.0, 2.0):
        rs = load_e2(f"mackey_drift_cohg_lr0.003_amh{k:g}_s*.json")
        rmap = {r["seed"]: r for r in ref1}
        for r in rs:
            b = rmap.get(r["seed"])
            if b is None:
                continue
            oa = round(r["coord_open_frac"] * 6 * r["steps"])
            ob = round(b["coord_open_frac"] * 6 * b["steps"])
            lost_tot[k] += max(ob - oa, 0)
            li = (r["losses"] == b["losses"])
            W(f"| {k:g} | {r['seed']} | {oa} | {ob} | {ob - oa} | "
              f"{r['nmse']:.6f} / {b['nmse']:.6f} | {r['events']} / "
              f"{b['events']} | {li} | "
              f"{r.get('failclosed_closed_steps')} / "
              f"{b.get('failclosed_closed_steps')} |")
    W("")

    # ---- retrospective consistency --------------------------------------
    W("### Retrospective consistency of every certified opening\n")
    W("For each gate opening at step `t`: the envelope in force at `t`, and "
      "the NEXT probe's observed drift rate `M_obs`. The opening is "
      "*retrospectively consistent* iff that following observation does not "
      "exceed the envelope the opening was certified under.\n")
    W("`cold-start openings` are openings that happen BEFORE the run's first "
      "`M_obs` exists (the first probe pair completes at step "
      "`2 * probe_every`), so the only envelope available to certify them is "
      "the unverified floor. `consistent under the FINAL envelope` re-checks "
      "each opening against the envelope the run ends with -- the one that is "
      "never below any diagnostic observed over the whole run.\n")
    W("| KAPPA | seed | openings | first / last open step | cold-start "
      "openings | with a following probe | consistent vs envelope in force | "
      "frac | consistent vs FINAL envelope | worst next-M_obs / envelope | "
      "envelope at first / last opening |")
    W("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in (1.0, 2.0):
        for r in load_e2(f"mackey_drift_cohg_lr0.003_amh{k:g}_s*.json"):
            ok, tot, nn, worst, cold, okf = retro_consistency(r)
            envs = r.get("gate_open_env") or []
            gs = r.get("gate_open_steps") or []
            W(f"| {k:g} | {r['seed']} | {len(gs)} | "
              f"{(gs[0] if gs else '-')} / {(gs[-1] if gs else '-')} | "
              f"{cold} | {tot} | {ok} | "
              f"{(ok / tot if tot else float('nan')):.4f} | {okf} | "
              f"{worst:.4f} | "
              f"{(envs[0] if envs else float('nan')):.4g} / "
              f"{(envs[-1] if envs else float('nan')):.4g} |")
    W("")

    # ---- envelope trajectory --------------------------------------------
    W("### Envelope trajectory (seed 0)\n")
    for k in (1.0, 2.0):
        rs = load_e2(f"mackey_drift_cohg_lr0.003_amh{k:g}_s0.json")
        if not rs:
            continue
        lg = rs[0].get("adapt_probe_log") or []
        with_obs = [r for r in lg if r[2] is not None]
        W(f"KAPPA={k:g}: {len(lg)} probes, {len(with_obs)} of them yielding an "
          f"`M_obs`; envelope {rs[0].get('adapt_mh_final'):.6g} at the end "
          f"(floor {E2_FLOOR}); {rs[0].get('adapt_mh_raises')} raises.\n")
        W("| probe step | envelope before | M_obs | envelope after | closed |")
        W("|---|---|---|---|---|")
        shown = 0
        prev = None
        for row in lg:
            t, mb, mo, ma, cl = row
            if prev is not None and ma == prev and shown > 12 and not cl:
                continue
            prev = ma
            W(f"| {t} | {mb:.5g} | {'-' if mo is None else f'{mo:.5g}'} | "
              f"{ma:.5g} | {cl} |")
            shown += 1
            if shown >= 25:
                W("| ... | | | | |")
                break
        W("")

    # ---- paired tests ----------------------------------------------------
    W("### Paired tests vs the fixed-prior fail-closed arm (exact sign-flip)\n")
    W("| KAPPA | d NMSE | p | d events | p | d coord-open rate | d HVPs |")
    W("|---|---|---|---|---|---|---|")
    rmap = {r["seed"]: r for r in ref1}
    for k in (1.0, 2.0):
        rs = load_e2(f"mackey_drift_cohg_lr0.003_amh{k:g}_s*.json")
        pairs = [(r, rmap[r["seed"]]) for r in rs if r["seed"] in rmap]
        if len(pairs) < 2:
            continue
        d1 = [a["nmse"] - b["nmse"] for a, b in pairs]
        d2 = [float(a["events"] - b["events"]) for a, b in pairs]
        d3 = [a["coord_open_frac"] - b["coord_open_frac"] for a, b in pairs]
        d4 = [float(a["hvp_total"] - b["hvp_total"]) for a, b in pairs]
        p1, m1 = signflip_p(d1)
        p2, m2 = signflip_p(d2)
        W(f"| {k:g} | {m1:+.6f} | {p1:.4f} | {m2:+.1f} | {p2:.4f} | "
          f"{sum(d3) / len(d3):+.3e} | {sum(d4) / len(d4):+.0f} |")
    W("")

    # =================================================================== E1
    W("## E1 -- teacher/student `kw_drift`, EXACT ground truth, seeds 0-4\n")
    W("A violation here is a genuine failure of the anytime certificate: "
      "`e_t < ||S_t - Shat_t||_F` on any step, checked against a parallel "
      "exact forward-mode recursion in fp64.\n")
    e1 = {}
    for tag, label in [("_fixed_fc0", "fixed prior M_H* = 2.27609, no FC"),
                       ("_fixed_fc1", "fixed prior M_H* = 2.27609, "
                                      "fail-closed"),
                       ("_amh1", "**adaptive envelope KAPPA=1**"),
                       ("_amh2", "**adaptive envelope KAPPA=2**")]:
        p = os.path.join(E1A, f"teacher_kw_drift{tag}.json")
        if not os.path.exists(p):
            p = os.path.join(MIS, f"teacher_kw_drift_x1_fc"
                             f"{tag[-1] if tag.startswith('_fixed') else ''}.json")
            if not os.path.exists(p):
                continue
        e1[label] = json.load(open(p))
    W("| arm | n | violation rate | worst true-err / bound | valid rate | "
      "closure frac | probes / nominal | KW HVPs | final e_t | "
      "final envelope | raises | max M_obs |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for label, rs in e1.items():
        n = len(rs)
        W(f"| {label} | {n} | "
          f"{mstd([r['violation_rate'] for r in rs])[0]:.4f} | "
          f"{max(r['worst_true_over_bound'] for r in rs):.4f} | "
          f"{mstd([r['valid_rate'] for r in rs])[0]:.4f} | "
          f"{(mstd([r['closed_frac'] for r in rs])[0] if rs[0].get('closed_frac') is not None else float('nan')):.4f} | "
          f"{mstd([r['probe_overhead'] for r in rs])[0]:.4f} | "
          f"{mstd([float(r['kw_hvp']) for r in rs])[0]:.0f} | "
          f"{mstd([r['final_e'] for r in rs])[0]:.4g} | "
          f"{(mstd([r.get('adapt_mh_final') for r in rs])[0]):.5g} | "
          f"{(mstd([r.get('adapt_mh_raises') for r in rs])[0]):.1f} | "
          f"{(mstd([r.get('adapt_mobs_max') for r in rs])[0]):.5g} |")
    W("")
    W("### E1 envelope trajectory (seed 0)\n")
    for tag, k in (("_amh1", 1), ("_amh2", 2)):
        p = os.path.join(E1A, f"teacher_kw_drift{tag}.json")
        if not os.path.exists(p):
            continue
        rs = json.load(open(p))
        r0 = [r for r in rs if r["seed"] == 0][0]
        lg = r0.get("adapt_probe_log") or []
        W(f"KAPPA={k}: {len(lg)} probes; envelope "
          f"{r0['adapt_mh_final']:.6g} at the end (floor {E1_FLOOR:.5f}); "
          f"{r0['adapt_mh_raises']} raises; max M_obs "
          f"{r0['adapt_mobs_max']:.6g}; closure fraction "
          f"{r0['closed_frac']:.4f}.\n")
        W("| probe step | envelope before | M_obs | envelope after | closed |")
        W("|---|---|---|---|---|")
        prev, shown = None, 0
        for row in lg:
            t, mb, mo, ma, cl = row
            if prev is not None and ma == prev and shown > 10 and not cl:
                continue
            prev = ma
            W(f"| {t} | {mb:.5g} | {'-' if mo is None else f'{mo:.5g}'} | "
              f"{ma:.5g} | {cl} |")
            shown += 1
            if shown >= 22:
                W("| ... | | | | |")
                break
        W("")

    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
