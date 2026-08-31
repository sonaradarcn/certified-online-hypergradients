"""Appends the explicit answers to the two key questions to unified_metrics.md.

Run AFTER _reanalyze.py (reads the CSVs it produced). Idempotent: the section
is rewritten, never duplicated.
"""
import os, csv, math
from collections import defaultdict

OUT = os.path.dirname(os.path.abspath(__file__))
MARK = "\n## Do the tail-safety claims survive the unified metric?\n"


def rd(name):
    return list(csv.DictReader(open(os.path.join(OUT, name))))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def mstd(v):
    v = [x for x in v if not math.isnan(x)]
    n = len(v)
    if not n:
        return float("nan"), float("nan")
    m = sum(v) / n
    if n < 2:
        return m, float("nan")
    return m, math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))


def s(v, p=2):
    m, sd = mstd(v)
    if not math.isfinite(m):
        return "inf"
    return f"{m:.{p}f}" + ("" if math.isnan(sd) else f"+-{sd:.{p}f}")


def main():
    e4 = rd("unified_metrics_e4.csv")
    e2 = rd("unified_metrics_e2_drift.csv")
    e3 = rd("unified_metrics_e3.csv")

    g4 = defaultdict(list)
    for r in e4:
        g4[r["arm"]].append(r)

    L = [MARK.strip(), ""]
    A = L.append

    A("### Validation of the unified rule\n")
    A(f"Applied post hoc to the E2 traces, the rule reproduces the stored "
      f"`events` counter **exactly on all {len(e2)}/{len(e2)} E2 drift runs** "
      "(zero mismatches). The post-hoc rule is therefore identical to the rule "
      "E2 applied online, and the E3/E4 numbers below are computed with that "
      "same rule rather than with the non-finite-only counter those two "
      "experiments used.\n")

    A("### Q1 -- E4 (GPT-2): do the gated runs stay clean under (a)-(c) while "
      "the ungated degraded seed shows up?\n")
    gat = g4["gpt2_cohg_r0_lr0.001"] + g4["gpt2_cohg_lr0.001"]
    ng = g4["gpt2_cohg_nogate_lr0.001"]
    ml200 = g4["gpt2_hd_lr0.001_ml200"]
    A("**Partly -- and the honest answer is metric-dependent.**\n")
    A("| metric | gated COHG (n=6: cohg + cohg_r0) | ungated (n=3) | ungated "
      "degraded seed 1 | HD meta-lr 200 (n=3) |")
    A("|---|---|---|---|---|")
    A(f"| (a) spikes | {s([f(r['n_spike']) for r in gat])} | "
      f"{s([f(r['n_spike']) for r in ng])} | "
      f"{[r for r in ng if r['seed']=='1'][0]['n_spike']} | "
      f"{s([f(r['n_spike']) for r in ml200])} |")
    A(f"| (b) max-excess | {s([f(r['max_excess']) for r in gat])} | "
      f"{s([f(r['max_excess']) for r in ng])} | "
      f"{f([r for r in ng if r['seed']=='1'][0]['max_excess']):.2f} | "
      f"{s([f(r['max_excess']) for r in ml200])} |")
    A(f"| (c) worst-window PPL | {s([f(r['worst_window_ppl']) for r in gat])} | "
      f"{s([f(r['worst_window_ppl']) for r in ng])} | "
      f"{f([r for r in ng if r['seed']=='1'][0]['worst_window_ppl']):.2f} | inf |")
    A(f"| online PPL | {s([f(r['online_ppl']) for r in gat],3)} | "
      f"{s([f(r['online_ppl']) for r in ng],3)} | "
      f"{f([r for r in ng if r['seed']=='1'][0]['online_ppl']):.3f} | inf |")
    A("")
    A("- **(a) spike count does not transfer to the GPT-2 regime.** Every COHG "
      "run (gated and ungated) records **0 spikes**, because the loss is a "
      "token log-loss around 3 nats and a 10x-median excursion (loss > ~30 nats) "
      "is essentially unreachable without an outright numerical blow-up. Metric "
      "(a) therefore neither vindicates nor indicts the gate on E4; it is "
      "uninformative there. It is *not* vacuous on E4 as a whole: it newly "
      "flags the HD meta-lr=200 arm with "
      f"{s([f(r['n_spike']) for r in ml200],1)} spikes per run, an arm whose "
      "stored `events` was 0 in every seed. So the unified rule strictly "
      "*adds* detections on E4 relative to the non-finite-only counter used in "
      "the paper.\n")
    A("- **(b) and (c) do carry the claim.** The gated runs are tightly "
      f"clustered (max-excess {s([f(r['max_excess']) for r in gat])}, "
      f"worst-window PPL {s([f(r['worst_window_ppl']) for r in gat])}), while "
      "the ungated degraded seed 1 is a clear outlier on both "
      f"(max-excess {f([r for r in ng if r['seed']=='1'][0]['max_excess']):.2f}, "
      f"worst-window PPL {f([r for r in ng if r['seed']=='1'][0]['worst_window_ppl']):.2f}, "
      f"i.e. {f([r for r in ng if r['seed']=='1'][0]['worst_window_ppl'])/max(f(r['worst_window_ppl']) for r in gat):.1f}x "
      "the worst gated run). Ungated seed 2 is also flagged by max-excess "
      f"({f([r for r in ng if r['seed']=='2'][0]['max_excess']):.2f}) even though "
      "its final online PPL (20.23) looks benign -- a transient the aggregate "
      "PPL hides.\n")
    A("- **Conclusion.** The tail-safety narrative survives on E4, but it must "
      "be stated in terms of *tail magnitude* (max-excess / worst-window), not "
      "in terms of a spike **count**. On GPT-2 the spike count is 0 for both "
      "gated and ungated COHG and so cannot be the evidence. The paper should "
      "not claim 'no instability events' as differentiating on E4 -- for that "
      "arm the honest statement is that no COHG variant produced a spike or a "
      "non-finite loss, and the gate's benefit is a 14x smaller worst window "
      "and a 2.1x smaller max-excess.\n")

    A("### Q2 -- E3 ewc1000: COHG vs HD vs fixed under (b) max-excess\n")
    A("**Not computable.** `code/experiments/e3_continual.py` never persists a "
      "per-step loss trace (saved keys: `acc_matrix, avg_acc, bwt, events, "
      "gate_open_frac, coord_open_frac, hvp_total, lam_hist, wall_s`), and no "
      "other E3 artifact directory (`e3_cal`, `e3_prefreeze`, `e3_probe`, "
      "`e3_unfair`, `e3_smoke`) stores one either; the run logs record only "
      "per-task accuracies. Metric (b) -- and (a) and (c) -- would require "
      "re-running E3 with loss logging enabled.\n")
    A("What the stored data *does* support, for the ewc0=1000 arms (n=10 each):\n")
    g3 = defaultdict(list)
    for r in e3:
        g3[r["arm"]].append(r)
    A("| arm | avg_acc | BWT | non-finite events | seeds ever triggering | "
      "collapse rate (avg_acc<0.15) | worst-seed avg_acc | max forgetting |")
    A("|---|---|---|---|---|---|---|---|")
    for a in ("cifar100_cohg_lr0.05_ewc1000", "cifar100_hd_lr0.05_ewc1000",
              "cifar100_fixed_lr0.05_ewc1000",
              "cifar100_cohg_nogate_lr0.05_ewc1000"):
        rs = g3[a]
        ev = [f(r["events_stored"]) for r in rs]
        A(f"| {a} | {s([f(r['avg_acc']) for r in rs],4)} | "
          f"{s([f(r['bwt']) for r in rs],4)} | {s(ev,1)} | "
          f"{sum(1 for x in ev if x>0)}/{len(rs)} | "
          f"{sum(f(r['collapse']) for r in rs)/len(rs):.2f} | "
          f"{min(f(r['avg_acc']) for r in rs):.4f} | "
          f"{s([f(r['max_forget']) for r in rs],4)} |")
    A("")
    A("- On the mis-set EWC regime (ewc0=1000) the *ordering under every "
      "computable stability proxy is HD first*: HD has 0/10 seeds with a "
      "non-finite loss, the highest mean accuracy (0.3805+-0.0219) and the "
      "smallest spread, while COHG (0.3124+-0.1150, 5/10 seeds triggering, "
      "2/10 collapsed) and fixed (0.2900+-0.1329, 5/10 triggering, 3/10 "
      "collapsed) are both unstable. COHG beats fixed on the mean (+0.022) and "
      "on collapse rate (0.20 vs 0.30) but the gap is far inside the seed "
      "spread.\n")
    A("- The gate is therefore **not** what rescues this regime; it is the one "
      "E3 setting where an ungated first-order baseline (HD) dominates COHG on "
      "both accuracy and stability. Any claim that COHG is the tail-safe "
      "method at ewc1000 is not supported by the stored artifacts.\n")

    A("### Additional finding: the event counter misses a whole failure mode\n")
    A("`cifar100_cohg_nogate_lr0.4_ewc10` has **0 non-finite events in all 10 "
      "seeds** yet collapses in **7/10** (avg_acc 0.1393+-0.0185, i.e. near "
      "chance for 10-way task-IL). A stability metric built only on non-finite "
      "losses -- and, on E3, that is all that exists -- scores this arm as "
      "perfectly stable. This is the strongest single argument in the data for "
      "reporting a degradation metric that is not an event count.\n")

    A("### E2 under the unified metric: the gate is not uniformly best\n")
    g2 = defaultdict(list)
    for r in e2:
        g2[r["arm"]].append(r)
    A("| arm | spikes | max-excess | NMSE |")
    A("|---|---|---|---|")
    for a in ("lorenz_drift_fixed_lr0.03", "lorenz_drift_cohg_lr0.03",
              "lorenz_drift_cohg_nogate_lr0.03",
              "mackey_drift_fixed_lr0.03", "mackey_drift_cohg_lr0.03",
              "mackey_drift_cohg_nogate_lr0.03"):
        rs = g2[a]
        A(f"| {a} | {s([f(r['n_spike']) for r in rs])} | "
          f"{s([f(r['max_excess']) for r in rs])} | "
          f"{s([f(r['nmse']) for r in rs],4)} |")
    A("")
    A("COHG sits **between** fixed and ungated on both spike count and "
      "max-excess at lr0=0.03: it buys a large NMSE improvement over fixed "
      "(0.0048 vs 0.0086 on Lorenz) at the cost of roughly 2x the spikes and "
      "2x the max-excess, while removing the gate costs another ~3x on spikes "
      "and ~11x on max-excess. Note that at lr0=0.03 the ungated arm also has "
      "the *best* mean NMSE (0.0016 Lorenz, 0.0028 Mackey), so on these two "
      "arms the gate trades accuracy for tail control; the ungated arm's "
      "liability shows up at lr0=0.003 on Mackey-Glass, where one seed diverges "
      "and drags the arm to NMSE 5.98+-18.92 (max-excess 1.07e7) against "
      "COHG's 0.0150+-0.0023. The defensible claim is 'the gate removes most "
      "of the ungated tail at a modest cost in mean accuracy', not 'the gate "
      "is as safe as a fixed step size' and not 'the gate is free'.\n")

    p = os.path.join(OUT, "unified_metrics.md")
    txt = open(p, encoding="utf-8").read()
    i = txt.find(MARK.strip())
    if i >= 0:
        txt = txt[:i]
    open(p, "w", encoding="utf-8").write(txt.rstrip() + "\n\n" + "\n".join(L) + "\n")
    print("appended key-questions section to", p)

    # ---------------------------------------------------- censored highlights
    ce = rd("censored_e2_drift.csv")
    gc = defaultdict(list)
    for r in ce:
        gc[r["arm"]].append(r)
    H = ["## Highlights -- does recovery confound method identity?", ""]
    trig_arms = {a: rs for a, rs in gc.items()
                 if any(int(r["triggered"]) for r in rs)}
    ntrig_runs = sum(int(r["triggered"]) for r in ce)
    H.append(
        f"1. **On E2 the confound is small and one-sided.** Only "
        f"{ntrig_runs}/{len(ce)} drift runs ever trigger recovery, and they are "
        f"confined to {len(trig_arms)} baseline arms "
        f"({', '.join('`'+a+'`' for a in sorted(trig_arms))}). "
        "**No COHG, cohg_nogate, cohg_r0, fixed, fmd or hdm run at lr0 in "
        "{0.003, 0.03} ever produces a non-finite loss**, so for every arm the "
        "paper uses in its headline comparison the recovery heuristic is never "
        "invoked and the censored numbers are identical to the reported ones. "
        "Method identity is *not* confounded by recovery for COHG vs fixed vs "
        "ungated on E2.\n")
    H.append(
        "2. **Where it does fire, it fires almost immediately.** Every "
        "triggering seed trips within the first ~600 of 12000 steps "
        "(t = 33, 34, 40, 54, 91, 106, 144, 147, 194, 402, 419, 572), i.e. "
        ">=95% of those runs are post-recovery. For `hd_scalar` (3/10 Lorenz, "
        "5/10 Mackey) and `tfmd` the reported end-of-run numbers therefore "
        "describe the backed-off controller, not the method as specified. Those "
        "baselines' results should be read as 'diverges within ~200 steps, then "
        "runs at a halved step size', which is a stronger statement than a "
        "large NMSE.\n")
    H.append(
        "3. **Divergence without a trigger is the common failure mode.** "
        "`lorenz_drift_hd_lr0.003` triggers in 1/10 seeds but reaches "
        "pre-trigger NMSE ~1e34; `lorenz_drift_hd_lr0.03` likewise. The loss "
        "grows to astronomical but still-finite values, so the non-finite "
        "detector never fires. This is the E2 analogue of the E3 "
        "`cohg_nogate_lr0.4_ewc10` collapse (7/10 seeds near chance, 0 events) "
        "and it is why an event count alone cannot carry a stability claim.\n")
    H.append(
        "4. **On E3 the confound is real and symmetric between COHG and "
        "fixed.** At ewc0=1000, 5/10 seeds trigger for COHG (107.2 triggers per "
        "seed averaged over all 10 seeds, i.e. 214 per triggering seed) and "
        "5/10 for fixed (167.3, i.e. 335 per triggering seed). Triggering seeds end far "
        "worse than clean ones within the same arm (COHG 0.2528+-0.1431 vs "
        "0.3720+-0.0200; fixed 0.1995+-0.1370 vs 0.3806+-0.0212), so the arm "
        "means are a mixture of two populations rather than a method effect. "
        "Because E3 stores no loss trace, the pre-trigger prefix cannot be "
        "isolated -- the clean-seed subgroup means above are the closest "
        "available substitute, and on them COHG and fixed are "
        "indistinguishable (0.3720 vs 0.3806).\n")
    q = os.path.join(OUT, "censored_analysis.md")
    t2 = open(q, encoding="utf-8").read()
    j = t2.find(H[0])
    if j >= 0:
        t2 = t2[:j]
    open(q, "w", encoding="utf-8").write(t2.rstrip() + "\n\n" + "\n".join(H) + "\n")
    print("appended highlights to", q)


if __name__ == "__main__":
    main()
