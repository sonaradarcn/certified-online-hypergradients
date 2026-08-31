"""Round-4 experiment 1 analysis: LATE AMPLITUDE SHIFT (`--scale-shift F`).

Reads results/e2_shift/*.json (and results/e2_shift/pilot/*.json for the pilot
table) and writes results/reanalysis/round4_shift.md.

`--seeds a,b,c` (or `--seeds 1-9`) restricts the EVALUATION set to those seeds;
`--out NAME.md` names the output file.  Seed 0 is the factor-selection pilot
seed, so the confirmatory analysis is run with `--seeds 1-9`.  The pilot table
itself is always the seed-0 pilot sweep and is not affected by the filter.

Everything is ddof=1, every paired test is an EXACT two-sided sign-flip
permutation test over all 2^n sign assignments of the per-seed differences.
Per-segment NMSE is normalized by THAT SEGMENT's target variance (recorded in
the run's `seg_stats`), so the three thirds are comparable even though the
middle third's amplitude is F times larger.
"""
from __future__ import annotations

import glob
import itertools
import json
import math
import os
import sys
from collections import defaultdict, deque

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SHIFT = os.path.join(ROOT, "results", "e2_shift")
PILOT = os.path.join(SHIFT, "pilot")
CTL = os.path.join(ROOT, "results", "e2_controls")
OUT = os.path.join(HERE, "round4_shift.md")
SEEDS = None  # None = all seeds; otherwise a set of evaluation seeds

from _reanalyze import unified_metrics  # noqa: E402


# --------------------------------------------------------------- helpers ---
def mstd(v):
    v = [x for x in v if x is not None and not (isinstance(x, float)
                                                and math.isnan(x))]
    if not v:
        return float("nan"), float("nan"), 0
    n = len(v)
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else float("nan")
    return m, sd, n


def g1(x, p=4):
    """Compact rendering that survives 1e33 blow-ups."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    if not math.isfinite(x):
        return "inf"
    if x != 0 and (abs(x) >= 1e4 or abs(x) < 1e-4):
        return f"{x:.3g}"
    return f"{x:.{p}f}"


def f(m, s, p=4):
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "nan"
    if not math.isfinite(m):
        return "inf"
    if not math.isfinite(s):
        return g1(m, p)
    return f"{g1(m, p)}+-{g1(s, p)}"


def med(v):
    v = sorted(x for x in v if x is not None
               and not (isinstance(x, float) and math.isnan(x)))
    if not v:
        return float("nan")
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def signflip_p(d):
    n = len(d)
    obs = abs(sum(d) / n)
    cnt = 0
    for signs in itertools.product([1, -1], repeat=n):
        if abs(sum(s * x for s, x in zip(signs, d)) / n) >= obs - 1e-15:
            cnt += 1
    return cnt / 2 ** n, sum(d) / n


def spikes_in(losses, lo, hi):
    """Unified spike rule run over the WHOLE trace, counted in [lo, hi)."""
    win = deque(maxlen=500)
    out = []
    for t, x in enumerate(losses):
        if not math.isfinite(x):
            out.append(t)
            continue
        if len(win) >= 100:
            w = sorted(win)
            if x > 10.0 * w[len(w) // 2]:
                out.append(t)
        win.append(x)
    return [t for t in out if lo <= t < hi], out


def seg_nmse(d):
    return [s["nmse"] for s in d["seg_stats"]]


def arm_of(d):
    m = d["method"]
    if m == "cohg":
        return "cohg_fc1" if d.get("fail_closed") else "cohg_fc0"
    if m == "fixed":
        return f"fixed_lr{d['lr0']:g}"
    return m


def load(dirpath, seed_filter=True):
    rows = []
    for p in sorted(glob.glob(os.path.join(dirpath, "*.json"))):
        d = json.load(open(p))
        if seed_filter and SEEDS is not None and d.get("seed") not in SEEDS:
            continue
        rows.append(d)
    return rows


# ------------------------------------------------------------ F=1 control ---
def f1_control():
    """Per-segment NMSE of the UNSHIFTED reference arms, reconstructed from
    results/e2_controls (whose runs predate the seg_stats logging).  The stream
    is deterministic, so the per-segment target variance is recomputed exactly
    by replaying it."""
    sys.path.insert(0, os.path.join(ROOT, "code", "experiments"))
    import torch
    import data as D  # noqa: E402
    cache = {}

    def seg_var(seed, steps=12000):
        if seed in cache:
            return cache[seed]
        series = D.mackey_glass_drift(25_000, seed=seed)
        st = D.OrderedWindowStream(series, steps, window=20, batch_size=64,
                                   seed=seed + 1, device="cpu")
        seg = 25_000 // 3
        n_win, T = st.n, steps
        b = [next(t for t in range(T)
                  if int(t / (T - 1) * (n_win - 1)) >= tg) for tg in (seg, 2 * seg)]
        # lightweight exact replay: the generator is consumed ONLY by the
        # randint, so skipping the input-window stack keeps the draw identical
        acc = [[0.0, 0.0, 0] for _ in range(3)]
        for t in range(steps):
            center = int(t / max(st.T - 1, 1) * (st.n - 1))
            lo = max(0, center - st.band)
            hi = min(st.n, center + 1)
            idx = lo + torch.randint(0, max(hi - lo, 1), (st.bs,),
                                     generator=st.gen)
            y = st.x[idx + st.window]
            i = 0 if t < b[0] else (1 if t < b[1] else 2)
            acc[i][0] += float(y.sum())
            acc[i][1] += float((y ** 2).sum())
            acc[i][2] += y.numel()
        var = []
        for su, sq, n in acc:
            mu = su / n
            var.append(sq / n - mu ** 2)
        cache[seed] = (b, var)
        return cache[seed]

    out = defaultdict(list)
    pats = {
        "cohg_fc0": "mackey_drift_cohg_lr0.003_mh5_fc0_s*.json",
        "cohg_fc1": "mackey_drift_cohg_lr0.003_mh5_fc1_s*.json",
        "cohg_nogate": "mackey_drift_cohg_nogate_lr0.003_a0.4_s*.json",
        "absgate": "mackey_drift_absgate_lr0.003_a0.4_s*.json",
    }
    for arm, pat in pats.items():
        for p in sorted(glob.glob(os.path.join(CTL, pat))):
            d = json.load(open(p))
            if SEEDS is not None and d["seed"] not in SEEDS:
                continue
            b, var = seg_var(d["seed"])
            L = d["losses"]
            segs = []
            for i, (lo, hi) in enumerate([(0, b[0]), (b[0], b[1]),
                                          (b[1], len(L))]):
                fin = [x for x in L[lo:hi] if math.isfinite(x)]
                ml = sum(fin) / len(fin) if fin else float("nan")
                segs.append(ml * 2.0 / max(var[i], 1e-12))
            out[arm].append(dict(seed=d["seed"], nmse=d["nmse"],
                                 events=d["events"], seg=segs,
                                 bounds=b))
    return out


# ----------------------------------------------------------------- main ----
def main():
    runs = load(SHIFT)
    runs = [d for d in runs if d.get("seg_stats")]
    Fs = sorted({d.get("scale_shift") for d in runs if d.get("scale_shift")})
    by = defaultdict(list)
    for d in runs:
        by[(d.get("scale_shift"), arm_of(d))].append(d)
    for k in by:
        by[k].sort(key=lambda d: d["seed"])

    L = []
    W = L.append
    W("# Round-4 experiment 1: certified re-adaptation under a LATE amplitude shift\n")
    if SEEDS is not None:
        W("**Evaluation seed set: " + ", ".join(str(x) for x in sorted(SEEDS))
          + f" (n = {len(SEEDS)}).** Seed 0 is the factor-selection pilot "
          "seed and is excluded from every statistic below except the pilot "
          "table, which is the seed-0 sweep itself. Exact two-sided sign-flip "
          f"permutation tests enumerate all 2^{len(SEEDS)} = "
          f"{2 ** len(SEEDS)} sign assignments, so the smallest attainable "
          f"p-value is 2/{2 ** len(SEEDS)} = {2 / 2 ** len(SEEDS):.4f}. "
          "All spreads are ddof=1.\n")

    # ---- pilot -----------------------------------------------------------
    pil = load(PILOT, seed_filter=False)
    if pil:
        W("## Pilot (seed 0, `fixed` arm at the mis-set lr0 = 0.003)\n")
        W("`--scale-shift F` multiplies the middle third of the series "
          "(inputs AND targets) by F. Segment NMSE is normalized by that "
          "segment's own target variance, so the three thirds are directly "
          "comparable. Stream-step boundaries `t1 = 4004`, `t2 = 8007` "
          "(identical to the tau regime switches).\n")
        W("| F | seg-0 NMSE | seg-1 NMSE (shifted) | seg-2 NMSE | overall NMSE"
          " | events | spikes in [t1,t1+500) | spikes in [t2,t2+500) | "
          "non-finite | max loss |")
        W("|---|---|---|---|---|---|---|---|---|---|")
        for d in sorted(pil, key=lambda x: (x.get("scale_shift") or 1.0)):
            F = d.get("scale_shift") or 1.0
            b = d["seg_bounds"]
            sn = seg_nmse(d)
            sp1, allsp = spikes_in(d["losses"], b[0], b[0] + 500)
            sp2, _ = spikes_in(d["losses"], b[1], b[1] + 500)
            fin = [x for x in d["losses"] if math.isfinite(x)]
            W(f"| {F:g} | {sn[0]:.4f} | {sn[1]:.4f} | {sn[2]:.4f} | "
              f"{d['nmse']:.4f} | {d['events']} | {len(sp1)} | {len(sp2)} | "
              f"{len(d['losses']) - len(fin)} | {max(fin):.3g} |")
        W("")

    if not runs:
        open(OUT, "w", encoding="utf-8").write("\n".join(L))
        print("pilot only")
        return

    # ---- F=1 control -----------------------------------------------------
    W("## F = 1 control (no amplitude shift) -- reconstructed from "
      "`results/e2_controls`\n")
    ctl = f1_control()
    if ctl:
        W("| arm | n | overall NMSE | events | seg-0 NMSE | seg-1 NMSE | "
          "seg-2 NMSE |")
        W("|---|---|---|---|---|---|---|")
        for arm in ("cohg_fc0", "cohg_fc1", "cohg_nogate", "absgate"):
            rs = ctl.get(arm, [])
            if not rs:
                continue
            W(f"| {arm} | {len(rs)} | {f(*mstd([r['nmse'] for r in rs])[:2])} "
              f"| {f(*mstd([float(r['events']) for r in rs])[:2], 1)} | "
              + " | ".join(f(*mstd([r['seg'][i] for r in rs])[:2])
                           for i in range(3)) + " |")
        W("")

    # ---- per-F tables ----------------------------------------------------
    ARM_ORDER = ["fixed_lr0.003", "fixed_lr0.01", "fixed_lr0.03",
                 "fixed_lr0.1", "fixed_lr0.3", "hd", "cohg_fc0", "cohg_fc1",
                 "cohg_nogate", "absgate"]
    summary = {}
    for F in Fs:
        W(f"## F = {F:g}\n")
        b = by[(F, "cohg_fc0")][0]["seg_bounds"] if by[(F, "cohg_fc0")] \
            else next(iter(by[(F, a)] for a in ARM_ORDER if by[(F, a)]))[0]["seg_bounds"]
        W(f"Segment boundaries (stream steps): t1 = {b[0]}, t2 = {b[1]}.\n")
        W("### (b) per-segment NMSE, events, and adaptation timing\n")
        W("`events outside transition` excludes the 500 steps after each "
          "boundary, where the amplitude jump itself (a factor F^2 in the "
          "loss) trips the running-median spike rule for every arm. "
          "`diverged` counts seeds with overall NMSE > 1.\n")
        W("| arm | n | overall NMSE | median | diverged | seg-0 NMSE | "
          "**seg-1 NMSE (shifted)** | seg-2 NMSE | events | events outside "
          "transition | ev.out seg0 | ev.out seg1 | ev.out seg2 | "
          "coord-open seg0/seg1/seg2 | open steps in seg1 / seg2 "
          "(seeds with >=1) | HVPs |")
        W("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        rows = {}
        for arm in ARM_ORDER:
            rs = by[(F, arm)]
            if not rs:
                continue
            sn = [[r["seg_stats"][i]["nmse"] for r in rs] for i in range(3)]
            ev = [float(r["events"]) for r in rs]
            evx = []
            evseg = [[], [], []]
            for r in rs:
                _, allsp = spikes_in(r["losses"], 0, 0)
                bb = r["seg_bounds"]
                keep = [t for t in allsp
                        if not (bb[0] <= t < bb[0] + 500
                                or bb[1] <= t < bb[1] + 500)]
                evx.append(float(len(keep)))
                T = len(r["losses"])
                lims = [(0, bb[0]), (bb[0], bb[1]), (bb[1], T)]
                for i, (lo, hi) in enumerate(lims):
                    evseg[i].append(float(sum(1 for t in keep if lo <= t < hi)))
            cof = []
            for i in range(3):
                v = [r["seg_stats"][i]["coord_open_frac"] for r in rs
                     if r["seg_stats"][i]["coord_open_frac"] is not None]
                cof.append(sum(v) / len(v) if v else None)
            fo, lo_, n1, n2 = [], [], [], []
            for r in rs:
                gs = r.get("gate_open_steps") or []
                bb = r["seg_bounds"]
                aft = [t for t in gs if t >= bb[0]]
                fo.append(aft[0] if aft else None)
                lo_.append(gs[-1] if gs else None)
                n1.append(sum(1 for t in gs if bb[0] <= t < bb[1]))
                n2.append(sum(1 for t in gs if t >= bb[1]))
            nm = [r["nmse"] for r in rs]
            rows[arm] = dict(nmse=nm, seg=sn, ev=ev, evx=evx, evseg=evseg,
                             cof=cof, first_open=fo, last_open=lo_, n1=n1,
                             n2=n2, runs=rs)
            cofs = "/".join("-" if c is None else f"{c:.2e}" for c in cof)
            gated = rs[0].get("coord_open_counts") is not None
            opn = (f"{sum(n1)} ({sum(1 for x in n1 if x)}/{len(n1)}) / "
                   f"{sum(n2)} ({sum(1 for x in n2 if x)}/{len(n2)})"
                   if gated else "-")
            W(f"| {arm} | {len(rs)} | {f(*mstd(nm)[:2])} | {g1(med(nm))} | "
              f"{sum(1 for x in nm if x > 1)} | "
              + " | ".join(f(*mstd(sn[i])[:2]) for i in range(3))
              + f" | {f(*mstd(ev)[:2], 1)} | {f(*mstd(evx)[:2], 1)} | "
              + " | ".join(f(*mstd(evseg[i])[:2], 1) for i in range(3))
              + f" | {cofs} | {opn} | {med([r['hvp_total'] for r in rs]):.0f} |")
        summary[F] = rows
        W("")

        # post-hoc best fixed per segment
        W("### (b') post-hoc best FIXED learning rate per segment (oracle)\n")
        W("| segment | best fixed lr | its seg NMSE | COHG seg NMSE | "
          "fixed lr0=0.003 seg NMSE |")
        W("|---|---|---|---|---|")
        for i in range(3):
            best, bv = None, float("inf")
            for arm in ARM_ORDER:
                if not arm.startswith("fixed_") or arm not in rows:
                    continue
                m = mstd(rows[arm]["seg"][i])[0]
                if m < bv:
                    best, bv = arm, m
            cm = mstd(rows["cohg_fc0"]["seg"][i])[0] if "cohg_fc0" in rows else float("nan")
            fm = mstd(rows["fixed_lr0.003"]["seg"][i])[0] if "fixed_lr0.003" in rows else float("nan")
            W(f"| {i} | {best} | {bv:.4f} | {cm:.4f} | {fm:.4f} |")
        W("")

        # ---- (a) re-opening detail -------------------------------------
        W("### (a) does COHG re-open at/after the shift?\n")
        for arm in ("cohg_fc0", "cohg_fc1"):
            if arm not in rows:
                continue
            rr = rows[arm]
            W(f"**{arm}** -- {sum(1 for x in rr['n1'] if x)} of "
              f"{len(rr['n1'])} seeds re-open at least once in segment 1 "
              f"(the shifted third) and {sum(1 for x in rr['n2'] if x)} of "
              f"{len(rr['n2'])} in segment 2 (after the reversal); "
              f"{sum(rr['n1'])} and {sum(rr['n2'])} open STEPS in total.\n")
            W("Per-seed gate-opening steps and the lambda response "
              "(lam sampled every 10 steps):\n")
            W("| seed | all open steps | opens in seg0/seg1/seg2 | "
              "lam at t1-10 (min..max over the 6 groups) | lam at t1+200 | "
              "lam at t2+200 | lam final | net d(log lr) in seg1 | closed "
              "steps seg0/1/2 |")
            W("|---|---|---|---|---|---|---|---|---|")
            for r in rows[arm]["runs"]:
                gs = r.get("gate_open_steps") or []
                bb = r["seg_bounds"]
                c = [sum(1 for t in gs if lo <= t < hi)
                     for lo, hi in [(0, bb[0]), (bb[0], bb[1]),
                                    (bb[1], r["steps"])]]
                lh = {row[0]: row[1:] for row in r["lam_hist"]}

                def lam_at(t):
                    ks = [k for k in lh if k <= t]
                    return lh[max(ks)] if ks else None

                def rng(v):
                    return "-" if v is None else f"[{min(v):.3f}, {max(v):.3f}]"
                a1, a2, a3 = lam_at(bb[0] - 10), lam_at(bb[0] + 200), lam_at(bb[1] + 200)
                af = lam_at(r["steps"])
                net = ("-" if (a1 is None or a2 is None)
                       else f"{max(x - y for x, y in zip(a2, a1)):+.4f} / "
                            f"{min(x - y for x, y in zip(a2, a1)):+.4f}")
                cs = "/".join(str(r["seg_stats"][i]["closed_steps"])
                              for i in range(3))
                if len(gs) <= 18:
                    gss = ",".join(str(x) for x in gs)
                else:
                    late = [t for t in gs if t >= bb[0]]
                    gss = (",".join(str(x) for x in gs[:6]) + ",... ("
                           + str(len(gs)) + " total; first >= t1: "
                           + (str(late[0]) if late else "none") + ", last: "
                           + str(gs[-1]) + ")")
                W(f"| {r['seed']} | {gss or 'none'} | {c[0]}/{c[1]}/{c[2]} | "
                  f"{rng(a1)} | {rng(a2)} | {rng(a3)} | {rng(af)} | {net} | "
                  f"{cs} |")
            W("")

        # ---- (c) transferred static threshold --------------------------
        W("### (c) the TRANSFERRED static threshold (absgate, thr = "
          "0.05806520209, no recalibration)\n")
        if "absgate" in rows:
            W("`top-400 |ghat|` is the 400th largest |ghat_{t,j}| the run "
              "recorded, the order statistic the frozen threshold "
              "0.05806520209 is read against.\n")
            W("| seed | coord-open frac seg0 | seg1 | seg2 | steps-with-open "
              "seg0/1/2 | NMSE seg0/seg1/seg2 | top-400 |ghat| | events |")
            W("|---|---|---|---|---|---|---|---|")
            tops = []
            for r in rows["absgate"]["runs"]:
                ss = r["seg_stats"]
                gt = r.get("ghat_abs_top")
                tv = min(gt) if gt else float("nan")
                tops.append(tv)
                W(f"| {r['seed']} | "
                  + " | ".join(f"{ss[i]['coord_open_frac']:.3e}"
                               for i in range(3))
                  + " | " + "/".join(str(ss[i]["steps_open"]) for i in range(3))
                  + " | " + "/".join(f"{ss[i]['nmse']:.4f}" for i in range(3))
                  + f" | {g1(tv)} | {r['events']} |")
            W("")
            W(f"Median top-400 |ghat| over the evaluation seeds: "
              f"**{g1(med(tops))}**, against the frozen threshold 0.05807 "
              f"(ratio {med(tops) / 0.05806520209:.3g}). Range "
              f"{g1(min(tops))} to {g1(max(tops))}.\n")

        # ---- paired tests ----------------------------------------------
        W("### Paired comparisons vs `cohg_fc0` (exact sign-flip permutation; "
          "negative delta = the other arm is BETTER)\n")
        W("| arm | d overall NMSE | p | d seg-1 NMSE | p | d events | p | "
          "d ev.out | p |")
        W("|---|---|---|---|---|---|---|---|---|")
        ref = rows.get("cohg_fc0")
        if ref:
            for arm in ARM_ORDER:
                if arm == "cohg_fc0" or arm not in rows:
                    continue
                o = rows[arm]
                n = min(len(ref["nmse"]), len(o["nmse"]))
                if n < 2:
                    continue
                d1 = [o["nmse"][i] - ref["nmse"][i] for i in range(n)]
                d2 = [o["seg"][1][i] - ref["seg"][1][i] for i in range(n)]
                d3 = [o["ev"][i] - ref["ev"][i] for i in range(n)]
                d4 = [o["evx"][i] - ref["evx"][i] for i in range(n)]
                p1, m1 = signflip_p(d1)
                p2, m2 = signflip_p(d2)
                p3, m3 = signflip_p(d3)
                p4, m4 = signflip_p(d4)
                W(f"| {arm} | {m1:+.5f} | {p1:.4f} | {m2:+.5f} | {p2:.4f} | "
                  f"{m3:+.1f} | {p3:.4f} | {m4:+.1f} | {p4:.4f} |")
        W("")

        # ---- degradation metrics ---------------------------------------
        W("### Unified degradation metrics (whole trace)\n")
        W("| arm | worst-window mean | max-excess (max finite / median) | "
          "non-finite steps | cert viol / checked | cert max ratio |")
        W("|---|---|---|---|---|---|")
        for arm in ARM_ORDER:
            if arm not in rows:
                continue
            um = [unified_metrics(r["losses"]) for r in rows[arm]["runs"]]
            ww = mstd([u["worst_window_mean"] for u in um])
            me = mstd([u["max_excess"] for u in um])
            nf = sum(u["n_nonfinite"] for u in um)
            cv = sum((r.get("cert_violations") or 0)
                     for r in rows[arm]["runs"])
            cc = sum((r.get("cert_checked") or 0) for r in rows[arm]["runs"])
            cr = [r.get("cert_max_ratio") for r in rows[arm]["runs"]
                  if r.get("cert_max_ratio") is not None]
            cvs = f"{cv} / {cc}" if cc else "-"
            crs = f"{max(cr):.4f}" if cr else "-"
            W(f"| {arm} | {f(*ww[:2])} | {f(*me[:2], 2)} | {nf} | {cvs} | "
              f"{crs} |")
        W("")

    # ---- global re-opening tally over the evaluation seed set -----------
    W("## Gate re-opening tally over the evaluation seed set\n")
    W("An opening counts as a re-opening when its stream step is at or after "
      "t1, the step at which the first shifted batch arrives. Openings at "
      "steps below t1 are in segment 0 by construction.\n")
    W("| F | arm | n runs | openings at step >= t1 | seeds with >=1 | "
      "openings at step >= t2 | all late opening steps |")
    W("|---|---|---|---|---|---|---|")
    tot_runs = tot_late = 0
    for F in Fs:
        for arm in ("cohg_fc0", "cohg_fc1"):
            rs = by[(F, arm)]
            if not rs:
                continue
            late, late2, seeds_hit = [], 0, 0
            for r in rs:
                gs = r.get("gate_open_steps") or []
                bb = r["seg_bounds"]
                a = [t for t in gs if t >= bb[0]]
                if a:
                    seeds_hit += 1
                late += [(r["seed"], t) for t in a]
                late2 += sum(1 for t in gs if t >= bb[1])
            tot_runs += len(rs)
            tot_late += len(late)
            steps = ", ".join(f"s{sd}:{t}" for sd, t in late) or "none"
            W(f"| {F:g} | {arm} | {len(rs)} | {len(late)} | {seeds_hit}/"
              f"{len(rs)} | {late2} | {steps} |")
    W("")
    W(f"**Total: {tot_late} opening(s) at or after the shift across "
      f"{tot_runs} gated runs.**\n")

    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", OUT)


def _parse_seeds(spec):
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


if __name__ == "__main__":
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == "--seeds":
            SEEDS = _parse_seeds(argv[i + 1])
            i += 2
        elif argv[i] == "--out":
            OUT = (argv[i + 1] if os.path.isabs(argv[i + 1])
                   else os.path.join(HERE, argv[i + 1]))
            i += 2
        else:
            raise SystemExit(f"unknown argument {argv[i]!r}")
    main()
