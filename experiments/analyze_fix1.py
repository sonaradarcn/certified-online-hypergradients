"""Recompute paper tables from raw artifacts after the fix-1 campaign.

Outputs (stdout, ready to transcribe into LaTeX):
  1. tab:e2  — mackey_drift @ lr0=0.003, ALL seeds on disk, mean+-std,
               incl. new arms hdm/tfmd/fmd/cohg_ogd and per-method HVPs/step.
  2. permutation tests — two-sided exact sign-flip on per-seed differences,
               favorable AND unfavorable pairs.
  3. tab:e4  — results/e4_v2, mean+-std over seeds for PPL/events/GB/wall,
               plus gate_open_frac for cohg arms.
Usage:  python analyze_fix1.py [--e2-only|--e4-only]
"""
from __future__ import annotations

import glob
import itertools
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")


def load(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        try:
            out.append(json.load(open(f)))
        except Exception as e:  # corrupt artifact: report, skip
            print(f"  [skip corrupt] {f}: {e}")
    return out


def mstd(vals):
    n = len(vals)
    m = sum(vals) / n
    s = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return m, s, n


def perm_p(diffs):
    """Two-sided exact sign-flip permutation p for mean of paired diffs."""
    n = len(diffs)
    obs = abs(sum(diffs))
    cnt = 0
    for signs in itertools.product((1, -1), repeat=n):
        if abs(sum(s * d for s, d in zip(signs, diffs))) >= obs - 1e-12:
            cnt += 1
    return cnt / 2 ** n


def e2_table():
    print("=" * 70)
    print("tab:e2  mackey_drift lr0=0.003 (mis-set), 12k steps")
    print("=" * 70)
    arms = [
        ("oracle-stable-fixed", None),   # resolved from fixed grid below
        ("hd", f"{R}/e2/mackey_drift_hd_lr0.003_s*.json"),
        ("hdm", f"{R}/e2/mackey_drift_hdm_lr0.003_s*.json"),
        ("tfmd", f"{R}/e2/mackey_drift_tfmd_lr0.003_s*.json"),
        ("fmd", f"{R}/e2/mackey_drift_fmd_lr0.003_s*.json"),
        ("cohg_nogate", f"{R}/e2/mackey_drift_cohg_nogate_lr0.003_s*.json"),
        ("cohg (sign)", f"{R}/e2/mackey_drift_cohg_lr0.003_s*.json"),
        ("cohg_r0", f"{R}/e2/mackey_drift_cohg_r0_lr0.003_s*.json"),
        ("cohg_ogd", f"{R}/e2/mackey_drift_cohg_ogd_lr0.003_s*.json"),
    ]
    per_seed = {}
    for name, pat in arms:
        if pat is None:
            continue
        runs = load(pat)
        if not runs:
            print(f"{name:22s}  -- no artifacts yet --")
            continue
        nm = [r["nmse"] for r in runs]
        ev = [r["events"] for r in runs]
        hv = [r["hvp_total"] / r["steps"] for r in runs]
        per_seed[name] = {r["seed"]: r for r in runs}
        mn, sn, n = mstd(nm)
        me, se, _ = mstd(ev)
        mh = sum(hv) / len(hv)
        med = sorted(nm)[len(nm) // 2]
        print(f"{name:22s} n={n:2d} NMSE {mn:.4g}+-{sn:.4g} (median {med:.4g}) "
              f"events {me:.1f}+-{se:.1f}  HVP/step {mh:.1f}")
        print(f"{'':22s}    per-seed NMSE: {[round(v,4) for v in sorted(nm)]}")
        print(f"{'':22s}    per-seed events: {sorted(ev)}")
    # oracle-stable-fixed / oracle-fixed from the fixed grid on this stream
    fixed = {}
    for f in sorted(glob.glob(f"{R}/e2/mackey_drift_fixed_lr*_s*.json")):
        d = json.load(open(f))
        fixed.setdefault(d["lr0"], []).append(d)
    stable, best = None, None
    for lr, runs in sorted(fixed.items()):
        nm, _, n = mstd([r["nmse"] for r in runs])
        ev, _, _ = mstd([r["events"] for r in runs])
        tag = f"fixed lr={lr:g} n={n} NMSE {nm:.4g} events {ev:.1f}"
        if best is None or nm < best[0]:
            best = (nm, ev, tag)
        if ev == 0 and (stable is None or nm < stable[0]):
            stable = (nm, ev, tag)
        print("  " + tag)
    if stable:
        print(f"oracle-stable-fixed -> {stable[2]}")
    if best:
        print(f"oracle-fixed        -> {best[2]}")

    print("-" * 70)
    print("Permutation tests (two-sided exact, paired by seed, NMSE / events):")
    pairs = [("cohg (sign)", "cohg_nogate"), ("cohg (sign)", "hdm"),
             ("cohg (sign)", "hd"), ("cohg (sign)", "fmd"),
             ("cohg (sign)", "cohg_ogd")]
    for a, b in pairs:
        if a not in per_seed or b not in per_seed:
            print(f"  {a} vs {b}: missing arm")
            continue
        seeds = sorted(set(per_seed[a]) & set(per_seed[b]))
        for metric in ("nmse", "events"):
            diffs = [per_seed[a][s][metric] - per_seed[b][s][metric]
                     for s in seeds]
            direction = "A<B" if sum(diffs) < 0 else "A>B"
            print(f"  {a} vs {b} [{metric}] n={len(seeds)} "
                  f"p={perm_p(diffs):.4g} ({direction})")


def e4_table():
    print("=" * 70)
    print("tab:e4  results/e4_v2 (three domains, drift@[1000,2000])")
    print("=" * 70)
    groups = {}
    for f in sorted(glob.glob(f"{R}/e4_v2/*.json")):
        if f.endswith(".claim"):
            continue
        d = json.load(open(f))
        key = os.path.basename(f).rsplit("_s", 1)[0]
        groups.setdefault(key, []).append(d)
    for key, runs in sorted(groups.items()):
        ppl = [r.get("online_ppl", r.get("final_ppl", r.get("ppl"))) for r in runs]
        ev = [r.get("events", 0) for r in runs]
        gb = [r.get("peak_mem_gb", r.get("peak_gb", 0)) for r in runs]
        wall = [r.get("wall_s", 0) / 3600 for r in runs]
        gof = [r.get("gate_open_frac") for r in runs
               if r.get("gate_open_frac") is not None]
        if None in ppl:
            print(f"{key}: missing ppl key, raw keys: {list(runs[0])}")
            continue
        mp, sp, n = mstd(ppl)
        mw, sw, _ = mstd(wall)
        line = (f"{key:28s} n={n} PPL {mp:.2f}+-{sp:.2f} "
                f"events {sum(ev)/len(ev):.1f} GB {max(gb):.1f} "
                f"wall {mw:.1f}+-{sw:.1f}h")
        if gof:
            line += f" gate_open {sum(gof)/len(gof):.4g}"
        print(line)
        print(f"{'':28s}   per-seed PPL: {[round(p,2) for p in ppl]}")


if __name__ == "__main__":
    if "--e4-only" not in sys.argv:
        e2_table()
    if "--e2-only" not in sys.argv:
        e4_table()
