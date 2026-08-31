"""E2 analysis: per-dataset method comparison (paper Table 1 + stats).

For each dataset:
- fixed family: NMSE per lr; oracle-fixed = best mean-NMSE lr (post hoc)
- adaptive methods at well-set / mis-set inits: NMSE, events, gate stats
- adaptation benefit: NMSE(method @ mis-set lr0) vs NMSE(fixed @ same lr)
- paired seed-level comparisons (COHG vs HD / vs oracle-fixed):
  exact permutation test on paired differences (instability counts and NMSE)

Writes results/e2/SUMMARY.md. Rerun any time (uses whatever JSONs exist).
"""

from __future__ import annotations

import glob
import itertools
import json
import math
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E2 = os.path.join(HERE, "..", "results", "e2")

runs = []
for path in glob.glob(os.path.join(E2, "*.json")):
    try:
        with open(path) as f:
            r = json.load(f)
        r.pop("losses", None)
        r.pop("lam_hist", None)
        runs.append(r)
    except Exception as e:
        print("skip", os.path.basename(path), e)

by = defaultdict(dict)  # (ds, method, lr) -> {seed: run}
for r in runs:
    by[(r["dataset"], r["method"], r["lr0"])][r["seed"]] = r


def agg(ds, method, lr):
    d = by.get((ds, method, lr), {})
    if not d:
        return None
    nm = [x["nmse"] for x in d.values()]
    ev = [x["events"] for x in d.values()]
    gates = [x["gate_open_frac"] for x in d.values()
             if x.get("gate_open_frac") is not None]
    return {"n": len(d), "nmse_mean": float(np.mean(nm)),
            "nmse_std": float(np.std(nm)), "events_mean": float(np.mean(ev)),
            "gate": float(np.mean(gates)) if gates else None,
            "seeds": {s: (x["nmse"], x["events"]) for s, x in d.items()}}


def perm_test_paired(a: dict, b: dict):
    """Exact sign-flip permutation test on paired per-seed differences."""
    common = sorted(set(a) & set(b))
    if len(common) < 4:
        return None, len(common)
    d = np.array([a[s] - b[s] for s in common], dtype=float)
    obs = d.mean()
    n = len(d)
    flips = list(itertools.product([1, -1], repeat=n))
    stats = [np.mean(d * np.array(f)) for f in flips]
    p = np.mean([abs(s) >= abs(obs) - 1e-15 for s in stats])
    return float(p), n


FIXED_LRS = [0.003, 0.01, 0.03, 0.1, 0.3, 0.6, 1.0]
DATASETS = ["mackey", "lorenz", "sunspot", "santafe",
            "mackey_drift", "lorenz_drift"]
ADAPT = ["hd", "hd_scalar", "hdm", "tfmd", "fmd", "cohg", "cohg_r0",
         "cohg_nogate", "cohg_ogd"]

L = ["# E2 Summary (auto-generated)", "",
     f"runs loaded: {len(runs)}", ""]
for ds in DATASETS:
    L.append(f"\n## {ds}")
    L.append("| system | lr0 | n | NMSE mean±std | events | gate |")
    L.append("|---|---|---|---|---|---|")
    fixed = {lr: agg(ds, "fixed", lr) for lr in FIXED_LRS}
    oracle_lr, oracle = None, None
    for lr, a in fixed.items():
        if a:
            L.append(f"| fixed | {lr:g} | {a['n']} | "
                     f"{a['nmse_mean']:.4f}±{a['nmse_std']:.4f} | "
                     f"{a['events_mean']:.1f} | — |")
            if oracle is None or a["nmse_mean"] < oracle["nmse_mean"]:
                oracle, oracle_lr = a, lr
    if oracle:
        L.append(f"| **oracle-fixed** | {oracle_lr:g} | {oracle['n']} | "
                 f"{oracle['nmse_mean']:.4f} | {oracle['events_mean']:.1f} | — |")
    for meth in ADAPT:
        for lr in [0.003, 0.03, 0.3, 1.0]:
            a = agg(ds, meth, lr)
            if a:
                g = f"{a['gate']:.2f}" if a["gate"] is not None else "—"
                L.append(f"| {meth} | {lr:g} | {a['n']} | "
                         f"{a['nmse_mean']:.4f}±{a['nmse_std']:.4f} | "
                         f"{a['events_mean']:.1f} | {g} |")
    # paired tests at well-set init
    cohg = agg(ds, "cohg", 0.03)
    for rival in ["hd", "hdm", "tfmd", "fmd", "cohg_r0", "cohg_nogate"]:
        rv = agg(ds, rival, 0.03)
        if cohg and rv:
            a_n = {s: v[0] for s, v in cohg["seeds"].items()}
            b_n = {s: v[0] for s, v in rv["seeds"].items()}
            p, n = perm_test_paired(a_n, b_n)
            if p is not None:
                L.append(f"- paired NMSE cohg vs {rival} (n={n}): "
                         f"Δ={np.mean([a_n[s] - b_n[s] for s in set(a_n) & set(b_n)]):+.4f}, "
                         f"p={p:.3f}")
    # adaptation benefit at mis-set inits
    for lr in [0.003, 1.0]:
        c, fx = agg(ds, "cohg", lr), fixed.get(lr)
        if c and fx:
            L.append(f"- mis-set lr0={lr:g}: cohg {c['nmse_mean']:.4f} vs "
                     f"fixed {fx['nmse_mean']:.4f} "
                     f"(benefit {fx['nmse_mean'] - c['nmse_mean']:+.4f}), "
                     f"gate={c['gate']:.2f}" if c["gate"] is not None else "")

out = os.path.join(E2, "SUMMARY.md")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print("\n".join(L))
print(f"\n-> {out}")
