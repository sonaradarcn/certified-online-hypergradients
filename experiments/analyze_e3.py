"""E3 analysis: fixed grid (10 seeds) + adaptive arms when present."""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E3 = os.path.join(HERE, "..", "results", "e3")

runs = []
for p in glob.glob(os.path.join(E3, "*.json")):
    with open(p) as f:
        r = json.load(f)
    r.pop("acc_matrix", None)
    r.pop("lam_hist", None)
    runs.append(r)

by = defaultdict(list)
for r in runs:
    by[(r["method"], r["lr0"], r["ewc0"])].append(r)

print(f"{len(runs)} runs\n")
print("| method | lr0 | ewc0 | n | avg_acc | bwt | events | gate |")
print("|---|---|---|---|---|---|---|---|")
for key in sorted(by):
    g = by[key]
    acc = [x["avg_acc"] for x in g]
    bwt = [x["bwt"] for x in g]
    ev = [x["events"] for x in g]
    gates = [x.get("gate_open_frac") for x in g
             if x.get("gate_open_frac") is not None]
    gate = f"{np.mean(gates):.2f}" if gates else "-"
    print(f"| {key[0]} | {key[1]:g} | {key[2]:g} | {len(g)} | "
          f"{np.mean(acc):.4f}±{np.std(acc):.4f} | {np.mean(bwt):+.4f} | "
          f"{np.mean(ev):.1f} | {gate} |")
