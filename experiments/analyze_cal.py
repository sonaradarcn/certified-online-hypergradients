"""Pick the frozen META_LR per method from the calibration sweep.

Criterion (pre-registered): lowest mean NMSE across {lr0=0.003, lr0=0.03} x
{seed 100, 101}; ties broken toward the smaller meta-LR (stability).
"""

from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.join(HERE, "..", "results", "e2_cal")

pat = re.compile(r"mackey_(\w+)_ml([\d.]+)_lr([\d.]+)_s(\d+)\.json")
groups = defaultdict(list)
for path in glob.glob(os.path.join(CAL, "*.json")):
    m = pat.match(os.path.basename(path))
    if not m:
        continue
    meth, ml, lr0, seed = m.group(1), float(m.group(2)), float(m.group(3)), int(m.group(4))
    with open(path) as f:
        r = json.load(f)
    groups[(meth, ml)].append((lr0, seed, r["nmse"], r["events"]))

best = {}
for meth in sorted({k[0] for k in groups}):
    rows = []
    for (m2, ml), vals in sorted(groups.items()):
        if m2 != meth:
            continue
        nm = np.mean([v[2] for v in vals])
        ev = np.mean([v[3] for v in vals])
        rows.append((ml, nm, ev, len(vals)))
        print(f"{meth:12s} ml={ml:<5g} n={len(vals):2d} "
              f"nmse={nm:.4f} events={ev:.1f}")
    if rows:
        # AMENDED RULE (findings F11): select lowest NMSE among the configs
        # with the minimal event count for this method (never trade
        # stability for NMSE inside calibration); ties -> smaller meta-LR
        min_ev = min(r[2] for r in rows)
        safe = [r for r in rows if r[2] <= min_ev + 1e-9]
        ml_best = min(safe, key=lambda x: (round(x[1], 5), x[0]))[0]
        best[meth] = ml_best
        print(f"  -> {meth}: META_LR = {ml_best} (amended rule)\n")

# ablation arms inherit the parent method's meta-LR by design
if "cohg" in best:
    best["cohg_nogate"] = best["cohg"]
    best["cohg_r0"] = best["cohg"]

print("FROZEN:", json.dumps(best))
out = os.path.join(CAL, "FROZEN_META_LR.json")
with open(out, "w") as f:
    json.dump(best, f, indent=1)
print("->", out)
