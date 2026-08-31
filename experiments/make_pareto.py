"""Fig 1b: safety-performance Pareto on drifting streams + cost normalization
(reviewer fix #7). Three visual families only (reviewer fix #11):
fixed-lambda spectrum / ungated adaptive / certified gated.

X = instability events (log scale, +1 offset), Y = online NMSE (log),
marker size ~ HVP cost per step (cost normalization made visible).
Data: results/e2 mackey_drift + lorenz_drift, seeds paired.
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150,
                     "savefig.bbox": "tight"})

FAMILY = {
    "fixed": ("fixed $\\lambda$ spectrum", "#7f7f7f", "o"),
    "hd": ("ungated adaptive", "#d62728", "^"),
    "hd_scalar": ("ungated adaptive", "#d62728", "^"),
    "hdm": ("ungated adaptive", "#d62728", "^"),
    "tfmd": ("ungated adaptive", "#d62728", "^"),
    "fmd": ("ungated adaptive", "#d62728", "^"),
    "cohg_nogate": ("ungated adaptive", "#d62728", "^"),
    "cohg": ("certified gated (ours)", "#1f77b4", "*"),
    "cohg_r0": ("certified gated (ours)", "#1f77b4", "*"),
}

rows = defaultdict(list)
for p in glob.glob(os.path.join(RES, "e2", "*_drift_*.json")):
    try:
        r = json.load(open(p))
    except Exception:
        continue
    if r["lr0"] not in (0.03, 0.003, 0.01, 0.1, 0.3, 0.6):
        continue  # exclude the divergence-region 1.0 arms
    key = (r["method"], r["lr0"])
    rows[key].append((r["nmse"], r["events"],
                      r.get("hvp_total", 0) / max(r["steps"], 1)))

fig, ax = plt.subplots(figsize=(4.6, 3.4))
seen_labels = set()
summary = []
for (meth, lr), vals in sorted(rows.items()):
    if len(vals) < 4 or meth not in FAMILY:
        continue
    nm = np.median([v[0] for v in vals])
    ev = np.mean([v[1] for v in vals])
    hvp = np.mean([v[2] for v in vals])
    if not np.isfinite(nm) or nm > 1:
        continue  # divergent cells annotated separately in caption
    label, color, marker = FAMILY[meth]
    show = label if label not in seen_labels else None
    seen_labels.add(label)
    size = 30 + 25 * np.sqrt(hvp)
    ax.scatter(ev + 1, nm, s=size, c=color, marker=marker, label=show,
               alpha=0.85, edgecolors="k", linewidths=0.4, zorder=3)
    summary.append((meth, lr, nm, ev, hvp))
    if meth in ("cohg", "hdm", "fmd") or (meth == "fixed" and lr in (0.003, 0.3)):
        ax.annotate(f"{meth}@{lr:g}" if meth != "fixed" else f"fix@{lr:g}",
                    (ev + 1, nm), textcoords="offset points", xytext=(5, 4),
                    fontsize=6.5)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("instability events + 1 (12k steps, mean over seeds)")
ax.set_ylabel("online NMSE (median over seeds)")
ax.set_title("Safety-performance Pareto on drifting streams\n"
             "(marker size $\\propto$ HVPs/step: cost-normalized view)",
             fontsize=8.5)
ax.legend(frameon=False, fontsize=7, loc="upper right")
fig.savefig(os.path.join(FIG, "fig1b_pareto.pdf"))
fig.savefig(os.path.join(FIG, "fig1b_pareto.png"))
print("fig1b saved")
print("\n| method | lr0 | NMSE(med) | events | HVP/step |")
for m, lr, nm, ev, hvp in sorted(summary, key=lambda x: x[3]):
    print(f"| {m} | {lr:g} | {nm:.4f} | {ev:.1f} | {hvp:.2f} |")
