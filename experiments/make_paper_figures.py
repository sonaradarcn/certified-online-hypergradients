"""Paper figures 4-7 from final data -> results/figures/."""

from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 7.5, "figure.dpi": 150,
                     "savefig.bbox": "tight"})
C_OURS, C_BAD, C_FIX = "#1f77b4", "#d62728", "#7f7f7f"


def load(path):
    with open(path) as f:
        return json.load(f)


# ---------------- Fig 4: E2 lambda trajectories + gate (drift, mis-set) ----
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
for ax, meth, title in [
        (axes[0], "cohg", "(a) certified gated (ours)"),
        (axes[1], "cohg_nogate", "(b) same estimator, gate OFF")]:
    p = os.path.join(RES, "e2", f"mackey_drift_{meth}_lr0.003_s0.json")
    if not os.path.exists(p):
        continue
    r = load(p)
    lam = np.array(r["lam_hist"])  # [t, m...]
    T = r["steps"]
    for j in range(1, lam.shape[1]):
        ax.plot(lam[:, 0], lam[:, j] / np.log(10), lw=0.9)
    ax.axhline(np.log10(0.003), ls=":", c="k", lw=0.8)
    for b in (T // 3, 2 * T // 3):
        ax.axvline(b, ls="--", c="gray", lw=0.7)
    ax.set_xlabel("step")
    ax.set_title(f"{title}\nevents={r['events']}, NMSE={r['nmse']:.3g}",
                 fontsize=8)
axes[0].set_ylabel("$\\log_{10}\\eta_j$ (per group)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig4_lambda_traj.pdf"))
fig.savefig(os.path.join(FIG, "fig4_lambda_traj.png"))
plt.close(fig)
print("fig4 done")

# ---------------- Fig 5: E3 per-task accuracy (gate on/off) ----------------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
cells = [("cohg", "lr0.05_ewc10", C_OURS, "certified gated"),
         ("cohg_nogate", "lr0.05_ewc10", C_BAD, "gate OFF"),
         ("fixed", "lr0.05_ewc10", C_FIX, "fixed $\\lambda$")]
for meth, cfg, color, label in cells:
    accs_last, accs_first = [], []
    for p in glob.glob(os.path.join(RES, "e3", f"cifar100_{meth}_{cfg}_s*.json")):
        r = load(p)
        am = r.get("acc_matrix")
        if not am:
            continue
        accs_last.append(am[-1])                    # final acc per task
        accs_first.append([am[k][k] for k in range(len(am))])  # acc right after task k
    if not accs_last:
        continue
    last = np.mean(np.array(accs_last), axis=0)
    first = np.mean(np.array(accs_first), axis=0)
    x = np.arange(1, len(last) + 1)
    axes[0].plot(x, first, "-o", ms=3, color=color, label=label)
    axes[1].plot(x, last, "-o", ms=3, color=color, label=label)
axes[0].set_title("(a) accuracy right after learning task $k$", fontsize=8.5)
axes[1].set_title("(b) final accuracy per task (after task 10)", fontsize=8.5)
for ax in axes:
    ax.set_xlabel("task index")
    ax.set_ylabel("accuracy")
axes[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig5_e3_tasks.pdf"))
fig.savefig(os.path.join(FIG, "fig5_e3_tasks.png"))
plt.close(fig)
print("fig5 done")

# ---------------- Fig 6: E4 online PPL + drift markers ---------------------
fig, ax = plt.subplots(figsize=(7.0, 2.8))


def smooth_ppl(losses, w=100):
    v = np.array([x for x in losses], dtype=float)
    out = []
    for i in range(0, len(v), 20):
        seg = v[max(0, i - w):i + 1]
        seg = seg[np.isfinite(seg)]
        out.append((i, np.exp(seg.mean()) if len(seg) else np.nan))
    return np.array(out)


shown = set()
for meth, color, label in [("fixed", C_FIX, "fixed $\\lambda$ (lr=1e-3)"),
                           ("cohg_r0", C_OURS, "certified gated r=0 (ours)"),
                           ("cohg_nogate", C_BAD, "gate OFF")]:
    p = os.path.join(RES, "e4", f"gpt2_{meth}_lr0.001_s0.json")
    if not os.path.exists(p):
        continue
    r = load(p)
    sm = smooth_ppl(r["losses"])
    ax.plot(sm[:, 0], sm[:, 1], color=color, lw=1.2,
            label=label + f" (final {r['online_ppl']:.1f})")
    for b in r.get("drift_steps", []):
        if b < r["steps"]:
            ax.axvline(b, ls="--", c="gray", lw=0.8)
ax.set_xlabel("stream step")
ax.set_ylabel("online perplexity (rolling)")
ax.set_title("GPT-2 124M streaming TTA under domain shift "
             "(wiki $\\to$ news $\\to$ code)", fontsize=9)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig6_gpt2.pdf"))
fig.savefig(os.path.join(FIG, "fig6_gpt2.png"))
plt.close(fig)
print("fig6 done")

# ---------------- Fig 7: M4 ablations ---------------------------------------
m4 = defaultdict(list)
for p in glob.glob(os.path.join(RES, "m4", "*.json")):
    m = re.search(r"(mackey_drift|lorenz)_(\w+?)_lr[\d.]+(_[\w.]+)_s(\d+)\.json",
                  os.path.basename(p))
    if not m:
        continue
    r = load(p)
    m4[(m.group(1), m.group(2), m.group(3))].append((r["nmse"], r["events"]))

fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.5))


def panel(ax, tags, xvals, xlabel, title, log_x=False):
    for ds, mk, color in [("mackey_drift", "o", C_OURS), ("lorenz", "s", "#2ca02c")]:
        nm = [np.median([v[0] for v in m4.get((ds, "cohg", t), [(np.nan, 0)])])
              for t in tags]
        ev = [np.mean([v[1] for v in m4.get((ds, "cohg", t), [(np.nan, 0)])])
              for t in tags]
        ax.plot(xvals, nm, "-" + mk, ms=4, color=color, label=ds)
        ax2 = ax.twinx()
        ax2.plot(xvals, ev, ":" + mk, ms=3, color=color, alpha=0.5)
        ax2.set_yticks([])
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=8.5)


panel(axes[0], [f"_r{r}" for r in [0, 1, 2, 4, 8, 16]], [0, 1, 2, 4, 8, 16],
      "sketch rank $r$", "(a) rank: FLAT (honest A3)")
panel(axes[1], [f"_K{k}" for k in [1, 5, 10, 20, 50]], [1, 5, 10, 20, 50],
      "refresh period $K$", "(b) laziness stabilizes (K=1 unstable)", log_x=True)
panel(axes[2], [f"_c{c:g}" for c in [1.0, 1.5, 2.0, 3.0]], [1.0, 1.5, 2.0, 3.0],
      "gate threshold $c$", "(c) gate threshold")
panel(axes[3], [f"_g{g:g}" for g in [0.8, 0.9, 0.95]], [0.8, 0.9, 0.95],
      "discount $\\gamma$", "(d) discount")
axes[0].set_ylabel("online NMSE (solid)\nevents (dotted, right)")
axes[0].legend(frameon=False, fontsize=6.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig7_ablations.pdf"))
fig.savefig(os.path.join(FIG, "fig7_ablations.png"))
plt.close(fig)
print("fig7 done ->", FIG)
