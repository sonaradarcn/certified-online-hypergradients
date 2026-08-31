"""Draft paper figures from finalized E0/E1 data -> results/figures/.

Fig 1 (E0, motivation): cross-group sensitivity residual is large AND low-rank.
  (a) best-rank-r capture curves per architecture (2nd-half mean +- range)
  (b) residual share + stable rank per architecture (bar/dot summary)
Fig 2 (E1, certificate): validity & tightness.
  (a) certificate e_t vs true error trace (teacher, kw_drift, r=4, K=5)
  (b) tightness distribution across configs (gamma x K), teacher kw_drift
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "figure.dpi": 150, "savefig.bbox": "tight",
})
ARCHS = ["mlp", "resnet", "transformer", "gru"]
NICE = {"mlp": "MLP / MNIST", "resnet": "ResNet-8 / CIFAR-10",
        "transformer": "Transformer / PTB-char", "gru": "GRU / Mackey-Glass"}
COLORS = dict(zip(ARCHS, plt.cm.tab10.colors))

# ---------------------------------------------------------------- Fig 1
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
summary_rows = []
for arch in ARCHS:
    path = os.path.join(RES, "e0", f"{arch}_seed0.json")
    if not os.path.exists(path):
        continue
    blob = json.load(open(path))
    snaps = [s for s in blob["snapshots"] if s["t"] >= blob["summary"]["steps"] // 2]
    m = blob["summary"]["m"]
    caps = np.array([s["capture"] for s in snaps])  # (n_snap, m)
    ranks = np.arange(1, m + 1)
    med = np.median(caps, axis=0)
    lo, hi = caps.min(axis=0), caps.max(axis=0)
    axes[0].plot(ranks / m, med, label=NICE[arch], color=COLORS[arch])
    axes[0].fill_between(ranks / m, lo, hi, alpha=0.15, color=COLORS[arch])
    summary_rows.append((arch,
                         blob["summary"]["mean_res_share_2nd_half"],
                         blob["summary"]["mean_stable_rank_2nd_half"], m))
axes[0].axhline(0.8, ls=":", c="gray", lw=0.8)
axes[0].axvline(0.25, ls=":", c="gray", lw=0.8)
axes[0].set_xlabel("rank fraction $r/m$")
axes[0].set_ylabel("residual capture $\\|R_r\\|_F/\\|R\\|_F$")
axes[0].set_title("(a) low-rank structure of cross-group residual")
axes[0].legend(loc="lower right", frameon=False)

x = np.arange(len(summary_rows))
share = [r[1] for r in summary_rows]
srank = [r[2] for r in summary_rows]
ax2 = axes[1]
ax2.bar(x - 0.18, share, width=0.36, label="residual share", color="#4878b0")
ax2.set_ylabel("residual share $\\|R\\|_F/\\|S\\|_F$", color="#4878b0")
ax2.set_ylim(0, 1)
ax3 = ax2.twinx()
ax3.bar(x + 0.18, srank, width=0.36, label="stable rank", color="#d1885c")
ax3.set_ylabel("stable rank of $R$", color="#d1885c")
ax2.set_xticks(x)
ax2.set_xticklabels([r[0] for r in summary_rows], rotation=15)
ax2.set_title("(b) coupling is heavy yet low-rank")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_e0_structure.pdf"))
fig.savefig(os.path.join(FIG, "fig1_e0_structure.png"))
plt.close(fig)
print("fig1 done")

# ---------------------------------------------------------------- Fig 2
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
# (a) trace from kw_drift g90, config r4_K5, seed 0
for path in [os.path.join(RES, "e1", "teacher_kw_drift_g90.json")]:
    rows = json.load(open(path))
    tr = next((r for r in rows
               if r["r"] == 4 and r["K"] == 5 and r.get("traces")), None)
    if tr:
        ts = [x["t"] for x in tr["traces"]]
        axes[0].semilogy(ts, [x["e"] for x in tr["traces"]],
                         label="certificate $e_t$", color="#c44e52")
        axes[0].semilogy(ts, [max(x["true_err"], 1e-12) for x in tr["traces"]],
                         label="true error $\\|\\tilde S_t-\\hat S_t\\|_F$",
                         color="#4878b0")
        axes[0].set_xlabel("step $t$")
        axes[0].set_title("(a) anytime certificate vs. true error\n"
                          "(teacher-student, $\\gamma$=0.9, r=4, K=5)")
        axes[0].legend(frameon=False)

# (b) tightness distributions across (gamma, K), r=4
box_data, box_labels = [], []
for gam, fname in [("0.9", "teacher_kw_drift_g90.json"),
                   ("0.95", "teacher_kw_drift_g95.json")]:
    path = os.path.join(RES, "e1", fname)
    if not os.path.exists(path):
        continue
    rows = json.load(open(path))
    for K in [1, 5, 10, 20]:
        med = [r["tight_med"] for r in rows if r["r"] == 4 and r["K"] == K]
        if med:
            box_data.append(med)
            box_labels.append(f"$\\gamma$={gam}\nK={K}")
axes[1].boxplot(box_data, tick_labels=box_labels, showfliers=False)
axes[1].set_yscale("log")
axes[1].axhline(10, ls=":", c="gray", lw=0.8)
axes[1].set_ylabel("tightness $e_t/\\|E_t\\|_F$ (median per run)")
axes[1].set_title("(b) tightness across discount & laziness\n"
                  "(10 seeds; $\\gamma{=}0.95$, K$\\geq$10 crosses "
                  "$\\gamma\\bar\\rho{=}1$)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_e1_certificate.pdf"))
fig.savefig(os.path.join(FIG, "fig2_e1_certificate.png"))
plt.close(fig)
print("fig2 done ->", FIG)
