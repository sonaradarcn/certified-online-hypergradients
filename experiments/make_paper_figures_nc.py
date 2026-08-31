"""Print-quality paper figures for Neurocomputing (Elsevier cas-dc).

Regenerates all 7 paper figures into paper/main/figs/ at exact print size:
  single column = 3.3 in wide, double column (fig7) = 6.85 in wide.
Data loading is identical to make_figures.py / make_pareto.py /
make_paper_figures.py -- only presentation changed (sizes, fonts,
Okabe-Ito palette, no titles, readable labels, fixed collisions).
"""

from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "paper", "main", "figs")
FIG_PNG = os.path.join(HERE, "..", "results", "figures")
os.makedirs(FIG, exist_ok=True)
os.makedirs(FIG_PNG, exist_ok=True)

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.0,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.45,
    "ytick.minor.width": 0.45,
    "figure.dpi": 150,
})

# Okabe-Ito color-blind-safe palette
OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
      "#E69F00", "#56B4E9", "#F0E442", "#000000"]
C_OURS = OI[0]   # blue      -> certified gated (ours)
C_BAD = OI[1]    # vermilion -> ungated adaptive / gate OFF
C_FIX = OI[7]    # black     -> fixed lambda
C_GREEN = OI[2]

SC_W = 3.3    # single-column width (in)
DC_W = 6.85   # double-column width (in)


def load(path):
    with open(path) as f:
        return json.load(f)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"),
                bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(FIG_PNG, name + "_nc.png"),
                bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print(name, "done")


def tag(ax, s, x=0.02, y=0.97, va="top", ha="left"):
    ax.text(x, y, s, transform=ax.transAxes, va=va, ha=ha,
            fontsize=8, fontweight="bold")


# ====================================================================
# Fig 1 (E0): cross-group sensitivity residual is large AND low-rank
# ====================================================================
ARCHS = ["mlp", "resnet", "transformer", "gru"]
NICE = {"mlp": "MLP / MNIST", "resnet": "ResNet-8 / CIFAR-10",
        "transformer": "Transformer / PTB-char", "gru": "GRU / Mackey-Glass"}
SHORT = {"mlp": "MLP", "resnet": "ResNet-8",
         "transformer": "Transformer", "gru": "GRU"}
ACOL = dict(zip(ARCHS, [OI[0], OI[1], OI[2], OI[3]]))

fig, axes = plt.subplots(2, 1, figsize=(SC_W, 4.3))
summary_rows = []
for arch in ARCHS:
    path = os.path.join(RES, "e0", f"{arch}_seed0.json")
    if not os.path.exists(path):
        continue
    blob = load(path)
    snaps = [s for s in blob["snapshots"]
             if s["t"] >= blob["summary"]["steps"] // 2]
    m = blob["summary"]["m"]
    caps = np.array([s["capture"] for s in snaps])
    ranks = np.arange(1, m + 1)
    med = np.median(caps, axis=0)
    lo, hi = caps.min(axis=0), caps.max(axis=0)
    axes[0].plot(ranks / m, med, label=NICE[arch], color=ACOL[arch])
    axes[0].fill_between(ranks / m, lo, hi, alpha=0.15, color=ACOL[arch],
                         lw=0)
    summary_rows.append((arch,
                         blob["summary"]["mean_res_share_2nd_half"],
                         blob["summary"]["mean_stable_rank_2nd_half"], m))
axes[0].axhline(0.8, ls=":", c="gray", lw=0.7)
axes[0].axvline(0.25, ls=":", c="gray", lw=0.7)
axes[0].set_xlabel("rank fraction $r/m$")
axes[0].set_ylabel("residual capture $\\|R_r\\|_F/\\|R\\|_F$")
axes[0].legend(loc="lower right", frameon=False)
tag(axes[0], "(a)")

x = np.arange(len(summary_rows))
share = [r[1] for r in summary_rows]
srank = [r[2] for r in summary_rows]
ax2 = axes[1]
ax2.bar(x - 0.18, share, width=0.36, color=OI[0])
ax2.set_ylabel("residual share $\\|R\\|_F/\\|S\\|_F$", color=OI[0])
ax2.tick_params(axis="y", colors=OI[0])
ax2.set_ylim(0, 1)
ax3 = ax2.twinx()
ax3.bar(x + 0.18, srank, width=0.36, color=OI[1])
ax3.set_ylabel("stable rank of $R$", color=OI[1])
ax3.tick_params(axis="y", colors=OI[1])
ax3.spines["left"].set_visible(False)
ax2.set_xticks(x)
ax2.set_xticklabels([SHORT[r[0]] for r in summary_rows],
                    rotation=12, ha="right", rotation_mode="anchor")
tag(ax2, "(b)")
fig.tight_layout(h_pad=1.2)
save(fig, "fig1_e0_structure")

# ====================================================================
# Fig 2 (E1): certificate validity & tightness
# ====================================================================
fig, axes = plt.subplots(2, 1, figsize=(SC_W, 4.3))
# (a) trace from kw_drift g90, config r4_K5
rows = load(os.path.join(RES, "e1", "teacher_kw_drift_g90.json"))
tr = next((r for r in rows
           if r["r"] == 4 and r["K"] == 5 and r.get("traces")), None)
if tr:
    ts = [z["t"] for z in tr["traces"]]
    axes[0].semilogy(ts, [z["e"] for z in tr["traces"]],
                     label="certificate $e_t$", color=OI[1])
    axes[0].semilogy(ts, [max(z["true_err"], 1e-12) for z in tr["traces"]],
                     label="true error $\\|\\tilde S_t-\\hat S_t\\|_F$",
                     color=OI[0])
    axes[0].set_xlabel("step $t$")
    axes[0].set_ylabel("Frobenius norm")
    axes[0].legend(frameon=False, loc="center right")
tag(axes[0], "(a)", x=0.03, y=0.05, va="bottom")

# (b) tightness distributions across (gamma, K), r=4
box_data, k_labels, g_groups = [], [], []
for gam, fname in [("0.9", "teacher_kw_drift_g90.json"),
                   ("0.95", "teacher_kw_drift_g95.json")]:
    path = os.path.join(RES, "e1", fname)
    if not os.path.exists(path):
        continue
    rows = load(path)
    for K in [1, 5, 10, 20]:
        med = [r["tight_med"] for r in rows if r["r"] == 4 and r["K"] == K]
        # K=1 medians are all-NaN (exact refresh): the box was always empty,
        # so drop those slots instead of showing phantom ticks.
        med = [v for v in med if np.isfinite(v)]
        if med:
            box_data.append(med)
            k_labels.append(f"{K}")
            g_groups.append(gam)
axb = axes[1]
bp = axb.boxplot(box_data, tick_labels=k_labels, showfliers=False,
                 widths=0.55,
                 boxprops=dict(lw=0.7), whiskerprops=dict(lw=0.7),
                 capprops=dict(lw=0.7),
                 medianprops=dict(lw=1.0, color=OI[1]))
axb.set_yscale("log")
axb.axhline(10, ls=":", c="gray", lw=0.7)
axb.set_xlabel("refresh period $K$")
axb.set_ylabel("tightness $e_t/\\|E_t\\|_F$")
# group separators + gamma labels along the top
n_g1 = g_groups.count("0.9")
if 0 < n_g1 < len(box_data):
    axb.axvline(n_g1 + 0.5, ls="-", c="0.8", lw=0.6)
    axb.text((1 + n_g1) / 2, 0.96, "$\\gamma=0.9$",
             transform=axb.get_xaxis_transform(), ha="center", va="top",
             fontsize=7)
    axb.text((n_g1 + 1 + len(box_data)) / 2, 0.96, "$\\gamma=0.95$",
             transform=axb.get_xaxis_transform(), ha="center", va="top",
             fontsize=7)
tag(axb, "(b)", x=0.02, y=0.85)
fig.tight_layout(h_pad=1.2)
save(fig, "fig2_e1_certificate")

# ====================================================================
# Fig 1b: safety-performance Pareto (E2 drifting streams)
# ====================================================================
FAMILY = {
    "fixed": ("fixed $\\lambda$ spectrum", C_FIX, "o"),
    "hd": ("ungated adaptive", C_BAD, "^"),
    "hd_scalar": ("ungated adaptive", C_BAD, "^"),
    "hdm": ("ungated adaptive", C_BAD, "^"),
    "tfmd": ("ungated adaptive", C_BAD, "^"),
    "fmd": ("ungated adaptive", C_BAD, "^"),
    "cohg_nogate": ("ungated adaptive", C_BAD, "^"),
    "cohg": ("certified gated (ours)", C_OURS, "*"),
    "cohg_r0": ("certified gated (ours)", C_OURS, "*"),
}

rows2 = defaultdict(list)
for p in glob.glob(os.path.join(RES, "e2", "*_drift_*.json")):
    try:
        r = load(p)
    except Exception:
        continue
    if r["lr0"] not in (0.03, 0.003, 0.01, 0.1, 0.3, 0.6):
        continue  # exclude the divergence-region 1.0 arms
    rows2[(r["method"], r["lr0"])].append(
        (r["nmse"], r["events"], r.get("hvp_total", 0) / max(r["steps"], 1)))


def msize(hvp):
    return 16.0 + 15.0 * np.sqrt(hvp)


# annotation plan: label -> (xytext offset in points, ha)
ANN = {
    ("fixed", 0.003): ("fixed 0.003", (7, -3), "left"),
    ("fixed", 0.3): ("fixed 0.3", (9, -9), "left"),
    ("cohg", 0.003): ("COHG (lr 0.003)", (6, 7), "left"),
    ("cohg", 0.03): ("COHG (lr 0.03)", (8, 9), "left"),
    ("fmd", 0.03): ("FMD", (10, -3), "left"),
    ("hdm", 0.03): ("HDM", (0, 13), "center"),
}

fig, ax = plt.subplots(figsize=(SC_W, 2.9))
seen_labels = set()
for (meth, lr), vals in sorted(rows2.items()):
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
    ax.scatter(ev + 1, nm, s=msize(hvp), c=color, marker=marker, label=show,
               alpha=0.85, edgecolors="k", linewidths=0.4, zorder=3)
    if (meth, lr) in ANN:
        text, off, ha = ANN[(meth, lr)]
        ax.annotate(text, (ev + 1, nm), textcoords="offset points",
                    xytext=off, fontsize=7, ha=ha, zorder=4,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="0.45",
                                    shrinkA=1, shrinkB=3))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.68, 3200)
ax.set_ylim(0.0007, 0.4)
ax.set_xlabel("instability events + 1 (mean over seeds)")
ax.set_ylabel("online NMSE (median over seeds)")
leg1 = ax.legend(frameon=False, loc="lower left", handletextpad=0.3,
                 borderaxespad=0.2, labelspacing=0.35)
ax.add_artist(leg1)
# size key: marker area encodes HVPs per step
size_handles = [
    ax.scatter([], [], s=msize(h), facecolors="none", edgecolors="k",
               linewidths=0.5, marker="o") for h in (0, 1, 8)]
ax.legend(size_handles, ["0", "1", "8"], title="HVPs/step",
          frameon=False, loc="upper left", labelspacing=0.5,
          handletextpad=0.3, borderaxespad=0.2, title_fontsize=7)
save(fig, "fig1b_pareto")

# ====================================================================
# Fig 4 (E2): lambda trajectories, gate ON vs gate OFF
# ====================================================================
fig, axes = plt.subplots(2, 1, figsize=(SC_W, 4.0), sharex=True)
GCOL = [OI[0], OI[1], OI[2], OI[3], OI[4], OI[5]]  # per-group colors
handles = []
for ax, meth, label in [(axes[0], "cohg", "(a)"),
                        (axes[1], "cohg_nogate", "(b)")]:
    p = os.path.join(RES, "e2", f"mackey_drift_{meth}_lr0.003_s0.json")
    if not os.path.exists(p):
        continue
    r = load(p)
    lam = np.array(r["lam_hist"])  # [t, groups...]
    T = r["steps"]
    for j in range(1, lam.shape[1]):
        (ln,) = ax.plot(lam[:, 0], lam[:, j] / np.log(10), lw=0.8,
                        color=GCOL[(j - 1) % len(GCOL)])
        if ax is axes[0]:
            handles.append(ln)
    ax.axhline(np.log10(0.003), ls=":", c="k", lw=0.7)
    for b in (T // 3, 2 * T // 3):
        ax.axvline(b, ls="--", c="gray", lw=0.6)
    ax.text(0.0, 1.02, label, transform=ax.transAxes, ha="left",
            va="bottom", fontsize=8, fontweight="bold")
    ax.text(1.0, 1.02, f"events = {r['events']},  NMSE = {r['nmse']:.3g}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    ax.set_ylabel("$\\log_{10}\\eta_j$")
axes[1].set_xlabel("step")
fig.legend(handles, [f"group {j}" for j in range(1, len(handles) + 1)],
           frameon=False, ncol=3, loc="upper center",
           bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handlelength=1.2,
           handletextpad=0.4)
fig.tight_layout(h_pad=1.6, rect=(0, 0, 1, 0.91))
save(fig, "fig4_lambda_traj")

# ====================================================================
# Fig 5 (E3): per-task accuracy (gate on/off)
# ====================================================================
fig, axes = plt.subplots(2, 1, figsize=(SC_W, 4.0), sharex=True)
cells = [("cohg", "lr0.05_ewc10", C_OURS, "certified gated (ours)"),
         ("cohg_nogate", "lr0.05_ewc10", C_BAD, "gate-off"),
         ("fixed", "lr0.05_ewc10", C_FIX, "fixed $\\lambda$")]
for meth, cfg, color, label in cells:
    accs_last, accs_first = [], []
    for p in glob.glob(os.path.join(RES, "e3",
                                    f"cifar100_{meth}_{cfg}_s*.json")):
        r = load(p)
        am = r.get("acc_matrix")
        if not am:
            continue
        accs_last.append(am[-1])
        accs_first.append([am[k][k] for k in range(len(am))])
    if not accs_last:
        continue
    last = np.mean(np.array(accs_last), axis=0)
    first = np.mean(np.array(accs_first), axis=0)
    x = np.arange(1, len(last) + 1)
    axes[0].plot(x, first, "-o", ms=2.5, color=color, label=label)
    axes[1].plot(x, last, "-o", ms=2.5, color=color, label=label)
tag(axes[0], "(a)", x=0.98, y=0.04, va="bottom", ha="right")
tag(axes[1], "(b)", x=0.98, y=0.04, va="bottom", ha="right")
axes[1].set_xlabel("task index $k$")
axes[1].set_xticks(np.arange(1, 11))
for ax in axes:
    ax.set_ylabel("accuracy")
axes[0].legend(frameon=False, loc="upper left", borderaxespad=0.3,
               labelspacing=0.35)
fig.tight_layout(h_pad=0.8)
save(fig, "fig5_e3_tasks")

# ====================================================================
# Fig 6 (E4): GPT-2 online perplexity
# NOTE: rolling window w = 100 steps (trailing mean of the log-loss,
# evaluated every 20 steps, then exponentiated).
# ====================================================================
fig, ax = plt.subplots(figsize=(SC_W, 2.4))


def smooth_ppl(losses, w=100):
    v = np.array([z for z in losses], dtype=float)
    out = []
    for i in range(0, len(v), 20):
        seg = v[max(0, i - w):i + 1]
        seg = seg[np.isfinite(seg)]
        out.append((i, np.exp(seg.mean()) if len(seg) else np.nan))
    return np.array(out)


for meth, seed, color, ls, label in [
        ("fixed", 0, C_FIX, "-", "fixed $\\lambda$"),
        ("cohg_r0", 0, C_OURS, "-", "certified gated (ours)"),
        ("cohg_nogate", 0, C_BAD, "-", "gate-off, seed 1"),
        ("cohg_nogate", 1, C_BAD, ":", "gate-off, seed 2")]:
    p = os.path.join(RES, "e4_v2", f"gpt2_{meth}_lr0.001_s{seed}.json")
    if not os.path.exists(p):
        continue
    r = load(p)
    sm = smooth_ppl(r["losses"])
    ax.plot(sm[:, 0], sm[:, 1], color=color, ls=ls, lw=1.0,
            label=label + f", final {r['online_ppl']:.1f}")
    if seed == 0 and meth == "fixed":
        for b in r.get("drift_steps", []):
            if b < r["steps"]:
                ax.axvline(b, ls="--", c="gray", lw=0.7)
ax.set_xlabel("stream step")
ax.set_ylabel("online perplexity (rolling)")
ax.set_yscale("log")
ax.legend(frameon=False, loc="upper left", borderaxespad=0.3,
          fontsize=7)
save(fig, "fig6_gpt2")

# ====================================================================
# Fig 7 (M4): ablations, double column
# ====================================================================
m4 = defaultdict(list)
for p in glob.glob(os.path.join(RES, "m4", "*.json")):
    m = re.search(r"(mackey_drift|lorenz)_(\w+?)_lr[\d.]+(_[\w.]+)_s(\d+)\.json",
                  os.path.basename(p))
    if not m:
        continue
    r = load(p)
    m4[(m.group(1), m.group(2), m.group(3))].append((r["nmse"], r["events"]))

DSETS = [("mackey_drift", "Mackey (drift)", "o", OI[0]),
         ("lorenz", "Lorenz", "s", OI[2])]

fig, axes = plt.subplots(1, 4, figsize=(DC_W, 2.05))


def panel(ax, tags_, xvals, xlabel, ptag, log_x=False, xticks=None):
    ax2 = ax.twinx()
    ev_all = []
    for ds, _, mk, color in DSETS:
        nm = [np.median([v[0] for v in m4.get((ds, "cohg", t), [(np.nan, 0)])])
              for t in tags_]
        ev = [np.mean([v[1] for v in m4.get((ds, "cohg", t), [(np.nan, 0)])])
              for t in tags_]
        ev_all += [e for e in ev if np.isfinite(e)]
        ax.plot(xvals, nm, "-" + mk, ms=3, color=color)
        ax2.plot(xvals, ev, "--d", ms=3, color=color, lw=0.9,
                 markerfacecolor="none")
    if log_x:
        ax.set_xscale("log")
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_xlabel(xlabel)
    ax.margins(y=0.18)  # keep solid NMSE lines away from tags/edges
    if ev_all:  # headroom so dashed event curves do not sit on solid lines
        top = max(ev_all)
        ax2.set_ylim(-0.05 * max(top, 1), max(top, 1) * 1.55)
    ax.tick_params(axis="both", labelsize=7)
    ax2.tick_params(axis="y", labelsize=7)
    tag(ax, ptag)
    return ax2


tw = []
tw.append(panel(axes[0], [f"_r{r}" for r in [0, 1, 2, 4, 8, 16]],
                [0, 1, 2, 4, 8, 16], "sketch rank $r$", "(a)",
                xticks=[0, 4, 8, 16]))
tw.append(panel(axes[1], [f"_K{k}" for k in [1, 5, 10, 20, 50]],
                [1, 5, 10, 20, 50], "refresh period $K$", "(b)", log_x=True))
tw.append(panel(axes[2], [f"_c{c:g}" for c in [1.0, 1.5, 2.0, 3.0]],
                [1.0, 1.5, 2.0, 3.0], "gate threshold $c$", "(c)",
                xticks=[1, 2, 3]))
tw.append(panel(axes[3], [f"_g{g:g}" for g in [0.8, 0.9, 0.95]],
                [0.8, 0.9, 0.95], "discount $\\gamma$", "(d)",
                xticks=[0.8, 0.9, 0.95]))
axes[3].set_xticklabels(["0.8", "0.9", "0.95"])
axes[0].set_ylabel("online NMSE (solid)")
tw[-1].set_ylabel("events (dashed)")
leg_handles = [
    Line2D([], [], color=OI[0], marker="o", ms=3, lw=1.0,
           label="Mackey (drift)"),
    Line2D([], [], color=OI[2], marker="s", ms=3, lw=1.0, label="Lorenz"),
    Line2D([], [], color="0.3", ls="-", lw=1.0, label="NMSE (left axis)"),
    Line2D([], [], color="0.3", ls="--", marker="d", ms=3, lw=0.9,
           markerfacecolor="none", label="events (right axis)"),
]
fig.legend(handles=leg_handles, frameon=False, ncol=4, loc="upper center",
           bbox_to_anchor=(0.5, 1.02), columnspacing=1.4, handlelength=1.6,
           handletextpad=0.5)
fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.90))
save(fig, "fig7_ablations")

print("all figures ->", os.path.abspath(FIG))
