"""Fig. 10: drift-prior (M_H) misspecification curves, E1 + E2 series.

Panels, all against log10(M_H / M_H_dep), the prior relative to the value
DEPLOYED in that regime:
  (a) certificate violation rate       (b) worst true-error / bound ratio
  (c) fail-closed closed-gate fraction (d) probe overhead, %

Series
  E1 (solid)  teacher/kw_drift, gamma 0.9, K 10, r 4, 1000 steps, 5 seeds,
      EXACT ExactFMD ground truth; violation = e_t < ||S_t - Shat_t||_F on any
      step.  Deployed prior M_H_dep = 2.27609, calibrated as the max
      probe-to-probe rate M_obs over 297 probes on 3 calibration seeds
      (results/e1_misspec/teacher_kw_drift_cal.json), disjoint from the five
      evaluation seeds.
  E2 (dashed) mackey_drift GRU, the M_H in {5, 0.5, 0.05} points measured in
      results/e2_controls; violation = |ghat_j - g_true_j| > beta_col_j against
      a parallel exact discounted FMD.  Deployed prior M_H_dep = 5, the value
      of Table tab:certparams.

  Both series are therefore plotted at 1, 1/10, 1/100 of the value actually
  run, which is what "a hundred times too small" means in the text.  The
  earlier axis normalised each series by the MAXIMUM observed rate on its own
  probe population, which is not the same statistic on the two streams (E1: a
  calibration max on held-out seeds; E2: an in-sample max on the deployed arm,
  761.5) and put the two sweeps two decades apart for that reason alone.

  Fail-closed monitor ON = filled square, OFF = open circle.

-> paper/main/figs/fig10_misspec.pdf (+ .png in results/figures/)
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
MIS = os.path.join(RES, "e1_misspec")
CTL = os.path.join(RES, "e2_controls")
FIG = os.path.join(ROOT, "paper", "main", "figs")
FIG_PNG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
os.makedirs(FIG_PNG, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "lines.linewidth": 1.0, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.size": 1.3, "ytick.minor.size": 1.3,
    "figure.dpi": 150, "pdf.fonttype": 42, "ps.fonttype": 42,
})
# Okabe-Ito
BLUE, ORANGE, GREEN, BLACK = "#0072B2", "#D55E00", "#009E73", "#000000"
DC_W = 6.85
MH_FACTORS = [1.0, 0.3, 0.1, 0.03, 0.01]
E2_MH = [5.0, 0.5, 0.05]
E2_MH_DEP = 5.0        # Table tab:certparams, the shipped E2 prior
E1_MH_DEP = 2.27609    # E1 calibration max on seeds 100-102


def e1_series():
    """-> {fc: dict of arrays}"""
    out = {}
    for fc in (0, 1):
        x, vr, wr, cf, po = [], [], [], [], []
        for f in MH_FACTORS:
            p = os.path.join(MIS, f"teacher_kw_drift_x{f:g}_fc{fc}.json")
            if not os.path.exists(p):
                continue
            rows = json.load(open(p))
            x.append(f)
            vr.append(float(np.mean([r["violation_rate"] for r in rows])))
            wr.append(float(np.max([r["worst_true_over_bound"] for r in rows])))
            cf.append(float(np.mean([r["closed_frac"] or 0.0 for r in rows])))
            po.append(100.0 * (float(np.mean([r["probe_overhead"]
                                              for r in rows])) - 1.0))
        o = np.argsort(x)
        out[fc] = tuple(np.asarray(v)[o] for v in (x, vr, wr, cf, po))
    return out


def e2_series():
    def arm(mh, fc):
        return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(
            CTL, f"mackey_drift_cohg_lr0.003_mh{mh:g}_fc{fc}_s*.json")))]

    ref = arm(5.0, 1)
    mobs_max = max(r["m_obs_stats"]["max"] for r in ref)
    mobs_med = float(np.mean([r["m_obs_stats"]["median"] for r in ref]))
    base_hvp = float(np.mean([r["hvp_total"] for r in arm(5.0, 0)]))
    out = {}
    for fc in (0, 1):
        x, vr, wr, cf, po = [], [], [], [], []
        for mh in E2_MH:
            rs = arm(mh, fc)
            if not rs:
                continue
            x.append(mh / E2_MH_DEP)
            vr.append(float(np.mean([r["cert_violation_frac"] for r in rs])))
            wr.append(float(np.max([r["cert_max_ratio"] for r in rs])))
            cf.append(float(np.mean(
                [(r["failclosed_closed_steps"] or 0) / r["steps"]
                 for r in rs])))
            po.append(100.0 * (float(np.mean([r["hvp_total"] for r in rs]))
                               / base_hvp - 1.0))
        o = np.argsort(x)
        out[fc] = tuple(np.asarray(v)[o] for v in (x, vr, wr, cf, po))
    return out, mobs_max, mobs_med


e1 = e1_series()
e2, E2_MOBS_MAX, E2_MOBS_MED = e2_series()

# style: line style = experiment, marker + colour = fail-closed on/off
EXP = {"E1": dict(ls="-"), "E2": dict(ls=(0, (3.2, 1.6)))}
FCS = {0: dict(marker="o", ms=3.2, mfc="none", mew=0.9, color=ORANGE),
       1: dict(marker="s", ms=3.0, color=BLUE)}

fig, axes = plt.subplots(1, 4, figsize=(DC_W, 2.0))
YLAB = ["certificate violation rate", "worst true err / bound",
        "closed-gate fraction", "probe overhead (%)"]
PANEL = ["(a)", "(b)", "(c)", "(d)"]

for k, (ax, ylab) in enumerate(zip(axes, YLAB)):
    for fc in (1, 0):
        for name, ser in (("E1", e1), ("E2", e2)):
            if fc not in ser or not len(ser[fc][0]):
                continue
            ax.plot(ser[fc][0], ser[fc][k + 1],
                    **EXP[name], **FCS[fc], zorder=3 - fc,
                    clip_on=False)
    ax.set_xscale("log")
    ax.set_xlabel(r"$M_H\,/\,M_H^{\mathrm{dep}}$", labelpad=1.0)
    ax.set_ylabel(ylab, labelpad=2.0)
    ax.set_xlim(6e-3, 1.7)
    ax.set_xticks([1e-2, 1e-1, 1e0])
    ax.set_xticklabels([r"$10^{-2}$", r"$10^{-1}$", r"$1$"])
    ax.grid(alpha=0.22, lw=0.4)
    ax.set_axisbelow(True)
    ax.text(0.0, 1.015, PANEL[k], transform=ax.transAxes, fontsize=7.5,
            va="bottom", ha="left")

axes[0].set_ylim(-0.05, 1.0)
axes[0].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
axes[0].text(0.50, 0.50, "zero violations\nat every point",
             transform=axes[0].transAxes, fontsize=7.0, color=BLACK,
             ha="center", va="center")

axes[1].set_ylim(0.0, 1.14)
axes[1].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
axes[1].axhline(1.0, color=BLACK, lw=0.7, ls=":", zorder=1)
axes[1].text(0.97, 1.005, "bound violated above", transform=axes[1].get_yaxis_transform(),
             fontsize=7.0, color=BLACK, ha="right", va="bottom")

axes[2].set_ylim(-0.05, 1.05)
axes[2].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

axes[3].set_ylim(-5, 118)
axes[3].set_yticks([0, 25, 50, 75, 100])
axes[3].axhline(100.0, color=BLACK, lw=0.7, ls=":", zorder=1)
axes[3].text(0.97, 101.0, "2$\\times$ probe cap",
             transform=axes[3].get_yaxis_transform(),
             fontsize=7.0, color=BLACK, ha="right", va="bottom")

handles = [
    Line2D([], [], color="0.30", ls="-", lw=1.1,
           label=r"teacher–student audit, $M_H^{\mathrm{dep}}{=}2.28$"),
    Line2D([], [], color="0.30", ls=(0, (3.2, 1.6)), lw=1.1,
           label=r"drifting Mackey–Glass, $M_H^{\mathrm{dep}}{=}5$"),
    Line2D([], [], color=ORANGE, ls="none", marker="o", ms=3.2, mfc="none",
           mew=0.9, label="monitor off"),
    Line2D([], [], color=BLUE, ls="none", marker="s", ms=3.0,
           label="fail-closed monitor on"),
]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
           handlelength=2.0, columnspacing=1.5, handletextpad=0.5,
           borderpad=0.0, borderaxespad=0.0, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(pad=0.30, w_pad=1.0, rect=(0, 0.115, 1, 0.955))

pdf = os.path.join(FIG, "fig10_misspec.pdf")
fig.savefig(pdf, bbox_inches="tight", pad_inches=0.01)
fig.savefig(os.path.join(FIG_PNG, "fig10_misspec.png"), dpi=300,
            bbox_inches="tight", pad_inches=0.01)
plt.close(fig)

try:
    from pypdf import PdfReader
    box = PdfReader(pdf).pages[0].mediabox
    w_in = float(box.width) / 72.0
    h_in = float(box.height) / 72.0
    print(f"natural size: {w_in:.4f} in x {h_in:.4f} in  "
          f"({'OK' if w_in <= 6.85 + 1e-6 else 'TOO WIDE'} vs 6.85 in)")
except Exception as exc:   # pragma: no cover
    print("pypdf check skipped:", exc)

print(f"E1 deployed prior {E1_MH_DEP}; E2 deployed prior {E2_MH_DEP}")
print(f"E2 deployed-arm M_obs: median {E2_MOBS_MED:.4g}, "
      f"max {E2_MOBS_MAX:.4g}")
for fc in (0, 1):
    print("E1 fc", fc, [np.round(v, 4).tolist() for v in e1[fc]])
    print("E2 fc", fc, [np.round(v, 6).tolist() for v in e2[fc]])
print("->", pdf)
