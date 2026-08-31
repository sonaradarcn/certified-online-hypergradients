"""fig9: GPT-2 lambda trajectories under the ALTERNATE domain order code -> news -> wiki.

Companion to fig8 (standard order wiki -> news -> code), built from
results/e4_orders/gpt2order_cnw_*.json.

(a) certified gated COHG (cohg_r0), all 3 seeds overlaid, seed 0 bold; the
    single gate opening of each run is marked. It lands in (0, 20], i.e. at
    stream start -- never at or after a domain boundary (t = 1000, 2000).
(b) the worst ungated seed (cohg_nogate, seed 2, online PPL 1102.4): lambda
    slams between the clamps for the whole run.

Print size: single column 3.3in, 8pt, Okabe-Ito, no titles.
"""
from __future__ import annotations
import os, json, math, glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EO = os.path.join(ROOT, "results", "e4_orders")
FIG = os.path.join(ROOT, "paper", "main", "figs")
PNG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)
os.makedirs(PNG, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "lines.linewidth": 1.0, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "figure.dpi": 150, "pdf.fonttype": 42, "ps.fonttype": 42,
})

OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
COORD = ["emb", "h0-2", "h3-5", "h6-8", "h9-11", "ln_f"]
BOUND = [1000, 2000]
# Width chosen so the *tight-cropped* PDF MediaBox lands just inside the
# 3.3in (238pt) single column, i.e. LaTeX does not shrink the fonts.
SC_W = 3.35
SC_H = 3.75
ANN = 7          # in-panel annotation size (>= 7pt at print size)


def load(meth):
    out = {}
    for p in sorted(glob.glob(os.path.join(EO, f"gpt2order_cnw_{meth}_lr0.001_s*.json"))):
        d = json.load(open(p))
        out[d["seed"]] = d
    return out


def lam_arr(d):
    h = np.asarray(d["lam_hist"], dtype=float)
    return h[:, 0], h[:, 1:]


def opens(d):
    t, L = lam_arr(d)
    ev = []
    for i in range(1, len(t)):
        ch = np.nonzero(np.abs(L[i] - L[i - 1]) > 1e-9)[0]
        if len(ch):
            ev.append(dict(t_lo=int(t[i - 1]), t_hi=int(t[i]),
                           coords=[COORD[c] for c in ch],
                           delta=[float(L[i][c] - L[i - 1][c]) for c in ch]))
    return ev


def main():
    r0 = load("cohg_r0")
    ng = load("cohg_nogate")

    fig, ax = plt.subplots(2, 1, figsize=(SC_W, SC_H), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.0, 1.45]))

    # ---------------------------------------------------------- (a) gated
    OFS = 0.018
    a = ax[0]
    for b in BOUND:
        a.axvline(b, color="0.6", lw=0.7, ls="--", zorder=1)
    for s, d in sorted(r0.items()):
        t, L = lam_arr(d)
        for c in range(6):
            y = L[:, c] + c * OFS
            if s == 0:
                a.plot(t, y, color=OI[c], lw=1.2, zorder=3, label=COORD[c])
            else:
                a.plot(t, y, color=OI[c], lw=0.45, alpha=0.6, zorder=2)
    tmark = None
    for s, d in sorted(r0.items()):
        for e in opens(d):
            tmark = e["t_hi"]
            a.axvline(tmark, color="0.35", lw=0.6, ls=":", zorder=1)
    a.set_ylim(-7.66, -6.02)
    if tmark is not None:
        a.annotate("only gate opening of the run\n"
                   "(one accepted meta-update, $t\\leq20$)",
                   xy=(tmark, -6.88), xytext=(330, -6.88), fontsize=ANN,
                   color="0.2", va="center", ha="left", linespacing=1.2,
                   arrowprops=dict(arrowstyle="->", lw=0.5, color="0.35"))
    a.set_ylabel(r"$\lambda_j$  (log LR)")
    a.text(0.015, 0.975,
           "(a) certified gate ON (COHG); curves offset",
           transform=a.transAxes, fontsize=ANN, va="top",
           bbox=dict(fc="w", ec="none", alpha=0.85, pad=1.0))

    # -------------------------------------------------------- (b) ungated
    b_ = ax[1]
    sdeg = max(ng, key=lambda s: ng[s]["online_ppl"])
    d = ng[sdeg]
    t, L = lam_arr(d)
    for b in BOUND:
        b_.axvline(b, color="0.6", lw=0.7, ls="--", zorder=1)
    for c in range(6):
        b_.plot(t, L[:, c], color=OI[c], lw=0.8)
    b_.axhline(math.log(1e-6), color="0.7", lw=0.5, ls="-.")
    b_.axhline(math.log(0.1), color="0.7", lw=0.5, ls="-.")
    b_.set_ylim(-14.6, -0.1)
    b_.set_ylabel(r"$\lambda_j$  (log LR)")
    b_.set_xlabel("stream step $t$")
    b_.text(0.015, 0.975,
            f"(b) gate OFF, seed {sdeg}: online PPL "
            f"{d['online_ppl']:.0f} (gated {r0[0]['online_ppl']:.1f})\n"
            r"dash-dot: clamps $[\log 10^{-6},\log 10^{-1}]$",
            transform=b_.transAxes, fontsize=ANN, va="top",
            linespacing=1.25,
            bbox=dict(fc="w", ec="none", alpha=0.72, pad=1.0))

    for a_ in ax:
        a_.set_xlim(0, 3000)
        a_.tick_params(length=2.5)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    for xc, lab in ((500, "code"), (1500, "news"), (2500, "wiki")):
        ax[0].text(xc, 1.03, lab, transform=ax[0].get_xaxis_transform(),
                   ha="center", va="bottom", fontsize=ANN, color="0.4")

    # six-entry coordinate legend, reflowed 2 rows x 3 columns below the axes
    handles = [Line2D([0], [0], color=OI[c], lw=1.4) for c in range(6)]
    fig.tight_layout(pad=0.3, h_pad=0.55, rect=(0.0, 0.075, 1.0, 1.0))
    fig.legend(handles, COORD, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.004), frameon=False,
               handlelength=1.5, columnspacing=1.6, handletextpad=0.45,
               labelspacing=0.3, fontsize=7)
    fig.savefig(os.path.join(FIG, "fig9_gpt2_order.pdf"), bbox_inches="tight",
                pad_inches=0.02)
    fig.savefig(os.path.join(PNG, "fig9_gpt2_order.png"), dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("worst ungated seed:", sdeg, ng[sdeg]["online_ppl"])
    print("wrote fig9_gpt2_order.{pdf,png}")


if __name__ == "__main__":
    main()
