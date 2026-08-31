"""fig8: GPT-2 per-coordinate lambda trajectories (adaptation visibility).

(a) certified gated COHG (cohg_r0), all 3 seeds overlaid thin, seed 0 bold,
    gate-open events marked.
(b) the ungated seed that degraded (cohg_nogate, seed 1): lambda drifts freely.

Also emits results/reanalysis/gate_open_timing.md with per-seed gate-open
counts and their location relative to the domain boundaries (t=1000, 2000).
"""
from __future__ import annotations
import os, json, math, glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
E4 = os.path.join(ROOT, "results", "e4_v2")
FIG = os.path.join(ROOT, "paper", "main", "figs")
PNG = os.path.join(ROOT, "results", "figures")
OUT = os.path.join(ROOT, "results", "reanalysis")
os.makedirs(FIG, exist_ok=True); os.makedirs(PNG, exist_ok=True)

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
# Figure width is chosen so that the *tight-cropped* PDF MediaBox lands just
# inside the 3.3in single column (238pt); see the pypdf check in the header
# comment of this file's sibling generators.
SC_W = 3.35
SC_H = 3.75
ANN = 7          # in-panel annotation size (>= 7pt after LaTeX scaling)


def load(meth):
    out = {}
    for p in sorted(glob.glob(os.path.join(E4, f"gpt2_{meth}_lr0.001_s*.json"))):
        d = json.load(open(p))
        out[d["seed"]] = d
    return out


def lam_arr(d):
    h = np.asarray(d["lam_hist"], dtype=float)
    return h[:, 0], h[:, 1:]           # t (S,), lam (S, 6)


def opens(d):
    """Sample intervals in which lambda changed, i.e. gate-open events.

    lam_hist is sampled every 20 steps, so an event is localised to the
    half-open interval (t_prev, t_cur]; we report t_cur as its upper bound.
    """
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
    ck = load("cohg")

    # --------------------------------------------------------------- figure
    fig, ax = plt.subplots(2, 1, figsize=(SC_W, SC_H), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.0, 1.45]))

    # (a) gated -- all six coordinates coincide exactly, so we offset them
    #     vertically by OFS for visibility (stated in the caption/label).
    OFS = 0.012
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
    ymark = None
    for s, d in sorted(r0.items()):
        for e in opens(d):
            ymark = e["t_hi"]
            a.axvline(ymark, color="0.35", lw=0.6, ls=":", zorder=1)
    a.set_ylim(-6.99, -6.34)
    if ymark is not None:
        a.annotate("only gate opening of the run", xy=(ymark, -6.70),
                   xytext=(360, -6.70), fontsize=ANN, color="0.2",
                   va="center", ha="left",
                   arrowprops=dict(arrowstyle="->", lw=0.5, color="0.35"))
    a.set_ylabel(r"$\lambda_j$  (log LR)")
    a.text(0.015, 0.035,
           "(a) certified gate ON (COHG); curves offset",
           transform=a.transAxes, fontsize=ANN, va="bottom",
           bbox=dict(fc="w", ec="none", alpha=0.85, pad=1.0))

    # (b) ungated, degraded seed
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
            f"{d['online_ppl']:.1f} (gated {r0[0]['online_ppl']:.1f})\n"
            r"dash-dot: clamps $[\log 10^{-6},\log 10^{-1}]$",
            transform=b_.transAxes, fontsize=ANN, va="top",
            linespacing=1.25,
            bbox=dict(fc="w", ec="none", alpha=0.72, pad=1.0))

    for a_ in ax:
        a_.set_xlim(0, 3000)
        a_.tick_params(length=2.5)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    # domain labels along the top of panel (a)
    for xc, lab in ((500, "domain 1"), (1500, "domain 2"), (2500, "domain 3")):
        ax[0].text(xc, 1.03, lab, transform=ax[0].get_xaxis_transform(),
                   ha="center", va="bottom", fontsize=ANN, color="0.4")

    # six-entry coordinate legend, reflowed 2 rows x 3 columns below the axes
    handles = [Line2D([0], [0], color=OI[c], lw=1.4) for c in range(6)]
    fig.tight_layout(pad=0.3, h_pad=0.55, rect=(0.0, 0.075, 1.0, 1.0))
    fig.legend(handles, COORD, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.004), frameon=False,
               handlelength=1.5, columnspacing=1.6, handletextpad=0.45,
               labelspacing=0.3, fontsize=7)
    fig.savefig(os.path.join(FIG, "fig8_gpt2_lambda.pdf"),
                bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(PNG, "fig8_gpt2_lambda.png"), dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # ------------------------------------------------------------- timing md
    L = []
    A = L.append
    A("# GPT-2 (E4, results/e4_v2): gate-open timing\n")
    A("`lam_hist` is sampled every 20 steps, so a lambda change between two "
      "samples localises a gate-open event to the half-open interval "
      "`(t_prev, t_cur]`. Domain boundaries are at t=1000 and t=2000 "
      "(3 domains: D1 t<1000, D2 1000<=t<2000, D3 t>=2000).\n")
    A("The stored `gate_open_frac` gives the exact number of accepted meta-updates: "
      "`gate_open_frac x steps`.\n")
    A("| arm | seed | gate_open_frac | exact #opens | detected change intervals | "
      "coords moved | delta lambda | domain |")
    A("|---|---|---|---|---|---|---|---|")
    for name, runs in (("cohg_r0 (rank-0, gated)", r0),
                       ("cohg (rank-4, gated)", ck),
                       ("cohg_nogate (gate forced open)", ng)):
        for s, d in sorted(runs.items()):
            ev = opens(d)
            nexact = d["gate_open_frac"] * d["steps"]
            if len(ev) <= 4:
                iv = "; ".join(f"({e['t_lo']},{e['t_hi']}]" for e in ev) or "none"
                cd = "; ".join(",".join(e["coords"]) for e in ev) or "-"
                dl = "; ".join(",".join(f"{x:+.2f}" for x in e["delta"]) for e in ev) or "-"
                dom = "; ".join("D1" if e["t_hi"] <= 1000 else
                                ("D2" if e["t_hi"] <= 2000 else "D3") for e in ev) or "-"
            else:
                iv = f"{len(ev)} intervals (every sample)"
                cd = "all 6"; dl = "+-0.40 per step"; dom = "D1,D2,D3"
            A(f"| {name} | {s} | {d['gate_open_frac']:.6g} | {nexact:.0f} | {iv} | "
              f"{cd} | {dl} | {dom} |")

    A("\n## Per-seed summary (gated arms)\n")
    for name, runs in (("cohg_r0", r0), ("cohg", ck)):
        for s, d in sorted(runs.items()):
            ev = opens(d)
            n_before = sum(1 for e in ev if e["t_hi"] <= 1000)
            n_mid = sum(1 for e in ev if 1000 < e["t_hi"] <= 2000)
            n_after = sum(1 for e in ev if e["t_hi"] > 2000)
            A(f"- `{name}` seed {s}: exact opens = "
              f"{d['gate_open_frac']*d['steps']:.0f} / {d['steps']} steps "
              f"({100*d['gate_open_frac']:.3f}% of steps); detected at "
              f"{[ (e['t_lo'], e['t_hi']) for e in ev ]}; "
              f"D1={n_before}, D2={n_mid}, D3={n_after}. "
              f"Total lambda displacement per coord: "
              f"{[round(float(x),3) for x in (lam_arr(d)[1][-1] - lam_arr(d)[1][0])]}")
    A("\n## Ungated arm (cohg_nogate)\n")
    for s, d in sorted(ng.items()):
        t, Lm = lam_arr(d)
        A(f"- seed {s}: gate forced open every step "
          f"({d['gate_open_frac']*d['steps']:.0f}/{d['steps']}); "
          f"online PPL {d['online_ppl']:.3f}; lambda start "
          f"{Lm[0].round(3).tolist()} -> end {Lm[-1].round(3).tolist()}; "
          f"per-coord range (max-min) {np.round(Lm.max(0)-Lm.min(0),3).tolist()}; "
          f"lambda at boundaries t=1000 -> {Lm[np.argmin(np.abs(t-1000))].round(3).tolist()}, "
          f"t=2000 -> {Lm[np.argmin(np.abs(t-2000))].round(3).tolist()}")
    open(os.path.join(OUT, "gate_open_timing.md"), "w").write("\n".join(L) + "\n")
    print("degraded ungated seed:", sdeg)
    print("wrote fig8 + gate_open_timing.md")


if __name__ == "__main__":
    main()
