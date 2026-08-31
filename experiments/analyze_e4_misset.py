"""Phase-2 deliverable: results/e4_misset/REPORT.md (review P7).

"Does COHG adapt when adaptation is actually needed?"  The E4 GPT-2 stream is
rerun from a deliberately mis-set initialisation eta0 = 1e-4 -- ten times below
the 1e-3 operating point of the shipped grid, mirroring the E2 mis-set design.
The post-hoc best fixed learning rate on this stream is 3e-3.

Arms (results/e4_misset, seeds 0-2):
    fixed        lr 1e-4      no adaptation at all
    cohg_r0      lr 1e-4      gated, corrected (vector) held bound
    cohg_nogate  lr 1e-4      same hypergradient, gate always open

References (results/e4_v2): fixed lr 1e-3 (seeds 0-7) and fixed lr 3e-3
(seeds 0-2).

Answers, per arm: number of gate openings and at which steps, how far lambda
climbs (and the learning rate that implies), whether the gate re-closes once
the learning rate has been corrected, and the final online perplexity against
each fixed reference.
"""

from __future__ import annotations

import json
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
MIS = os.path.join(ROOT, "results", "e4_misset")
V2 = os.path.join(ROOT, "results", "e4_v2")
SEEDS = (0, 1, 2)


def load(d, method, lr, seed):
    p = os.path.join(d, "gpt2_%s_lr%g_s%d.json" % (method, lr, seed))
    return json.load(open(p)) if os.path.exists(p) else None


def group(d, method, lr, seeds):
    return [x for x in (load(d, method, lr, s) for s in seeds) if x is not None]


def ppl_stat(runs):
    v = [r["online_ppl"] for r in runs]
    if not v:
        return None
    return (st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0), len(v))


def lam_summary(d):
    """(lambda_0, lambda_max, lambda_final, lr_final, step of lambda_max)."""
    rows = [(int(r[0]), [float(x) for x in r[1:]]) for r in d["lam_hist"]]
    l0 = max(rows[0][1])
    best_t, best = max(((t, max(v)) for t, v in rows), key=lambda kv: kv[1])
    lf = max(rows[-1][1])
    return l0, best, lf, math.exp(lf), best_t


def openings(d):
    gs = d.get("gate_open_steps")
    n = (int(round(d["gate_open_frac"] * d["steps"]))
         if d.get("gate_open_frac") is not None else 0)
    return n, (gs if gs is not None else [])


def main():
    L = []
    A = L.append
    A("# GPT-2 mis-set initialisation: adaptation when it is needed "
      "(review P7)\n")
    A("Stream, flags and seeds are the e4_v2 standard configuration "
      "(`--tokens-per-domain 512000 --max-steps 3000 --batch 2 --seq-len 256 "
      "--probe-every 100 --kw-eps 0.15 --meta-lr 0.4`, order "
      "`wiki,news,code`, drift at steps 1000 and 2000); the only change is "
      "`--lr 1e-4`, i.e. an initial learning rate ten times too small.  All "
      "certificate arms use the corrected vector-valued held bound "
      "(Phase 1 / `--legacy-hold` off).\n")

    arms = [("fixed", 1e-4, MIS, "fixed lr 1e-4 (mis-set, no adaptation)"),
            ("cohg_r0", 1e-4, MIS, "COHG r0 lr 1e-4 (gated)"),
            ("cohg_nogate", 1e-4, MIS, "COHG no-gate lr 1e-4 (ungated)"),
            ("fixed", 1e-3, V2, "fixed lr 1e-3 (shipped operating point)"),
            ("fixed", 3e-3, V2, "fixed lr 3e-3 (post-hoc best fixed)")]

    A("## Online perplexity\n")
    A("| arm | seeds | online_ppl mean | sd |")
    A("|---|---|---|---|")
    stats = {}
    for m, lr, d, label in arms:
        seeds = range(8) if (d is V2 and lr == 1e-3) else SEEDS
        runs = group(d, m, lr, seeds)
        s = ppl_stat(runs)
        stats[(m, lr)] = (s, runs)
        A("| %s | %s | %s | %s |"
          % (label, (s[2] if s else 0),
             ("%.4f" % s[0]) if s else "-", ("%.4f" % s[1]) if s else "-"))
    A("")

    base = stats[("fixed", 1e-4)][0]
    if base:
        A("Relative to the mis-set fixed baseline (%.4f):\n" % base[0])
        A("| arm | online_ppl | delta vs fixed 1e-4 | %% of the fixed-1e-4 "
          "gap to fixed 3e-3 closed |")
        A("|---|---|---|---|")
        best = stats[("fixed", 3e-3)][0]
        gap = (base[0] - best[0]) if best else None
        for m, lr, d, label in arms:
            s = stats[(m, lr)][0]
            if not s:
                continue
            frac = ("%.1f%%" % (100.0 * (base[0] - s[0]) / gap)
                    if gap else "-")
            A("| %s | %.4f | %+.4f | %s |"
              % (label, s[0], s[0] - base[0], frac))
        A("")

    A("## Gate behaviour and the lambda trajectory\n")
    A("lambda is the per-group log learning rate; lambda_0 = log(1e-4) = "
      "%.4f, and the shipped operating point is log(1e-3) = %.4f, the "
      "post-hoc best log(3e-3) = %.4f.\n"
      % (math.log(1e-4), math.log(1e-3), math.log(3e-3)))
    A("| arm | seed | gate openings | opening steps | lambda_max | "
      "step of lambda_max | lambda_final | lr_final | gate re-closes? |")
    A("|---|---|---|---|---|---|---|---|---|")
    for m in ("cohg_r0", "cohg_nogate"):
        for s in SEEDS:
            d = load(MIS, m, 1e-4, s)
            if d is None:
                A("| %s | %d | *missing* | | | | | | |" % (m, s))
                continue
            n, steps = openings(d)
            l0, lmax, lf, lrf, tmax = lam_summary(d)
            shown = (str(steps[:12]) + (" ..." if len(steps) > 12 else "")
                     if steps else "n/a")
            if m == "cohg_nogate":
                recl = "n/a (gate always open)"
            elif not steps:
                recl = "never opened"
            else:
                last = steps[-1]
                recl = ("yes, last opening at step %d of %d (closed for the "
                        "final %.0f%% of the stream)"
                        % (last, d["steps"],
                           100.0 * (d["steps"] - last) / d["steps"]))
            A("| %s | %d | %d | %s | %.4f | %d | %.4f | %.3g | %s |"
              % (m, s, n, shown, lmax, tmax, lf, lrf, recl))
    A("")

    A("## Reading\n")
    cg = [load(MIS, "cohg_r0", 1e-4, s) for s in SEEDS]
    cg = [d for d in cg if d is not None]
    if cg:
        ns = [openings(d)[0] for d in cg]
        firsts = [openings(d)[1][0] for d in cg if openings(d)[1]]
        lasts = [openings(d)[1][-1] for d in cg if openings(d)[1]]
        lrs = [lam_summary(d)[3] for d in cg]
        A("* The gate opens %s times per seed (first opening at step %s, last "
          "at step %s)."
          % (sorted(set(ns)), sorted(set(firsts)), sorted(set(lasts))))
        A("* lambda climbs to a final learning rate of %s, against the mis-set "
          "1e-4 start and the post-hoc best 3e-3."
          % ", ".join("%.3g" % x for x in lrs))
    A("")
    A("<sub>generated by `code/experiments/analyze_e4_misset.py`</sub>")

    txt = "\n".join(L) + "\n"
    os.makedirs(MIS, exist_ok=True)
    out = os.path.join(MIS, "REPORT.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)
    print("-> %s" % out)


if __name__ == "__main__":
    main()
