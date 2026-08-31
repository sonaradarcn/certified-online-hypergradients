"""Round-3 GPU reanalysis: E4 mis-set init, E4 reverse-order n=8, E3 traced.

Reads raw run JSONs from results/e4_misset, results/e4_orders, results/e3_traced
(with results/e4_v2 and results/e3 as reference sets) and writes
results/reanalysis/round3_gpu.md.  Analysis only; nothing under results/ other
than round3_gpu.md is written.

Metric conventions are byte-for-byte the ones in _reanalyze.py:
  * ddof = 1 for every std;
  * unified spike rule: deque of the last 500 FINITE losses, once it holds >=100
    entries a step is a spike iff loss_t > 10 * median(window); every non-finite
    loss is also an event;
  * max-excess = max(finite loss) / median(finite loss);
  * worst-window = max_e mean(loss[e-100:e]) over e = 100..n, finite entries only
    (a window with no finite entry contributes +inf), exp(.) on GPT-2;
  * paired tests are EXACT two-sided sign-flip tests over all 2^n assignments.
"""
import os, re, json, math, glob, itertools
from collections import deque

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTMD = os.path.join(ROOT, "results", "reanalysis", "round3_gpu.md")


# ----------------------------------------------------------------- metrics
def unified_metrics(losses):
    n = len(losses)
    fin = [x for x in losses if math.isfinite(x)]
    n_nf = n - len(fin)
    win = deque(maxlen=500); n_spike = 0; first_spike = None
    for t, x in enumerate(losses):
        if not math.isfinite(x):
            continue
        if len(win) >= 100:
            w = sorted(win); m = w[len(w) // 2]
            if x > 10.0 * m:
                n_spike += 1
                if first_spike is None:
                    first_spike = t
        win.append(x)
    if fin:
        s = sorted(fin); med_all = s[len(s) // 2]; max_fin = max(fin)
        max_excess = max_fin / med_all if med_all > 0 else float("inf")
    else:
        med_all = max_fin = float("nan"); max_excess = float("inf")
    W = 100
    worst = float("-inf"); worst_fin = float("-inf"); n_empty = 0
    if n >= W:
        for e in range(W, n + 1):
            seg = losses[e - W:e]
            sf = [x for x in seg if math.isfinite(x)]
            if sf:
                m = sum(sf) / len(sf)
                worst_fin = max(worst_fin, m)
            else:
                m = float("inf"); n_empty += 1
            worst = max(worst, m)
    else:
        worst = worst_fin = (sum(fin) / len(fin)) if fin else float("inf")
    first_nf = next((t for t, x in enumerate(losses) if not math.isfinite(x)), None)
    return dict(n_steps=n, n_finite=len(fin), n_nonfinite=n_nf, n_spike=n_spike,
                n_event_unified=n_spike + n_nf, first_spike_t=first_spike,
                median_loss=med_all, max_finite_loss=max_fin,
                max_excess=max_excess, worst_window_mean=worst,
                worst_window_finite=worst_fin, n_empty_windows=n_empty,
                first_nonfinite_t=first_nf,
                mean_loss_finite=(sum(fin) / len(fin)) if fin else float("nan"))


def sexp(x):
    try:
        return math.exp(x)
    except OverflowError:
        return float("inf")


def mstd(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return float("nan"), float("nan"), 0
    n = len(v); m = sum(v) / n
    if n < 2:
        return m, float("nan"), n
    return m, math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)), n


def med(v):
    s = sorted(v); n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def signflip_p(d):
    n = len(d); obs = abs(sum(d) / n); cnt = 0
    for signs in itertools.product([1, -1], repeat=n):
        if abs(sum(s * x for s, x in zip(signs, d)) / n) >= obs - 1e-15:
            cnt += 1
    return cnt / 2 ** n, sum(d) / n


def f(m, s, p=4):
    if not math.isfinite(m):
        return "inf"
    if not math.isfinite(s):
        return f"{m:.{p}f}"
    return f"{m:.{p}f} +- {s:.{p}f}"


def g(x, p=3):
    """Compact rendering that survives 1e38 blow-ups."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if not math.isfinite(x):
        return "**inf**"
    return f"{x:.{p}f}" if abs(x) < 1e5 else f"{x:.2e}"


def lg(x):
    """log10, robust to inf / non-positive."""
    if not math.isfinite(x):
        return float("inf")
    return math.log10(x) if x > 0 else float("-inf")


OUT = []
def A(x=""):
    OUT.append(x)


# ========================================================== preamble
A("# Round-3 GPU reanalysis: mis-set initialization, reverse-order n = 8, traced E3")
A()
A("Analysis only -- no paper text and no result artifact was edited. Every number "
  "below is recomputed from the raw run JSONs by "
  "`results/reanalysis/_round3_gpu.py`, which reuses the metric code of "
  "`_reanalyze.py` verbatim.")
A()
A("**Conventions.** `ddof = 1` for every std. **Unified spike rule**, identical in "
  "every regime: maintain a deque of the last 500 *finite* losses; once it holds "
  ">= 100 entries a step counts as a spike iff `loss_t > 10 x median(window)`; every "
  "non-finite loss is additionally an event; `unified events = spikes + non-finite`. "
  "**max-excess** = `max(finite loss) / median(finite loss)`, in loss space. "
  "**worst-window** = `max_e mean(loss[e-100 .. e-1])` over all `e = 100..n`, "
  "averaging finite entries only; a window containing *no* finite entry contributes "
  "`+inf` (this is what produces the `inf` cells in section 3 -- a separate "
  "finite-restricted variant is tabulated alongside so the arms remain comparable). "
  "On GPT-2 worst-window is reported as `exp(.)` = perplexity; on CIFAR-100 it stays "
  "in raw cross-entropy. **Paired tests** are exact two-sided sign-flip "
  "(randomization) tests enumerating all `2^n` sign assignments; `p` floors are "
  "`2/2^n` = 0.25 (n=3), 0.0078125 (n=8), 0.001953125 (n=10).")
A()
A("Three completed GPU result sets are covered, all launched by "
  "`code/experiments/launch_r3_chain.py` on the local 2x RTX 3080:")
A()
A("| set | directory | runs | phase |")
A("|---|---|---|---|")
A("| mis-set initialization | `results/e4_misset/` | 9 | p2_misset (review P7) |")
A("| reverse-order expansion | `results/e4_orders/` (+5 new) | 17 total | p2_misset |")
A("| traced E3 | `results/e3_traced/` | 80 | p3_e3traced (review P5/P9) |")
A()
A("### The three plain answers up front")
A()
A("1. **Mis-set init (P7).** *The certified gate does adapt when adaptation is "
  "genuinely needed, and it does stop when it stops -- but it stops far too early.* "
  "From lr0 = 1e-4 it opens the gate **twice** (steps 1 and 2, all three seeds) "
  "versus **once** from the well-set 1e-3, and every accepted move raises the LR. It "
  "then never opens again for 2997 steps and never at a domain boundary. The LR ends "
  "at only 1.49x-2.06x the mis-set init against the 10x needed, recovering 23.2% of "
  "the mis-set PPL penalty (24.888 -> 24.006 against 21.092 well-set). Safety is "
  "untouched: 0/3 degraded, 0 spikes, 0 non-finite, max-excess indistinguishable from "
  "the fixed baseline, while the ungated ablation from the same start degrades 1/3 "
  "with a worst-window PPL of 3.7e6.")
A("2. **Reverse order at n = 8.** *The \"+0.30 n.s. code-first penalty\" does not "
  "change in kind.* The paired test is arithmetically unchanged (n = 3, "
  "+0.3043 PPL, p = 0.5) because `fixed` gained no seeds; the unpaired n = 8 view "
  "softens the gap to +0.2837 PPL, still smaller than the gated arm's own seed std "
  "(0.3881) and still unsupported. Gate timing for the five new seeds is now **exact** "
  "and confirms the finding: `gate_open_steps = [1]`, exactly one opening, well "
  "before step 20, zero at either boundary, in every one of the five. "
  "Degraded runs: **0 / 8**.")
A("3. **Traced E3.** *Both earlier claims hold, in opposite regimes.* At **ewc10** "
  "the gate separation is real and the trace metrics strengthen it: `cohg` is not "
  "merely as safe as `fixed` and `hd` but measurably safer (lower max-excess on 9/10 "
  "and 10/10 seeds respectively), while the ungated ablation blows up in 6/10 seeds. "
  "At **ewc1000** HD's dominance holds and hardens: `hd` triggers 0/10 and blows up "
  "1/10, `cohg` triggers **6/10** and blows up 6/10 -- the worst trigger rate of any "
  "arm anywhere in E3. **Caveat:** the traced set's ewc1000 block does not reproduce "
  "the canonical `results/e3` seed for seed (max |delta acc| 0.296); the ewc10 block "
  "does (max 0.041).")
A()

# ================================================================ PART 1
MIS = os.path.join(ROOT, "results", "e4_misset")
E4V2 = os.path.join(ROOT, "results", "e4_v2")
LAM_MIN, LAM_MAX = math.log(1e-6), math.log(0.1)
COORDS = ["emb", "h0-2", "h3-5", "h6-8", "h9-11", "ln_f"]


def load_e4(path):
    d = json.load(open(path))
    d["_um"] = unified_metrics(d["losses"])
    return d


mis = {}
for p in sorted(glob.glob(os.path.join(MIS, "*.json"))):
    m = re.match(r"^gpt2_(?P<meth>.+)_lr0\.0001_s(?P<s>\d+)\.json$", os.path.basename(p))
    mis[(m["meth"], int(m["s"]))] = load_e4(p)

ARMS1 = ["fixed", "cohg_r0", "cohg_nogate"]
LABEL1 = {"fixed": "fixed lr 1e-4", "cohg_r0": "cohg_r0 (gated) lr0 1e-4",
          "cohg_nogate": "cohg_nogate lr0 1e-4"}

A("---")
A()
A("## 1. E4 mis-set initialization (`results/e4_misset/`, lr0 = 1e-4, seeds 0-2)")
A()
A("Review item P7, *\"adaptation when it is actually needed\"*: the stream and every "
  "other flag are the `e4_v2` standard (wiki -> news -> code, 2999 steps, boundaries "
  "t = 1000 / 2000, 512k tokens per domain, `meta_lr = 0.4`, `K = 20`, `gamma = 0.9`, "
  "`probe_every = 100`, `kw_eps = 0.15`), and the *only* change is that the initial "
  "learning rate is set to **1e-4, i.e. 10x below the well-set default 1e-3 and 30x "
  "below the post-hoc-best 3e-3**. All 9 runs carry `legacy_hold = False`, "
  "`held_bound = vector_prop10` (the corrected Proposition-10 vector bound), so the "
  "whole block is post-fix provenance.")
A()
A("`lam_hist` is 150 rows `[t, lam_0..lam_5]` sampled every 20 steps "
  "(t = 0, 20, ..., 2980), and both COHG arms additionally store "
  "`gate_open_steps`, so gate timing here is **exact**, not interval-localised. "
  "Lambda init = `log 1e-4 = -9.2103` on all six LR groups "
  "(emb, h0-2, h3-5, h6-8, h9-11, ln_f); clamps `[log 1e-6, log 1e-1] = "
  "[-13.8155, -2.3026]`. The certified step is "
  "`lam_j <- lam_j - meta_lr * s_j * sign(ghat_j)` with "
  "`s_j = min(1, (|ghat_j| - beta_j)/|ghat_j|) <= 1`, so an accepted move is *at "
  "most* 0.40 in lambda (a factor 1.4918 in LR) per coordinate.")
A()

A("### 1.1 Online PPL, worst-window, max-excess, events")
A()
A("| arm | n | online PPL mean +- std | median | min | max | worst-window PPL | "
  "max-excess | spikes | non-finite | unified events | events (stored) |")
A("|---|---|---|---|---|---|---|---|---|---|---|---|")
rows1 = {}
for a in ARMS1:
    rs = [mis[(a, s)] for s in range(3)]
    ppl = [r["online_ppl"] for r in rs]
    ww = [sexp(r["_um"]["worst_window_mean"]) for r in rs]
    mx = [r["_um"]["max_excess"] for r in rs]
    sp = [r["_um"]["n_spike"] for r in rs]
    nf = [r["_um"]["n_nonfinite"] for r in rs]
    ue = [r["_um"]["n_event_unified"] for r in rs]
    ev = [r["events"] for r in rs]
    rows1[a] = dict(ppl=ppl, ww=ww, mx=mx, sp=sp)
    A(f"| {LABEL1[a]} | 3 | **{f(*mstd(ppl)[:2])}** | {med(ppl):.4f} | {min(ppl):.4f} | "
      f"{max(ppl):.4f} | {g(mstd(ww)[0], 2)} +- {g(mstd(ww)[1], 2)} | "
      f"{f(*mstd(mx)[:2])} | {f(*mstd(sp)[:2], p=2)} | {f(*mstd(nf)[:2], p=2)} | "
      f"{f(*mstd(ue)[:2], p=2)} | {f(*mstd(ev)[:2], p=2)} |")
A()
A("Per-seed online PPL / worst-window PPL / max-excess / spikes:")
A()
A("| arm | s0 | s1 | s2 |")
A("|---|---|---|---|")
for a in ARMS1:
    r = rows1[a]
    A(f"| {LABEL1[a]} | " + " | ".join(
        f"{r['ppl'][s]:.4f} / {g(r['ww'][s], 2)} / {r['mx'][s]:.4f} / {r['sp'][s]}"
        for s in range(3)) + " |")
A()

ref = {}
for tag, pat in [("fixed 1e-3", "gpt2_fixed_lr0.001_s*.json"),
                 ("fixed 3e-3", "gpt2_fixed_lr0.003_s*.json"),
                 ("cohg_r0 1e-3", "gpt2_cohg_r0_lr0.001_s*.json"),
                 ("hd ml2 1e-3", "gpt2_hd_lr0.001_ml2_s*.json")]:
    ds = [load_e4(p) for p in sorted(glob.glob(os.path.join(E4V2, pat)))]
    ref[tag] = dict(n=len(ds), ppl=[d["online_ppl"] for d in ds],
                    ww=[sexp(d["_um"]["worst_window_mean"]) for d in ds],
                    mx=[d["_um"]["max_excess"] for d in ds])

A("### 1.2 Where the mis-set arms land relative to the well-set reference points")
A()
A("Reference rows recomputed from `results/e4_v2/` through the identical code path.")
A()
A("| operating point | n | online PPL | worst-window PPL (mean) | max-excess (mean) |")
A("|---|---|---|---|---|")
order12 = [("fixed lr **1e-4** (mis-set, 10x too small)", rows1["fixed"]),
           ("**cohg_r0 from lr0 = 1e-4** (gated, must adapt)", rows1["cohg_r0"]),
           ("cohg_nogate from lr0 = 1e-4", rows1["cohg_nogate"])]
for lbl, r in order12:
    A(f"| {lbl} | 3 | **{f(*mstd(r['ppl'])[:2])}** | {g(mstd(r['ww'])[0], 2)} | "
      f"{g(mstd(r['mx'])[0], 4)} |")
for tag, lbl in [("fixed 1e-3", "fixed lr 1e-3 (well-set default)"),
                 ("cohg_r0 1e-3", "cohg_r0 from lr0 = 1e-3 (well-set)"),
                 ("hd ml2 1e-3", "hd ml2 from lr0 = 1e-3"),
                 ("fixed 3e-3", "fixed lr 3e-3 (**post-hoc best** grid point)")]:
    r = ref[tag]
    A(f"| {lbl} | {r['n']} | {f(*mstd(r['ppl'])[:2])} | {g(mstd(r['ww'])[0], 2)} | "
      f"{g(mstd(r['mx'])[0], 4)} |")
A()

pairs1 = []
for a, b in [("cohg_r0", "fixed"), ("cohg_nogate", "fixed"), ("cohg_nogate", "cohg_r0")]:
    d = [mis[(a, s)]["online_ppl"] - mis[(b, s)]["online_ppl"] for s in range(3)]
    p, mn = signflip_p(d)
    pairs1.append((a, b, d, mn, p))
A("Paired sign-flip tests inside the mis-set block (common seeds 0-2, n = 3, "
  "p floor 0.25):")
A()
A("| contrast | per-seed diffs (s0, s1, s2) | mean diff | sign pattern | p (exact) |")
A("|---|---|---|---|---|")
for a, b, d, mn, p in pairs1:
    A(f"| `{a}` - `{b}` | {d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f} | **{mn:+.4f}** | "
      f"{sum(1 for x in d if x < 0)} neg / {sum(1 for x in d if x > 0)} pos | {p:g} |")
A()
fixmis = mstd(rows1["fixed"]["ppl"])[0]
r0mis = mstd(rows1["cohg_r0"]["ppl"])[0]
fix3 = mstd(ref["fixed 1e-3"]["ppl"])[0]
best3 = mstd(ref["fixed 3e-3"]["ppl"])[0]
gap_fix = fixmis - r0mis
A(f"**Recovery accounting.** Mis-setting the LR costs "
  f"`fixed 1e-4 - fixed 1e-3` = **{fixmis - fix3:+.4f} PPL** "
  f"({fixmis:.4f} vs {fix3:.4f}). The certified gate recovers **{gap_fix:.4f} PPL** "
  f"of that, i.e. **{100 * gap_fix / (fixmis - fix3):.1f}%** of the distance back to "
  f"the well-set fixed baseline; it still finishes **{r0mis - fix3:+.4f} PPL** above "
  f"well-set fixed 1e-3 and **{r0mis - best3:+.4f} PPL** above the post-hoc-best "
  f"3e-3 point. For scale, the gate's benefit at the *well-set* start was only "
  f"-0.4166 PPL, so mis-setting doubles the gate's payoff in absolute PPL while "
  f"leaving ~77% of the mis-set penalty on the table.")
A()

A("### 1.3 Gate behaviour of `cohg_r0` at lr0 = 1e-4 -- the key question")
A()
A("| seed | `gate_open_frac` | exact #opens / 2999 | `gate_open_steps` | "
  "`coord_open_frac` | accepted coord-moves (of 2999 x 6) | lambda end | "
  "implied end LR |")
A("|---|---|---|---|---|---|---|---|")
gate_detail = {}
for s in range(3):
    d = mis[("cohg_r0", s)]
    lh = d["lam_hist"]
    lam0, lamE = lh[0][1:], lh[-1][1:]
    gate_detail[s] = dict(lam0=lam0, lamE=lamE)
    A(f"| {s} | {d['gate_open_frac']:.9g} | **{round(d['gate_open_frac'] * 2999)}** | "
      f"{d['gate_open_steps']} | {d['coord_open_frac']:.9g} | "
      f"**{round(d['coord_open_frac'] * 2999 * 6)}** | "
      f"[{', '.join(f'{v:.4f}' for v in lamE)}] | "
      f"[{', '.join(f'{math.exp(v):.3e}' for v in lamE)}] |")
A()
A("Compare the well-set arm: `results/e4_v2/gpt2_cohg_r0_lr0.001_s*` has "
  "`gate_open_frac = 3.3344448e-04` = **1** open and **6** accepted coordinate-moves "
  "in all 8 seeds. Starting 10x too low, the gate opens **twice** and accepts "
  "**seven** coordinate-moves -- it does respond to the larger error, but by exactly "
  "one extra coordinate-move.")
A()
A("Per-coordinate lambda displacement from init (`log 1e-4 = -9.2103`) and the "
  "resulting LR multiplier:")
A()
A("| seed | quantity | " + " | ".join(COORDS) + " |")
A("|---|---|" + "---|" * 6)
for s in range(3):
    gd = gate_detail[s]
    A(f"| {s} | delta lambda | " +
      " | ".join(f"{e - b:+.4f}" for b, e in zip(gd["lam0"], gd["lamE"])) + " |")
    A(f"| {s} | LR multiplier | " +
      " | ".join(f"{math.exp(e - b):.3f}x" for b, e in zip(gd["lam0"], gd["lamE"])) + " |")
A()
A("The structure is identical in all three seeds and decomposes exactly: **open 1 "
  "(step 1) moves all six coordinates by the full +0.400; open 2 (step 2) moves only "
  "`emb`, and only by ~+0.30** (the trust factor `s_j` < 1 there). "
  "6 + 1 = the 7 accepted coordinate-moves in the table above.")
A()
A("Trajectory check over all 150 `lam_hist` samples (t = 0, 20, ..., 2980):")
A()
for s in range(3):
    d = mis[("cohg_r0", s)]
    lh = d["lam_hist"]
    changes = [(lh[i - 1][0], lh[i][0]) for i in range(1, len(lh))
               if any(abs(lh[i][1 + j] - lh[i - 1][1 + j]) > 1e-9 for j in range(6))]
    const = all(all(abs(lh[i][1 + j] - lh[1][1 + j]) < 1e-9 for j in range(6))
                for i in range(1, len(lh)))
    A(f"- **seed {s}**: lambda changes in **{len(changes)}** of 149 sample gaps "
      f"({changes}); bit-constant from the t = 20 sample through the t = 2980 sample: "
      f"**{const}**. Exact opens `{d['gate_open_steps']}` -- both inside the first "
      f"three steps of the stream. Opens at t >= 1000 (news boundary): **0**. "
      f"Opens at t >= 2000 (code boundary): **0**.")
A()
maxlam = max(max(gate_detail[s]["lamE"]) for s in range(3))
A(f"Highest lambda any coordinate reaches in any seed: **{maxlam:.4f}** "
  f"(LR = {math.exp(maxlam):.3e}, i.e. {math.exp(maxlam) / 1e-4:.2f}x the init). "
  f"Targets: `log 1e-3 = {math.log(1e-3):.4f}`, `log 3e-3 = {math.log(3e-3):.4f}`. "
  f"Remaining climb to 1e-3: **{math.log(1e-3) - maxlam:.4f}** in lambda "
  f"({math.exp(math.log(1e-3) - maxlam):.2f}x more LR), which at <= 0.40 per accepted "
  f"move needs **>= {math.ceil((math.log(1e-3) - maxlam) / 0.4)} further gate "
  f"openings**; to reach the post-hoc-best 3e-3 it needs "
  f">= {math.ceil((math.log(3e-3) - maxlam) / 0.4)}. It makes **zero**.")
A()

A("### 1.4 `cohg_nogate` at lr0 = 1e-4: lambda range, clamp saturation, degradation")
A()
A("The gate is forced open at every step (`gate_open_frac = 1.0`, 2999/2999). "
  "`coord_open_frac = 0.0` here means the certified controller is **never "
  "consulted** (`CoordGatedController.maybe_update` is bypassed, so its counters stay "
  "at zero) -- it does *not* mean zero coordinates were certified. Clamps "
  "`[-13.8155, -2.3026]`, admissible span 11.5129.")
A()
A("| seed | online PPL | worst-window PPL | max-excess | spikes | per-coord lambda "
  "range (max-min) | frac of lam samples at lower clamp | at upper clamp | "
  "lambda end | peak per-step log-loss (step) |")
A("|---|---|---|---|---|---|---|---|---|---|")
for s in range(3):
    d = mis[("cohg_nogate", s)]
    lh, um = d["lam_hist"], d["_um"]
    cols = [[r[1 + j] for r in lh] for j in range(6)]
    rng = [max(c) - min(c) for c in cols]
    lo = sum(1 for r in lh for j in range(6) if abs(r[1 + j] - LAM_MIN) < 1e-6) / (len(lh) * 6)
    hi = sum(1 for r in lh for j in range(6) if abs(r[1 + j] - LAM_MAX) < 1e-6) / (len(lh) * 6)
    pk = max(range(len(d["losses"])), key=lambda i: d["losses"][i])
    A(f"| {s} | {d['online_ppl']:.4f} | {g(sexp(um['worst_window_mean']), 2)} | "
      f"{um['max_excess']:.4f} | {um['n_spike']} | "
      f"[{', '.join(f'{v:.3f}' for v in rng)}] | {lo:.4f} | {hi:.4f} | "
      f"[{', '.join(f'{v:.3f}' for v in lh[-1][1:])}] | {d['losses'][pk]:.4f} (t={pk}) |")
A()
touch_lo = sum(1 for s in range(3) for j in range(6)
               if any(abs(r[1 + j] - LAM_MIN) < 1e-6 for r in mis[("cohg_nogate", s)]["lam_hist"]))
touch_hi = sum(1 for s in range(3) for j in range(6)
               if any(abs(r[1 + j] - LAM_MAX) < 1e-6 for r in mis[("cohg_nogate", s)]["lam_hist"]))
A(f"**{touch_lo} / 18** coordinate-seed pairs touch the lower clamp exactly and "
  f"**{touch_hi} / 18** touch the upper clamp exactly; 38-43% of all lambda samples "
  f"sit pinned at the lower clamp and 9-10% at the upper one. Started 10x *below* "
  f"the well-set LR -- the one situation in which climbing is unambiguously the right "
  f"move -- the ungated meta-optimizer still does not climb and settle: it bang-bangs "
  f"across the full admissible span, exactly as it does from the well-set start.")
A()
A("Per-domain PPL (D1 wiki t < 1000, D2 news 1000 <= t < 2000, D3 code t >= 2000):")
A()
A("| arm | seed | D1 wiki | D2 news | D3 code |")
A("|---|---|---|---|---|")
for a in ARMS1:
    for s in range(3):
        L = mis[(a, s)]["losses"]
        pdm = []
        for seg in (L[:1000], L[1000:2000], L[2000:]):
            sf = [x for x in seg if math.isfinite(x)]
            pdm.append(sexp(sum(sf) / len(sf)) if sf else float("inf"))
        A(f"| {LABEL1[a]} | {s} | {g(pdm[0], 3)} | {g(pdm[1], 3)} | {g(pdm[2], 3)} |")
A()
thr_ppl_mis = fixmis + 2.0
thr_ww_mis = 2.0 * med(rows1["cohg_r0"]["ww"])
A(f"**Degraded-run classification** (definition of `e4_expansion.md` section 0, "
  f"anchored on the fixed baseline of the *same configuration*, i.e. fixed lr 1e-4): "
  f"PPL threshold = {fixmis:.4f} + 2.0 = **{thr_ppl_mis:.4f}**; worst-window "
  f"threshold = 2 x median worst-window of `cohg_r0` = 2 x "
  f"{med(rows1['cohg_r0']['ww']):.2f} = **{thr_ww_mis:.2f}**.")
A()
A("| arm | degraded / n | which seeds (criterion) |")
A("|---|---|---|")
for a in ARMS1:
    r = rows1[a]; who = []
    for s in range(3):
        c = []
        if r["ppl"][s] > thr_ppl_mis:
            c.append("PPL")
        if r["ww"][s] > thr_ww_mis:
            c.append("worst-window")
        if c:
            who.append(f"s{s} ({'+'.join(c)})")
    A(f"| {LABEL1[a]} | **{len(who)} / 3** | {', '.join(who) if who else '-'} |")
A()
A("Cost:")
A()
A("| arm | wall-clock h | peak GB | HVPs |")
A("|---|---|---|---|")
for a in ARMS1:
    rs = [mis[(a, s)] for s in range(3)]
    A(f"| {LABEL1[a]} | {mstd([r['wall_s'] / 3600 for r in rs])[0]:.2f} | "
      f"{mstd([r['peak_mem_gb'] for r in rs])[0]:.2f} | "
      f"{mstd([r['hvp_total'] for r in rs])[0]:.0f} |")
A()

A("### 1.5 Plain answer to review P7")
A()
A("**Does the certified gate adapt when adaptation is genuinely needed? Partly -- it "
  "moves in the right direction, immediately, and it stops; but it moves far too "
  "little, and it stops long before the adaptation is finished.**")
A()
A("1. *It notices, and it notices fast.* Started 10x too low, `cohg_r0` opens the "
  "gate at steps **1 and 2** in all three seeds, and every accepted coordinate-move "
  "**raises** the LR (`+0.400` on all six groups at open 1, a further `+0.30` on "
  "`emb` at open 2). Under the well-set start the same rule opens **once** and "
  "accepts six moves; mis-set, it opens **twice** and accepts seven. So the gate is "
  "measurably more active when the initialization is worse -- but the extra activity "
  "amounts to a single extra coordinate-move.")
A("2. *It stops -- and stays stopped.* After step 2 the gate never opens again for "
  "the remaining 2997 steps; lambda is bit-constant from the t = 20 sample to the "
  "t = 2980 sample in every seed. There are **zero** openings at or after either "
  "domain boundary (t = 1000, t = 2000). So the answer to \"does it stop when done\" "
  "is an unambiguous *yes* -- the gate does not chatter and does not drift.")
A("3. *But it stops before it is done.* The final LR is only **1.49x-2.06x** the "
  "mis-set init (2.0e-4 on `emb`, 1.49e-4 elsewhere), against the 10x needed to reach "
  "1e-3 and the 30x needed to reach the post-hoc best 3e-3. Reaching 1e-3 would "
  f"require at least {math.ceil((math.log(1e-3) - maxlam) / 0.4)} more openings; it "
  "makes none. Online PPL lands at **24.006 +- 0.020**, versus 24.888 +- 0.014 for "
  "fixed 1e-4, 21.092 +- 0.031 for well-set fixed 1e-3 and 19.840 +- 0.027 for the "
  "post-hoc best. The gate recovers **23.2%** of the mis-set penalty and leaves 77% "
  "of it standing.")
A("4. *The safety claim survives the stress test intact.* 0/3 degraded gated runs, "
  "0 spikes, 0 non-finite losses, max-excess 1.309 +- 0.011 -- statistically "
  "indistinguishable from the fixed baseline's 1.301 +- 0.012, and its worst-window "
  "(48.09) is *below* the fixed baseline's (50.71). Removing the gate from the same "
  "starting point produces 1/3 degraded runs, a seed at 43.1 PPL with a worst-window "
  "PPL of 3.7e6 and a single-step log-loss of 142.6, and 5.7 +- 9.8 spikes per run.")
A("5. *Honest framing for the paper.* This block is the strongest available evidence "
  "that COHG's gate is not merely inert -- it does fire more, and in the correct "
  "direction, when the LR is genuinely mis-set, and it beats the mis-set fixed "
  "baseline on **3/3 seeds** with a mean gap of -0.882 PPL (p = 0.25 only because "
  "n = 3 floors it there; the effect is 43x the seed std). But it is *not* evidence "
  "of full recovery: a method that certifiably adapts would have to keep opening, and "
  "this one certifies its way to a stop after two steps. The correct claim is "
  "\"certified partial recovery from a mis-set LR, at zero cost in stability\", not "
  "\"the gate finds the right LR\".")
A()

# ================================================================ PART 2
ORD = os.path.join(ROOT, "results", "e4_orders")
ordd = {}
for p in sorted(glob.glob(os.path.join(ORD, "*.json"))):
    b = os.path.basename(p)[:-5]
    m = re.match(r"^gpt2order_cnw_(?P<meth>.+)_lr0\.001(?:_ml2)?_s(?P<s>\d+)$", b)
    ordd[(m["meth"], int(m["s"]))] = load_e4(p)

A("---")
A()
A("## 2. Reverse-order stream (code -> news -> wiki) refreshed at n = 8")
A()
A("`results/e4_orders/`, files `gpt2order_cnw_*`. `cohg_r0` now covers seeds 0-7; "
  "`fixed`, `hd ml2` and `cohg_nogate` are still at seeds 0-2. D1 = code (t < 1000), "
  "D2 = news, D3 = wiki (t >= 2000).")
A()
A("### 2.0 Provenance of the eight `cohg_r0` seeds (stated up front, as required)")
A()
A("| seed | `legacy_hold` | `held_bound` | `gate_open_steps` | `gate_open_frac` | "
  "`coord_open_frac` | events | online PPL | wall h |")
A("|---|---|---|---|---|---|---|---|---|")
for s in range(8):
    d = ordd[("cohg_r0", s)]
    A(f"| {s} | {d.get('legacy_hold', '*(key absent)*')} | "
      f"{d.get('held_bound', '*(key absent)*')} | "
      f"{d.get('gate_open_steps', '*(key absent)*')} | {d['gate_open_frac']:.9g} | "
      f"{d['coord_open_frac']:.9g} | {d['events']} | {d['online_ppl']:.4f} | "
      f"{d['wall_s'] / 3600:.2f} |")
A()
A("Seeds **0-2** were produced by the pre-fix driver revision and carry **no** "
  "`legacy_hold` / `held_bound` / `gate_open_steps` keys. Seeds **3-7** were launched "
  "by `launch_r3_chain.py` phase `p2_misset` *after* the held-bound correction "
  "(review P4: the full vector-valued Proposition-10 drift-hold, "
  "`dh.probe(..., eta_vec=eta)` / `dh.bounds(eta)`, replacing the scalar path) and "
  "record `legacy_hold = False`, `held_bound = vector_prop10`. **The two sub-blocks "
  "are therefore not identical provenance and every n = 8 aggregate below must carry "
  "that caveat.**")
A()
A("What is checkable is that the correction did not change the gate's realized "
  "behaviour on this stream: `gate_open_frac` and `coord_open_frac` are "
  "**bit-identical** (3.3344448e-04, i.e. exactly one open and six accepted "
  "coordinate-moves in 2999 steps) across all eight seeds, old and new, and every "
  "seed has `events = 0`. The new seeds also run ~1.5x faster in wall-clock "
  "(3.0-3.7 h vs 4.6-5.2 h), consistent with a less contended card rather than a "
  "different amount of work (`hvp_total = 4100` in all eight).")
A()

r0 = [ordd[("cohg_r0", s)] for s in range(8)]
r0_ppl = [d["online_ppl"] for d in r0]
r0_ww = [sexp(d["_um"]["worst_window_mean"]) for d in r0]
r0_mx = [d["_um"]["max_excess"] for d in r0]
fx_ppl = [ordd[("fixed", s)]["online_ppl"] for s in range(3)]
fixmean = mstd(fx_ppl)[0]

A("### 2.1 The refreshed `cohg_r0` row at n = 8")
A()
A("| quantity | n = 3 (seeds 0-2, as published) | **n = 8 (seeds 0-7)** |")
A("|---|---|---|")
A(f"| online PPL mean +- std | {f(*mstd(r0_ppl[:3])[:2])} | **{f(*mstd(r0_ppl)[:2])}** |")
A(f"| online PPL median | {med(r0_ppl[:3]):.4f} | **{med(r0_ppl):.4f}** |")
A(f"| online PPL min / max | {min(r0_ppl[:3]):.4f} / {max(r0_ppl[:3]):.4f} | "
  f"**{min(r0_ppl):.4f} / {max(r0_ppl):.4f}** |")
A(f"| worst-window PPL mean +- std | {f(*mstd(r0_ww[:3])[:2])} | **{f(*mstd(r0_ww)[:2])}** |")
A(f"| worst-window median / min / max | {med(r0_ww[:3]):.4f} / {min(r0_ww[:3]):.4f} / "
  f"{max(r0_ww[:3]):.4f} | **{med(r0_ww):.4f} / {min(r0_ww):.4f} / {max(r0_ww):.4f}** |")
A(f"| max-excess mean +- std | {f(*mstd(r0_mx[:3])[:2])} | **{f(*mstd(r0_mx)[:2])}** |")
A(f"| spikes / non-finite / stored events (totals) | 0 / 0 / 0 | "
  f"**{sum(d['_um']['n_spike'] for d in r0)} / "
  f"{sum(d['_um']['n_nonfinite'] for d in r0)} / {sum(d['events'] for d in r0)}** |")
A(f"| **degraded runs** | 0 / 3 | **0 / 8** |")
A()
A("Per-seed detail, all eight seeds:")
A()
A("| seed | online PPL | worst-window PPL | max-excess | max finite log-loss | "
  "median log-loss | spikes | non-finite | degraded? |")
A("|---|---|---|---|---|---|---|---|---|")
thr_ppl2 = fixmean + 2.0
thr_ww2_n8 = 2.0 * med(r0_ww)
thr_ww2_n3 = 2.0 * med(r0_ww[:3])
for s in range(8):
    d = ordd[("cohg_r0", s)]; um = d["_um"]
    deg = (d["online_ppl"] > thr_ppl2) or (sexp(um["worst_window_mean"]) > thr_ww2_n8)
    A(f"| {s} | {d['online_ppl']:.4f} | {sexp(um['worst_window_mean']):.4f} | "
      f"{um['max_excess']:.4f} | {um['max_finite_loss']:.4f} | {um['median_loss']:.4f} | "
      f"{um['n_spike']} | {um['n_nonfinite']} | {'**yes**' if deg else 'no'} |")
A()
A(f"**Degraded-run definition** (e4_expansion.md section 0): degraded iff online PPL "
  f"> (same-order fixed mean + 2.0) **or** worst-window PPL > 2 x (median "
  f"worst-window of the gated arm of the same order). The fixed mean is unchanged at "
  f"{fixmean:.4f} -> **PPL threshold {thr_ppl2:.4f}**. The gated arm's own median "
  f"worst-window moves from {med(r0_ww[:3]):.4f} (n = 3) to {med(r0_ww):.4f} (n = 8), "
  f"so its worst-window threshold moves from **{thr_ww2_n3:.4f}** to "
  f"**{thr_ww2_n8:.4f}**. **No classification changes under either threshold.**")
A()
A("| arm | n | degraded / n (n=8 threshold) | degraded / n (n=3 threshold) | which seeds |")
A("|---|---|---|---|---|")
LBL2 = {"fixed": "fixed lr 1e-3", "hd": "hd ml2", "cohg_r0": "cohg_r0 (gated)",
        "cohg_nogate": "cohg_nogate"}
for a in ["fixed", "hd", "cohg_r0", "cohg_nogate"]:
    ss = sorted(s for (m_, s) in ordd if m_ == a)
    ppls = [ordd[(a, s)]["online_ppl"] for s in ss]
    wws = [sexp(ordd[(a, s)]["_um"]["worst_window_mean"]) for s in ss]
    who8, n3 = [], 0
    for i, s in enumerate(ss):
        c = []
        if ppls[i] > thr_ppl2:
            c.append("PPL")
        if wws[i] > thr_ww2_n8:
            c.append("WW")
        if c:
            who8.append(f"s{s} ({'+'.join(c)})")
        if ppls[i] > thr_ppl2 or wws[i] > thr_ww2_n3:
            n3 += 1
    A(f"| {LBL2[a]} | {len(ss)} | **{len(who8)} / {len(ss)}** | {n3} / {len(ss)} | "
      f"{', '.join(who8) if who8 else '-'} |")
A()

A("### 2.2 Paired test on common seeds -- does the \"+0.30 n.s. code-first penalty\" change?")
A()
d3 = [ordd[("cohg_r0", s)]["online_ppl"] - ordd[("fixed", s)]["online_ppl"] for s in range(3)]
p3, m3 = signflip_p(d3)
d3h = [ordd[("cohg_r0", s)]["online_ppl"] - ordd[("hd", s)]["online_ppl"] for s in range(3)]
p3h, m3h = signflip_p(d3h)
d3n = [ordd[("cohg_nogate", s)]["online_ppl"] - ordd[("fixed", s)]["online_ppl"] for s in range(3)]
p3n, m3n = signflip_p(d3n)
A("**No.** `fixed` exists only for seeds 0-2 on this stream, so the *paired* "
  "contrast is still n = 3 and is arithmetically **unchanged** by the refresh -- "
  "adding seeds to one arm cannot alter a paired statistic whose pairs did not "
  "change.")
A()
A("| contrast | n | per-seed diffs (s0, s1, s2) | mean diff | sign | p (exact) |")
A("|---|---|---|---|---|---|")
A(f"| `cohg_r0` - `fixed` | 3 | {', '.join(f'{x:+.4f}' for x in d3)} | **{m3:+.4f}** | "
  f"2 pos / 1 neg | {p3:g} |")
A(f"| `cohg_r0` - `hd ml2` | 3 | {', '.join(f'{x:+.4f}' for x in d3h)} | {m3h:+.4f} | "
  f"3 pos | {p3h:g} |")
A(f"| `cohg_nogate` - `fixed` | 3 | {', '.join(f'{x:+.4f}' for x in d3n)} | {m3n:+.4f} | "
  f"2 pos / 1 neg | {p3n:g} |")
A()
A("### 2.3 The n = 8 summary, reported separately (not a paired test)")
A()
unp = mstd(r0_ppl)[0] - fixmean
A("| quantity | value |")
A("|---|---|")
A(f"| `cohg_r0` n = 8 mean +- std | **{f(*mstd(r0_ppl)[:2])}** |")
A(f"| `fixed` n = 3 mean +- std | {f(*mstd(fx_ppl)[:2])} |")
A(f"| unpaired difference of means | **{unp:+.4f} PPL** |")
A(f"| same difference at n = 3 (published) | {mstd(r0_ppl[:3])[0] - fixmean:+.4f} PPL |")
A(f"| shift caused by the five new seeds | {unp - (mstd(r0_ppl[:3])[0] - fixmean):+.4f} PPL |")
A(f"| `cohg_r0` std, n = 3 -> n = 8 | {mstd(r0_ppl[:3])[1]:.4f} -> {mstd(r0_ppl)[1]:.4f} |")
A()
A(f"The five new seeds (21.6223, 21.6452, 21.6098, 21.2949, 20.6776) land inside the "
  f"existing n = 3 range [20.9062, 21.6637] except s7 = 20.6776, which is a new "
  f"minimum and is *below* the fixed baseline. So the direction of the finding is "
  f"unchanged (`cohg_r0` still costs ~+0.28 PPL against fixed under code-first, "
  f"versus -0.42 PPL under wiki-first) but the magnitude softens from +0.3043 to "
  f"**{unp:+.4f}** and the arm's dispersion stays large "
  f"({mstd(r0_ppl)[1]:.4f}, versus 0.0386 for the same arm under wiki-first at "
  f"n = 8 -- a 10x wider seed spread). Two of eight seeds (s0, s7) beat fixed; six do "
  f"not. **The sign flip relative to the standard order survives at n = 8, and it "
  f"remains statistically unsupported.**")
A()

A("### 2.4 Gate-open timing, including the five new seeds")
A()
A("Seeds 3-7 store `gate_open_steps` explicitly, so their timing is exact. Seeds 0-2 "
  "lack the key and are localised from `lam_hist` to the half-open interval "
  "`(t_prev, t_cur]`.")
A()
A("| seed | exact #opens / 2999 | open step(s) | before step 20? | opens at t >= 1000 | "
  "opens at t >= 2000 | lambda end | coords raised / lowered |")
A("|---|---|---|---|---|---|---|---|")
for s in range(8):
    d = ordd[("cohg_r0", s)]
    lh = d["lam_hist"]; lam0, lamE = lh[0][1:], lh[-1][1:]
    gos = d.get("gate_open_steps")
    nop = round(d["gate_open_frac"] * 2999)
    if gos is not None:
        steps = f"`{gos}` (exact)"
        b20 = all(t < 20 for t in gos)
        n1k = sum(1 for t in gos if t >= 1000); n2k = sum(1 for t in gos if t >= 2000)
    else:
        chg = [(lh[i - 1][0], lh[i][0]) for i in range(1, len(lh))
               if any(abs(lh[i][1 + j] - lh[i - 1][1 + j]) > 1e-9 for j in range(6))]
        steps = "(0, 20] (localised)" if chg == [(0, 20)] else str(chg)
        b20 = (chg == [(0, 20)])
        n1k = sum(1 for _, b_ in chg if b_ > 1000); n2k = sum(1 for _, b_ in chg if b_ > 2000)
    up = sum(1 for b_, e_ in zip(lam0, lamE) if e_ > b_)
    A(f"| {s} | **{nop}** | {steps} | **{b20}** | {n1k} | {n2k} | "
      f"[{', '.join(f'{v:.4f}' for v in lamE)}] | {up} / {6 - up} |")
A()
A("Per-coordinate delta lambda (init `log 1e-3 = -6.9078`, meta_lr 0.4):")
A()
A("| seed | " + " | ".join(COORDS) + " | # lowered |")
A("|---|" + "---|" * 7)
for s in range(8):
    lh = ordd[("cohg_r0", s)]["lam_hist"]
    dl = [lh[-1][1 + j] - lh[0][1 + j] for j in range(6)]
    A(f"| {s} | " + " | ".join(f"{v:+.4f}" for v in dl) +
      f" | {sum(1 for v in dl if v < 0)} |")
A()
allconst = all(all(all(abs(ordd[("cohg_r0", s)]["lam_hist"][i][1 + j] -
                           ordd[("cohg_r0", s)]["lam_hist"][1][1 + j]) < 1e-9
                       for j in range(6))
                   for i in range(1, 150)) for s in range(8))
A(f"Lambda is bit-constant from the t = 20 sample onwards in **all eight** seeds: "
  f"**{allconst}**. So: exactly one accepted meta-update per run, at step 1 (verified "
  f"exactly for s3-s7, localised to (0, 20] for s0-s2), then 2979 steps of frozen "
  f"lambda. **Zero** openings at either domain boundary in any of the eight seeds.")
A()
A("The five new seeds also reproduce the *direction* finding: s3, s4, s5 lower five "
  "of six coordinates (identical pattern to s1 and s2), s6 lowers three, s7 lowers "
  "only `emb`. Across all eight seeds `emb` is lowered **8/8** times and `ln_f` is "
  "raised **8/8** times; the four middle blocks split. The published claim -- "
  "\"under code-first the same certified rule usually *lowers* the LR, whereas "
  "wiki-first it always raises it\" -- holds at n = 8, with the sharper statement "
  "that the embedding group is lowered in every single seed of this order and in no "
  "seed of the standard order.")
A()

A("### 2.5 Plain answer")
A()
A(f"**The \"+0.30 n.s. code-first penalty\" does not change in kind, only in size, "
  f"and the gate's timing finding is now exact rather than inferred.** The paired "
  f"test is untouched at +0.3043 PPL, p = {p3:g} (the fixed arm gained no seeds, so "
  f"the pairs are literally the same three). The unpaired n = 8 view softens the "
  f"penalty to **{unp:+.4f} PPL** with a std of {mstd(r0_ppl)[1]:.4f}, i.e. the "
  f"penalty is smaller than one seed-to-seed standard deviation of the gated arm and "
  f"remains not significant by any available test. Meanwhile the safety side "
  f"strengthens: **0 / 8 degraded gated runs**, 0 spikes and 0 non-finite losses "
  f"across all eight seeds, max-excess {mstd(r0_mx)[0]:.4f} +- {mstd(r0_mx)[1]:.4f}, "
  f"worst-window max {max(r0_ww):.2f} against the fixed baseline's "
  f"{max(sexp(ordd[('fixed', s)]['_um']['worst_window_mean']) for s in range(3)):.2f} "
  f"-- versus 2/3 degraded ungated runs with a worst-window of 4.2e5. And the timing "
  f"answer is now certain for the new seeds: **exactly 1 opening, at step 1, in every "
  f"one of the five**, well before step 20 and 999 steps before the first domain "
  f"boundary.")
A()

# ================================================================ PART 3
TR = os.path.join(ROOT, "results", "e3_traced")
CAN = os.path.join(ROOT, "results", "e3")


def load_e3(p, traces=True):
    d = json.load(open(p))
    if traces and "losses" in d:
        d["_um"] = unified_metrics(d["losses"])
    return d


tr, can = {}, {}
for p in sorted(glob.glob(os.path.join(TR, "*.json"))):
    m = re.match(r"^cifar100_(?P<meth>.+)_lr0\.05_ewc(?P<e>[0-9.]+)_s(?P<s>\d+)\.json$",
                 os.path.basename(p))
    tr[(m["meth"], m["e"], int(m["s"]))] = load_e3(p)
for p in sorted(glob.glob(os.path.join(CAN, "*.json"))):
    m = re.match(r"^cifar100_(?P<meth>.+)_lr0\.05_ewc(?P<e>[0-9.]+)_s(?P<s>\d+)\.json$",
                 os.path.basename(p))
    if m:
        can[(m["meth"], m["e"], int(m["s"]))] = load_e3(p, traces=False)

METH3 = ["fixed", "hd", "cohg", "cohg_nogate"]
LBL3 = {"fixed": "fixed", "hd": "hd", "cohg": "**cohg (gated)**", "cohg_nogate": "cohg_nogate"}
EW = ["10", "1000"]
BLOW = 100.0   # a run "blows up" if its max finite loss exceeds this

A("---")
A()
A("## 3. E3 traced re-run (`results/e3_traced/`, 80 runs, lr0 = 0.05, seeds 0-9)")
A()
A("{`cohg`, `cohg_nogate`, `hd`, `fixed`} x {ewc0 = 10, ewc0 = 1000} x seeds 0-9, "
  "all launched with `--log-losses` in the retained-holdout condition (E3 default), "
  "so the per-step loss trace that `results/e3` never stored is now available and "
  "spikes / max-excess / worst-window become computable for E3 for the first time. "
  "Hyperparameters were verified identical to the canonical set seed by seed: "
  "`meta_lr` (fixed 0.1, hd 0.02, cohg 0.4, cohg_nogate 0.4), `rank = 4`, `K = 10`, "
  "`gamma = 0.9`, `lr0 = 0.05`, matching `ewc0`. Traces are 3040 steps "
  "(10 tasks x 304 steps).")
A()
A("E3's recovery rule differs from E4's and matters for the censored analysis: on a "
  "non-finite loss the driver does **not** restore a checkpoint. It halves the LR "
  "scale *and* the EWC strength (`lam <- clamp(lam - log 2)`) and resets the "
  "estimator state, then continues.")
A()

A("### 3.1 (a) Verification of the traced re-run against the untraced canonical `results/e3`")
A()
A("| arm | ewc | n matched | max abs delta avg_acc | mean abs delta avg_acc | "
  "max abs delta BWT | seeds with delta events != 0 | **flag: seeds with "
  "\\|delta acc\\| > 0.02** |")
A("|---|---|---|---|---|---|---|---|")
verif = {}
for e in EW:
    for m in METH3:
        ss = [s for s in range(10) if (m, e, s) in tr and (m, e, s) in can]
        da = [abs(tr[(m, e, s)]["avg_acc"] - can[(m, e, s)]["avg_acc"]) for s in ss]
        db = [abs(tr[(m, e, s)]["bwt"] - can[(m, e, s)]["bwt"]) for s in ss]
        de = {s: tr[(m, e, s)]["events"] - can[(m, e, s)]["events"] for s in ss
              if tr[(m, e, s)]["events"] != can[(m, e, s)]["events"]}
        flag = [s for s, x in zip(ss, da) if x > 0.02]
        verif[(m, e)] = (max(da), max(db), de, flag)
        A(f"| {LBL3[m]} | {e} | {len(ss)} | **{max(da):.4f}** | {sum(da) / len(da):.4f} | "
          f"{max(db):.4f} | {('s' + ', s'.join(str(k) for k in sorted(de))) if de else 'none'} | "
          f"{('**' + str(len(flag)) + ': s' + ', s'.join(str(s) for s in flag) + '**') if flag else 'none'} |")
A()
A("**This is the headline verification result and it is not a clean pass.** The four "
  "*stable* configurations reproduce well: `fixed`/`hd`/`cohg` at ewc = 10 and `hd` at "
  "ewc = 1000 agree to max |delta acc| = 0.016 / 0.020 / 0.041 / 0.029 with mean "
  "|delta| <= 0.012, and 0 event-count differences. The four *unstable* configurations "
  "do not: `fixed` @ ewc1000 differs by up to **0.251** accuracy, `cohg` @ ewc1000 by "
  "up to **0.296**, `cohg_nogate` by up to 0.090 (ewc10) and 0.182 (ewc1000), with "
  "event counts swinging by hundreds in both directions.")
A()
A("Per-seed rows where |delta avg_acc| > 0.02:")
A()
A("| arm | ewc | seed | canonical avg_acc | traced avg_acc | delta | events can / tr |")
A("|---|---|---|---|---|---|---|")
for e in EW:
    for m in METH3:
        for s in range(10):
            if (m, e, s) in tr and (m, e, s) in can:
                d_ = tr[(m, e, s)]["avg_acc"] - can[(m, e, s)]["avg_acc"]
                if abs(d_) > 0.02:
                    A(f"| {LBL3[m]} | {e} | {s} | {can[(m, e, s)]['avg_acc']:.4f} | "
                      f"{tr[(m, e, s)]['avg_acc']:.4f} | **{d_:+.4f}** | "
                      f"{can[(m, e, s)]['events']} / {tr[(m, e, s)]['events']} |")
A()
A("**Diagnosis.** What is verifiable from the artifacts: every stored hyperparameter "
  "matches seed by seed (`method`, `seed`, `lr0`, `ewc0`, `meta_lr`, `rank`, `K`, "
  "`gamma`), `hvp_total` matches exactly (so the two sets do the same amount of "
  "work), and the flag sets differ only by `--log-losses`, which is write-only. The "
  "execution environments were nevertheless not identical: mean wall-clock per arm "
  "differs by 1.14x-7.05x (traced/canonical), largest on `hd` @ ewc10 "
  "(791 s -> 5574 s), consistent with the traced phase running two E3 jobs per card "
  "(`SLOTS_PER_GPU = 2` in `launch_r3_chain.py`) rather than with extra computation. "
  "No device identifier is stored in either set of JSONs and no run log survives for "
  "the canonical set, so the hardware attribution cannot be closed from the "
  "artifacts; what can be said is that the two runs used different execution "
  "conditions and therefore different floating-point reduction orders. The mechanism "
  "is the one already quantified in `results/reanalysis/device_sensitivity.md`: a "
  "float-reassociation perturbation of order 1e-7 leaves stable trajectories intact "
  "(there, |dNMSE|/NMSE ~ 1e-7 on 5 of 10 seeds and identical event counts on 10/10) "
  "but is amplified without bound once a run enters the divergent regime -- the same "
  "study shows a `cohg_nogate` seed moving from NMSE 59.8 to 0.0048 and an event "
  "count from 298 to 53 under nothing but a device change on identical code and "
  "seed. E3 @ ewc1000 is exactly that divergent regime.")
A()
A("**Consequence, which must be stated wherever E3 numbers are used.** At ewc = 10 "
  "the two sets are interchangeable and the traced set can be quoted as canonical. "
  "At ewc = 1000 they are **not** interchangeable: the outcome of an individual seed "
  "is device-determined, and any ewc1000 claim must be phrased over the distribution "
  "(e.g. \"k of 10 seeds blow up\"), never over a specific seed's accuracy. Notably "
  "the traced set is the *more favourable* of the two for `fixed` (+0.075 mean acc, "
  "3 canonical collapses at 0.1000 disappear) and for `cohg` (+0.037, the canonical "
  "s3 collapse disappears), and the *less* favourable for `cohg_nogate` (-0.044).")
A()
A("Arm-level means, canonical vs traced:")
A()
A("| arm | ewc | canonical avg_acc (n=10) | traced avg_acc (n=10) | delta means | "
  "canonical BWT | traced BWT | delta means | canonical collapses | traced collapses |")
A("|---|---|---|---|---|---|---|---|---|---|")
for e in EW:
    for m in METH3:
        ss = [s for s in range(10) if (m, e, s) in tr and (m, e, s) in can]
        ca, ta = mstd([can[(m, e, s)]["avg_acc"] for s in ss]), mstd([tr[(m, e, s)]["avg_acc"] for s in ss])
        cb, tb = mstd([can[(m, e, s)]["bwt"] for s in ss]), mstd([tr[(m, e, s)]["bwt"] for s in ss])
        cc = sum(1 for s in ss if can[(m, e, s)]["avg_acc"] < 0.15)
        tc = sum(1 for s in ss if tr[(m, e, s)]["avg_acc"] < 0.15)
        A(f"| {LBL3[m]} | {e} | {f(*ca[:2])} | {f(*ta[:2])} | **{ta[0] - ca[0]:+.4f}** | "
          f"{f(*cb[:2])} | {f(*tb[:2])} | **{tb[0] - cb[0]:+.4f}** | {cc}/10 | {tc}/10 |")
A()

A("### 3.2 (b) Unified trace-level degradation metrics, per arm")
A()
A("Now computable from the stored `losses`. Worst-window is in raw cross-entropy. "
  "Losses in the divergent arms reach ~1e38 (float32 overflow threshold) while still "
  "being *finite*, so arithmetic means of max-excess and worst-window are dominated "
  "by one or two seeds and are reported only for completeness; **the median, the "
  "maximum, and the blow-up count are the readable summaries.** A run is called a "
  "*blow-up* here if its maximum finite loss exceeds 100 (ambient cross-entropy on "
  "this task is ~1.7-2.1).")
A()
A("| arm | ewc | n | spikes mean +- std | non-finite mean +- std | unified events | "
  "blow-ups (max finite loss > 100) | max-excess median | max-excess max (seed) | "
  "worst-window median | worst-window max (seed) | median loss |")
A("|---|---|---|---|---|---|---|---|---|---|---|---|")
e3agg = {}
for e in EW:
    for m in METH3:
        rs = [tr[(m, e, s)] for s in range(10)]
        u = [r["_um"] for r in rs]
        sp = [x["n_spike"] for x in u]; nf = [x["n_nonfinite"] for x in u]
        ue = [x["n_event_unified"] for x in u]
        mx = [x["max_excess"] for x in u]; ww = [x["worst_window_mean"] for x in u]
        wwf = [x["worst_window_finite"] for x in u]
        mfl = [x["max_finite_loss"] for x in u]; ml = [x["median_loss"] for x in u]
        blow = [s for s in range(10) if mfl[s] > BLOW]
        e3agg[(m, e)] = dict(sp=sp, nf=nf, ue=ue, mx=mx, ww=ww, wwf=wwf, mfl=mfl,
                             ml=ml, blow=blow,
                             acc=[r["avg_acc"] for r in rs],
                             bwt=[r["bwt"] for r in rs])
        imx = max(range(10), key=lambda i: mx[i]); iww = max(range(10), key=lambda i: ww[i])
        A(f"| {LBL3[m]} | {e} | 10 | {f(*mstd(sp)[:2], p=2)} | {f(*mstd(nf)[:2], p=2)} | "
          f"{f(*mstd(ue)[:2], p=2)} | **{len(blow)} / 10** | {g(med(mx))} | "
          f"{g(max(mx))} (s{imx}) | {g(med(ww))} | {g(max(ww))} (s{iww}) | "
          f"{f(*mstd(ml)[:2], p=3)} |")
A()
A("`worst-window = inf` arises when some 100-step window contains **no** finite loss "
  "at all. The finite-restricted variant (max over windows holding at least one "
  "finite entry) and the count of all-non-finite windows:")
A()
A("| arm | ewc | worst-window (strict) median / max | worst-window (finite-restricted) "
  "median / max | all-non-finite windows, total over 10 seeds |")
A("|---|---|---|---|---|")
for e in EW:
    for m in METH3:
        a_ = e3agg[(m, e)]
        ne = sum(tr[(m, e, s)]["_um"]["n_empty_windows"] for s in range(10))
        A(f"| {LBL3[m]} | {e} | {g(med(a_['ww']))} / {g(max(a_['ww']))} | "
          f"{g(med(a_['wwf']))} / {g(max(a_['wwf']))} | {ne} |")
A()
A("The clean, comparable view -- restricted to the arms/seeds that never blow up, "
  "plus the blow-up counts:")
A()
A("| arm | ewc | non-blow-up seeds | max-excess over those (mean +- std) | "
  "worst-window over those (mean +- std) | blow-up seeds |")
A("|---|---|---|---|---|---|")
for e in EW:
    for m in METH3:
        a_ = e3agg[(m, e)]
        ok = [s for s in range(10) if s not in a_["blow"]]
        A(f"| {LBL3[m]} | {e} | {len(ok)} / 10 | "
          f"{f(*mstd([a_['mx'][s] for s in ok])[:2], p=3) if ok else '-'} | "
          f"{f(*mstd([a_['ww'][s] for s in ok])[:2], p=3) if ok else '-'} | "
          f"{('s' + ', s'.join(str(s) for s in a_['blow'])) if a_['blow'] else '-'} |")
A()
for e in EW:
    A(f"Per-seed detail, ewc = {e} (max-excess / worst-window / spikes / non-finite):")
    A()
    A("| arm | " + " | ".join(f"s{s}" for s in range(10)) + " |")
    A("|---|" + "---|" * 10)
    for m in METH3:
        a_ = e3agg[(m, e)]
        A(f"| {LBL3[m]} | " + " | ".join(
            f"{g(a_['mx'][s], 2)}<br>{g(a_['ww'][s], 2)}<br>{a_['sp'][s]} / {a_['nf'][s]}"
            for s in range(10)) + " |")
    A()

A("### 3.3 (c) Censored analysis: time to first non-finite trigger")
A()
A("A *trigger* is the first non-finite loss, i.e. exactly the step at which the E3 "
  "driver halves lambda and the EWC strength. Metrics are computed on the trace "
  "strictly before that step; never-triggering runs are right-censored at the full "
  "3040-step horizon.")
A()
A("| arm | ewc | frac triggering | mean steps to 1st trigger (triggering seeds) | "
  "min / max | mean steps survived (all, censored) | frac of horizon survived |")
A("|---|---|---|---|---|---|---|")
cens = {}
for e in EW:
    for m in METH3:
        rs = [tr[(m, e, s)] for s in range(10)]
        fnf = [r["_um"]["first_nonfinite_t"] for r in rs]
        n = len(rs[0]["losses"])
        surv = [(t if t is not None else n) for t in fnf]
        trig = [t for t in fnf if t is not None]
        cens[(m, e)] = dict(fnf=fnf, surv=surv, n=n, trig=trig)
        A(f"| {LBL3[m]} | {e} | **{len(trig) / 10:.2f}** ({len(trig)}/10) | "
          f"{f(*mstd(trig)[:2], p=1) if trig else '-'} | "
          f"{(str(min(trig)) + ' / ' + str(max(trig))) if trig else '-'} | "
          f"{f(*mstd(surv)[:2], p=1)} | {mstd(surv)[0] / n:.4f} |")
A()
A("Per-seed first-trigger step (triggering seeds only; task boundaries every 304 "
  "steps):")
A()
for e in EW:
    for m in METH3:
        ts = [(s, cens[(m, e)]["fnf"][s]) for s in range(10)
              if cens[(m, e)]["fnf"][s] is not None]
        if ts:
            A(f"- `{m}` @ ewc {e}: " +
              ", ".join(f"s{s}@t={t} (task {t // 304 + 1})" for s, t in ts))
A()
A("Performance conditional on the trigger status:")
A()
A("| arm | ewc | avg_acc, all 10 | avg_acc, **never-triggering** | avg_acc, "
  "triggering | pre-trigger mean loss (median over seeds) | pre-trigger max-excess "
  "(median) | pre-trigger worst-window (median) | pre-trigger blow-ups |")
A("|---|---|---|---|---|---|---|---|---|")
for e in EW:
    for m in METH3:
        rs = [tr[(m, e, s)] for s in range(10)]
        fnf = cens[(m, e)]["fnf"]
        clean = [rs[s]["avg_acc"] for s in range(10) if fnf[s] is None]
        dirty = [rs[s]["avg_acc"] for s in range(10) if fnf[s] is not None]
        pm, px, pw, pb = [], [], [], 0
        for s in range(10):
            L = rs[s]["losses"]
            pre = L[:(fnf[s] if fnf[s] is not None else len(L))]
            pf = [x for x in pre if math.isfinite(x)]
            pm.append(sum(pf) / len(pf) if pf else float("nan"))
            u = unified_metrics(pre) if pre else None
            px.append(u["max_excess"] if u else float("nan"))
            pw.append(u["worst_window_mean"] if u else float("nan"))
            if u and math.isfinite(u["max_finite_loss"]) and u["max_finite_loss"] > BLOW:
                pb += 1
        A(f"| {LBL3[m]} | {e} | {f(*mstd([r['avg_acc'] for r in rs])[:2])} | "
          f"{f(*mstd(clean)[:2]) if clean else '-'} | "
          f"{f(*mstd(dirty)[:2]) if dirty else '-'} | {g(med(pm))} | {g(med(px))} | "
          f"{g(med(pw))} | {pb} / 10 |")
A()
A("Read-out of the censored block:")
A()
A("* At **ewc = 10** three of four arms never trigger at all (`fixed`, `hd`, `cohg`: "
  "0/10 each, full 3040-step survival, zero spikes, zero non-finite). Only "
  "`cohg_nogate` triggers (1/10, s2 @ t = 733), and its damage is far larger than one "
  "trigger suggests: **6** of its 10 seeds blow up with finite losses up to 5.1e37, and "
  "its pre-trigger statistics are already destroyed. The certificate gate is the "
  "difference between an arm that never leaves the safe regime and one that leaves it "
  "in 6 of 10 seeds.")
A("* At **ewc = 1000** the ordering inverts on this metric. `hd` triggers **0/10** "
  "and blows up in only **1/10** seeds (s8). `fixed` triggers 4/10 and blows up 4/10. "
  "`cohg_nogate` triggers only 3/10 but blows up **8/10** -- the two counts come "
  "apart because most of its damage is finite-but-enormous loss rather than overflow "
  "to `inf`. **`cohg` (gated) triggers 6/10 -- "
  "the worst of the four -- and blows up 6/10**, with first triggers spread from "
  "t = 653 to t = 2932 (mean 1817 +- 876), i.e. the gate does not delay the onset "
  "either.")
A("* Conditioning on survival is informative: every arm's never-triggering seeds are "
  "healthy and near-identical (`fixed` 0.3823 +- 0.0185, `hd` 0.3796 +- 0.0221, "
  "`cohg` 0.3872 +- 0.0209), so the ewc1000 accuracy spread is entirely a "
  "composition effect -- *which* seeds fell over, not how well the survivors did. "
  "`cohg`'s 4 surviving seeds are the best-performing survivor set of the four arms; "
  "its problem at ewc1000 is exclusively that 6 of 10 seeds enter the divergent "
  "regime.")
A()

A("### 3.4 Accuracy / BWT / gate activity beside the trace metrics")
A()
A("| arm | ewc | avg_acc | BWT | collapse rate (avg_acc<0.15) | `gate_open_frac` | "
  "`coord_open_frac` | blow-ups | non-finite triggers | spikes (median) |")
A("|---|---|---|---|---|---|---|---|---|---|")
for e in EW:
    for m in METH3:
        rs = [tr[(m, e, s)] for s in range(10)]
        a_ = e3agg[(m, e)]
        gof = [r.get("gate_open_frac") for r in rs if r.get("gate_open_frac") is not None]
        cof = [r.get("coord_open_frac") for r in rs if r.get("coord_open_frac") is not None]
        A(f"| {LBL3[m]} | {e} | {f(*mstd(a_['acc'])[:2])} | {f(*mstd(a_['bwt'])[:2])} | "
          f"{sum(1 for x in a_['acc'] if x < 0.15) / 10:.2f} | "
          f"{f(*mstd(gof)[:2], p=5) if gof else '-'} | "
          f"{f(*mstd(cof)[:2], p=5) if cof else '-'} | {len(a_['blow'])}/10 | "
          f"{len(cens[(m, e)]['trig'])}/10 | {med(a_['sp']):.1f} |")
A()
A("(`coord_open_frac = 0` for `cohg_nogate` means the certified controller is never "
  "consulted, not that zero coordinates were certified.)")
A()
A("### 3.5 Paired sign-flip tests on the traced runs (common seeds 0-9, n = 10)")
A()
A("Accuracy and BWT are tested directly. max-excess and worst-window span 38 orders "
  "of magnitude, so they are tested on **log10** (a monotone transform: the sign "
  "pattern, and hence the exact sign-flip test's evidence, is about the ordering, and "
  "log10 keeps the mean statistic from being decided by a single 1e38 seed). Seeds "
  "with `worst-window = inf` are excluded from the worst-window test only; the "
  "excluded count is shown.")
A()
A("| ewc | contrast | mean d avg_acc | p | mean d BWT | p | mean d log10 max-excess | "
  "p | mean d log10 worst-window (n used) | p |")
A("|---|---|---|---|---|---|---|---|---|---|")
for e in EW:
    for a, b in [("cohg", "fixed"), ("cohg", "hd"), ("cohg", "cohg_nogate"),
                 ("hd", "fixed"), ("fixed", "cohg_nogate")]:
        da = [tr[(a, e, s)]["avg_acc"] - tr[(b, e, s)]["avg_acc"] for s in range(10)]
        db = [tr[(a, e, s)]["bwt"] - tr[(b, e, s)]["bwt"] for s in range(10)]
        dm = [lg(tr[(a, e, s)]["_um"]["max_excess"]) - lg(tr[(b, e, s)]["_um"]["max_excess"])
              for s in range(10)]
        keep = [s for s in range(10)
                if math.isfinite(tr[(a, e, s)]["_um"]["worst_window_mean"])
                and math.isfinite(tr[(b, e, s)]["_um"]["worst_window_mean"])]
        dw = [lg(tr[(a, e, s)]["_um"]["worst_window_mean"]) -
              lg(tr[(b, e, s)]["_um"]["worst_window_mean"]) for s in keep]
        pa, ma = signflip_p(da); pb_, mb = signflip_p(db)
        pm, mm = signflip_p(dm)
        pw, mw = (signflip_p(dw) if dw else (float("nan"), float("nan")))
        A(f"| {e} | `{a}` - `{b}` | **{ma:+.4f}** | {pa:.4g} | {mb:+.4f} | {pb_:.4g} | "
          f"**{mm:+.4f}** | {pm:.4g} | **{mw:+.4f}** (n={len(dw)}) | {pw:.4g} |")
A()

A("### 3.6 (d) Plain answer: how do the four arms compare under trace-level metrics?")
A()
A("**The two EWC regimes give opposite verdicts, and both earlier claims survive -- "
  "one of them strengthened, the other narrowed.**")
A()
A("**At ewc = 10, the gate separation is real and the trace metrics make it "
  "stronger than the accuracy numbers did.** All of `fixed`, `hd` and `cohg` are "
  "perfectly clean: 0 spikes, 0 non-finite losses, 0 blow-ups, full 3040-step "
  "survival, 0/10 each. Their accuracies are statistically indistinguishable "
  "(`cohg` - `fixed` = -0.006, p = 0.39; `cohg` - `hd` = +0.008, p = 0.44). But on "
  "the degradation metrics `cohg` is the *best* of the three, not merely equal: "
  "max-excess 5.92 +- 0.90 versus 7.19 +- 1.40 (fixed) and 6.97 +- 1.26 (hd), and "
  "worst-window 2.716 +- 0.080 versus 2.904 +- 0.105 and 2.748 +- 0.071. The paired "
  "tests confirm it: `cohg` - `fixed` on log10 max-excess is negative on 9/10 seeds "
  "(p = 0.0059) and on log10 worst-window on **10/10** (p = 0.00195); `cohg` - `hd` "
  "on log10 max-excess is negative on **10/10** (p = 0.00195) and on log10 "
  "worst-window on 8/10 (p = 0.014). Against the ungated ablation the separation is "
  "categorical: `cohg_nogate` blows up in **6 of 10** seeds (s1-s5, s8; max finite "
  "loss up to 5.1e37), records 68.9 +- 105.0 spikes per run against `cohg`'s exact "
  "zero, triggers a non-finite recovery in s2, and loses 0.149 accuracy on 10/10 "
  "seeds (p = 0.00195). **At ewc10 the certificate gate is what keeps a method that would "
  "otherwise diverge in 6 of 10 seeds at exactly zero degradation events -- and it "
  "does so while being marginally *safer* than the fixed baseline, which the "
  "accuracy-only view could not show.**")
A()
A("**At ewc = 1000, HD's dominance over COHG holds and is reinforced by the trace "
  "metrics.** On accuracy alone the gap is small and insignificant: `cohg` 0.3497 +- "
  "0.0929 versus `hd` 0.3796 +- 0.0221, paired difference -0.030, p = 0.44. On "
  "stability it is not small. `hd` never produces a non-finite loss (0/10 triggers) "
  "and blows up in a single seed (s8); `cohg` triggers in **6/10** seeds and blows up "
  "in **6/10**, the worst trigger rate of any arm in either regime, worse even than "
  "the ungated ablation (3/10 triggers, though 8/10 blow-ups) and than `fixed` "
  "(4/10 and 4/10). Paired on log10 max-excess, `cohg` - `hd` is positive on 6/10 "
  "seeds with p = 0.031; on log10 worst-window, positive on 7/10 with p = 0.027. "
  "And the "
  "conditional analysis says the gate is not buying a softer failure either: `cohg`'s "
  "first triggers span t = 653-2932 with mean 1817 +- 876, no later than `fixed`'s "
  "1793 +- 850. **So the honest ewc1000 statement is: the certified gate does not "
  "protect against the EWC-1000 failure mode at all, HD does, and COHG is if anything "
  "the most trigger-prone arm there.** The single mitigating fact is the "
  "conditional-on-survival result: `cohg`'s 4 surviving seeds average 0.3872 +- 0.0209, "
  "the best survivor set of the four arms -- the ewc1000 problem is composition (how "
  "many seeds diverge), not quality (how the survivors do).")
A()
A("**A finding that cuts across both claims and should be stated in the paper.** At "
  "ewc = 1000 the *fixed* baseline itself blows up in 4/10 seeds and triggers 4/10. "
  "The ewc1000 instability is therefore a property of the **operating point**, not of "
  "adaptive LR control: it is the EWC penalty at strength 1000 that makes the loss "
  "surface divergent, and `hd`'s small, AdaGrad-normalised meta-steps happen to stay "
  "out of the divergent basin while both `cohg` variants and the fixed LR walk into "
  "it. Framing ewc1000 as \"COHG fails where HD succeeds\" is only half true; "
  "\"ewc1000 is a divergent operating point in which only HD's step size is small "
  "enough to stay out, and COHG's certificate offers no protection there\" is the "
  "supported statement.")
A()
A("**Caveat that limits (b), (c) and (d).** All of section 3 is computed on the "
  "traced set, whose ewc1000 block is device-divergent from the canonical set "
  "(section 3.1). Seed-level ewc1000 statements are not portable between the two "
  "sets; the ewc10 block is. Because the *ranking* of the arms at ewc1000 is the same "
  "in both sets (`hd` best and clean, `cohg` and `cohg_nogate` unstable, `fixed` "
  "intermediate), the qualitative conclusion above is robust to the device, but the "
  "specific counts (6/10, 4/10, 3/10, 1/10) are device-conditional and should be "
  "quoted as \"on the traced set\".")
A()

# ------------------------------------------------------------------ write
os.makedirs(os.path.dirname(OUTMD), exist_ok=True)
open(OUTMD, "w", encoding="utf-8").write("\n".join(OUT) + "\n")
print("wrote", OUTMD, "lines:", len(OUT))
