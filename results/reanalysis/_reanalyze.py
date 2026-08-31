"""Unified degradation metrics + censored (pre-recovery) analysis.

Reads raw run JSONs from results/e2 (drift arms, lr in {0.003,0.03}),
results/e3, results/e4_v2 and writes per-run CSVs + markdown summaries
into results/reanalysis/.

Spike rule (IDENTICAL everywhere, replicates the in-run E2 rule):
    maintain a deque of the last 500 FINITE losses; for each finite loss,
    if len(window) >= 100 and loss > 10 * median(window) -> spike.
    Every non-finite loss counts as an event too.
"""
import os, re, json, math, glob, csv
from collections import deque, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "results", "reanalysis")
os.makedirs(OUT, exist_ok=True)


# ----------------------------------------------------------------- metrics
def unified_metrics(losses):
    """Return dict of unified degradation metrics from a raw loss trace."""
    n = len(losses)
    fin = [x for x in losses if math.isfinite(x)]
    n_nonfinite = n - len(fin)

    # (a) spike events, same rule everywhere
    win = deque(maxlen=500)
    n_spike = 0
    first_spike = None
    for t, x in enumerate(losses):
        if not math.isfinite(x):
            continue
        if len(win) >= 100:
            w = sorted(win)
            med = w[len(w) // 2]
            if x > 10.0 * med:
                n_spike += 1
                if first_spike is None:
                    first_spike = t
        win.append(x)

    # (b) max-excess (finite only)
    if fin:
        s = sorted(fin)
        med_all = s[len(s) // 2]
        max_fin = max(fin)
        max_excess = max_fin / med_all if med_all > 0 else float("inf")
    else:
        med_all = float("nan"); max_fin = float("nan"); max_excess = float("inf")

    # (c) worst trailing-100-step window mean (mean over finite entries in win)
    W = 100
    worst_win = float("-inf"); worst_win_end = -1
    if n >= W:
        for e in range(W, n + 1):
            seg = losses[e - W:e]
            sf = [x for x in seg if math.isfinite(x)]
            m = (sum(sf) / len(sf)) if sf else float("inf")
            if m > worst_win:
                worst_win = m; worst_win_end = e - 1
    else:
        worst_win = (sum(fin) / len(fin)) if fin else float("inf")
        worst_win_end = n - 1

    # first recovery trigger = first non-finite loss
    first_nf = next((t for t, x in enumerate(losses) if not math.isfinite(x)), None)

    return dict(
        n_steps=n, n_finite=len(fin), n_nonfinite=n_nonfinite,
        n_spike=n_spike, n_event_unified=n_spike + n_nonfinite,
        first_spike_t=first_spike,
        median_loss=med_all, max_finite_loss=max_fin, max_excess=max_excess,
        worst_window_mean=worst_win, worst_window_end_t=worst_win_end,
        first_nonfinite_t=first_nf,
        mean_loss_finite=(sum(fin) / len(fin)) if fin else float("nan"),
    )


def censored(losses, nmse_full=None):
    """Metrics up to (excluding) the first non-finite loss."""
    first_nf = next((t for t, x in enumerate(losses) if not math.isfinite(x)), None)
    n = len(losses)
    triggered = first_nf is not None
    steps_survived = first_nf if triggered else n
    pre = losses[:steps_survived]
    pf = [x for x in pre if math.isfinite(x)]
    fin_all = [x for x in losses if math.isfinite(x)]
    mean_pre = (sum(pf) / len(pf)) if pf else float("nan")
    mean_all = (sum(fin_all) / len(fin_all)) if fin_all else float("nan")
    # NMSE-so-far: nmse_full = mean_all * 2 / y_var  =>  scale by mean ratio
    nmse_pre = (nmse_full * mean_pre / mean_all
                if (nmse_full is not None and mean_all and math.isfinite(mean_all)
                    and mean_all > 0 and math.isfinite(mean_pre)) else float("nan"))
    um = unified_metrics(pre) if pre else None
    return dict(
        triggered=int(triggered), steps_survived=steps_survived,
        frac_survived=steps_survived / n if n else float("nan"),
        pre_mean_loss=mean_pre, pre_nmse=nmse_pre,
        pre_median_loss=(sorted(pf)[len(pf) // 2] if pf else float("nan")),
        pre_max_loss=(max(pf) if pf else float("nan")),
        pre_max_excess=(um["max_excess"] if um else float("nan")),
        pre_n_spike=(um["n_spike"] if um else 0),
        pre_worst_window=(um["worst_window_mean"] if um else float("nan")),
    )


def mstd(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return float("nan"), float("nan"), 0
    n = len(v)
    m = sum(v) / n
    if n < 2:
        return m, float("nan"), n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    return m, sd, n


def fmt(m, s, prec=3):
    if not math.isfinite(m):
        return "inf"
    if not math.isfinite(s):
        return f"{m:.{prec}f}"
    return f"{m:.{prec}f}+-{s:.{prec}f}"


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ------------------------------------------------------------------- E2
E2_DIR = os.path.join(ROOT, "results", "e2")
LRS = {"0.003", "0.03"}


def parse_e2(fname):
    b = fname[:-5]
    m = re.match(r"^(?P<ds>lorenz|mackey|santafe|sunspot)_(?P<rest>.+)_s(?P<seed>\d+)$", b)
    if not m:
        return None
    ds, rest, seed = m["ds"], m["rest"], int(m["seed"])
    if not rest.startswith("drift_"):
        return None
    rest = rest[len("drift_"):]
    mm = re.match(r"^(?P<meth>.+)_lr(?P<lr>[0-9.]+)$", rest)
    if not mm:
        return None
    return dict(dataset=ds, method=mm["meth"], lr0=mm["lr"], seed=seed)


def run_e2():
    rows, crows = [], []
    for p in sorted(glob.glob(os.path.join(E2_DIR, "*.json"))):
        meta = parse_e2(os.path.basename(p))
        if meta is None or meta["lr0"] not in LRS:
            continue
        try:
            d = json.load(open(p))
        except Exception:
            print("SKIP unreadable", p); continue
        L = d["losses"]
        um = unified_metrics(L)
        arm = f"{meta['dataset']}_drift_{meta['method']}_lr{meta['lr0']}"
        r = dict(regime="E2-drift", arm=arm, **meta,
                 nmse=d["nmse"], events_stored=d["events"],
                 gate_open_frac=d.get("gate_open_frac"), **um)
        rows.append(r)
        c = censored(L, nmse_full=d["nmse"])
        crows.append(dict(regime="E2-drift", arm=arm, **meta, nmse_full=d["nmse"], **c))
    write_csv(os.path.join(OUT, "unified_metrics_e2_drift.csv"), rows)
    write_csv(os.path.join(OUT, "censored_e2_drift.csv"), crows)
    return rows, crows


# ------------------------------------------------------------------- E3
E3_DIR = os.path.join(ROOT, "results", "e3")


def run_e3():
    rows = []
    for p in sorted(glob.glob(os.path.join(E3_DIR, "*.json"))):
        b = os.path.basename(p)[:-5]
        m = re.match(r"^cifar100_(?P<meth>.+)_lr(?P<lr>[0-9.]+)_ewc(?P<ewc>[0-9.]+)_s(?P<seed>\d+)$", b)
        if not m:
            print("SKIP name", b); continue
        d = json.load(open(p))
        A = d["acc_matrix"]
        final = A[-1]
        avg = d["avg_acc"]
        # accuracy-space degradation proxies (loss traces are NOT stored for E3)
        diag = [A[k][k] for k in range(len(A))]
        drops = [A[-1][j] - A[j][j] for j in range(len(A) - 1)]
        # running-average-accuracy trajectory: mean acc over seen tasks after task k
        traj = [sum(a) / len(a) for a in A]
        rows.append(dict(
            regime="E3", arm=f"cifar100_{m['meth']}_lr{m['lr']}_ewc{m['ewc']}",
            method=m["meth"], lr0=m["lr"], ewc0=m["ewc"], seed=int(m["seed"]),
            avg_acc=avg, bwt=d["bwt"],
            collapse=int(avg < 0.15),
            events_stored=d["events"],       # non-finite recovery triggers only
            n_nonfinite=d["events"],
            losses_available=0,              # <-- no per-step loss trace stored
            n_spike="NA", max_excess="NA", worst_window_mean="NA",
            min_task_acc=min(final), max_forget=(-min(drops) if drops else 0.0),
            worst_running_avg_acc=min(traj), final_task_acc=diag[-1],
            gate_open_frac=d.get("gate_open_frac"),
        ))
    write_csv(os.path.join(OUT, "unified_metrics_e3.csv"), rows)
    return rows


# ------------------------------------------------------------------- E4
E4_DIR = os.path.join(ROOT, "results", "e4_v2")


def sexp(x):
    try:
        return math.exp(x)
    except OverflowError:
        return float("inf")


def run_e4():
    rows = []
    for p in sorted(glob.glob(os.path.join(E4_DIR, "*.json"))):
        b = os.path.basename(p)[:-5]
        m = re.match(r"^gpt2_(?P<rest>.+)_s(?P<seed>\d+)$", b)
        d = json.load(open(p))
        L = d["losses"]
        um = unified_metrics(L)
        rows.append(dict(
            regime="E4-gpt2", arm="gpt2_" + m["rest"], method=d["method"],
            seed=int(m["seed"]), online_ppl=d["online_ppl"],
            events_stored=d["events"], gate_open_frac=d.get("gate_open_frac"),
            **um,
            worst_window_ppl=sexp(um["worst_window_mean"]),
            median_ppl=sexp(um["median_loss"]),
            max_finite_ppl=sexp(um["max_finite_loss"]),
        ))
    write_csv(os.path.join(OUT, "unified_metrics_e4.csv"), rows)
    return rows


# --------------------------------------------------------------- summaries
def summarize(rows, keys, arm_key="arm"):
    out = {}
    g = defaultdict(list)
    for r in rows:
        g[r[arm_key]].append(r)
    for a, rs in sorted(g.items()):
        out[a] = {"n": len(rs)}
        for k in keys:
            vals = [r[k] for r in rs if isinstance(r[k], (int, float))]
            out[a][k] = mstd(vals)
    return out


def main():
    e2, e2c = run_e2()
    e3 = run_e3()
    e4 = run_e4()

    lines = []
    A = lines.append
    A("# Unified degradation metrics (reanalysis)\n")
    A("Generated by `results/reanalysis/_reanalyze.py` from raw run artifacts.\n")
    A("**Spike rule, identical in every regime** (replicates the rule that E2 "
      "applied online, now applied post hoc to E3/E4 as well): maintain a window "
      "of the last 500 *finite* losses; once the window holds >=100 entries, a step "
      "counts as a spike if `loss_t > 10 x median(window)`. Every non-finite loss "
      "also counts as an event. `unified events = spikes + non-finite`.\n")
    A("**max-excess** = `max_t finite loss_t / median(all finite losses)`.\n")
    A("**worst-window** = `max_t mean(loss[t-99..t])` over finite entries; for E4 "
      "reported as perplexity `exp(.)`.\n")
    A("All aggregates are mean+-std with ddof=1 over seeds.\n")

    # ---- E2
    A("\n## E2 (Lorenz / Mackey-Glass, drift arms, lr0 in {0.003, 0.03})\n")
    A("Per-run CSV: `unified_metrics_e2_drift.csv`\n")
    ks = ["n_spike", "n_nonfinite", "n_event_unified", "events_stored",
          "max_excess", "worst_window_mean", "nmse"]
    s = summarize(e2, ks)
    A("| arm | n | spikes (a) | non-finite | unified events | events (stored) | "
      "max-excess (b) | worst-window (c) | NMSE |")
    A("|---|---|---|---|---|---|---|---|---|")
    for a, v in s.items():
        A(f"| {a} | {v['n']} | {fmt(*v['n_spike'][:2],prec=2)} | "
          f"{fmt(*v['n_nonfinite'][:2],prec=2)} | {fmt(*v['n_event_unified'][:2],prec=2)} | "
          f"{fmt(*v['events_stored'][:2],prec=2)} | {fmt(*v['max_excess'][:2],prec=2)} | "
          f"{fmt(*v['worst_window_mean'][:2],prec=4)} | {fmt(*v['nmse'][:2],prec=4)} |")

    # ---- E3
    A("\n## E3 (Split-CIFAR-100 continual)\n")
    A("Per-run CSV: `unified_metrics_e3.csv`\n")
    A("> **Data limitation (important).** The E3 driver "
      "(`code/experiments/e3_continual.py`) does **not** persist a per-step loss "
      "trace: the saved keys are "
      "`acc_matrix, avg_acc, bwt, events, gate_open_frac, coord_open_frac, "
      "hvp_total, lam_hist, wall_s`. Metrics (a) spikes, (b) max-excess and "
      "(c) worst-window are loss-trace quantities and are therefore **not "
      "computable for E3 from the stored artifacts** (they are marked `NA`). "
      "`events` for E3 counts non-finite losses only. Below we report the stored "
      "non-finite count, the collapse indicator, and accuracy-space degradation "
      "proxies computed from `acc_matrix`.\n")
    ks3 = ["avg_acc", "bwt", "events_stored", "collapse", "min_task_acc",
           "max_forget", "worst_running_avg_acc"]
    s3 = summarize(e3, ks3)
    A("| arm | n | avg_acc | BWT | non-finite events | collapse rate "
      "(avg_acc<0.15) | min final-task acc | max forgetting | worst running-avg acc |")
    A("|---|---|---|---|---|---|---|---|---|")
    for a, v in s3.items():
        A(f"| {a} | {v['n']} | {fmt(*v['avg_acc'][:2],prec=4)} | "
          f"{fmt(*v['bwt'][:2],prec=4)} | {fmt(*v['events_stored'][:2],prec=2)} | "
          f"{v['collapse'][0]:.2f} | {fmt(*v['min_task_acc'][:2],prec=4)} | "
          f"{fmt(*v['max_forget'][:2],prec=4)} | "
          f"{fmt(*v['worst_running_avg_acc'][:2],prec=4)} |")

    # ---- E4
    A("\n## E4 (GPT-2 test-time adaptation, results/e4_v2)\n")
    A("Per-run CSV: `unified_metrics_e4.csv`\n")
    ks4 = ["n_spike", "n_nonfinite", "n_event_unified", "events_stored",
           "max_excess", "worst_window_ppl", "online_ppl", "max_finite_loss"]
    s4 = summarize(e4, ks4)
    A("| arm | n | spikes (a) | non-finite | unified events | events (stored) | "
      "max-excess (b) | worst-window PPL (c) | online PPL |")
    A("|---|---|---|---|---|---|---|---|---|")
    for a, v in s4.items():
        A(f"| {a} | {v['n']} | {fmt(*v['n_spike'][:2],prec=2)} | "
          f"{fmt(*v['n_nonfinite'][:2],prec=2)} | {fmt(*v['n_event_unified'][:2],prec=2)} | "
          f"{fmt(*v['events_stored'][:2],prec=2)} | {fmt(*v['max_excess'][:2],prec=2)} | "
          f"{fmt(*v['worst_window_ppl'][:2],prec=2)} | {fmt(*v['online_ppl'][:2],prec=3)} |")
    A("\n### E4 per-seed detail (spikes are seed-specific)\n")
    A("| run | spikes | max-excess | worst-window PPL | online PPL | max finite loss |")
    A("|---|---|---|---|---|---|")
    for r in e4:
        A(f"| {r['arm']}_s{r['seed']} | {r['n_spike']} | {r['max_excess']:.2f} | "
          f"{r['worst_window_ppl']:.2f} | {r['online_ppl']:.3f} | {r['max_finite_loss']:.3f} |")

    open(os.path.join(OUT, "unified_metrics.md"), "w").write("\n".join(lines) + "\n")

    # ------------------------------------------------------ censored report
    cl = []
    B = cl.append
    B("# Censored (pre-recovery) analysis\n")
    B("Generated by `results/reanalysis/_reanalyze.py`.\n")
    B("A *recovery trigger* is the first non-finite loss, which is exactly where "
      "the drivers restore the checkpoint and back off lambda. Metrics below are "
      "computed on the trace **strictly before** that step, so they compare methods "
      "on the part of the run where method identity, not the shared recovery "
      "heuristic, determines behaviour. Runs that never trigger are right-censored "
      "at the full horizon.\n")
    B("`NMSE-so-far` is reconstructed exactly as "
      "`nmse_full * mean(pre-trigger finite losses) / mean(all finite losses)`, "
      "which is algebraically identical to recomputing NMSE on the pre-trigger "
      "prefix (the target variance normaliser is a per-run constant).\n")

    B("\n## E2 drift arms (per-run CSV: `censored_e2_drift.csv`)\n")
    B("| arm | n seeds | frac triggering | mean steps to 1st trigger (triggering seeds) | "
      "mean steps survived (all, censored) | pre-trigger NMSE | pre-trigger mean loss | "
      "pre-trigger spikes | full-run NMSE |")
    B("|---|---|---|---|---|---|---|---|---|")
    g = defaultdict(list)
    for r in e2c:
        g[r["arm"]].append(r)
    nmse_full_by_arm = defaultdict(list)
    for r in e2:
        nmse_full_by_arm[r["arm"]].append(r["nmse"])
    for a, rs in sorted(g.items()):
        trig = [r for r in rs if r["triggered"]]
        frac = len(trig) / len(rs)
        mst = mstd([r["steps_survived"] for r in trig])
        msa = mstd([r["steps_survived"] for r in rs])
        pn = mstd([r["pre_nmse"] for r in rs])
        pm = mstd([r["pre_mean_loss"] for r in rs])
        ps = mstd([r["pre_n_spike"] for r in rs])
        fn = mstd(nmse_full_by_arm[a])
        B(f"| {a} | {len(rs)} | {frac:.2f} ({len(trig)}/{len(rs)}) | "
          f"{fmt(*mst[:2],prec=1) if trig else '-'} | {fmt(*msa[:2],prec=1)} | "
          f"{fmt(*pn[:2],prec=4)} | {fmt(*pm[:2],prec=4)} | {fmt(*ps[:2],prec=2)} | "
          f"{fmt(*fn[:2],prec=4)} |")

    B("\n### E2: per-seed first-trigger steps (triggering seeds only)\n")
    for a, rs in sorted(g.items()):
        trig = sorted([(r["seed"], r["steps_survived"]) for r in rs if r["triggered"]])
        if trig:
            B(f"- `{a}`: " + ", ".join(f"s{s}@t={t}" for s, t in trig))

    B("\n## E3 (per-run CSV: `unified_metrics_e3.csv`)\n")
    B("> E3 stores no loss trace and no timestamp for recovery triggers, so "
      "*steps to first trigger* and *pre-trigger performance* are **not "
      "recoverable** from the artifacts. What is recoverable is whether a seed "
      "ever triggered (`events > 0`, non-finite losses) and its end-of-run "
      "accuracy. The per-task `acc_matrix` gives an accuracy-space analogue of "
      "'performance before things go wrong'.\n")
    B("| arm | n seeds | frac with >=1 recovery trigger | mean triggers/seed | "
      "avg_acc (all seeds) | avg_acc (never-triggering seeds) | avg_acc "
      "(triggering seeds) | collapse rate |")
    B("|---|---|---|---|---|---|---|---|")
    g3 = defaultdict(list)
    for r in e3:
        g3[r["arm"]].append(r)
    for a, rs in sorted(g3.items()):
        trig = [r for r in rs if r["events_stored"] > 0]
        clean = [r for r in rs if r["events_stored"] == 0]
        B(f"| {a} | {len(rs)} | {len(trig)/len(rs):.2f} ({len(trig)}/{len(rs)}) | "
          f"{sum(r['events_stored'] for r in rs)/len(rs):.2f} | "
          f"{fmt(*mstd([r['avg_acc'] for r in rs])[:2],prec=4)} | "
          f"{fmt(*mstd([r['avg_acc'] for r in clean])[:2],prec=4) if clean else '-'} | "
          f"{fmt(*mstd([r['avg_acc'] for r in trig])[:2],prec=4) if trig else '-'} | "
          f"{sum(r['collapse'] for r in rs)/len(rs):.2f} |")

    open(os.path.join(OUT, "censored_analysis.md"), "w").write("\n".join(cl) + "\n")
    print("wrote", OUT)
    print("E2 rows", len(e2), "E3 rows", len(e3), "E4 rows", len(e4))


if __name__ == "__main__":
    main()
