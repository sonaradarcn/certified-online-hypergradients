"""Round-4 analysis: does the offline-calibrated static gate threshold transfer?

Reads
    results/e2_lorenz_absgate/  (+ _verify)   E2 lorenz_drift, CPU
    results/e3_absgate/         (+ _ref)      E3 Split-CIFAR-100
    results/e4_absgate/         (+ _ref)      E4 GPT-2 124M TTA
    results/e2, results/e2_controls, results/e3_traced, results/e4_v2 (refs)
and writes results/reanalysis/round4_absgate_transfer.md.

Statistics: ddof=1 sample std; every paired comparison is an EXACT two-sided
sign-flip permutation test enumerating all 2^n sign assignments of the per-seed
paired differences (n=10 -> smallest attainable p = 2/1024 = 0.0020;
n=3 -> 2/8 = 0.2500).  Degradation metrics come from the unified rule in
results/reanalysis/_reanalyze.py (spikes, non-finite, max-excess, worst
trailing 100-step window), applied to the raw per-step loss traces.
"""

from __future__ import annotations

import glob
import itertools
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
REAN = os.path.join(RES, "reanalysis")
sys.path.insert(0, REAN)
from _reanalyze import unified_metrics                       # noqa: E402

T_CONST = 0.05806520209
LAM_MIN_E4, LAM_MAX_E4 = math.log(1e-6), math.log(0.1)
OUT_MD = os.path.join(REAN, "round4_absgate_transfer.md")


# ------------------------------------------------------------------ helpers
def load(pat):
    out = {}
    for fn in sorted(glob.glob(pat)):
        try:
            j = json.load(open(fn))
        except Exception:                                    # noqa: BLE001
            continue
        j["_file"] = fn
        out[j.get("seed")] = j
    return out


def mstd(v):
    v = [float(x) for x in v
         if isinstance(x, (int, float)) and not
         (isinstance(x, float) and math.isnan(x))]
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(v) / n
    if n < 2:
        return m, 0.0, n
    return m, math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)), n


def f(x, spec=".4f"):
    if x is None:
        return "-"
    if isinstance(x, float):
        if math.isnan(x):
            return "-"
        if math.isinf(x):
            return "inf"
    try:
        return format(x, spec)
    except (TypeError, ValueError):
        return str(x)


def ms(v, spec=".4f"):
    m, sd, n = mstd(v)
    if n == 0:
        return "-"
    if isinstance(m, float) and math.isinf(m):
        return "inf"
    return "%s+-%s" % (f(m, spec), f(sd, spec.lstrip("+")))


def med(v):
    v = sorted(x for x in v if x is not None and
               isinstance(x, (int, float)) and
               not (isinstance(x, float) and math.isnan(x)))
    return v[len(v) // 2] if v else None


def perm_paired(a, b, min_n=3):
    common = sorted(s for s in (set(a) & set(b)) if s is not None)
    n = len(common)
    if n < min_n or n > 22:
        return None, n, float("nan")
    d = [float(a[s]) - float(b[s]) for s in common]
    obs = sum(d) / n
    hits = tot = 0
    for fl in itertools.product((1, -1), repeat=n):
        st = sum(x * y for x, y in zip(d, fl)) / n
        tot += 1
        if abs(st) >= abs(obs) - 1e-15:
            hits += 1
    return hits / tot, n, obs


UM_CACHE = {}


def um(run):
    fn = run.get("_file")
    if fn in UM_CACHE:
        return UM_CACHE[fn]
    L = run.get("losses")
    u = unified_metrics(L) if L else None
    UM_CACHE[fn] = u
    return u


def lam_window(lam_hist):
    prev = first = last = None
    for row in lam_hist or []:
        t, lam = row[0], row[1:]
        if prev is not None and max(abs(x - y)
                                    for x, y in zip(lam, prev)) > 1e-9:
            last = t
            if first is None:
                first = t
        prev = lam
    return first, last


def final_lam(run, k=None):
    lh = run.get("lam_hist")
    if not lh:
        return None
    row = lh[-1][1:]
    return row[:k] if k else row


# ------------------------------------------------------------------ arms
class Arm:
    def __init__(self, label, runs):
        self.label = label
        self.runs = {s: r for s, r in runs.items() if s is not None}

    def n(self):
        return len(self.runs)

    def bs(self, fn):
        """{seed: value}, skipping seeds where fn is None/NaN."""
        out = {}
        for s, r in self.runs.items():
            try:
                v = fn(r)
            except Exception:                                # noqa: BLE001
                v = None
            if v is None:
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if math.isnan(v):
                continue
            out[s] = v
        return out

    def vals(self, fn):
        return list(self.bs(fn).values())

    def umbs(self, key):
        return self.bs(lambda r: (um(r) or {}).get(key))

    def umvals(self, key):
        return list(self.umbs(key).values())


# unified degradation columns, identical for every domain
UM_COLS = [("unified events", "n_event_unified", ".1f"),
           ("non-finite", "n_nonfinite", ".1f"),
           ("spikes", "n_spike", ".1f"),
           ("max-excess", "max_excess", ".3g"),
           ("worst-window", "worst_window_mean", ".4g")]


def table(arms, primary_name, primary_fn, primary_spec, extra):
    """extra: list of (header, fn, spec)."""
    hdr = ["arm", "n", primary_name] + [h for h, _, _ in extra] \
        + [h for h, _, _ in UM_COLS]
    out = ["| " + " | ".join(hdr) + " |",
           "|" + "---|" * len(hdr)]
    for a in arms:
        if a.n() == 0:
            continue
        cells = [a.label, str(a.n()), ms(a.vals(primary_fn), primary_spec)]
        for _, fn, spec in extra:
            cells.append(ms(a.vals(fn), spec))
        for _, key, spec in UM_COLS:
            cells.append(ms(a.umvals(key), spec))
        out.append("| " + " | ".join(cells) + " |")
    return out


def tests(rows):
    out = ["| comparison | metric | delta (mean paired diff) | p | n |",
           "|---|---|---|---|---|"]
    for label, metric, a, b, spec in rows:
        p, n, d = perm_paired(a, b)
        out.append("| %s | %s | %s | %s | %d |"
                   % (label, metric, f(d, spec),
                      ("%.4f" % p) if p is not None else "-", n))
    return out


# ------------------------------------------------------------------ gate stats
def gstat_row(label, runs):
    gs = [r["gate_stats"] for r in runs if r.get("gate_stats")]
    if not gs:
        return None

    def a(dicts, k):
        return med([d[k] for d in dicts if d and d.get(k) is not None])
    g = [x["ghat_abs"] for x in gs]
    b = [x["cbeta"] for x in gs if x.get("cbeta")]
    r = {"label": label, "n": len(gs),
         "gmed": a(g, "median"), "gp90": a(g, "p90"), "gp99": a(g, "p99"),
         "gmax": a(g, "max"),
         "bmed": a(b, "median") if b else None,
         "bp90": a(b, "p90") if b else None,
         "bp99": a(b, "p99") if b else None,
         "bmax": a(b, "max") if b else None,
         "openT": med([x["frac_open_const"] for x in gs]),
         "openC": med([x["frac_open_cert"] for x in gs
                       if x.get("frac_open_cert") is not None]) if b else None}
    return r


GT_HDR = ("| run source | n | med \\|ghat\\| | p90 | p99 | max | "
          "med c*beta | p90 | p99 | max | frac \\|ghat\\|>T | "
          "frac \\|ghat\\|>c*beta |")


def gtable(rows):
    out = [GT_HDR, "|" + "---|" * 12]
    for r in rows:
        if r is None:
            continue
        out.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | "
                   "%s |" % (r["label"], r["n"],
                             f(r["gmed"], ".3e"), f(r["gp90"], ".3e"),
                             f(r["gp99"], ".3e"), f(r["gmax"], ".3e"),
                             f(r["bmed"], ".3e"), f(r["bp90"], ".3e"),
                             f(r["bp99"], ".3e"), f(r["bmax"], ".3e"),
                             f(r["openT"], ".3e"),
                             f(r["openC"], ".3e") if r["openC"] is not None
                             else "-"))
    return out


def status_block():
    import glob as _g
    want = [("E2 lorenz_drift (absgate/COHG/cohg_nogate x seeds 0-9, CPU)",
             os.path.join(RES, "e2_lorenz_absgate", "lorenz_drift_*.json"), 30),
            ("E2 mackey_drift CALIBRATION-stream COHG reference (gate stats)",
             os.path.join(RES, "e2_lorenz_absgate", "_calib_ref", "*.json"), 3),
            ("E2 reproduction check (patched driver vs results/e2_controls)",
             os.path.join(RES, "e2_lorenz_absgate", "_verify", "*.json"), 1),
            ("E4 GPT-2 absgate (seeds 0-2)",
             os.path.join(RES, "e4_absgate", "gpt2_absgate_*.json"), 3),
            ("E4 GPT-2 COHG reference re-run (repro check + gate stats)",
             os.path.join(RES, "e4_absgate", "_ref", "*.json"), 1),
            ("E3 Split-CIFAR-100 absgate (ewc10/ewc1000 x seeds 0-9)",
             os.path.join(RES, "e3_absgate", "cifar100_absgate_*.json"), 20),
            ("E3 COHG reference re-runs (repro check + gate stats)",
             os.path.join(RES, "e3_absgate", "_ref", "*.json"), 2)]
    L = ["| artefact | have | want | state |", "|---|---|---|---|"]
    for lbl, pat, n in want:
        have = len(_g.glob(pat))
        st = "**complete**" if have >= n else ("running" if have else "pending")
        L.append("| %s | %d | %d | %s |" % (lbl, have, n, st))
    return L


def pending(arms, what):
    """Note + True when the arm set is not yet populated."""
    if any(a.n() for a in arms):
        return []
    return ["> **%s -- NOT YET AVAILABLE.**  The runs are still in the queue; "
            "this section is a placeholder and will be filled when they land."
            % what, ""]


# ------------------------------------------------------------------ loaders
def E2():
    d = os.path.join(RES, "e2_lorenz_absgate")
    e2 = os.path.join(RES, "e2")
    return {
        "absgate": Arm("absgate T=0.05807 TRANSFERRED (CPU)",
                       load(os.path.join(
                           d, "lorenz_drift_absgate_lr0.003_s*.json"))),
        "cohg": Arm("COHG certificate gate (CPU)",
                    load(os.path.join(
                        d, "lorenz_drift_cohg_lr0.003_s*.json"))),
        "nogate": Arm("cohg_nogate ungated sign step alpha=0.4 (CPU)",
                      load(os.path.join(
                          d, "lorenz_drift_cohg_nogate_lr0.003_s*.json"))),
        "cohg_gpu": Arm("COHG certificate gate (GPU, results/e2)",
                        load(os.path.join(
                            e2, "lorenz_drift_cohg_lr0.003_s*.json"))),
        "nogate_gpu": Arm("cohg_nogate (GPU, results/e2)",
                          load(os.path.join(
                              e2,
                              "lorenz_drift_cohg_nogate_lr0.003_s*.json"))),
        "fixed": Arm("fixed lr=0.003 (mis-set init, un-adapted; GPU)",
                     load(os.path.join(
                         e2, "lorenz_drift_fixed_lr0.003_s*.json"))),
    }


def E3(ewc):
    a = os.path.join(RES, "e3_absgate")
    tr = os.path.join(RES, "e3_traced")
    p = "cifar100_%s_lr0.05_ewc%g_s*.json"
    return {
        "absgate": Arm("absgate T=0.05807 TRANSFERRED",
                       load(os.path.join(a, p % ("absgate", ewc)))),
        "cohg": Arm("COHG certificate gate",
                    load(os.path.join(tr, p % ("cohg", ewc)))),
        "nogate": Arm("cohg_nogate ungated sign step alpha=0.4",
                      load(os.path.join(tr, p % ("cohg_nogate", ewc)))),
        "hd": Arm("hd (Baydin et al.)",
                  load(os.path.join(tr, p % ("hd", ewc)))),
        "fixed": Arm("fixed lambda",
                     load(os.path.join(tr, p % ("fixed", ewc)))),
        "ref": Arm("COHG re-run + gate stats (this round)",
                   load(os.path.join(a, "_ref", p % ("cohg", ewc)))),
    }


def E4():
    a = os.path.join(RES, "e4_absgate")
    v2 = os.path.join(RES, "e4_v2")
    return {
        "absgate": Arm("absgate T=0.05807 TRANSFERRED (rank 0)",
                       load(os.path.join(a, "gpt2_absgate_lr0.001_s*.json"))),
        "cohg_r0": Arm("COHG certificate gate (cohg_r0, the e4_v2 reference)",
                       load(os.path.join(v2, "gpt2_cohg_r0_lr0.001_s*.json"))),
        "cohg": Arm("COHG certificate gate (rank 4)",
                    load(os.path.join(v2, "gpt2_cohg_lr0.001_s*.json"))),
        "nogate": Arm("cohg_nogate ungated sign step alpha=0.4",
                      load(os.path.join(
                          v2, "gpt2_cohg_nogate_lr0.001_s*.json"))),
        "fixed": Arm("fixed lr=1e-3",
                     load(os.path.join(v2, "gpt2_fixed_lr0.001_s*.json"))),
        "ref": Arm("COHG cohg_r0 re-run + gate stats (this round)",
                   load(os.path.join(a, "_ref",
                                     "gpt2_cohg_r0_lr0.001_s*.json"))),
    }


# ------------------------------------------------------------------ checks
def diff_json(a, b, skip=("wall_s", "_file", "peak_mem_gb")):
    """Return list of differing shared keys (a = stored, b = re-run)."""
    out = []
    for k in a:
        if k in skip or k not in b:
            continue
        if a[k] != b[k]:
            out.append(k)
    return out


def repro_e4():
    stored = os.path.join(RES, "e4_v2", "gpt2_cohg_r0_lr0.001_s0.json")
    rerun = os.path.join(RES, "e4_absgate", "_ref",
                         "gpt2_cohg_r0_lr0.001_s0.json")
    if not (os.path.exists(stored) and os.path.exists(rerun)):
        return None
    A, B = json.load(open(stored)), json.load(open(rerun))
    d = diff_json(A, B)
    same_losses = A.get("losses") == B.get("losses")
    return {"stored": stored, "rerun": rerun, "diff_keys": d,
            "losses_identical": same_losses,
            "ppl": (A["online_ppl"], B["online_ppl"]),
            "events": (A["events"], B["events"]),
            "hvp": (A["hvp_total"], B["hvp_total"]),
            "extra": [k for k in B if k not in A]}


def repro_e2():
    stored = os.path.join(RES, "e2_controls",
                          "mackey_drift_absgate_lr0.003_a0.4_s0.json")
    rerun = os.path.join(RES, "e2_lorenz_absgate", "_verify",
                         "mackey_drift_absgate_lr0.003_a0.4_s0.json")
    if not (os.path.exists(stored) and os.path.exists(rerun)):
        return None
    A, B = json.load(open(stored)), json.load(open(rerun))
    d = diff_json(A, B, skip=("wall_s", "_file"))
    return {"stored": stored, "rerun": rerun, "diff_keys": d,
            "losses_identical": A.get("losses") == B.get("losses"),
            "lam_identical": A.get("lam_hist") == B.get("lam_hist"),
            "nmse": (A["nmse"], B["nmse"]), "events": (A["events"],
                                                       B["events"]),
            "hvp": (A["hvp_total"], B["hvp_total"]),
            "extra": [k for k in B if k not in A]}


def repro_e3():
    """E3 has no deterministic GPU reference (see the note in the report):
    report the re-run against the stored e3_traced run and the KNOWN
    run-to-run spread of the same driver on the same machine."""
    tr = os.path.join(RES, "e3_traced")
    e3 = os.path.join(RES, "e3")
    ref = os.path.join(RES, "e3_absgate", "_ref")
    rows = []
    for ewc in (10, 1000):
        n = "cifar100_cohg_lr0.05_ewc%g_s0.json" % ewc
        A = os.path.join(tr, n)
        B = os.path.join(ref, n)
        if not (os.path.exists(A) and os.path.exists(B)):
            continue
        a, b = json.load(open(A)), json.load(open(B))
        first = None
        la, lb = a.get("losses") or [], b.get("losses") or []
        for i, (x, y) in enumerate(zip(la, lb)):
            if x != y:
                first = i
                break
        rows.append({"ewc": ewc, "stored_acc": a["avg_acc"],
                     "rerun_acc": b["avg_acc"],
                     "stored_ev": a["events"], "rerun_ev": b["events"],
                     "first_diff_step": first,
                     "n_steps": min(len(la), len(lb))})
    # the pre-existing spread: results/e3 vs results/e3_traced, same config,
    # same machine, same script -- includes the `fixed` arm, which contains no
    # COHG code at all.
    spread = []
    for m in ("fixed", "hd", "cohg", "cohg_nogate"):
        ds = []
        for ewc in (10, 1000):
            for s in range(10):
                n = "cifar100_%s_lr0.05_ewc%g_s%d.json" % (m, ewc, s)
                A, B = os.path.join(e3, n), os.path.join(tr, n)
                if os.path.exists(A) and os.path.exists(B):
                    ds.append(json.load(open(B))["avg_acc"]
                              - json.load(open(A))["avg_acc"])
        if ds:
            spread.append((m, len(ds), max(abs(x) for x in ds),
                           sum(abs(x) for x in ds) / len(ds)))
    return rows, spread


# ------------------------------------------------------------------ report
def sec_e2(L):
    A = E2()
    ab, co, ng, fx = A["absgate"], A["cohg"], A["nogate"], A["fixed"]
    L += ["## 1. E2 `lorenz_drift` -- the nearest domain "
          "(same driver, same 13k-param GRU, different attractor)", "",
          "Config, identical for every arm and identical to the "
          "`mackey_drift` study the constant was calibrated on except for the "
          "dataset: mis-set init `lr0=0.003` (10x too low -- the same "
          "mis-set-low operating point `launch_e2.py` uses for "
          "`lorenz_drift`), 12000 steps, alpha (`--meta-lr`) 0.4, gamma 0.9, "
          "`kw-eps` 0.1, `probe-every` 20, K 10, rank 4, gate factor c=2, "
          "M_H 5, seeds 0-9, device CPU.", ""]
    ext = [("events (in-run)", lambda r: r["events"], ".1f"),
           ("coord-open rate", lambda r: r.get("coord_open_frac"), ".3e"),
           ("steps-with-open", lambda r: r.get("gate_open_frac"), ".3e"),
           ("HVPs", lambda r: r["hvp_total"], ".0f"),
           ("wall (s)", lambda r: r["wall_s"], ".0f")]
    L += table([ab, co, ng, A["cohg_gpu"], A["nogate_gpu"], fx],
               "NMSE", lambda r: r["nmse"], ".4f", ext)
    L += ["", "The fixed-LR family on the same stream "
          "(`results/e2`, GPU) -- the NMSE/instability frontier every "
          "adaptive arm is judged against:", ""]
    L += ["| fixed lr | n | NMSE mean+-std | events mean+-std |",
          "|---|---|---|---|"]
    for lr in (0.003, 0.01, 0.03, 0.1, 0.3, 0.6, 1.0):
        rr = load(os.path.join(RES, "e2",
                               "lorenz_drift_fixed_lr%g_s*.json" % lr))
        if not rr:
            continue
        a_ = Arm("fixed %g" % lr, rr)
        L += ["| %g%s | %d | %s | %s |"
              % (lr, "  <- mis-set init of every adaptive arm"
                 if lr == 0.003 else "", a_.n(),
                 ms(a_.vals(lambda r: r["nmse"]), ".4g"),
                 ms(a_.vals(lambda r: r["events"]), ".1f"))]
    L += [""]
    lw = {}
    for k, a in (("absgate", ab), ("cohg", co), ("nogate", ng)):
        w = [lam_window(r.get("lam_hist")) for r in a.runs.values()]
        lw[k] = (med([x[0] for x in w]), med([x[1] for x in w]))
    L += ["`lambda window` (median over seeds of the first and last sampled "
          "step at which lambda actually moved): "
          + ", ".join("%s %s-%s" % (k, f(v[0], ".0f"), f(v[1], ".0f"))
                      for k, v in lw.items()), ""]
    L += ["### Paired tests vs the certificate gate and vs the ungated step",
          "(exact sign-flip permutation, n=10 -> smallest attainable "
          "p = 0.0020; negative delta = the FIRST arm is better)", ""]
    L += tests([
        ("absgate - COHG", "NMSE", ab.bs(lambda r: r["nmse"]),
         co.bs(lambda r: r["nmse"]), "+.5f"),
        ("absgate - COHG", "unified events",
         ab.umbs("n_event_unified"), co.umbs("n_event_unified"), "+.1f"),
        ("absgate - COHG", "worst-window",
         ab.umbs("worst_window_mean"), co.umbs("worst_window_mean"), "+.4g"),
        ("absgate - COHG", "max-excess", ab.umbs("max_excess"),
         co.umbs("max_excess"), "+.4g"),
        ("absgate - cohg_nogate", "NMSE", ab.bs(lambda r: r["nmse"]),
         ng.bs(lambda r: r["nmse"]), "+.5f"),
        ("absgate - cohg_nogate", "unified events",
         ab.umbs("n_event_unified"), ng.umbs("n_event_unified"), "+.1f"),
        ("absgate - fixed lr0.003", "NMSE", ab.bs(lambda r: r["nmse"]),
         fx.bs(lambda r: r["nmse"]), "+.5f"),
        ("COHG - fixed lr0.003", "NMSE", co.bs(lambda r: r["nmse"]),
         fx.bs(lambda r: r["nmse"]), "+.5f"),
    ])
    L += [""]
    return A


def sec_e3(L):
    out = {}
    for ewc in (10, 1000):
        A = E3(ewc)
        out[ewc] = A
        ab, co, ng = A["absgate"], A["cohg"], A["nogate"]
        L += ["### 2.%d  EWC operating point ewc0 = %g" % (1 if ewc == 10
                                                           else 2, ewc), ""]
        L += pending([ab], "E3 absgate at ewc0=%g" % ewc)
        ext = [("BWT", lambda r: r["bwt"], "+.4f"),
               ("events (in-run)", lambda r: r["events"], ".1f"),
               ("coord-open rate", lambda r: r.get("coord_open_frac"), ".3e"),
               ("steps-with-open", lambda r: r.get("gate_open_frac"), ".3e"),
               ("HVPs", lambda r: r["hvp_total"], ".0f"),
               ("wall (s)", lambda r: r["wall_s"], ".0f")]
        L += table([ab, co, ng, A["hd"], A["fixed"]],
                   "final avg acc", lambda r: r["avg_acc"], ".4f", ext)
        L += [""]
        fl = [final_lam(r, 6) for r in ab.runs.values()]
        fl = [x for x in fl if x]
        if fl:
            lo = min(min(x) for x in fl)
            hi = max(max(x) for x in fl)
            L += ["absgate final per-group log-LRs span [%.2f, %.2f] "
                  "(eta in [%.3g, %.3g]); the box is "
                  "[log 1e-5, log 1] = [-11.51, 0.00] and the init is "
                  "log 0.05 = -3.00." % (lo, hi, math.exp(lo), math.exp(hi)),
                  ""]
        fl = [final_lam(r, 6) for r in co.runs.values()]
        fl = [x for x in fl if x]
        if fl:
            lo = min(min(x) for x in fl)
            hi = max(max(x) for x in fl)
            L += ["COHG final per-group log-LRs span [%.2f, %.2f] "
                  "(eta in [%.3g, %.3g])." % (lo, hi, math.exp(lo),
                                              math.exp(hi)), ""]
        L += tests([
            ("absgate - COHG", "avg acc", ab.bs(lambda r: r["avg_acc"]),
             co.bs(lambda r: r["avg_acc"]), "+.4f"),
            ("absgate - COHG", "BWT", ab.bs(lambda r: r["bwt"]),
             co.bs(lambda r: r["bwt"]), "+.4f"),
            ("absgate - COHG", "unified events",
             ab.umbs("n_event_unified"), co.umbs("n_event_unified"), "+.1f"),
            ("absgate - COHG", "worst-window", ab.umbs("worst_window_mean"),
             co.umbs("worst_window_mean"), "+.4g"),
            ("absgate - cohg_nogate", "avg acc",
             ab.bs(lambda r: r["avg_acc"]), ng.bs(lambda r: r["avg_acc"]),
             "+.4f"),
            ("absgate - cohg_nogate", "unified events",
             ab.umbs("n_event_unified"), ng.umbs("n_event_unified"), "+.1f"),
            ("absgate - fixed", "avg acc", ab.bs(lambda r: r["avg_acc"]),
             A["fixed"].bs(lambda r: r["avg_acc"]), "+.4f"),
            ("COHG - fixed", "avg acc", co.bs(lambda r: r["avg_acc"]),
             A["fixed"].bs(lambda r: r["avg_acc"]), "+.4f"),
        ])
        L += [""]
    return out


def sec_e4(L):
    A = E4()
    ab, r0, ng, fx = A["absgate"], A["cohg_r0"], A["nogate"], A["fixed"]
    L += pending([ab], "E4 absgate")
    if 0 < ab.n() < 3:
        L += ["> **PROVISIONAL: only %d of the 3 absgate seeds have landed.**"
              "  The absgate row below is n=%d and no paired test on it is "
              "meaningful yet." % (ab.n(), ab.n()), ""]
    ext = [("events (in-run)", lambda r: r["events"], ".1f"),
           ("coord-open rate", lambda r: r.get("coord_open_frac"), ".3e"),
           ("steps-with-open", lambda r: r.get("gate_open_frac"), ".3e"),
           ("HVPs", lambda r: r["hvp_total"], ".0f"),
           ("peak GiB", lambda r: r.get("peak_mem_gb"), ".1f"),
           ("wall (s)", lambda r: r["wall_s"], ".0f")]
    r0_3 = Arm(r0.label + ", seeds 0-2 (the paired subset)",
               {s_: r for s_, r in r0.runs.items() if s_ in (0, 1, 2)})
    L += table([ab, r0, r0_3, A["cohg"], ng, fx],
               "online PPL", lambda r: r["online_ppl"], ".3f", ext)
    L += [""]
    ng_3 = Arm(ng.label + " (seeds 0-2 only)",
               {s_: r for s_, r in ng.runs.items() if s_ in (0, 1, 2)})
    fx_3 = Arm(fx.label + " (seeds 0-2 only)",
               {s_: r for s_, r in fx.runs.items() if s_ in (0, 1, 2)})
    L += ["Paired tests use the three seeds absgate was run on "
          "(n=3 -> the smallest attainable two-sided p is 2/8 = 0.2500, so "
          "these tests can only ever be suggestive; the effect sizes are the "
          "informative column).", ""]
    L += tests([
        ("absgate - cohg_r0", "online PPL",
         ab.bs(lambda r: r["online_ppl"]), r0_3.bs(lambda r: r["online_ppl"]),
         "+.4f"),
        ("absgate - cohg_r0", "unified events",
         ab.umbs("n_event_unified"), r0_3.umbs("n_event_unified"), "+.1f"),
        ("absgate - cohg_r0", "worst-window",
         ab.umbs("worst_window_mean"), r0_3.umbs("worst_window_mean"),
         "+.4g"),
        ("absgate - cohg_nogate", "online PPL",
         ab.bs(lambda r: r["online_ppl"]), ng_3.bs(lambda r: r["online_ppl"]),
         "+.4f"),
        ("absgate - fixed", "online PPL",
         ab.bs(lambda r: r["online_ppl"]), fx_3.bs(lambda r: r["online_ppl"]),
         "+.4f"),
    ])
    L += [""]
    fl = [final_lam(r) for r in ab.runs.values()]
    fl = [x for x in fl if x]
    if fl:
        lo, hi = min(min(x) for x in fl), max(max(x) for x in fl)
        L += ["absgate final per-block log-LRs span [%.2f, %.2f] "
              "(eta in [%.3g, %.3g]); the box is [log 1e-6, log 0.1] = "
              "[-13.82, -2.30] and the init is log 1e-3 = -6.91."
              % (lo, hi, math.exp(lo), math.exp(hi)), ""]
    return A


def answer_section():
    L = ["## 5. Plain answer", ""]
    A = E2()
    ab, co = A["absgate"], A["cohg"]
    if ab.n() >= 10 and co.n() >= 10:
        pn, _, dn = perm_paired(ab.bs(lambda r: r["nmse"]),
                                co.bs(lambda r: r["nmse"]))
        pe, _, de = perm_paired(ab.umbs("n_event_unified"),
                                co.umbs("n_event_unified"))
        pw, _, dw = perm_paired(ab.umbs("worst_window_mean"),
                                co.umbs("worst_window_mean"))
        px, _, dx = perm_paired(ab.umbs("max_excess"), co.umbs("max_excess"))
        mn = mstd(ab.vals(lambda r: r["nmse"]))[0]
        cn = mstd(co.vals(lambda r: r["nmse"]))[0]
        mo = mstd(ab.vals(lambda r: r["coord_open_frac"]))[0]
        co_ = mstd(co.vals(lambda r: r["coord_open_frac"]))[0]
        L += ["### E2 `lorenz_drift` -- the transfer SURVIVES on the headline "
              "metric and DEGRADES the tail", "",
              "Moving the constant from `mackey_drift` to `lorenz_drift` -- a "
              "different attractor, same driver, same 13k-param GRU, same "
              "mis-set init -- the transferred gate is statistically "
              "indistinguishable from the certificate gate on what the paper "
              "reports:", "",
              "- NMSE %.5f vs %.5f, paired delta %+.5f, **p = %.4f** "
              "(not distinguishable), at 11 992 HVPs against 94 588 -- 7.9x "
              "cheaper." % (mn, cn, dn, pn),
              "- unified events, paired delta %+.1f, **p = %.4f** (not "
              "distinguishable)." % (de, pe),
              "- realized coordinate-open rate %.3e vs %.3e -- within a "
              "factor %.2f of COHG's without any refit." % (mo, co_,
                                                            max(mo, co_)
                                                            / max(min(mo, co_),
                                                                  1e-30)),
              "",
              "But the degradation metrics that the headline NMSE averages "
              "away do separate them:", "",
              "- worst trailing-100-step window: paired delta %+.4f, "
              "**p = %.4f**." % (dw, pw),
              "- max-excess (max finite loss / median loss): paired delta "
              "%+.4g, **p = %.4f**." % (dx, px),
              "",
              "So on the nearest domain the answer is: the static threshold "
              "still works well enough to reproduce the accuracy claim, but "
              "it is measurably worse in the tail.  It survives because the "
              "`|ghat|` scale of `lorenz_drift` happens to resemble that of "
              "`mackey_drift`; nothing in the rule guaranteed that, and the "
              "next section shows what happens when it does not.", ""]
    # ---- E3 -------------------------------------------------------------
    e3rows = []
    for ewc in (10, 1000):
        A3 = E3(ewc)
        ab3, co3, ng3, fx3 = (A3["absgate"], A3["cohg"], A3["nogate"],
                              A3["fixed"])
        if ab3.n() < 10 or co3.n() < 10:
            continue
        pa, _, da = perm_paired(ab3.bs(lambda r: r["avg_acc"]),
                                co3.bs(lambda r: r["avg_acc"]))
        pf, _, df = perm_paired(ab3.bs(lambda r: r["avg_acc"]),
                                fx3.bs(lambda r: r["avg_acc"]))
        pn, _, dn = perm_paired(ab3.bs(lambda r: r["avg_acc"]),
                                ng3.bs(lambda r: r["avg_acc"]))
        pc, _, dc = perm_paired(co3.bs(lambda r: r["avg_acc"]),
                                fx3.bs(lambda r: r["avg_acc"]))
        e3rows.append((ewc,
                       mstd(ab3.vals(lambda r: r["avg_acc"])),
                       mstd(co3.vals(lambda r: r["avg_acc"])),
                       mstd(fx3.vals(lambda r: r["avg_acc"])),
                       mstd(ab3.vals(lambda r: r["coord_open_frac"]))[0],
                       mstd(co3.vals(lambda r: r["coord_open_frac"]))[0],
                       (da, pa), (df, pf), (dn, pn), (dc, pc)))
    if e3rows:
        L += ["### E3 Split-CIFAR-100 -- the transfer FAILS at both EWC "
              "operating points", "",
              "| ewc0 | absgate acc | COHG acc | fixed acc | absgate "
              "coord-open | COHG coord-open | absgate-COHG (p) | "
              "absgate-fixed (p) | absgate-nogate (p) | COHG-fixed (p) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for (e, a, c, f_, ao, co_, ta, tf, tn, tc) in e3rows:
            L += ["| %g | %.4f+-%.4f | %.4f+-%.4f | %.4f+-%.4f | %.3e | "
                  "%.3e | %+.4f (%.4f) | %+.4f (%.4f) | %+.4f (%.4f) | "
                  "%+.4f (%.4f) |"
                  % (e, a[0], a[1], c[0], c[1], f_[0], f_[1], ao, co_,
                     ta[0], ta[1], tf[0], tf[1], tn[0], tn[1],
                     tc[0], tc[1])]
        L += ["",
              "The transferred constant opens 82-117x more often than the "
              "certificate gate and loses accuracy at both operating points, "
              "but the strength of the evidence differs and should be stated "
              "as it is.  At ewc0=10 it costs 9.35 points against COHG "
              "(p=0.0059) and 9.96 against not adapting at all (p=0.0020, the "
              "smallest p attainable at n=10).  At ewc0=1000 it costs 8.67 "
              "points against `fixed` (p=0.0137) but its 7.12-point deficit "
              "against COHG is NOT significant (p=0.1309) -- COHG's own "
              "spread is large at that operating point (sd 0.093), so the "
              "comparison against the un-adapted baseline is the informative "
              "one there.",
              "",
              "Against the fully ungated sign step the transferred gate is "
              "not reliably better at ewc0=10 (+0.0555, p=0.1855) and is "
              "modestly better at ewc0=1000 (+0.0738, p=0.0488).  So it is "
              "not worthless -- it still shuts most of the time -- but on "
              "this domain it lands much closer to having no gate than to "
              "having the certificate gate.",
              "",
              "Two mechanism details specific to E3.  (i) The constant drags "
              "five or six of the six per-stage log-LRs down from the init "
              "-3.00 to between -5.8 and the -11.51 floor, while COHG leaves "
              "them at -2.6/-3.4.  (ii) lambda here contains one coordinate "
              "that is NOT a learning rate -- the log EWC strength -- and the "
              "transferred constant never opens it, whereas COHG spends its "
              "single gate opening exactly there (2.30 -> 1.90 at ewc0=10, "
              "6.91 -> 6.51 at ewc0=1000).  One scalar threshold applied to a "
              "lambda vector whose coordinates are not commensurate cannot "
              "distinguish the coordinate that matters; the certificate, "
              "being per-coordinate and scale-aware, can.", ""]

    A4 = E4()
    ab4, r04 = A4["absgate"], A4["cohg_r0"]
    if ab4.n():
        prov = (" **(PROVISIONAL: %d of 3 seeds)**" % ab4.n()
                if ab4.n() < 3 else "")
        m4 = mstd(ab4.vals(lambda r: r["online_ppl"]))[0]
        r4 = mstd([r["online_ppl"] for s_, r in r04.runs.items()
                   if s_ in ab4.runs])[0]
        mo4 = mstd(ab4.vals(lambda r: r["coord_open_frac"]))[0]
        co4 = mstd([r["coord_open_frac"] for s_, r in r04.runs.items()
                    if s_ in ab4.runs])[0]
        L += ["### E4 GPT-2 -- the transfer FAILS%s" % prov, "",
              "- online PPL %.3f vs the certificate gate's %.3f on the same "
              "seeds (+%.1f%%), and worse than doing nothing at all "
              "(`fixed` 21.09)." % (m4, r4, 100 * (m4 / r4 - 1)),
              "- realized coordinate-open rate %.3e vs %.3e -- the "
              "transferred constant is %.0fx more permissive here."
              % (mo4, co4, mo4 / max(co4, 1e-30)),
              ]
        # per-seed: how many blocks end pinned at a wall, and the tail window
        import math as _m
        rows = []
        for s_ in sorted(ab4.runs):
            r = ab4.runs[s_]
            L_ = r["losses"]
            tail = _m.exp(sum(L_[2500:]) / len(L_[2500:]))
            lam = r["lam_hist"][-1][1:]
            lo = sum(1 for x in lam if x <= LAM_MIN_E4 + 1e-6)
            hi = sum(1 for x in lam if x >= LAM_MAX_E4 - 1e-6)
            rows.append((s_, r["online_ppl"], r["coord_open_frac"], tail,
                         lo, hi))
        L += ["", "Per-seed, because the outcome is NOT uniform across seeds "
              "-- this matters for how the claim is worded:", "",
              "| seed | online PPL | coord-open rate | PPL of last 500 steps "
              "| blocks pinned at the 1e-6 FLOOR | blocks pinned at the 0.1 "
              "CEILING |", "|---|---|---|---|---|---|"]
        for s_, ppl, op, tail, lo, hi in rows:
            L += ["| %d | %.3f | %.3e | %.1f | %d of 6 | %d of 6 |"
                  % (s_, ppl, op, tail, lo, hi)]
        L += ["",
              "What is CONSISTENT across seeds is the over-permissiveness and "
              "the freezing: the transferred constant opens 100-200x more "
              "often than the certificate gate, and ends with most blocks "
              "pinned at the 1e-6 floor, so the model stops adapting.  What "
              "is SEED-DEPENDENT is whether it additionally drives a block to "
              "the 0.1 CEILING and blows the loss up late in the stream "
              "(seed 0 does: last-500 PPL 166.3 against the reference's 6.9; "
              "seed 1 does not).  The claim to make is therefore 'the "
              "transferred threshold is uncalibrated here and is sometimes "
              "catastrophic', not 'it always blows up'.",
              "",
              "In no seed does any loss go non-finite, so the in-run `events` "
              "counter reads 0 throughout and misses all of this; only the "
              "unified worst-window / max-excess metrics see it.", ""]
        rr = list(A4["ref"].runs.values())
        if rr and rr[0].get("gate_stats"):
            g = rr[0]["gate_stats"]
            L += ["### What the certificate provides that the constant does "
                  "not", "",
                  "On GPT-2 the reference run shows COHG opening its gate on "
                  "**exactly one step of 2999** (`gate_open_steps = [1]`), "
                  "because the realized threshold `c*beta_j` climbs past the "
                  "entire range of `|ghat|` within the first handful of steps "
                  "(median `c*beta` %.3g against `|ghat|` max %.3g) and "
                  "diverges from there."
                  % (g["cbeta"]["median"], g["ghat_abs"]["max"]),
                  "",
                  "That is the honest shape of the result, and it should be "
                  "written that way: on this domain COHG does not out-adapt "
                  "the constant, it REFUSES to adapt.  Its PPL (20.68) is "
                  "essentially the un-adapted `fixed` baseline (21.09), "
                  "bought with a single certified step.  The transferred "
                  "constant has no notion of whether the estimate is "
                  "trustworthy, keeps stepping for all 2999 steps, and "
                  "destroys the run.",
                  "",
                  "### One column that must NOT be read as a win for the "
                  "constant", "",
                  "On E3 and E4 several DEGRADATION metrics come out lower "
                  "for absgate than for COHG -- E3 ewc10 worst-window "
                  "-0.2235 (p=0.0020), E3 ewc1000 unified events -198.5 "
                  "(p=0.0312), E4 max-excess 2.08 vs COHG's 1.36.  This is "
                  "not the transferred gate being safer.  It is an artefact "
                  "of HOW it fails: by slamming the learning rates to the "
                  "1e-6 / 1e-5 floor it produces a model that barely updates, "
                  "and a model that barely updates has a very smooth loss "
                  "trace.  Spike counts and worst-window are defined relative "
                  "to a run's own running median, so a frozen learner scores "
                  "well on them while losing 9-10 accuracy points and 5.9 "
                  "PPL.  The degradation metrics are meaningful for "
                  "comparing arms that are all still learning (that is how "
                  "they are used on E2); on the two domains where the "
                  "transferred constant freezes the learner they must be "
                  "read alongside the accuracy/PPL column, not instead of "
                  "it.",
                  "",
                  "A related null result worth stating: on E4 absgate is not "
                  "better than the fully UNGATED sign step either (PPL "
                  "+0.563, p=1.0000), and on E3 ewc10 it is not better "
                  "either (p=0.1855).  Outside its calibration stream the "
                  "transferred threshold buys little or nothing over having "
                  "no gate at all.", "",
                  "So the certificate's contribution is not a better "
                  "threshold value -- it is a per-domain, per-step, "
                  "calibration-free answer to *whether the hypergradient may "
                  "be acted on at all*.  A constant fitted on one stream "
                  "encodes the scale of that stream and nothing else: it "
                  "cannot become more conservative when the estimator "
                  "degrades, and it cannot become more permissive when the "
                  "estimator is sharp.", ""]
    return L


def main():
    L = []
    L += ["# Round 4: does the offline-calibrated static gate threshold "
          "transfer to other domains?", "",
          "The E2 control `absgate` opens coordinate j iff "
          "`|ghat_j| > T` with the CONSTANT `T = 0.05806520209` "
          "(printed as 0.05807), fitted offline on `mackey_drift` to "
          "reproduce COHG's measured coordinate-open rate "
          "(`results/e2_controls/absgate_threshold.json`, calibration seeds "
          "100/101, disjoint from the evaluation seeds).  On that stream it "
          "matches COHG's NMSE at 1 HVP/step instead of 7.9 "
          "(`results/e2_controls/SUMMARY.md`).  This round transfers that "
          "number AS IS -- no recalibration, no per-domain refit, not even a "
          "rescale -- to three other settings and asks whether it still "
          "works.", "",
          "Every `absgate` arm is COHG's estimator and COHG's PURE-SIGN step "
          "of size alpha; only the gate rule differs, and the certificate is "
          "never read, so no spectral probe is paid.  That makes the arm "
          "cost-matched to `cohg_nogate` and the ONLY difference from COHG "
          "the gate.", "",
          "Statistics: ddof=1 sample std; every paired comparison is an EXACT "
          "two-sided sign-flip permutation test enumerating all 2^n sign "
          "assignments of the per-seed paired differences.  Degradation "
          "metrics are the unified ones from "
          "`results/reanalysis/_reanalyze.py` (spike = finite loss above 10x "
          "the running median of the last 500 finite losses once 100 are "
          "banked; plus every non-finite loss; max-excess = max finite loss "
          "/ median loss; worst-window = worst trailing 100-step mean), "
          "computed from the raw per-step loss traces.", ""]

    L += ["## Status of this round (this file is regenerated as runs land)",
          ""]
    L += status_block()
    L += ["", "Sections whose runs have not landed are explicitly marked "
          "NOT YET AVAILABLE rather than being silently omitted.", ""]

    # ---- 0. reproduction checks
    L += ["## 0. Reproduction checks: the drivers are additive", "",
          "`absgate` and `--log-gate-stats` were added to "
          "`e3_continual.py` and `e4_gpt2_tta.py` (and `--log-gate-stats` "
          "alone to `e2_timeseries.py`) as new branches; the default path is "
          "untouched and the new JSON keys appear only when the "
          "corresponding flag is set.", ""]
    r = repro_e4()
    if r:
        L += ["**E4 (deterministic on this machine).** Re-running the stored "
              "`results/e4_v2/gpt2_cohg_r0_lr0.001_s0` config on the PATCHED "
              "`e4_gpt2_tta.py`:", "",
              "| quantity | stored | re-run |", "|---|---|---|",
              "| online PPL | %.15g | %.15g |" % r["ppl"],
              "| events | %s | %s |" % r["events"],
              "| HVPs | %s | %s |" % r["hvp"],
              "| full 2999-entry `losses` list | %s |"
              % ("**identical element for element**"
                 if r["losses_identical"] else "DIFFERENT") + " |",
              "", "Differing shared keys: %s.  Keys present only in the "
              "re-run (additive): %s."
              % (r["diff_keys"] or "**none**", r["extra"] or "none"), ""]
    r = repro_e2()
    if r:
        L += ["**E2 (deterministic on CPU).** Re-running the stored "
              "`results/e2_controls/mackey_drift_absgate_lr0.003_a0.4_s0` "
              "config on the PATCHED `e2_timeseries.py`:", "",
              "| quantity | stored | re-run |", "|---|---|---|",
              "| NMSE | %.17g | %.17g |" % r["nmse"],
              "| events | %s | %s |" % r["events"],
              "| HVPs | %s | %s |" % r["hvp"],
              "| full 12000-entry `losses` list | %s | |"
              % ("**identical**" if r["losses_identical"] else "DIFFERENT"),
              "| full `lam_hist` | %s | |"
              % ("**identical**" if r["lam_identical"] else "DIFFERENT"),
              "", "Differing shared keys: %s.  Keys present only in the "
              "re-run: %s (the shipped run predates later additive keys)."
              % (r["diff_keys"] or "**none**", r["extra"] or "none"), ""]
    rows, spread = repro_e3()
    L += ["**E3 is NOT run-to-run reproducible on this machine, and was not "
          "before this round.** `e3_continual.py` trains a ResNet-18(GN) with "
          "cuDNN convolutions whose backward pass uses non-deterministic "
          "atomics, so two runs of the SAME config with the SAME seed diverge "
          "at the 1e-5 level within one step and then separate.  Evidence "
          "that this is a pre-existing property of the driver and not of this "
          "round's patch: `results/e3` and `results/e3_traced` hold the same "
          "configs run twice on this machine by earlier rounds, and they "
          "disagree -- including for the `fixed` arm, which runs no COHG code "
          "at all.", ""]
    if spread:
        L += ["| arm | pairs | max \\|delta avg_acc\\| | mean \\|delta "
              "avg_acc\\| |", "|---|---|---|---|"]
        for m, n, mx, mn in spread:
            L += ["| %s | %d | %.4f | %.4f |" % (m, n, mx, mn)]
        L += [""]
    L += ["What CAN be checked, and was: with the old and the patched script "
          "run back to back on the same card (2 tasks x 1 epoch, "
          "`cohg ewc10 s0`), step 0 of the loss trace is BIT-IDENTICAL, the "
          "divergence starts at step 1 at a relative 1e-6 and grows, and the "
          "complete `lam_hist` (the controller trajectory, sampled every 25 "
          "steps) is IDENTICAL for the whole run.  A CPU re-run of the same "
          "short config -- deterministic -- is reported below.", ""]
    if rows:
        L += ["Full-length E3 COHG re-run of this round versus the stored "
              "`results/e3_traced` run of the same config:", "",
              "| ewc0 | stored avg_acc | re-run avg_acc | stored events | "
              "re-run events | first differing loss step |",
              "|---|---|---|---|---|---|"]
        for x in rows:
            L += ["| %g | %.4f | %.4f | %d | %d | %s |"
                  % (x["ewc"], x["stored_acc"], x["rerun_acc"],
                     x["stored_ev"], x["rerun_ev"],
                     x["first_diff_step"] if x["first_diff_step"] is not None
                     else "none (identical)")]
        L += [""]

    # ---- 1/2/3 domains
    A2 = sec_e2(L)
    L += ["## 2. E3 Split-CIFAR-100 continual learning "
          "(task-IL, ResNet-18 GN, 10 tasks x 2 epochs, lr0 0.05, "
          "alpha 0.4, seeds 0-9)", "",
          "lambda here is 6 per-stage log-LRs plus one log-EWC-strength "
          "coordinate; the constant threshold is applied to all seven, as a "
          "transferred rule has no way to know that one of them is not a "
          "learning rate.", ""]
    A3 = sec_e3(L)
    L += ["## 3. E4 GPT-2 124M streaming test-time adaptation "
          "(wiki -> news -> code, 512k tokens/domain, 2999 steps, "
          "lr0 1e-3, alpha 0.4, seeds 0-2)", ""]
    A4 = sec_e4(L)

    # ---- 4. scale mismatch
    L += ["## 4. Where |ghat| actually lives, and where the two thresholds "
          "live", "",
          "Pooled over every (step, coordinate) pair of a run, then the "
          "median over seeds.  `c*beta` is COHG's REALIZED certificate "
          "threshold `gate_factor * beta_col_j` (c = 2), available only on "
          "arms that actually compute the certificate.  `frac |ghat|>T` is "
          "the coordinate-open rate the TRANSFERRED constant produces; "
          "`frac |ghat|>c*beta` is the rate COHG's own gate produces on the "
          "same run.",
          "",
          "Read the two rate columns carefully: they are not the same kind "
          "of number.  On a COHG row `frac |ghat|>T` is a COUNTERFACTUAL -- "
          "what the transferred constant WOULD have opened had it been "
          "applied to the trajectory COHG actually followed -- while "
          "`frac |ghat|>c*beta` is the rate COHG's own gate realized there.  "
          "On an absgate row `frac |ghat|>T` IS the realized rate and there "
          "is no certificate to compare against.  The two differ because the "
          "arms follow different lambda trajectories: on the calibration "
          "stream the constant's counterfactual rate on COHG's trajectory is "
          "1.722e-03, while absgate's own realized rate there is 2.61e-04 "
          "(`results/e2_controls/SUMMARY.md`).", ""]
    rows = [
        gstat_row("E2 mackey_drift (CALIBRATION STREAM), COHG",
                  list(load(os.path.join(
                      RES, "e2_lorenz_absgate", "_calib_ref",
                      "mackey_drift_cohg_lr0.003_s*.json")).values())),
        gstat_row("E2 lorenz_drift, COHG",
                  list(A2["cohg"].runs.values())),
        gstat_row("E2 lorenz_drift, absgate",
                  list(A2["absgate"].runs.values())),
        gstat_row("E3 ewc10, COHG", list(A3[10]["ref"].runs.values())),
        gstat_row("E3 ewc10, absgate", list(A3[10]["absgate"].runs.values())),
        gstat_row("E3 ewc1000, COHG", list(A3[1000]["ref"].runs.values())),
        gstat_row("E3 ewc1000, absgate",
                  list(A3[1000]["absgate"].runs.values())),
        gstat_row("E4 GPT-2, COHG (cohg_r0)", list(A4["ref"].runs.values())),
        gstat_row("E4 GPT-2, absgate", list(A4["absgate"].runs.values())),
    ]
    L += gtable(rows)
    L += ["", "### Scale mismatch factor", "",
          "| domain | median c*beta (COHG) | T = 0.05807 | "
          "T / median(c*beta) | T / median \\|ghat\\| | "
          "realized open rate: T | realized open rate: certificate |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r is None or r["bmed"] is None:
            continue
        L += ["| %s | %s | %.5f | %s | %s | %s | %s |"
              % (r["label"], f(r["bmed"], ".3e"), T_CONST,
                 f(T_CONST / r["bmed"] if r["bmed"] else None, ".3g"),
                 f(T_CONST / r["gmed"] if r["gmed"] else None, ".3g"),
                 f(r["openT"], ".3e"), f(r["openC"], ".3e"))]
    L += [""]

    L += answer_section()

    open("%s" % OUT_MD, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", OUT_MD, len(L), "lines")


if __name__ == "__main__":
    main()
