import os, json, math, glob, re, itertools
from collections import deque, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NH = os.path.join(ROOT, "results", "e3_noholdout")
HO = os.path.join(ROOT, "results", "e3")
OUT = os.path.join(ROOT, "results", "reanalysis", "e3_noholdout.md")

# ---- unified spike rule, identical to _reanalyze.py
def unified_metrics(losses):
    n = len(losses)
    fin = [x for x in losses if math.isfinite(x)]
    n_nonfinite = n - len(fin)
    win = deque(maxlen=500); n_spike = 0; first_spike = None
    for t, x in enumerate(losses):
        if not math.isfinite(x): continue
        if len(win) >= 100:
            w = sorted(win); med = w[len(w)//2]
            if x > 10.0*med:
                n_spike += 1
                if first_spike is None: first_spike = t
        win.append(x)
    if fin:
        s = sorted(fin); med_all = s[len(s)//2]; max_fin = max(fin)
        max_excess = max_fin/med_all if med_all > 0 else float("inf")
    else:
        med_all = float("nan"); max_fin = float("nan"); max_excess = float("inf")
    W = 100; worst_win = float("-inf"); worst_end = -1
    if n >= W:
        for e in range(W, n+1):
            seg = losses[e-W:e]
            sf = [x for x in seg if math.isfinite(x)]
            m = (sum(sf)/len(sf)) if sf else float("inf")
            if m > worst_win: worst_win = m; worst_end = e-1
    else:
        worst_win = (sum(fin)/len(fin)) if fin else float("inf"); worst_end = n-1
    return dict(n_steps=n, n_finite=len(fin), n_nonfinite=n_nonfinite,
                n_spike=n_spike, n_event_unified=n_spike+n_nonfinite,
                first_spike_t=first_spike, median_loss=med_all,
                max_finite_loss=max_fin, max_excess=max_excess,
                worst_window_mean=worst_win, worst_window_end_t=worst_end,
                mean_loss_finite=(sum(fin)/len(fin)) if fin else float("nan"))

def mstd(v):
    v = [x for x in v if x is not None and not (isinstance(x,float) and math.isnan(x))]
    if not v: return float("nan"), float("nan"), 0
    n=len(v); m=sum(v)/n
    if n<2: return m, float("nan"), n
    sd = math.sqrt(sum((x-m)**2 for x in v)/(n-1))
    return m, sd, n

def fmtp(m,s,prec=3):
    if not math.isfinite(m): return "inf"
    if not math.isfinite(s): return "%.*f" % (prec, m)
    return "%.*f +- %.*f" % (prec, m, prec, s)

def signflip(d):
    n = len(d); obs = sum(d)/n; a = abs(obs); cnt = 0
    for signs in itertools.product((1,-1), repeat=n):
        m = sum(s*x for s,x in zip(signs,d))/n
        if abs(m) >= a - 1e-15: cnt += 1
    return obs, cnt/(2**n), 2**n

def load(d, with_losses):
    rows = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        b = os.path.basename(p)[:-5]
        m = re.match(r"^cifar100_(?P<meth>.+)_lr(?P<lr>[0-9.]+)_ewc(?P<ewc>[0-9.]+)_s(?P<seed>\d+)$", b)
        if m is None or m.group("lr") != "0.05" or m.group("ewc") != "10":
            continue
        j = json.load(open(p))
        r = dict(method=m.group("meth"), seed=int(m.group("seed")), avg_acc=j["avg_acc"],
                 bwt=j["bwt"], events=j["events"], gate_open_frac=j.get("gate_open_frac"),
                 coord_open_frac=j.get("coord_open_frac"),
                 meta_lr=j.get("meta_lr"), no_holdout=j.get("no_holdout"),
                 lam_hist=j.get("lam_hist"), acc_matrix=j["acc_matrix"],
                 collapse=int(j["avg_acc"]<0.15), path=p)
        if with_losses:
            r.update(unified_metrics(j["losses"]))
            r["losses_n"] = len(j["losses"])
        rows[m.group("meth")][int(m.group("seed"))] = r
    return rows

nh = load(NH, True)
ho = load(HO, False)

ARMS = ["cohg","cohg_nogate","hd","fixed"]
NAME = {"cohg":"COHG (gate on)","cohg_nogate":"COHG w/o gate","hd":"HD","fixed":"Fixed"}

L=[]; A=L.append
A("# E3 no-retained-holdout condition (Split-CIFAR-100)\n")
A("Generated from `results/e3_noholdout/*.json` (40 runs: 4 arms x 10 seeds, flags")
A("`--no-holdout --log-losses`). The reference holdout condition is `results/e3/*.json`,")
A("same lr0=0.05, EWC0=10, seeds 0-9.\n")
A("In the no-holdout condition the hypergradient meta-objective is evaluated only on the")
A("incoming batch's prequential loss; no stored examples from past tasks are used at any")
A("point of the meta-update. Everything else (backbone, task order, lr0, EWC strength,")
A("meta-lr, gate, recovery heuristic) is unchanged. The 128-example holdout is still carved")
A("out of each task's training stream, so the data the learner trains on is bit-identical to")
A("the default condition and both conditions log the same 3040 steps; under `--no-holdout`")
A("those examples are simply never looked at again. EWC keeps its anchor and Fisher, which")
A("belong to the learner rather than to the meta-objective.\n")
A("The non-adaptive `fixed` arm therefore acts as a null control: the flag cannot touch it,")
A("and any holdout / no-holdout difference it shows is GPU nondeterminism. It shows")
A("-0.0027 accuracy at p = 0.5332, which sets the noise floor for reading the other rows.\n")
A("Spike rule, max-excess and worst-window follow `results/reanalysis/_reanalyze.py`:")
A("a window of the last 500 finite losses; once it holds >=100 entries a step is a spike if")
A("`loss_t > 10 x median(window)`; every non-finite loss is also an event;")
A("`unified events = spikes + non-finite`. `max-excess = max finite loss / median finite loss`;")
A("`worst-window = max over t of mean(loss[t-99..t])`. Aggregates are mean +- std, ddof=1,")
A("over the 10 seeds.\n")

A("\n## 0. Sanity checks\n")
bad=[]
exp_meta = {"cohg":0.4,"cohg_nogate":0.4,"hd":0.02,"fixed":None}
A("| arm | n runs | `no_holdout` true | meta_lr (all runs) | expected | steps logged |")
A("|---|---|---|---|---|---|")
for a in ARMS:
    rs=[nh[a][s] for s in sorted(nh[a])]
    nhf=all(r["no_holdout"] is True for r in rs)
    mls=sorted(set(r["meta_lr"] for r in rs))
    ns=sorted(set(r["losses_n"] for r in rs))
    exp = exp_meta[a]
    ok = (exp is None) or (len(mls)==1 and abs(mls[0]-exp)<1e-12)
    if not (nhf and ok): bad.append(a)
    A("| %s | %d | %s | %s | %s | %s |" % (NAME[a], len(rs), "yes" if nhf else "NO",
      ", ".join(str(x) for x in mls), exp if exp is not None else "n/a",
      ", ".join(str(x) for x in ns)))
A("")
A("All 40 runs carry `no_holdout=true`." if not bad else "**FAILED:** " + str(bad))
A("Seeds present: %s for every arm; the holdout arms use the same seed set, so every" % sorted(nh["cohg"]))
A("holdout / no-holdout comparison below is paired by seed.\n")

A("\n## 1. Per-arm summary, no-holdout condition (n = 10)\n")
A("| arm | avg_acc | BWT | non-finite events | collapse rate (avg_acc<0.15) | unified spikes | unified events | max-excess | worst-window |")
A("|---|---|---|---|---|---|---|---|---|")
S={}
for a in ARMS:
    rs=[nh[a][s] for s in sorted(nh[a])]
    S[a]=rs
    A("| %s | %s | %s | %s | %.2f (%d/10) | %s | %s | %s | %s |" % (
      NAME[a],
      fmtp(*mstd([r["avg_acc"] for r in rs])[:2],prec=4),
      fmtp(*mstd([r["bwt"] for r in rs])[:2],prec=4),
      fmtp(*mstd([r["events"] for r in rs])[:2],prec=2),
      sum(r["collapse"] for r in rs)/len(rs), sum(r["collapse"] for r in rs),
      fmtp(*mstd([r["n_spike"] for r in rs])[:2],prec=2),
      fmtp(*mstd([r["n_event_unified"] for r in rs])[:2],prec=2),
      fmtp(*mstd([r["max_excess"] for r in rs])[:2],prec=2),
      fmtp(*mstd([r["worst_window_mean"] for r in rs])[:2],prec=3)))
A("")
g=mstd([r["gate_open_frac"] for r in S["cohg"]])
gn=mstd([r["gate_open_frac"] for r in S["cohg_nogate"]])
c=mstd([r["coord_open_frac"] for r in S["cohg"]])
A("COHG gate_open_frac = %s (min %.6f, max %.6f); coord_open_frac = %s." % (
  fmtp(*g[:2],prec=6), min(r["gate_open_frac"] for r in S["cohg"]),
  max(r["gate_open_frac"] for r in S["cohg"]), fmtp(*c[:2],prec=6)))
A("COHG w/o gate gate_open_frac = %s (gate disabled; open at every step by construction).\n" % fmtp(*gn[:2],prec=6))

A("\n### Per-seed detail (no-holdout)\n")
A("| arm | seed | avg_acc | BWT | non-finite | spikes | max-excess | worst-window | gate_open_frac |")
A("|---|---|---|---|---|---|---|---|---|")
for a in ARMS:
    for r in S[a]:
        gof = "n/a" if r["gate_open_frac"] is None else "%.6f" % r["gate_open_frac"]
        A("| %s | %d | %.4f | %.4f | %d | %d | %.2f | %.3f | %s |" % (
          NAME[a], r["seed"], r["avg_acc"], r["bwt"], r["n_nonfinite"], r["n_spike"],
          r["max_excess"], r["worst_window_mean"], gof))

A("\n## 2. Paired comparisons within the no-holdout condition\n")
A("Exact two-sided sign-flip test over all 2^10 = 1024 sign assignments of the 10 paired")
A("per-seed differences.\n")
A("| comparison | metric | mean paired difference | p (exact) | direction |")
A("|---|---|---|---|---|")
comps=[("cohg","cohg_nogate","avg_acc"),("cohg","fixed","avg_acc"),
       ("cohg","hd","avg_acc"),("cohg","hd","bwt")]
PAIR={}
for x,y,k in comps:
    d=[nh[x][s][k]-nh[y][s][k] for s in sorted(nh[x])]
    obs,p,N=signflip(d)
    PAIR[(x,y,k)]=(obs,p)
    A("| %s vs %s | %s | %+.4f | %.4f | %s |" % (NAME[x],NAME[y],k,obs,p,
      "COHG higher" if obs>0 else "COHG lower"))
A("")

A("\n## 3. Holdout vs no-holdout, paired by seed\n")
A("| arm | metric | holdout | no-holdout | mean paired delta (no-holdout - holdout) | p (exact sign-flip) |")
A("|---|---|---|---|---|---|")
DELTA={}
for a in ARMS:
    for k,prec in (("avg_acc",4),("bwt",4),("events",2)):
        hv=[ho[a][s][k] for s in sorted(nh[a])]
        nv=[nh[a][s][k] for s in sorted(nh[a])]
        d=[b-c2 for b,c2 in zip(nv,hv)]
        if all(abs(x)<1e-15 for x in d):
            obs,p=0.0,1.0
        else:
            obs,p,_=signflip(d)
        DELTA[(a,k)]=(mstd(hv),mstd(nv),obs,p)
        A("| %s | %s | %s | %s | %+.4f | %.4f |" % (NAME[a],k,
          fmtp(*mstd(hv)[:2],prec=prec), fmtp(*mstd(nv)[:2],prec=prec), obs, p))
A("")
A("The `events` row for COHG w/o gate is driven by two of the ten holdout seeds (s2 with 153")
A("non-finite steps, s6 with 163); the other eight holdout seeds and all ten no-holdout seeds")
A("record zero. With only two nonzero paired differences the sign-flip p for that row is")
A("uninformative (0.5000 is its floor here); the seed-level count is the meaningful statistic:")
A("2/10 holdout seeds trigger recovery versus 0/10 no-holdout seeds.")
A("")
A("Collapse rates: " + "; ".join(
  "%s %d/10 holdout vs %d/10 no-holdout" % (NAME[a],
    sum(int(ho[a][s]["avg_acc"]<0.15) for s in sorted(nh[a])),
    sum(r["collapse"] for r in S[a])) for a in ARMS) + ".")
A("Holdout gate_open_frac (COHG) = %s; no-holdout = %s.\n" % (
  fmtp(*mstd([ho["cohg"][s]["gate_open_frac"] for s in sorted(nh["cohg"])])[:2],prec=6),
  fmtp(*g[:2],prec=6)))

A("\n### The same paired comparisons under the holdout condition, for reference\n")
A("| comparison | metric | holdout mean diff | holdout p | no-holdout mean diff | no-holdout p |")
A("|---|---|---|---|---|---|")
for x,y,k in comps:
    dh=[ho[x][s][k]-ho[y][s][k] for s in sorted(nh[x])]
    oh,ph,_=signflip(dh)
    on,pn=PAIR[(x,y,k)]
    A("| %s vs %s | %s | %+.4f | %.4f | %+.4f | %.4f |" % (NAME[x],NAME[y],k,oh,ph,on,pn))
A("")

A("\n## 4. What survives without retained past-task data\n")
A("**Survives.**\n")
A("1. COHG's accuracy is unchanged. 0.3804 +- 0.0198 without a holdout versus")
A("   0.3803 +- 0.0179 with one; the paired delta is +0.0001 (p = 0.9707). Its BWT is also")
A("   unchanged, -0.0730 +- 0.0246 versus -0.0711 +- 0.0165 (delta -0.0019, p = 0.6328).")
A("   The 128-example retained holdout buys COHG nothing on this benchmark.")
A("2. The gate-on / gate-off separation survives. COHG beats its ungated ablation by")
A("   +0.0877 accuracy (p = 0.0020) without a holdout, the same p as the holdout condition's")
A("   +0.1635. The gate is still the component that matters.")
A("3. COHG's forgetting advantage over HD survives and in fact sharpens: +0.0513 BWT")
A("   (p = 0.0020) without a holdout versus +0.0242 (p = 0.0898) with one, at matched or")
A("   better accuracy (+0.0186, p = 0.0410, versus +0.0142, p = 0.0918).")
A("4. COHG's near-zero gate duty cycle is identical in both conditions, 0.000658 of steps.")
A("   The meta-objective change does not make the certificate fire more often.")
A("")
A("**Does not survive, or was never there.**\n")
A("1. COHG's accuracy edge over a well-tuned fixed schedule is absent in both conditions")
A("   (+0.0044, p = 0.4375 without a holdout; +0.0016, p = 0.7695 with one). Removing the")
A("   holdout neither creates nor destroys an edge that the paper never claimed.")
A("2. The *magnitude* of the gate ablation gap shrinks by roughly half, from +0.1635 to")
A("   +0.0877, and its dramatic failure mode disappears. Under the holdout objective the")
A("   ungated ablation triggered non-finite recovery on 2/10 seeds (31.60 +- 66.66 events)")
A("   and collapsed below 0.15 accuracy on 1/10; under the prequential-only objective it")
A("   triggers on 0/10 seeds, collapses on 0/10, and gains +0.0759 accuracy (p = 0.0059).")
A("   So the strongest version of the instability claim, that an ungated hypergradient can")
A("   diverge outright, is a property of the holdout meta-objective specifically, not of")
A("   ungated hypergradients in general. The weaker claim, that gating improves accuracy")
A("   and reduces forgetting, holds in both.")
A("3. No arm produces a single spike or non-finite loss in the no-holdout condition")
A("   (0.00 +- 0.00 unified events across all 40 runs; max-excess 5.2 to 7.2, worst-window")
A("   2.49 to 2.91). The prequential-only meta-objective is a uniformly milder regime, so")
A("   the stability metrics cannot separate the arms here at all. Removing the holdout")
A("   costs COHG none of its stability, but it also removes the setting in which COHG's")
A("   stability advantage over the ungated ablation was visible.")
A("4. HD's behaviour is essentially unchanged. Accuracy -0.0042 (p = 0.1914), BWT -0.0023")
A("   (p = 0.5215). HD's larger forgetting relative to COHG is not a holdout artefact.")
A("")

open(OUT,"w",encoding="utf-8").write("\n".join(L)+"\n")
print("\n".join(L))
