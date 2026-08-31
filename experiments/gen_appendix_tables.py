"""Regenerate appendix result tables (B.7) from raw artifacts, ddof=1.

Prints LaTeX table bodies for:
  Table 10 (tab:e2stat)  stationary E2
  Table 11 (tab:e2drift) drifting E2
  Table 12 (tab:e3full)  E3 all configurations
"""
import glob
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")


def load(pat):
    out = []
    for f in sorted(glob.glob(pat)):
        try:
            out.append(json.load(open(f)))
        except Exception:
            pass
    return out


def ms(vals):
    n = len(vals)
    m = sum(vals) / n
    s = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return m, s, n


def cell(m, s, dec=4):
    return f"{m:.{dec}f}$\\pm${s:.{dec}f}"


def evcell(m, s):
    if m == 0 and s == 0:
        return "0"
    return f"{m:.1f}$\\pm${s:.1f}"


def row_e2(label, pat, dec=4, divthresh=1e24):
    runs = load(pat)
    if not runs:
        return f"% MISSING: {label} ({pat})"
    nm = [r["nmse"] for r in runs]
    ev = [r["events"] for r in runs]
    mn, sn, n = ms(nm)
    me, se, _ = ms(ev)
    if mn > divthresh:
        nc = "diverged"
    else:
        nc = cell(mn, sn, dec)
    return f"{label} & {nc} & {evcell(me, se)} & {n} \\\\"


E2 = os.path.join(R, "e2")

print("=" * 30, "TABLE 10: stationary", "=" * 30)
for ds, title in [("mackey", "Mackey-Glass"), ("lorenz", "Lorenz"),
                  ("sunspot", "Sunspots (monthly)"),
                  ("santafe", "Santa Fe laser")]:
    print(f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
    # oracle rows from fixed grid
    best, stab = None, None
    for lr in ["0.003", "0.01", "0.03", "0.1", "0.3", "0.6", "1"]:
        runs = load(os.path.join(E2, f"{ds}_fixed_lr{lr}_s*.json"))
        if not runs:
            continue
        mn, sn, n = ms([r["nmse"] for r in runs])
        me, se, _ = ms([r["events"] for r in runs])
        rec = (mn, sn, me, se, lr, n)
        if mn < 1e20 and (best is None or mn < best[0]):
            best = rec
        if me == 0 and (stab is None or mn < stab[0]):
            stab = rec
    if best:
        print(f"oracle-fixed ($\\eta{{=}}{best[4]}$) & {cell(best[0], best[1])} "
              f"& {evcell(best[2], best[3])} & {best[5]} \\\\")
    if stab and stab != best:
        print(f"oracle-stable ($\\eta{{=}}{stab[4]}$) & {cell(stab[0], stab[1])} "
              f"& {evcell(stab[2], stab[3])} & {stab[5]} \\\\")
    for meth, label in [("hd", "HD"), ("hd_scalar", "HD (scalar)"),
                        ("hdm", "HDM"), ("tfmd", "tfmd"), ("fmd", "fmd"),
                        ("cohg", "COHG"), ("cohg_r0", "COHG, $r{=}0$"),
                        ("cohg_nogate", "COHG, gate off")]:
        hits = sorted(set(
            os.path.basename(f).rsplit("_s", 1)[0]
            for f in glob.glob(os.path.join(E2, f"{ds}_{meth}_lr*_s*.json"))))
        for cfg in hits:
            lr = cfg.split("_lr")[-1]
            print(row_e2(f"{label}", os.path.join(E2, cfg + "_s*.json")),
                  f"% lr={lr}")
    print("\\midrule")

print("=" * 30, "TABLE 11: drift", "=" * 30)
for ds, title in [("mackey_drift", "Drifting Mackey-Glass ($\\tau$ switches)"),
                  ("lorenz_drift", "Drifting Lorenz ($\\rho$ switches)")]:
    print(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
    best, stab = None, None
    for lr in ["0.003", "0.01", "0.03", "0.1", "0.3", "0.6", "1"]:
        runs = load(os.path.join(E2, f"{ds}_fixed_lr{lr}_s*.json"))
        if not runs:
            continue
        mn, sn, n = ms([r["nmse"] for r in runs])
        me, se, _ = ms([r["events"] for r in runs])
        rec = (mn, sn, me, se, lr, n)
        if mn < 1e20 and (best is None or mn < best[0]):
            best = rec
        if me == 0 and (stab is None or mn < stab[0]):
            stab = rec
    if best:
        print(f"oracle-fixed & {best[4]} & {cell(best[0], best[1])} "
              f"& {evcell(best[2], best[3])} & {best[5]} \\\\")
    if stab:
        print(f"oracle-stable & {stab[4]} & {cell(stab[0], stab[1])} "
              f"& {evcell(stab[2], stab[3])} & {stab[5]} \\\\")
    for meth, label in [("hd", "HD"), ("hd_scalar", "HD (scalar)"),
                        ("hdm", "HDM"), ("tfmd", "tfmd"), ("fmd", "fmd"),
                        ("cohg", "COHG"), ("cohg_r0", "COHG, $r{=}0$"),
                        ("cohg_ogd", "COHG, proj.\\ grad."),
                        ("cohg_nogate", "COHG, gate off")]:
        hits = sorted(set(
            os.path.basename(f).rsplit("_s", 1)[0]
            for f in glob.glob(os.path.join(E2, f"{ds}_{meth}_lr*_s*.json"))))
        for cfg in hits:
            lr = cfg.split("_lr")[-1]
            runs = load(os.path.join(E2, cfg + "_s*.json"))
            nm = [r["nmse"] for r in runs]
            ev = [r["events"] for r in runs]
            mn, sn, n = ms(nm)
            me, se, _ = ms(ev)
            nc = "diverged" if mn > 1e24 else cell(mn, sn)
            print(f"{label} & {lr} & {nc} & {evcell(me, se)} & {n} \\\\")
    print("\\midrule")

print("=" * 30, "TABLE 12: E3", "=" * 30)
E3 = os.path.join(R, "e3")
cfgs = sorted(set(os.path.basename(f).rsplit("_s", 1)[0]
                  for f in glob.glob(os.path.join(E3, "cifar100_*_s*.json"))))
order = ["fixed", "hd", "tfmd", "cohg_r0", "cohg_nogate", "cohg"]
labels = {"fixed": "fixed", "hd": "HD", "tfmd": "tfmd", "cohg": "COHG",
          "cohg_r0": "COHG, $r{=}0$", "cohg_nogate": "COHG, gate off"}
rows = []
for cfg in cfgs:
    body = cfg[len("cifar100_"):]
    meth = None
    for m in ["cohg_nogate", "cohg_r0", "cohg", "tfmd", "hd", "fixed"]:
        if body.startswith(m + "_"):
            meth = m
            break
    rest = body[len(meth) + 1:]
    lr = rest.split("_ewc")[0][2:]
    ewc = rest.split("_ewc")[1]
    runs = load(os.path.join(E3, cfg + "_s*.json"))
    mls = sorted(set(r["meta_lr"] for r in runs))
    a, sa, n = ms([r["avg_acc"] for r in runs])
    b, sb, _ = ms([r["bwt"] for r in runs])
    e, se, _ = ms([r["events"] for r in runs])
    rows.append((meth, float(lr), float(ewc),
                 f"{labels[meth]} & ({lr}, {float(ewc):g}) & {cell(a, sa, 3)} & "
                 f"${b:+.3f}\\pm{sb:.3f}$ & {evcell(e, se)} \\\\ % n={n} ml={mls}"))
for meth in order:
    for r in sorted([x for x in rows if x[0] == meth], key=lambda x: (x[2], x[1])):
        print(r[3])
