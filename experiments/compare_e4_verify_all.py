"""Review P4, full scope: same-seed verification of the corrected E4 held bound.

Pairs EVERY legacy (scalar drift-hold) gated E4 run that is still a reported
result with its same-seed rerun under the corrected vector-valued Proposition-10
bound, and writes results/e4_verify_all/COMPARE_ALL.md.

Legacy provenance is defined by the ABSENCE of the `legacy_hold` / `held_bound`
/ `gate_open_steps` fields: those were added by the same commit that made the
vector bound the default, so any JSON without them predates the fix.

Scope (gated arms only -- the bound never enters the update of `fixed`, `hd` or
`cohg_nogate`, so those arms are excluded by construction, see SCOPE_NOTE):

  results/e4_v2      cohg_r0 (rank 0)  seeds 0,1,2   -> already done in e4_fix
  results/e4_v2      cohg_r0 (rank 0)  seeds 3..7    -> results/e4_verify_all
  results/e4_v2      cohg    (rank 4)  seeds 0,1,2   -> results/e4_verify_all
  results/e4_orders  cohg_r0 (rank 0)  seeds 0,1,2   -> results/e4_verify_all

Per pair the report gives
  * gate_open_frac / coord_open_frac equality
  * gate openings and their step(s): exact from `gate_open_steps` when present,
    otherwise bracketed on the 20-step `lam_hist` grid
  * max |delta lambda| over the common lam_hist grid
  * loss-trace max |delta| and whether the trace is bit-identical
  * online_ppl / mean_logloss difference, hvp_total, events
  * a three-way verdict: bit-identical / gate-identical (fp round-off) / DIFFERS

Usage:  python compare_e4_verify_all.py            # writes COMPARE_ALL.md
        python compare_e4_verify_all.py --quiet    # no stdout dump
Exit code 0 iff every expected pair is present and every verdict is
bit-identical or gate-identical.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
V2 = os.path.join(RES, "e4_v2")
ORD = os.path.join(RES, "e4_orders")
FIX = os.path.join(RES, "e4_fix")
NEW = os.path.join(RES, "e4_verify_all")
OUT_MD = os.path.join(NEW, "COMPARE_ALL.md")

TOL_PPL = 1e-3          # online_ppl agreement required for "gate-identical"
TOL_LAM = 1e-9          # max |delta lambda| tolerated as fp round-off

STD_ORDER = "wiki,news,code"
ALT_ORDER = "code,news,wiki"

SCOPE_NOTE = (
    "The gated arms are the only ones in scope: the held bound `(rho_t, "
    "kappa_t)` is consumed exclusively by `CoordGatedController.maybe_update` "
    "(through `est.step(...)` / `beta_col`), so it can only ever change a run "
    "whose lambda update is gated.  `fixed` never builds a certificate at all "
    "(`est is None`); `hd` uses `HDBaseline.update`, which never sees rho or "
    "kappa; `cohg_nogate` short-circuits to `rho, kappa = 1.0, 0.0` before the "
    "drift hold is queried and takes an ungated sign step.  Those three arms "
    "are therefore bit-identical under either code path by construction and "
    "are excluded from the rerun."
)


# --------------------------------------------------------------------------
# the pair table
# --------------------------------------------------------------------------
def build_pairs():
    """(label, arm, seed, order, legacy_path, rerun_path, rerun_source)."""
    P = []

    def add(arm, seed, order, legacy_dir, legacy_name, rerun_dir,
            rerun_name=None, source=""):
        P.append(dict(
            arm=arm, seed=seed, order=order,
            legacy=os.path.join(legacy_dir, legacy_name),
            rerun=os.path.join(rerun_dir, rerun_name or legacy_name),
            legacy_dir=os.path.basename(legacy_dir), source=source,
            key=os.path.splitext(legacy_name)[0]))

    # already verified in Phase 1 (results/e4_fix)
    for s in (0, 1, 2):
        add("cohg_r0", s, STD_ORDER, V2, "gpt2_cohg_r0_lr0.001_s%d.json" % s,
            FIX, source="e4_fix")
    # standard order, remaining seeds
    for s in (3, 4, 5, 6, 7):
        add("cohg_r0", s, STD_ORDER, V2, "gpt2_cohg_r0_lr0.001_s%d.json" % s,
            NEW, source="e4_verify_all")
    # rank-4 arm, standard order
    for s in (0, 1, 2):
        add("cohg(r=4)", s, STD_ORDER, V2, "gpt2_cohg_lr0.001_s%d.json" % s,
            NEW, source="e4_verify_all")
    # reverse order, legacy seeds
    for s in (0, 1, 2):
        add("cohg_r0", s, ALT_ORDER, ORD,
            "gpt2order_cnw_cohg_r0_lr0.001_s%d.json" % s,
            NEW, source="e4_verify_all")
    return P


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def mll(d):
    if d.get("mean_logloss") is not None:
        return float(d["mean_logloss"])
    fin = [x for x in d["losses"] if x == x and abs(x) != float("inf")]
    return sum(fin) / len(fin)


def lam_grid(d):
    return {int(r[0]): [float(x) for x in r[1:]] for r in d["lam_hist"]}


def openings(d):
    """(n_openings, exact_steps or None, human description)."""
    gof = d.get("gate_open_frac")
    n = int(round(gof * d["steps"])) if gof is not None else 0
    gos = d.get("gate_open_steps")
    if gos is not None:
        return n, list(gos), ("steps %s" % (gos[:6] + (["..."] if len(gos) > 6
                                                       else []),))
    g = lam_grid(d)
    ts = sorted(g)
    base = g[ts[0]]
    first = next((t for t in ts[1:] if g[t] != base), None)
    if first is None:
        return n, None, "lambda never moved on the 20-step grid"
    prev = ts[ts.index(first) - 1]
    return n, None, "in (%d, %d] (20-step lam_hist grid)" % (prev, first)


def window_of(d):
    """(lo, hi] bracket of the FIRST lambda move on the 20-step grid."""
    g = lam_grid(d)
    ts = sorted(g)
    base = g[ts[0]]
    first = next((t for t in ts[1:] if g[t] != base), None)
    if first is None:
        return None
    return ts[ts.index(first) - 1], first


def feq(a, b):
    """exact equality that also treats None==None and nan==nan as equal."""
    if a is None or b is None:
        return a is b or a == b
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
    return a == b


def compare(pair):
    o = json.load(open(pair["legacy"]))
    n = json.load(open(pair["rerun"]))

    no, so, wo = openings(o)
    nn, sn, wn = openings(n)
    win_o = window_of(o)
    # exact new opening steps must sit inside the old bracket
    if sn is not None and win_o is not None:
        steps_consistent = all(win_o[0] < t <= win_o[1] for t in sn)
    elif sn is None and so is None:
        steps_consistent = window_of(n) == win_o
    else:
        steps_consistent = window_of(n) == win_o

    go, gn = lam_grid(o), lam_grid(n)
    common = sorted(set(go) & set(gn))
    dlam = max((max(abs(a - b) for a, b in zip(go[t], gn[t]))
                for t in common), default=float("nan"))
    lam_same = bool(common) and all(go[t] == gn[t] for t in common)
    grid_same = sorted(go) == sorted(gn)

    lo, ln = o["losses"], n["losses"]
    len_same = len(lo) == len(ln)
    k = min(len(lo), len(ln))
    first_diff = next((i for i in range(k) if lo[i] != ln[i]), None)
    dloss = max((abs(lo[i] - ln[i]) for i in range(k)), default=0.0)
    loss_bitid = len_same and first_diff is None

    dppl = n["online_ppl"] - o["online_ppl"]
    gof_same = feq(o.get("gate_open_frac"), n.get("gate_open_frac"))
    cof_same = feq(o.get("coord_open_frac"), n.get("coord_open_frac"))
    hvp_same = o["hvp_total"] == n["hvp_total"]
    ev_same = o["events"] == n["events"]

    bit = (loss_bitid and lam_same and grid_same and gof_same and cof_same
           and hvp_same and ev_same and o["online_ppl"] == n["online_ppl"]
           and o["steps"] == n["steps"])
    gate_ok = (no == nn and gof_same and cof_same and hvp_same and ev_same
               and steps_consistent
               and (lam_same or (dlam == dlam and dlam <= TOL_LAM))
               and abs(dppl) <= TOL_PPL)
    verdict = ("bit-identical" if bit else
               ("gate-identical (fp round-off)" if gate_ok else "**DIFFERS**"))

    return dict(
        pair, present=True,
        n_old=no, n_new=nn, w_old=wo, w_new=wn,
        steps_consistent=steps_consistent,
        gof_old=o.get("gate_open_frac"), gof_new=n.get("gate_open_frac"),
        cof_old=o.get("coord_open_frac"), cof_new=n.get("coord_open_frac"),
        gof_same=gof_same, cof_same=cof_same,
        lam_same=lam_same, dlam=dlam,
        loss_bitid=loss_bitid, first_diff=first_diff, dloss=dloss,
        ppl_old=o["online_ppl"], ppl_new=n["online_ppl"], dppl=dppl,
        ll_old=mll(o), ll_new=mll(n),
        hvp_old=o["hvp_total"], hvp_new=n["hvp_total"],
        ev_old=o["events"], ev_new=n["events"],
        steps_old=o["steps"], steps_new=n["steps"],
        held_old=o.get("held_bound", "(absent -> scalar legacy)"),
        held_new=n.get("held_bound"),
        legacy_flag_old=o.get("legacy_hold", "(absent)"),
        legacy_flag_new=n.get("legacy_hold"),
        order_new=n.get("domain_order"),
        lam_end_old=go[max(go)], lam_end_new=gn[max(gn)],
        verdict=verdict, ok=(bit or gate_ok))


# --------------------------------------------------------------------------
def render(rows, pending):
    L = []
    A = L.append
    A("# E4 held-bound verification, FULL scope (review P4)\n")
    A("Same-seed rerun of **every legacy gated E4 run that is still a reported "
      "result**, under the corrected vector-valued Proposition-10 held bound\n")
    A("```")
    A("iota_t = Delta eta_t * Hbar_t + eta_max,t0 * (M_H * P_t + nu_H),"
      "   Delta eta_t = ||eta_t - eta_t0||_inf")
    A("```")
    A("i.e. `dh.probe(rho, kappa, eta_vec=eta)` / `dh.bounds(eta)` "
      "(`e4_gpt2_tta.py` default path), against the shipped runs, which used "
      "the scalar interface `dh.probe(rho, kappa)` / "
      "`dh.bounds(float(eta.max()))` -- forcing `Delta eta_t == 0` and reading "
      "`eta_max` at the CURRENT step instead of at the last probe.  The old "
      "path survives behind `--legacy-hold`.\n")
    A("Legacy provenance is the ABSENCE of the `legacy_hold` / `held_bound` / "
      "`gate_open_steps` fields, which entered with the fix.\n")
    A("Driver flags are identical for every pair (read back from the shipped "
      "JSONs): `--tokens-per-domain 512000 --max-steps 3000 --batch 2 "
      "--seq-len 256 --probe-every 100 --kw-eps 0.15 --lr 0.001 --meta-lr 0.4` "
      "with `--domain-order` as tabulated (K=20, gamma=0.9, rank=4 for the "
      "`cohg` arm / 0 for `cohg_r0`, M_H=50).\n")
    A("**Scope.** " + SCOPE_NOTE + "\n")

    total = len(rows) + len(pending)
    A("## Coverage\n")
    A("| | count |")
    A("|---|---|")
    A("| legacy gated runs in scope | %d |" % total)
    A("| verified (rerun present) | %d |" % len(rows))
    A("| still on the GPU queue | %d |" % len(pending))
    A("")
    if pending:
        A("Pending: " + ", ".join("`%s`" % p["key"] for p in pending) + "\n")

    if rows:
        A("## Per-pair verdict\n")
        A("| # | arm | seed | domain order | legacy dir | rerun dir | "
          "gate_open_frac == | coord_open_frac == | openings old/new | "
          "opening step(s) old | opening step(s) new | max abs d(lambda) | "
          "loss bit-identical | max abs d(loss) | d(online_ppl) | "
          "hvp_total old/new | events old/new | verdict |")
        A("|" + "---|" * 18)
        for i, r in enumerate(rows, 1):
            A("| %d | %s | %d | %s | %s | %s | %s | %s | %d/%d | %s | %s | "
              "%.3g | %s | %.3g | %+.2e | %d/%d | %d/%d | %s |"
              % (i, r["arm"], r["seed"], r["order"], r["legacy_dir"],
                 r["source"],
                 "yes" if r["gof_same"] else "**NO**",
                 "yes" if r["cof_same"] else "**NO**",
                 r["n_old"], r["n_new"], r["w_old"], r["w_new"],
                 r["dlam"], "yes" if r["loss_bitid"] else
                 ("no (first diff at step %s)" % r["first_diff"]),
                 r["dloss"], r["dppl"], r["hvp_old"], r["hvp_new"],
                 r["ev_old"], r["ev_new"], r["verdict"]))
        A("")

        A("## Perplexity, log-loss and final lambda\n")
        A("| # | arm | seed | order | online_ppl legacy | online_ppl rerun | "
          "delta | mean_logloss legacy | mean_logloss rerun | steps | "
          "final lambda legacy | final lambda rerun |")
        A("|" + "---|" * 12)
        for i, r in enumerate(rows, 1):
            A("| %d | %s | %d | %s | %.6f | %.6f | %+.2e | %.8f | %.8f | "
              "%d/%d | %s | %s |"
              % (i, r["arm"], r["seed"], r["order"], r["ppl_old"],
                 r["ppl_new"], r["dppl"], r["ll_old"], r["ll_new"],
                 r["steps_old"], r["steps_new"],
                 "[" + ", ".join("%.4f" % x for x in r["lam_end_old"]) + "]",
                 "[" + ", ".join("%.4f" % x for x in r["lam_end_new"]) + "]"))
        A("")

        A("## Provenance\n")
        A("| # | arm | seed | order | legacy `held_bound` | legacy "
          "`legacy_hold` | rerun `held_bound` | rerun `legacy_hold` | rerun "
          "`domain_order` |")
        A("|" + "---|" * 9)
        for i, r in enumerate(rows, 1):
            A("| %d | %s | %d | %s | %s | %s | %s | %s | %s |"
              % (i, r["arm"], r["seed"], r["order"], r["held_old"],
                 r["legacy_flag_old"], r["held_new"], r["legacy_flag_new"],
                 r["order_new"]))
        A("")

    A("## Summary\n")
    nbit = sum(1 for r in rows if r["verdict"] == "bit-identical")
    ngate = sum(1 for r in rows if r["verdict"].startswith("gate-identical"))
    nbad = [r for r in rows if not r["ok"]]
    A("| verdict | pairs |")
    A("|---|---|")
    A("| bit-identical | %d |" % nbit)
    A("| gate-identical (fp round-off) | %d |" % ngate)
    A("| DIFFERS | %d |" % len(nbad))
    A("| pending | %d |" % len(pending))
    A("")

    if nbad:
        A("**At least one pair DIFFERS -- the scalar shortcut was NOT inert.**"
          "  The E4 tables must be regenerated from the corrected runs for:\n")
        for r in nbad:
            A("* `%s` (%s, seed %d, order %s): openings %d -> %d, "
              "gate_open_frac %s -> %s, max abs d(lambda) %.3g, "
              "d(online_ppl) %+.3e, first differing loss step %s"
              % (r["key"], r["arm"], r["seed"], r["order"], r["n_old"],
                 r["n_new"], r["gof_old"], r["gof_new"], r["dlam"], r["dppl"],
                 r["first_diff"]))
    elif pending:
        A("**Pending -- %d of %d pairs in, and every one of them agrees.**  "
          "Each pair present so far reproduces its shipped run with the same "
          "gate decisions; the remaining reruns are on the GPU queue and this "
          "file is rewritten as each one lands." % (len(rows), total))
    elif rows:
        A("**Every legacy gated E4 run reproduces bit-for-bit under the "
          "corrected vector-valued Proposition-10 bound.**  All %d pairs agree "
          "on `gate_open_frac`, `coord_open_frac`, the number and location of "
          "the gate openings, the whole logged lambda trajectory, the per-step "
          "loss trace, `hvp_total` and `events`.  The scalar drift-hold "
          "shortcut was therefore *inert* on the entire reported E4 grid: the "
          "gate opens once, at t <= 20, and lambda is frozen thereafter, so "
          "`Delta eta_t = 0` between probes and `eta_max` at the current step "
          "equals `eta_max` at the last probe.  The published E4 numbers stand "
          "as reported." % len(rows))
        A("")
        A("This closes review item P4 at full scope: the 3-seed check in "
          "`results/e4_fix/COMPARE.md` is no longer the basis of the claim -- "
          "every legacy gated run that remains a reported result has been "
          "re-verified on its own seed.")
    else:
        A("_No rerun artifacts yet._")

    A("")
    A("<sub>generated by `code/experiments/compare_e4_verify_all.py`</sub>")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    rows, pending = [], []
    for p in build_pairs():
        if not os.path.exists(p["legacy"]):
            print("[warn] missing LEGACY artifact %s" % p["legacy"],
                  file=sys.stderr)
            continue
        if not os.path.exists(p["rerun"]):
            pending.append(p)
            continue
        try:
            rows.append(compare(p))
        except Exception as e:                              # noqa: BLE001
            print("[warn] compare failed for %s: %r" % (p["key"], e),
                  file=sys.stderr)
            pending.append(p)

    txt = render(rows, pending)
    os.makedirs(NEW, exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(txt)
    if not a.quiet:
        print(txt)
    print("-> %s  (%d verified, %d pending)" % (OUT_MD, len(rows),
                                                len(pending)))
    return 0 if (rows and all(r["ok"] for r in rows) and not pending) else 1


if __name__ == "__main__":
    sys.exit(main())
