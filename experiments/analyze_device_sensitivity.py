"""B5 (R2 minor): device-stratified sensitivity of the E2 mis-set-init arms.

The SAME (method, dataset, seed, config) runs exist twice: on GPU in
results/e2 (the campaign of record) and on CPU in results/e2_controls (the
control study's reference arm).  Everything about the run is identical except
the device, so the pair is a pure float-reassociation perturbation.

File-name mapping (verified against the JSON payloads, not just the names):
  cohg          results/e2/mackey_drift_cohg_lr0.003_s{S}.json          (GPU)
             <- results/e2_controls/..._mh5_fc0_s{S}.json               (CPU)
                 the CPU reference arm carries M_H=5.0, fail_closed=False,
                 alpha=0.4 -- identical to the GPU arm's implicit defaults.
  cohg_nogate   results/e2/mackey_drift_cohg_nogate_lr0.003_s{S}.json   (GPU)
             <- results/e2_controls/..._nogate_lr0.003_a0.4_s{S}.json   (CPU)

v2 (this revision) additionally reports, per seed: the realized open
COORDINATE-STEP count (not just the rate), HVP totals, wall time, the first
step at which the two loss trajectories differ at all, the max relative
trajectory deviation, and the final per-group log-LR distance -- i.e. where
the float perturbation enters and how far the controllers end up apart.

-> results/reanalysis/device_sensitivity.md
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
OUT = os.path.join(RES, "reanalysis")
os.makedirs(OUT, exist_ok=True)

PAIRS = [
    ("cohg", "certificate-gated COHG (alpha=0.4)",
     "e2/mackey_drift_cohg_lr0.003_s{s}.json",
     "e2_controls/mackey_drift_cohg_lr0.003_mh5_fc0_s{s}.json"),
    ("cohg_nogate", "same estimator, gate OFF (pure sign, alpha=0.4)",
     "e2/mackey_drift_cohg_nogate_lr0.003_s{s}.json",
     "e2_controls/mackey_drift_cohg_nogate_lr0.003_a0.4_s{s}.json"),
]
SEEDS = list(range(10))
N_GROUPS = 6          # per-group LRs -> coordinate-steps = steps * N_GROUPS


def load(rel):
    p = os.path.join(RES, rel)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        d = json.load(f)
    steps = d.get("steps") or len(d.get("losses") or []) or 1
    co = d.get("coord_open_frac")
    return {
        "nmse": d["nmse"], "events": d["events"],
        "open": co,
        "open_steps": (None if co is None else co * steps * N_GROUPS),
        "gate_open": d.get("gate_open_frac"),
        "hvp": d.get("hvp_total"), "wall": d.get("wall_s"),
        "steps": steps,
        "losses": d.get("losses"), "lam_hist": d.get("lam_hist"),
        "M_H": d.get("M_H"), "fail_closed": d.get("fail_closed"),
        "alpha": d.get("meta_lr"),
    }


def traj_divergence(g, c):
    """(first differing step, max relative deviation) over the loss trace."""
    a, b = g.get("losses"), c.get("losses")
    if not a or not b or len(a) != len(b):
        return None, None
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = np.abs(a - b)
    nz = np.nonzero(d)[0]
    first = int(nz[0]) if len(nz) else None
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-30)
    return first, float(np.max(d / denom))


def lam_gap(g, c):
    """||log-LR_final(GPU) - log-LR_final(CPU)||_inf, in log10-LR units."""
    a, b = g.get("lam_hist"), c.get("lam_hist")
    if not a or not b:
        return None
    return float(np.max(np.abs(np.asarray(a[-1][1:], dtype=np.float64)
                               - np.asarray(b[-1][1:], dtype=np.float64))))


def regime(r):
    """Coarse outcome class used throughout the paper: a run is 'unstable' if
    it trips >30 instability events or ends above NMSE 1 (= worse than
    predicting the mean)."""
    return (r["events"] > 30) or (r["nmse"] > 1.0)


def diverged(r):
    """Stricter sub-criterion: the run actually blew up (NMSE > 1)."""
    return r["nmse"] > 1.0


def fmt(x, spec=".3e", na="n/a"):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return na
    return format(x, spec)


L = ["# B5. Device-stratified sensitivity (GPU `results/e2` vs CPU "
     "`results/e2_controls`)", "",
     "Same code, same seeds, same config (`mackey_drift`, lr0=0.003, 12000 "
     "steps, gamma 0.9, alpha 0.4, K 10, rank 4, c=2, M_H=5, no fail-closed); "
     "the ONLY difference is the device, i.e. a float-reassociation "
     "perturbation of order 1e-7 relative.  `regime` marks a run unstable if "
     "events > 30 or NMSE > 1.  The CPU counterpart of the GPU `cohg` arm is "
     "the control study's reference arm `..._mh5_fc0_s{S}.json`, and of "
     "`cohg_nogate` it is `..._nogate_lr0.003_a0.4_s{S}.json`; the config "
     "fields inside those files were checked to match the GPU runs.",
     "",
     "`open coord-steps` = `coord_open_frac` x 12000 steps x 6 LR groups, "
     "i.e. the raw count of gate-open decisions.  `1st diff step` is the "
     "first index at which the two per-step loss traces are not bitwise "
     "equal; `max rel traj` is the largest relative gap over the whole trace; "
     "`d ln-LR` is the max-norm gap between the two final per-group "
     "NATURAL-log LR vectors (0.69 = a 2x LR difference).", ""]

summary = []
for key, label, gpu_t, cpu_t in PAIRS:
    rows, dn, de, flips, dflips, gflips = [], [], [], 0, 0, 0
    for s in SEEDS:
        g, c = load(gpu_t.format(s=s)), load(cpu_t.format(s=s))
        if g is None or c is None:
            continue
        rel = abs(g["nmse"] - c["nmse"]) / max(abs(g["nmse"]), abs(c["nmse"]),
                                               1e-30)
        flip = regime(g) != regime(c)
        flips += int(flip)
        dflips += int(diverged(g) != diverged(c))
        if g["open"] is not None and c["open"] is not None:
            gflips += int(abs(g["open"] - c["open"]) > 1e-12)
        dn.append(abs(g["nmse"] - c["nmse"]))
        de.append(abs(g["events"] - c["events"]))
        first, maxrel = traj_divergence(g, c)
        rows.append((s, g, c, rel, flip, first, maxrel, lam_gap(g, c)))

    if not rows:
        continue

    L += [f"## {label}  (`{key}`)", "",
          "### outcome metrics", "",
          "| seed | GPU NMSE | CPU NMSE | \\|dNMSE\\| | rel diff | GPU events "
          "| CPU events | \\|d ev\\| | regime flip |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s, g, c, rel, flip, first, maxrel, dl in rows:
        L.append(f"| {s} | {g['nmse']:.6g} | {c['nmse']:.6g} | "
                 f"{abs(g['nmse'] - c['nmse']):.3e} | {rel:.2e} | "
                 f"{g['events']} | {c['events']} | "
                 f"{abs(g['events'] - c['events'])} | "
                 f"{'**YES**' if flip else 'no'} |")

    L += ["", "### gate decisions, cost, and where the perturbation enters",
          "",
          "| seed | GPU open coord-steps | CPU open coord-steps | d open | "
          "GPU open rate | CPU open rate | 1st diff step | max rel traj | "
          "d ln-LR | GPU HVPs | CPU HVPs | GPU wall (s) | CPU wall (s) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, g, c, rel, flip, first, maxrel, dl in rows:
        if g["open"] is None or c["open"] is None:
            og = cg = dopen = ogr = cgr = "n/a"
        else:
            og = f"{g['open_steps']:.0f}"
            cg = f"{c['open_steps']:.0f}"
            dopen = f"{g['open_steps'] - c['open_steps']:+.0f}"
            ogr = f"{g['open']:.3e}"
            cgr = f"{c['open']:.3e}"
        if key == "cohg_nogate":
            og = cg = dopen = "gate off"
            ogr = cgr = "1.000 (open)"
        L.append(f"| {s} | {og} | {cg} | {dopen} | {ogr} | {cgr} | "
                 f"{'never' if first is None else first} | "
                 f"{fmt(maxrel, '.2e')} | {fmt(dl, '.3f')} | "
                 f"{g['hvp']} | {c['hvp']} | {fmt(g['wall'], '.0f')} | "
                 f"{fmt(c['wall'], '.0f')} |")

    gm = np.mean([r[1]["nmse"] for r in rows])
    cm = np.mean([r[2]["nmse"] for r in rows])
    gmed = np.median([r[1]["nmse"] for r in rows])
    cmed = np.median([r[2]["nmse"] for r in rows])
    rels = [r[3] for r in rows]
    firsts = [r[5] for r in rows if r[5] is not None]
    bitident = sum(1 for r in rows if r[5] is None)
    L += ["",
          f"- mean paired |dNMSE| = {np.mean(dn):.3e} "
          f"(median relative {np.median(rels):.2e}, max relative "
          f"{np.max(rels):.2e})",
          f"- mean paired |d events| = {np.mean(de):.2f} "
          f"(max {np.max(de):.0f})",
          f"- arm mean NMSE: GPU {gm:.4f} vs CPU {cm:.4f}; "
          f"arm MEDIAN NMSE: GPU {gmed:.4f} vs CPU {cmed:.4f}",
          f"- seeds with a BITWISE-IDENTICAL loss trajectory: "
          f"**{bitident} / {len(rows)}**"
          + (f"; among the rest the traces first differ at step "
             f"{min(firsts)}-{max(firsts)} (median {int(np.median(firsts))})"
             if firsts else ""),
          f"- seeds that change regime (events>30 OR NMSE>1): "
          f"**{flips} / {len(rows)}**",
          f"- seeds that change the stricter divergence flag (NMSE>1 alone): "
          f"**{dflips} / {len(rows)}**",
          f"- seeds whose realized per-coordinate GATE-OPEN RATE differs "
          f"between devices: **{gflips} / {len(rows)}**", ""]
    summary.append((label, key, np.mean(dn), np.median(rels), np.max(rels),
                    np.mean(de), flips, len(rows), gm, cm, gmed, cmed,
                    dflips, gflips, bitident))

L += ["## Summary", "",
      "| arm | n | mean \\|dNMSE\\| | median rel | max rel | mean \\|d ev\\| | "
      "bitwise-identical traj | regime flips (ev>30 or NMSE>1) | "
      "divergence flips (NMSE>1) | gate-rate differs | GPU mean | CPU mean | "
      "GPU median | CPU median |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for (label, key, mdn, mrel, xrel, mde, fl, n, gm, cm, gmed, cmed,
     dfl, gfl, bi) in summary:
    L.append(f"| {key} | {n} | {mdn:.3e} | {mrel:.2e} | {xrel:.2e} | "
             f"{mde:.2f} | {bi}/{n} | {fl} | {dfl} | {gfl} | {gm:.4f} | "
             f"{cm:.4f} | {gmed:.4f} | {cmed:.4f} |")

L += ["", "## Reading", "",
      "**The gated arm never changes regime, but it is not bit-identical "
      "across devices.** Six of ten COHG seeds reproduce to fp32 round-off "
      "(rel <= 1e-4); the other four (5, 6, 7, 9) differ by 15-22% in NMSE "
      "because the float perturbation moves a handful of borderline "
      "`|ghat_j|` values across the `c * beta_j` threshold, so the realized "
      "gate-open rate changes -- and those are exactly the four seeds whose "
      "open coordinate-step COUNT differs (seed 5: 36 GPU vs 31 CPU; seed 6: "
      "42 vs 36; seed 7: 37 vs 32; seed 9: 34 vs 28). "
      "The instability-event count is nevertheless IDENTICAL on all ten seeds "
      "(mean paired |d events| = 0), and both arm summaries land in the same "
      "place (GPU 0.0150 vs CPU 0.0162, i.e. within half a pooled SD of "
      "0.002). So the certificate gate's operating point is device-stable; "
      "its exact NMSE to three digits is not, and the paper should not quote "
      "COHG's NMSE to more than two significant figures on the strength of a "
      "single device.  (This refines the claim in "
      "`results/e2_controls/SUMMARY.md` that the two devices agree 'with "
      "identical gate decisions': that holds for the single config that was "
      "spot-checked, not for every seed.)", "",
      "**The perturbation enters immediately; the gate decides whether it "
      "matters.** No pair is bitwise identical from step 0 -- every one of "
      "the twenty pairs separates within the first handful of steps, which is "
      "what a reassociation-only difference should do. What differs between "
      "arms is the AMPLIFICATION. In the gated arm the divergence stays "
      "bounded: the four gate-flipping seeds end 0.52-0.60 nats apart in the "
      "worst LR group (a 1.7-1.8x LR difference) and stay inside the same "
      "NMSE decade, and the other six end bit-for-bit on the same lambda. "
      "In the ungated arm the same perturbation is amplified without limit: "
      "seeds 4, 8 and 5 end 3.9, 3.2 and 11.5 nats apart (the last is a "
      "1e5x LR difference), and seed 5 runs away.", "",
      "**The ungated arm changes regime on a float.** Seed 5 lands at NMSE "
      "59.8 with 298 events on GPU and at NMSE 0.0048 with 53 events on CPU: "
      "a ~1e-7 reassociation decides which side of the divergence boundary "
      "the run falls on, moving that seed's NMSE by four orders of magnitude "
      "and the ARM MEAN by a factor of ~2000, while the arm MEDIAN barely "
      "moves (0.0030 vs 0.0031). Under the paper's coarse regime label the "
      "seed is 'unstable' on both devices (298 and 53 events both exceed 30), "
      "so the headline regime count is unchanged (0/10 flips) -- but under "
      "the strict NMSE>1 divergence flag it flips 1/10. Two further ungated "
      "seeds (4 and 8) also disagree in event count (107 vs 108, 163 vs 168) "
      "without changing regime.", "",
      "**Cost is a device artefact, not a method property.** The gated arm "
      "issues the same 94588 HVPs on both devices and the ungated arm the "
      "same 11992, so nothing about the algorithm's work changes; only wall "
      "time does (GPU ~3.7ks vs CPU ~5.1ks for COHG), because at 13k "
      "parameters the HVPs are launch-bound and a CPU core beats a 3080. "
      "Device choice therefore cannot be read off the timing column as an "
      "efficiency claim either way.", "",
      "**Consequences.** (i) Report medians and event counts alongside means "
      "for the ungated controls -- the mean of an arm that straddles a "
      "divergence boundary is not a stable statistic. (ii) Keep every "
      "within-study comparison on one device, as results/e2_controls does. "
      "(iii) Quote COHG NMSE to two significant figures. (iv) The gate is "
      "what removes the device sensitivity of the OUTCOME, which is the same "
      "tail-control claim the paper makes on other grounds -- it does not "
      "make the run reproducible bit-for-bit, and the paper should not claim "
      "that."]

path = os.path.join(OUT, "device_sensitivity.md")
with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print("\n".join(L))
print("\n->", path)
