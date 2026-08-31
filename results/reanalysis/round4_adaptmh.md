# Round-4 experiment 2: an online-enforceable conservative drift envelope

Envelope in force at probe `t`: `M_H,t = max(M_H_floor, KAPPA * max_{s<=t} M_obs,s)`, re-stated at every probe, fail-closed monitor on. `M_H_floor` is the **deployed prior** (5.0 for E2, 2.2760914 for E1 -- the calibrated `M_H*` of the teacher/kw_drift stream), so the arm is never LESS conservative than the shipped certificate. The monitor is evaluated against the envelope in force over the interval just traversed (before the new observation is folded in): raising the envelope afterwards cannot retroactively excuse that interval.

## E2 -- GRU / `mackey_drift`, seeds 0-9, per-coordinate certificate audit (`--validate-cert`)

| arm | n | NMSE | events | coord-open rate | cert viol / checked | cert max ratio | closed steps | closure frac | HVPs | HVPs / no-FC | final envelope | envelope raises | max M_obs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed prior M_H=5, no FC | 10 | 0.0162+-0.0021 | 9.0+-12.6 | 4.722e-04 | 0 / 720000 | 0.8374 | nan | nan | 94588 | 1.000 | - | - | - |
| fixed prior M_H=5, fail-closed | 10 | 0.0162+-0.0021 | 9.0+-12.6 | 4.722e-04 | 0 / 720000 | 0.8374 | 939.7 | 0.0783 | 102602 | 1.085 | - | - | - |
| **adaptive envelope KAPPA=1** | 10 | 0.0162+-0.0021 | 9.0+-12.6 | 4.722e-04 | 0 / 720000 | 0.8374 | 8.6 | 0.0007 | 94816 | 1.002 | 29.92 | 1.4 | 29.92 |
| **adaptive envelope KAPPA=2** | 10 | 0.0162+-0.0021 | 9.0+-12.6 | 4.722e-04 | 0 / 720000 | 0.8374 | 1.0 | 0.0001 | 94740 | 1.002 | 39.74 | 1.1 | 19.87 |

### Gate decisions vs the fixed-prior arm (per seed)

`losses identical` / `lam identical` compare the FULL 12000-entry loss trace and the sampled lambda trajectory element for element against the fixed-prior fail-closed reference (`M_H = 5`, `results/e2_controls`). `open coord-steps` is `coord_open_frac * 6 * 12000`.

| KAPPA | seed | open coord-steps (adaptive) | (fixed prior) | openings lost | NMSE (adaptive / fixed) | events | losses identical | closed steps (adaptive / fixed) |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 33 | 33 | 0 | 0.014882 / 0.014882 | 0 / 0 | True | 1 / 1 |
| 1 | 1 | 26 | 26 | 0 | 0.018454 / 0.018454 | 0 / 0 | True | 20 / 1492 |
| 1 | 2 | 45 | 45 | 0 | 0.017283 / 0.017283 | 0 / 0 | True | 1 / 1 |
| 1 | 3 | 25 | 25 | 0 | 0.018540 / 0.018540 | 2 / 2 | True | 39 / 671 |
| 1 | 4 | 33 | 33 | 0 | 0.014573 / 0.014573 | 7 / 7 | True | 1 / 153 |
| 1 | 5 | 31 | 31 | 0 | 0.014854 / 0.014854 | 0 / 0 | True | 20 / 2412 |
| 1 | 6 | 36 | 36 | 0 | 0.019181 / 0.019181 | 0 / 0 | True | 1 / 1 |
| 1 | 7 | 32 | 32 | 0 | 0.014929 / 0.014929 | 28 / 28 | True | 1 / 268 |
| 1 | 8 | 51 | 51 | 0 | 0.012681 / 0.012681 | 28 / 28 | True | 1 / 3939 |
| 1 | 9 | 28 | 28 | 0 | 0.016280 / 0.016280 | 25 / 25 | True | 1 / 459 |
| 2 | 0 | 33 | 33 | 0 | 0.014882 / 0.014882 | 0 / 0 | True | 1 / 1 |
| 2 | 1 | 26 | 26 | 0 | 0.018454 / 0.018454 | 0 / 0 | True | 1 / 1492 |
| 2 | 2 | 45 | 45 | 0 | 0.017283 / 0.017283 | 0 / 0 | True | 1 / 1 |
| 2 | 3 | 25 | 25 | 0 | 0.018540 / 0.018540 | 2 / 2 | True | 1 / 671 |
| 2 | 4 | 33 | 33 | 0 | 0.014573 / 0.014573 | 7 / 7 | True | 1 / 153 |
| 2 | 5 | 31 | 31 | 0 | 0.014854 / 0.014854 | 0 / 0 | True | 1 / 2412 |
| 2 | 6 | 36 | 36 | 0 | 0.019181 / 0.019181 | 0 / 0 | True | 1 / 1 |
| 2 | 7 | 32 | 32 | 0 | 0.014929 / 0.014929 | 28 / 28 | True | 1 / 268 |
| 2 | 8 | 51 | 51 | 0 | 0.012681 / 0.012681 | 28 / 28 | True | 1 / 3939 |
| 2 | 9 | 28 | 28 | 0 | 0.016280 / 0.016280 | 25 / 25 | True | 1 / 459 |

### Retrospective consistency of every certified opening

For each gate opening at step `t`: the envelope in force at `t`, and the NEXT probe's observed drift rate `M_obs`. The opening is *retrospectively consistent* iff that following observation does not exceed the envelope the opening was certified under.

`cold-start openings` are openings that happen BEFORE the run's first `M_obs` exists (the first probe pair completes at step `2 * probe_every`), so the only envelope available to certify them is the unverified floor. `consistent under the FINAL envelope` re-checks each opening against the envelope the run ends with -- the one that is never below any diagnostic observed over the whole run.

| KAPPA | seed | openings | first / last open step | cold-start openings | with a following probe | consistent vs envelope in force | frac | consistent vs FINAL envelope | worst next-M_obs / envelope | envelope at first / last opening |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 8 | 1 / 8 | 8 | 8 | 0 | 0.0000 | 8 | 4.3535 | 5 / 5 |
| 1 | 1 | 6 | 1 / 6 | 6 | 6 | 0 | 0.0000 | 6 | 2.3459 | 5 / 5 |
| 1 | 2 | 12 | 1 / 12 | 12 | 12 | 0 | 0.0000 | 12 | 5.8048 | 5 / 5 |
| 1 | 3 | 6 | 1 / 6 | 6 | 6 | 0 | 0.0000 | 6 | 1.6082 | 5 / 5 |
| 1 | 4 | 8 | 1 / 8 | 8 | 8 | 0 | 0.0000 | 8 | 4.6807 | 5 / 5 |
| 1 | 5 | 7 | 1 / 7 | 7 | 7 | 0 | 0.0000 | 7 | 3.1619 | 5 / 5 |
| 1 | 6 | 9 | 1 / 9 | 9 | 9 | 0 | 0.0000 | 9 | 1.1858 | 5 / 5 |
| 1 | 7 | 8 | 1 / 8 | 8 | 8 | 0 | 0.0000 | 8 | 3.0251 | 5 / 5 |
| 1 | 8 | 13 | 1 / 15 | 13 | 13 | 0 | 0.0000 | 13 | 8.8076 | 5 / 5 |
| 1 | 9 | 7 | 1 / 7 | 7 | 7 | 0 | 0.0000 | 7 | 2.9853 | 5 / 5 |
| 2 | 0 | 8 | 1 / 8 | 8 | 8 | 0 | 0.0000 | 8 | 4.3535 | 5 / 5 |
| 2 | 1 | 6 | 1 / 6 | 6 | 6 | 0 | 0.0000 | 6 | 2.3459 | 5 / 5 |
| 2 | 2 | 12 | 1 / 12 | 12 | 12 | 0 | 0.0000 | 12 | 5.8048 | 5 / 5 |
| 2 | 3 | 6 | 1 / 6 | 6 | 6 | 0 | 0.0000 | 6 | 1.6082 | 5 / 5 |
| 2 | 4 | 8 | 1 / 8 | 8 | 8 | 0 | 0.0000 | 8 | 4.6807 | 5 / 5 |
| 2 | 5 | 7 | 1 / 7 | 7 | 7 | 0 | 0.0000 | 7 | 3.1619 | 5 / 5 |
| 2 | 6 | 9 | 1 / 9 | 9 | 9 | 0 | 0.0000 | 9 | 1.1858 | 5 / 5 |
| 2 | 7 | 8 | 1 / 8 | 8 | 8 | 0 | 0.0000 | 8 | 3.0251 | 5 / 5 |
| 2 | 8 | 13 | 1 / 15 | 13 | 13 | 0 | 0.0000 | 13 | 8.8076 | 5 / 5 |
| 2 | 9 | 7 | 1 / 7 | 7 | 7 | 0 | 0.0000 | 7 | 2.9853 | 5 / 5 |

### Envelope trajectory (seed 0)

KAPPA=1: 601 probes, 600 of them yielding an `M_obs`; envelope 21.7673 at the end (floor 5.0); 1 raises.

| probe step | envelope before | M_obs | envelope after | closed |
|---|---|---|---|---|
| 0 | 5 | - | 5 | 0 |
| 20 | 5 | 21.767 | 21.767 | 1 |
| 21 | 21.767 | 3.0437 | 21.767 | 0 |
| 40 | 21.767 | 1.242 | 21.767 | 0 |
| 60 | 21.767 | 1.7823 | 21.767 | 0 |
| 80 | 21.767 | 0.49505 | 21.767 | 0 |
| 100 | 21.767 | 0.57476 | 21.767 | 0 |
| 120 | 21.767 | 0.28824 | 21.767 | 0 |
| 140 | 21.767 | 0.019641 | 21.767 | 0 |
| 160 | 21.767 | 0.094808 | 21.767 | 0 |
| 180 | 21.767 | 0.026136 | 21.767 | 0 |
| 200 | 21.767 | 0.11704 | 21.767 | 0 |
| 220 | 21.767 | 1.2312 | 21.767 | 0 |

KAPPA=2: 601 probes, 600 of them yielding an `M_obs`; envelope 43.5346 at the end (floor 5.0); 1 raises.

| probe step | envelope before | M_obs | envelope after | closed |
|---|---|---|---|---|
| 0 | 5 | - | 5 | 0 |
| 20 | 5 | 21.767 | 43.535 | 1 |
| 21 | 43.535 | 3.0437 | 43.535 | 0 |
| 40 | 43.535 | 1.242 | 43.535 | 0 |
| 60 | 43.535 | 1.7823 | 43.535 | 0 |
| 80 | 43.535 | 0.49505 | 43.535 | 0 |
| 100 | 43.535 | 0.57476 | 43.535 | 0 |
| 120 | 43.535 | 0.28824 | 43.535 | 0 |
| 140 | 43.535 | 0.019641 | 43.535 | 0 |
| 160 | 43.535 | 0.094808 | 43.535 | 0 |
| 180 | 43.535 | 0.026136 | 43.535 | 0 |
| 200 | 43.535 | 0.11704 | 43.535 | 0 |
| 220 | 43.535 | 1.2312 | 43.535 | 0 |

### Paired tests vs the fixed-prior fail-closed arm (exact sign-flip)

| KAPPA | d NMSE | p | d events | p | d coord-open rate | d HVPs |
|---|---|---|---|---|---|---|
| 1 | +0.000000 | 1.0000 | +0.0 | 1.0000 | +0.000e+00 | -7786 |
| 2 | +0.000000 | 1.0000 | +0.0 | 1.0000 | +0.000e+00 | -7862 |

## E1 -- teacher/student `kw_drift`, EXACT ground truth, seeds 0-4

A violation here is a genuine failure of the anytime certificate: `e_t < ||S_t - Shat_t||_F` on any step, checked against a parallel exact forward-mode recursion in fp64.

| arm | n | violation rate | worst true-err / bound | valid rate | closure frac | probes / nominal | KW HVPs | final e_t | final envelope | raises | max M_obs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed prior M_H* = 2.27609, no FC | 5 | 0.0000 | 0.7095 | 1.0000 | nan | 1.0000 | 16476 | 0.6082 | nan | nan | nan |
| fixed prior M_H* = 2.27609, fail-closed | 5 | 0.0000 | 0.7095 | 1.0000 | 0.0490 | 1.0540 | 17470 | 0.5969 | nan | nan | nan |
| **adaptive envelope KAPPA=1** | 5 | 0.0000 | 0.7095 | 1.0000 | 0.0090 | 1.0100 | 16660 | 0.9073 | 9.8366 | 1.0 | 9.8366 |
| **adaptive envelope KAPPA=2** | 5 | 0.0000 | 0.7095 | 1.0000 | 0.0090 | 1.0100 | 16660 | 2.162 | 19.673 | 2.0 | 9.8366 |

### E1 envelope trajectory (seed 0)

KAPPA=1: 101 probes; envelope 12.9106 at the end (floor 2.27609); 1 raises; max M_obs 12.9106; closure fraction 0.0090.

| probe step | envelope before | M_obs | envelope after | closed |
|---|---|---|---|---|
| 0 | 2.2761 | - | 2.2761 | 0 |
| 10 | 2.2761 | 0.369 | 2.2761 | 0 |
| 20 | 2.2761 | 0.52447 | 2.2761 | 0 |
| 30 | 2.2761 | 0.22076 | 2.2761 | 0 |
| 40 | 2.2761 | 0.38338 | 2.2761 | 0 |
| 50 | 2.2761 | 0.2957 | 2.2761 | 0 |
| 60 | 2.2761 | 0.13042 | 2.2761 | 0 |
| 70 | 2.2761 | 0.92936 | 2.2761 | 0 |
| 80 | 2.2761 | 0.094795 | 2.2761 | 0 |
| 90 | 2.2761 | 1.426 | 2.2761 | 0 |
| 91 | 2.2761 | 12.911 | 12.911 | 1 |

KAPPA=2: 101 probes; envelope 25.8211 at the end (floor 2.27609); 2 raises; max M_obs 12.9106; closure fraction 0.0090.

| probe step | envelope before | M_obs | envelope after | closed |
|---|---|---|---|---|
| 0 | 2.2761 | - | 2.2761 | 0 |
| 10 | 2.2761 | 0.369 | 2.2761 | 0 |
| 20 | 2.2761 | 0.52447 | 2.2761 | 0 |
| 30 | 2.2761 | 0.22076 | 2.2761 | 0 |
| 40 | 2.2761 | 0.38338 | 2.2761 | 0 |
| 50 | 2.2761 | 0.2957 | 2.2761 | 0 |
| 60 | 2.2761 | 0.13042 | 2.2761 | 0 |
| 70 | 2.2761 | 0.92936 | 2.2761 | 0 |
| 80 | 2.2761 | 0.094795 | 2.2761 | 0 |
| 90 | 2.2761 | 1.426 | 2.8519 | 0 |
| 91 | 2.8519 | 12.911 | 25.821 | 1 |

## Design notes

* The envelope is re-stated at **every probe**, and the floor is the **deployed prior**, so the arm is never less conservative than the shipped certificate: `M_H,t = max(M_H_floor, KAPPA * max_{s<=t} M_obs,s)` with `M_H_floor = 5.0` for E2 (the paper's E2 prior) and `2.2760914236726824` for E1 (the calibrated `M_H*` of the teacher/kw_drift stream, i.e. the value round-3 B4 already deploys at factor 1). With `KAPPA >= 1` the envelope in force after any probe is by construction >= every `M_obs` the run has produced.
* The fail-closed monitor is evaluated against the envelope in force over the interval **just traversed**, i.e. BEFORE the new observation is folded in. Raising the envelope afterwards therefore cannot retroactively excuse an interval whose extrapolation was not justified; it only makes every subsequent interval conservative with respect to everything seen so far. Without that ordering the monitor would be vacuous at `KAPPA >= 1`.
* `--adaptive-mh` implies `--fail-closed`. E1 has no controller and therefore no gate, so for E1 the reported costs are closure fraction and probe overhead; openings are an E2-only quantity.

## Plain answers

**Can the deployed certificate be stated under an envelope that is never below any observed diagnostic, at what cost in openings and probes?**

**Yes -- at zero cost in openings and a NEGATIVE cost in probes. The catch is that under the strictest retrospective reading the openings that matter were never covered by the deployed prior either.**

**1. Validity is untouched (the hard requirement).** Zero violations everywhere:
* E1, exact fp64 ground truth (`e_t < ||S_t - Shat_t||_F` on any step): violation rate **0.0000** on all 5 seeds at both KAPPA, valid rate 1.0000, and the worst true-error / bound ratio is **0.7095** -- *exactly* the fixed-prior value. The bound is not approached within 29% in any arm.
* E2, per-coordinate audit against a parallel exact discounted FMD: **0 violations in 720000 checked coordinate-steps** at each KAPPA, worst `|ghat_j - g_true_j| / beta_col_j` = **0.8374**, identical to the fixed-prior arms.

**2. No openings are lost.** On E2 the adaptive-envelope runs are **bit-identical to the fixed-prior arm**: the full 12000-entry loss trace matches element for element on all 10 seeds at both KAPPA, `coord_open_frac` is 4.722e-04 in every arm, per-seed open coordinate-steps are 25-51 and **identical**, openings lost = **0**, NMSE and events identical to six decimals (paired d = 0.000000, p = 1.0000 on both axes). The reason is the round-2/3 mechanism: every opening happens in steps 1-15, long before the monitor has anything to say.

**3. The probe budget goes DOWN, not up.** The adaptive envelope is *less* trigger-happy than the fixed prior because the prior is too small:
* E2 closure fraction **0.0783 -> 0.0007** (KAPPA=1) / **0.0001** (KAPPA=2); closed steps 939.7 -> 8.6 / 1.0; HVPs 102602 -> 94816 / 94740, i.e. probe overhead over the no-monitor baseline **1.085x -> 1.002x**, a saving of 7786 / 7862 HVPs per run.
* E1 closure fraction **0.0490 -> 0.0090**; probes/nominal **1.054 -> 1.010**; KW HVPs 17470 -> 16660. The single closure per seed is the probe at which the envelope is raised.

**4. What the envelope reveals: the deployed priors are 4-6x too small.** The envelope is set once, at the first probe pair, and then essentially never moves (1.4 raises per run at KAPPA=1 on E2, 1.0 on E1):
* E2 ends at **29.92** (KAPPA=1) and **39.74** (KAPPA=2) against the deployed floor **5.0**; the largest observed `M_obs` is 29.92 / 19.87. (The two KAPPA differ in observed max because the forced-re-probe rule triggers at `M_obs > M_H/2`, so KAPPA=1 probes slightly more often and sees a slightly larger maximum: 94816 vs 94740 HVPs.)
* E1 ends at **9.84** (KAPPA=1) and **19.67** (KAPPA=2) against the floor **2.27609**, with max `M_obs` = 9.84 -- so even E1's *calibrated* `M_H*`, fitted on three held-out seeds, is exceeded by a factor **4.3** on the evaluation seeds. Seed 0's trajectory is typical: 100 probes with `M_obs` between 0.019 and 1.43, then a single 12.911 at probe step 91 which raises the envelope and closes the gate for that step.
* This is consistent with round-3 B4 and sharpens it: `M_H` is not what binds the certificate (worst ratio 0.71-0.84 regardless), so under-specifying it by 4-6x costs nothing in validity -- but the fixed-prior fail-closed monitor pays for the under-specification with 939.7 spurious closed steps per E2 run, and the adaptive envelope removes them.

**5. The cost, stated honestly: retrospective consistency is 0 of 84 openings per KAPPA on E2.** For every gate opening, "the envelope in force at that step" versus "the next probe's `M_obs`":
* All openings occur at steps 1-15, and the run's first `M_obs` only exists at probe step 20. So **100% of openings (84 of 84 open steps per KAPPA, 6-13 per seed) are cold-start openings**, certified under the unverified floor 5.0 with no diagnostic behind it.
* The very first observation then exceeds that floor on **all ten seeds**, by a factor **1.19x to 8.81x**. Under the strictest reading -- an opening is admissible only if the next observed drift rate does not exceed the envelope it was certified under -- **0 of 84 openings are admissible at either KAPPA**, and COHG would reduce to the fixed-LR arm.
* Under the run's FINAL envelope -- the one that is never below any diagnostic the run observed -- **84 of 84 openings (100%) are consistent**, at both KAPPA.

So the deployed certificate *can* be restated online in a form that is never below any observed diagnostic; doing so is free in decisions, saves 8% of the E2 probe budget and 5% of E1's, and never produces a violation against exact ground truth. What it cannot do is retro-certify the cold-start window, because on this configuration every certified update happens before the first drift diagnostic exists. Two honest remedies, neither of which this experiment ran: hold the gate shut until the first `M_obs` (which on this config would remove ALL of COHG's adaptation, since the burst is over by step 15), or state the floor as a calibrated `M_H*` from held-out seeds -- and the E1 numbers show that even a calibrated floor is exceeded 4.3x on unseen seeds, so the floor would have to carry its own margin. The defensible claim for the paper is therefore: *the drift envelope is an online-auditable quantity, the audit is cheap and never fires a false violation, and the fixed prior it replaces is measurably (4-6x) too small -- but the certificate's cold start is covered by the prior alone, and that is an assumption, not a measurement.*
