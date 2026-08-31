# B4. Drift-prior (M_H) misspecification curves

How far can the Hessian-drift prior `M_H` in assumption A4' be wrong before the anytime certificate stops holding, and does the fail-closed monitor notice?  Two streams, two ground truths.

| | E1 `teacher/kw_drift` | E2 `mackey_drift` GRU |
|---|---|---|
| model | 2-layer teacher-student, r=4, K=10, gamma=0.9, kw-eps=0.05 | 13k-param GRU, r=4, K=10, gamma=0.9, kw-eps=0.1, probe-every 20 |
| steps x seeds | 1000 x 5 (seeds 0-4) | 12000 x 10 (seeds 0-9) |
| ground truth | **exact** discounted `ExactFMD` sensitivity matrix `S_t`, fp64 | parallel exact discounted `ExactFMD` hypergradient, fp32 (1e-4 rel. tol.) |
| violation test | `e_t < ||S_t - Shat_t||_F` on ANY step | `|ghat_j - g_true_j| > beta_col_j` on ANY coordinate-step |
| ratio plotted | `||S_t - Shat_t||_F / e_t` | `|ghat_j - g_true_j| / beta_col_j` |
| lambda | frozen (certificate study) | live (the controller runs) |

## Definitions

Three quantities, kept apart because the earlier draft ran them together.

* **`M_obs`, the observed probe-to-probe drift rate.** `M_obs = |rho_probe - rho_prev| / (eta_max_t0 * D)`, where `D` is the parameter path length between the two probes and `eta_max_t0` the largest learning rate at the earlier probe (`DriftHoldFailClosed.probe`, `code/cohg/certificate.py`).  It is recorded ONLY by the fail-closed monitor.  The base `DriftHold` class stores a different field, `mh_observed = |Delta rho| / D`, which omits the `eta_max` factor and therefore estimates `eta_max * M_H`, not `M_H`; E3 and E4 use the base class, so their stored diagnostic is NOT on the `M_H` scale and no margin may be quoted from it.
* **`M_H`, the DEPLOYED prior.** The value actually passed to the certificate in the runs being reported: 5 on E2, 20 on E3, 50 on E4 (Table `tab:certparams`), and 2.27609 at factor 1 of the E1 sweep.  This is the denominator of the figure's x-axis for BOTH series, so `0.01` on that axis means one hundredth of what was run.
* **`M_obs^med`, `M_obs^p99`, `M_obs^max`.** Median, 99th percentile and maximum of `M_obs` on the DEPLOYED fail-closed arm of a regime, over that regime's named seeds.  These are properties of the stream, not settings.

**E1.** The deployed prior 2.27609 was CALIBRATED as `M_obs^max` over 297 probes on 3 calibration seeds ([100, 101, 102]), disjoint from the 5 evaluation seeds and measured at an effectively infinite prior (`M_H = 1e12`) so the probe schedule is undistorted.  On those calibration seeds: median 0.2777, p90 0.9936, p99 1.907, max 2.276.  On the five EVALUATION seeds at that same prior the rate is heavier: median 0.323, p99 14.4, max 32.1, and 5.6% of probes exceed the prior.  A calibration max on held-out seeds does not upper-bound fresh seeds.

**E2.** The deployed prior is 5 (Table `tab:certparams`); no separate calibration arm exists.  On the deployed fail-closed arm itself, over 6517 probes on the ten evaluation seeds, `M_obs` has median 0.7694 (per-seed medians 0.0984 to 2.71), p90 13.9, p99 79.1 and max 761.543, the last being a single extreme draw.  The deployed prior is thus 6.5x the median observed rate and well below its upper tail.  `761.5` is that in-sample maximum; it is NOT a calibration target and NOT the same statistic as E1's 2.276, and normalising the two sweeps by their own maxima is what put them two decades apart on the old axis.

## E1 -- exact ground truth (teacher/kw_drift, 5 seeds x 1000 steps)

`worst ratio` = max over seeds and steps of true error / certificate (>= 1 would be a violation).  Per-step ratio traces were not kept (`--trace-config none`), so a p99 of the ratio is NOT available for E1; the two quantile columns are inverted tightness quantiles stored per run (`p75 ratio` = mean over seeds of `1/tight_q25`).

| M_H/M_H_dep | M_H | fail-closed | seeds | violating steps | violation rate | worst ratio | p75 ratio | median ratio | closed-gate frac | probes | probe overhead | KW HVPs | FC events | re-probes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.276 | no | 5 | 0 / 5000 | 0.000 | 0.709 | 0.259 | 0.245 | -- | 100.0 | 1.000x | 16476 | -- | -- |
| 1 | 2.276 | yes | 5 | 0 / 5000 | 0.000 | 0.709 | 0.260 | 0.247 | 0.049 | 105.4 | 1.054x | 17470 | 5.4 | 10.8 |
| 0.3 | 0.6828 | no | 5 | 0 / 5000 | 0.000 | 0.730 | 0.279 | 0.264 | -- | 100.0 | 1.000x | 16476 | -- | -- |
| 0.3 | 0.6828 | yes | 5 | 0 / 5000 | 0.000 | 0.730 | 0.281 | 0.267 | 0.385 | 145.6 | 1.456x | 24954 | 31.2 | 89.0 |
| 0.1 | 0.2276 | no | 5 | 0 / 5000 | 0.000 | 0.735 | 0.286 | 0.268 | -- | 100.0 | 1.000x | 16476 | -- | -- |
| 0.1 | 0.2276 | yes | 5 | 0 / 5000 | 0.000 | 0.735 | 0.286 | 0.271 | 0.758 | 180.0 | 1.800x | 31520 | 29.0 | 159.2 |
| 0.03 | 0.06828 | no | 5 | 0 / 5000 | 0.000 | 0.738 | 0.288 | 0.270 | -- | 100.0 | 1.000x | 16476 | -- | -- |
| 0.03 | 0.06828 | yes | 5 | 0 / 5000 | 0.000 | 0.738 | 0.288 | 0.272 | 0.919 | 192.6 | 1.926x | 33942 | 11.6 | 185.2 |
| 0.01 | 0.02276 | no | 5 | 0 / 5000 | 0.000 | 0.738 | 0.288 | 0.270 | -- | 100.0 | 1.000x | 16476 | -- | -- |
| 0.01 | 0.02276 | yes | 5 | 0 / 5000 | 0.000 | 0.738 | 0.289 | 0.273 | 0.967 | 196.8 | 1.968x | 34758 | 4.6 | 193.6 |

Monitor diagnostics on the fail-closed arms (M_obs pooled over the 5 evaluation seeds):

| M_H/M_H_dep | M_H | median M_obs | p99 M_obs | max M_obs | frac. probes with M_obs > M_H | closed-gate frac |
|---|---|---|---|---|---|---|
| 1 | 2.276 | 0.323 | 14.4 | 32.1 | 0.056 | 0.049 |
| 0.3 | 0.6828 | 0.516 | 15.4 | 35.3 | 0.416 | 0.385 |
| 0.1 | 0.2276 | 0.785 | 18.6 | 35.3 | 0.784 | 0.758 |
| 0.03 | 0.06828 | 0.839 | 18.9 | 35.3 | 0.937 | 0.919 |
| 0.01 | 0.02276 | 0.882 | 19.2 | 35.3 | 0.982 | 0.967 |

## E2 -- GRU stream (mackey_drift, 10 seeds x 12000 steps)

`probe overhead` here is the total HVP count relative to the M_H=5 no-fail-closed arm (94588 HVPs); in E1 it is the probe COUNT relative to the nominal budget.  Both are capped at 2x by the 'a forced re-probe never forces another one' rule.

`worst ratio` is the MAX OVER SEEDS of each run's max ratio. `results/e2_controls/SUMMARY.md` reports the same quantity as the MEAN over seeds of the per-run max (0.705 / 0.706 / 0.706); both are right, and the max-over-seeds used here and in the figure is the conservative one.  The E1 table above uses the same max-over-seeds convention (its mean-over-seeds values are 0.646 / 0.672 / 0.680 / 0.683 / 0.684).

| M_H/M_H_dep | M_H/median M_obs | M_H | fail-closed | seeds | violations | violation rate | worst ratio | p99 ratio | median ratio | closed-gate frac | HVPs | probe overhead | FC events | re-probes | NMSE | events |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6.5 | 5 | no | 10 | 0 / 720000 | 0.000 | 0.837 | 0.170 | 0.056 | -- | 94588 | 1.000x | -- | -- | 0.0162+-0.0021 | 9.0 |
| 1 | 6.5 | 5 | yes | 10 | 0 / 720000 | 0.000 | 0.837 | 0.170 | 0.056 | 0.078 | 102602 | 1.085x | 45.6 | 103.8 | 0.0162+-0.0021 | 9.0 |
| 0.1 | 0.65 | 0.5 | no | 10 | 0 / 720000 | 0.000 | 0.838 | 0.183 | 0.054 | -- | 94588 | 1.000x | -- | -- | 0.0156+-0.0019 | 9.1 |
| 0.1 | 0.65 | 0.5 | yes | 10 | 0 / 720000 | 0.000 | 0.838 | 0.183 | 0.053 | 0.590 | 150869 | 1.595x | 126.3 | 726.1 | 0.0156+-0.0019 | 9.1 |
| 0.01 | 0.065 | 0.05 | no | 10 | 0 / 720000 | 0.000 | 0.838 | 0.185 | 0.053 | -- | 94588 | 1.000x | -- | -- | 0.0156+-0.0019 | 9.1 |
| 0.01 | 0.065 | 0.05 | yes | 10 | 0 / 720000 | 0.000 | 0.838 | 0.185 | 0.052 | 0.927 | 181564 | 1.920x | 62.3 | 1120.7 | 0.0156+-0.0019 | 9.1 |

## The two streams on a common axis

`M_H/M_H_dep` -- the prior relative to the value DEPLOYED in that regime -- is the figure axis and lines the two sweeps up directly.  `M_H / median M_obs`, the prior relative to the TYPICAL drift rate, which is what decides how often the monitor trips, agrees with it to within a factor of 1.3 at every matching point.  The old axis normalised each stream by the maximum rate on its own probe population, which is not the same statistic on the two streams and put them two decades apart for that reason alone.

| stream | M_H | M_H/M_H_dep | M_H/median M_obs | closed-gate frac | probe overhead | violations |
|---|---|---|---|---|---|---|
| E1 | 2.276 | 1 | 8.2 | 0.049 | 1.054x | 0 |
| E1 | 0.6828 | 0.3 | 2.46 | 0.385 | 1.456x | 0 |
| E1 | 0.2276 | 0.1 | 0.82 | 0.758 | 1.800x | 0 |
| E1 | 0.06828 | 0.03 | 0.246 | 0.919 | 1.926x | 0 |
| E1 | 0.02276 | 0.01 | 0.082 | 0.967 | 1.968x | 0 |
| E2 | 5 | 1 | 6.5 | 0.078 | 1.085x | 0 |
| E2 | 0.5 | 0.1 | 0.65 | 0.590 | 1.595x | 0 |
| E2 | 0.05 | 0.01 | 0.065 | 0.927 | 1.920x | 0 |

## Plain reading

**Does a too-small M_H ever produce a certificate violation on E1, where the ground truth is exact?  No -- not at any factor tested, down to 100x too small.** Across all 10 E1 points (5 M_H factors x fail-closed on/off, 5 seeds x 1000 steps each = 50000 certified steps), `valid_rate` is exactly 1.000 and `n_violations` is exactly 0. There is no crossover factor to report: the sweep bottoms out at M_H = 0.02276 (1/100 of the deployed 2.276) with the bound still holding on every step of every seed.

**And the margin barely moves.** The worst true-error/bound ratio rises monotonically but saturates: 0.709 -> 0.730 -> 0.735 -> 0.738 -> 0.738 as M_H shrinks by 100x, i.e. a 4% widening of the worst case and a convergent limit ~0.74 that is reached by factor 0.03. Removing the drift term entirely would therefore still leave ~26% headroom.  That is the diagnosis: in this regime the certificate is NOT bound by the A4' drift term at all -- it is bound by the rank-r truncation and the KW probe tolerance, both of which are independent of M_H.  The E2 audit says the same thing at 100x misspecification (max ratio 0.838, p99 ~0.18, 0 / 4.32e6 coordinate-step checks violated).

**Does the fail-closed monitor catch it?  Yes, loudly and monotonically -- it is a working detector of a condition that did not actually break anything here.** The closed-gate fraction on E1 runs 0.049 -> 0.385 -> 0.758 -> 0.919 -> 0.967 at factors 1 / 0.3 / 0.1 / 0.03 / 0.01, and E2 gives the same monotone shape ([0.078, 0.59, 0.927] at M_H = 5 / 0.5 / 0.05). Closure tracks the fraction of probes whose measured M_obs exceeds the prior almost exactly on E1 (0.056 / 0.416 / 0.784 / 0.937 / 0.982 vs the closure fractions above), so the signal is exactly what it claims to be. At a 100x-too-small prior the monitor is shut ~97% of the time on E1 and ~93% on E2: a misspecified drift prior is impossible to miss online, without any oracle.

**Even the DEPLOYED prior trips the monitor.** At factor 1 the E1 monitor still closes 4.9% of steps, because 5.6% of the evaluation seeds' probes measure M_obs above the deployed prior (up to 32.1, i.e. 14x it) -- the calibration max over 3 held-out seeds does not upper-bound the evaluation seeds. The certificate held anyway on every one of those steps, which is the concrete evidence that `M_obs > M_H` is a CONSERVATIVE trigger and not a certificate failure: M_obs divides a probe-to-probe rho change by a short path length and so over-reads the true Hessian-Lipschitz constant.  A practitioner should read a rising closed-gate fraction as 'recalibrate M_H', not as 'the bound just broke'.

**Cost.** The monitor is bounded by construction (a forced re-probe cannot force another), and the measurements sit under that cap: E1 probe overhead 1.054x -> 1.456x -> 1.800x -> 1.926x -> 1.968x, E2 HVP overhead [1.085, 1.595, 1.92]x at M_H = 5 / 0.5 / 0.05 (wall time 5.4ks -> 7.5ks -> 8.7ks against a 5.1ks baseline). So the price of the safety net is at most a 2x probe budget, paid only in proportion to how wrong the prior is, and nothing at all when the prior is right.  Turning the monitor ON never changes an E1 certificate outcome (violation rate and worst ratio are identical to the fc0 column at every factor) and never changes an E2 run outcome (NMSE, events and open rate are bit-identical to the fc0 arms).

**Caveats to state in the paper.** (i) Zero violations at 100x misspecification is evidence that the A4' term is slack in THESE two regimes, not that A4' is unnecessary; a stream where the drift term dominates the rank-truncation term would behave differently, and we did not construct one. (ii) E1 sweeps M_H DOWNWARD only -- an over-large M_H cannot break the bound (it only loosens it), which is why the sweep is one-sided, but that also means the figure says nothing about the cost of over-conservatism beyond factor 1. (iii) E1's lambda is frozen, so the misspecification cannot feed back into the controller; E2 supplies that half and shows the feedback is nil, because COHG finishes adapting in the first ~50 steps, before the monitor has two probes to compare. (iv) The deployed prior is NOT an upper bound on the observed rate in either regime: it exceeds the median by 6.5x on E2 and 7.0x on E1, and sits below the p99 (79.1 on E2, 14.4 on E1) and the max (761.5, 32.1).  The audit is the evidence we have that this did not matter here; it is not a demonstration that A4' held. (v) E3 and E4 record only the base-class diagnostic |Delta rho| / D, which is off the M_H scale by a factor of eta_max, so no margin can be quoted for those regimes at all.  On the GPT-2 runs that diagnostic has median 0.094 and max 1.80 over 319 stored probe windows at a frozen eta_max of 1.49e-3, which rescales to M_obs median 63 and max 1208 against a prior of 50.

Figure: `paper/main/figs/fig10_misspec.pdf` (preview `results/figures/fig10_misspec.png`), generated by `code/experiments/make_fig10_misspec.py`.
