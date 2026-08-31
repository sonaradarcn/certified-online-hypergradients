# E2 controls: certification vs generic conservatism

Config (identical for every arm): dataset `mackey_drift`, mis-set init `lr0=0.003` (10x too low), 12000 steps, gamma 0.9, kw-eps 0.1, probe-every 20, K 10, rank 4, gate factor c=2, seeds 0-9.
Device: CPU (13k-param GRU; HVPs are launch-bound, so a CPU core beats a 3080 here). Cross-device check against the GPU E2 run below.

Arms found: 18 | runs: 156

## Reference points

- COHG on GPU (`results/e2`): NMSE 0.0150+-0.0023, events 9.0+-12.6, coord-open 5.03e-04 (n=10)
- COHG on CPU (this study, arm `cohg_lr0.003_mh5_fc0`): NMSE 0.0162+-0.0021, events 9.0+-12.6, coord-open 4.72e-04 (n=10)

### Fixed-LR family on the same stream (`results/e2`, GPU) -- the NMSE/instability frontier the adaptive arms are judged against

| fixed lr | n | NMSE mean+-std | events mean+-std |
|---|---|---|---|
| 0.003  <- mis-set init used by every adaptive arm | 10 | 0.0522+-0.0022 | 0.0+-0.0 |
| 0.01 | 10 | 0.0209+-0.0008 | 8.5+-12.8 |
| 0.03 | 10 | 0.0100+-0.0003 | 14.1+-19.5 |
| 0.1 | 10 | 0.0043+-0.0002 | 39.3+-31.1 |
| 0.3 | 10 | 0.0020+-0.0002 | 88.0+-44.9 |

## Per-arm results

`lambda window` = median first-to-last sampled step (every 50) at which lambda actually moved -- i.e. WHEN the arm does its adaptation.

| arm | n | NMSE mean+-std | events mean+-std | coord-open rate | steps-with-open | lambda window | HVPs | wall (s) |
|---|---|---|---|---|---|---|---|---|
| **COHG reference (M_H=5, no fail-closed, cert-validated)** | | | | | | | | |
| COHG (alpha=0.4) | 10 | 0.0162+-0.0021 | 9.0+-12.6 | 4.72e-04 | 7.00e-04 | 50-50 | 94588 | 5102 |
| **A. ungated pure-SIGN sweep (no certificate at all)** | | | | | | | | |
| sign-nogate alpha=0.1 | 10 | 0.0056+-0.0015 | 98.1+-40.4 | 0.00e+00 | 1.00e+00 | 50-11950 | 11992 | 712 |
| sign-nogate alpha=0.2 | 10 | 0.0040+-0.0011 | 104.1+-38.8 | 0.00e+00 | 1.00e+00 | 50-11950 | 11992 | 683 |
| sign-nogate alpha=0.4 | 10 | 0.0031+-0.0011 | 105.0+-48.2 | 0.00e+00 | 1.00e+00 | 50-11950 | 11992 | 665 |
| sign-nogate alpha=0.8 | 10 | 0.0057+-0.0061 | 91.9+-43.1 | 0.00e+00 | 1.00e+00 | 50-11950 | 11992 | 837 |
| **B/C/D. matched-conservatism gates (same estimator, same alpha=0.4 sign step, open rate matched to COHG)** | | | | | | | | |
| B absgate |ghat|>thr | 10 | 0.0150+-0.0032 | 10.4+-13.2 | 2.61e-04 | 1.27e-03 | 100-150 | 11992 | 609 |
| C randgate i.i.d. | 10 | 0.0410+-0.0039 | 1.5+-2.9 | 5.31e-04 | 3.18e-03 | 325-11675 | 11992 | 766 |
| D periodicgate | 10 | 0.0453+-0.0018 | 0.7+-1.9 | 5.83e-04 | 5.83e-04 | 2000-11950 | 11992 | 759 |
| **E. Theorem-6 step condition enforced online** | | | | | | | | |
| t6clip rho_f=1 | 10 | 0.0507+-0.0017 | 0.0+-0.0 | 4.25e-04 | 7.00e-04 | 50-50 | 94588 | 4405 |
| t6clip rho_f=10 | 10 | 0.0520+-0.0021 | 0.0+-0.0 | 4.24e-04 | 7.00e-04 | 50-50 | 94588 | 4102 |
| **F. M_H misspecification x fail-closed** | | | | | | | | |
| COHG M_H=5 no FC | 10 | 0.0162+-0.0021 | 9.0+-12.6 | 4.72e-04 | 7.00e-04 | 50-50 | 94588 | 5102 |
| COHG M_H=5 fail-closed | 10 | 0.0162+-0.0021 | 9.0+-12.6 | 4.72e-04 | 7.00e-04 | 50-50 | 102602 | 5404 |
| COHG M_H=0.5 no FC | 10 | 0.0156+-0.0019 | 9.1+-12.6 | 4.81e-04 | 7.17e-04 | 50-50 | 94588 | 5173 |
| COHG M_H=0.5 fail-closed | 10 | 0.0156+-0.0019 | 9.1+-12.6 | 4.81e-04 | 7.17e-04 | 50-50 | 150869 | 7508 |
| COHG M_H=0.05 no FC | 10 | 0.0156+-0.0019 | 9.1+-12.6 | 4.81e-04 | 7.17e-04 | 50-50 | 94588 | 5403 |
| COHG M_H=0.05 fail-closed | 10 | 0.0156+-0.0019 | 9.1+-12.6 | 4.81e-04 | 7.17e-04 | 50-50 | 181564 | 8698 |

### Mechanism: adaptation is a burst, not a rate

The `lambda window` column is the single most informative number here. COHG moves lambda ONLY inside the first ~50 steps -- while the certificate e_col is still near zero and the hypergradient sign is therefore unambiguous -- lifting the six per-group LRs from the mis-set 0.003 to 0.0019-0.042, and then the gate stays shut for the remaining 11950 steps. Its ~35 open coordinate-steps are not spread thinly at a 5e-4 rate; they are concentrated exactly where they are certifiable. absgate reproduces that burst (window 100-150) and therefore reproduces the result. randgate (325-11675) and periodicgate (2000-11950) spend the same open budget uniformly over the stream, so the LR is wrong for most of the run and they land near the un-adapted fixed-0.003 level. t6clip opens in the same window as COHG but its steps are capped ~7x smaller, so the LRs never leave 0.003.


## Paired comparisons vs the COHG reference arm
(exact sign-flip permutation test on per-seed differences; negative delta = the control is BETTER)

| arm | delta NMSE | p | delta events | p |
|---|---|---|---|---|
| sign-nogate alpha=0.1 | -0.0106 | 0.0020 | +89.1 | 0.0020 |
| sign-nogate alpha=0.2 | -0.0122 | 0.0020 | +95.1 | 0.0020 |
| sign-nogate alpha=0.4 | -0.0131 | 0.0020 | +96.0 | 0.0020 |
| sign-nogate alpha=0.8 | -0.0105 | 0.0039 | +82.9 | 0.0020 |
| B absgate |ghat|>thr | -0.0011 | 0.4004 | +1.4 | 0.4375 |
| C randgate i.i.d. | +0.0248 | 0.0020 | -7.5 | 0.0625 |
| D periodicgate | +0.0291 | 0.0020 | -8.3 | 0.0625 |
| t6clip rho_f=1 | +0.0346 | 0.0020 | -9.0 | 0.0625 |
| t6clip rho_f=10 | +0.0359 | 0.0020 | -9.0 | 0.0625 |
| COHG M_H=5 fail-closed | +0.0000 | 1.0000 | +0.0 | 1.0000 |
| COHG M_H=0.5 no FC | -0.0005 | 0.0020 | +0.1 | 1.0000 |
| COHG M_H=0.5 fail-closed | -0.0005 | 0.0020 | +0.1 | 1.0000 |
| COHG M_H=0.05 no FC | -0.0006 | 0.0020 | +0.1 | 1.0000 |
| COHG M_H=0.05 fail-closed | -0.0006 | 0.0020 | +0.1 | 1.0000 |

## F. Certificate audit and fail-closed detection

`cert_violations` counts per-coordinate events `|ghat_j - g_true_j| > beta_col_j` against a parallel EXACT discounted sensitivity (ExactFMD, m HVPs/step), with a 1e-4 relative fp32 tolerance. `cert ratio p99/max` is `|ghat_j - g_true_j| / beta_col_j` (1.0 = the bound is exactly tight).

| arm | n | cert viol / checked | viol steps | cert ratio p99 | cert ratio max | failclosed events | closed steps | re-probes | obs M_H p99 |
|---|---|---|---|---|---|---|---|---|---|
| M_H=5 no FC | 10 | 0 / 720000 | 0 | 0.17 | 0.705 | - | - | - | - |
| M_H=5 FC | 10 | 0 / 720000 | 0 | 0.17 | 0.705 | 45.6 | 939.7 | 103.8 | 79.1 |
| M_H=0.5 no FC | 10 | 0 / 720000 | 0 | 0.183 | 0.706 | - | - | - | - |
| M_H=0.5 FC | 10 | 0 / 720000 | 0 | 0.183 | 0.706 | 126.3 | 7085.4 | 726.1 | 131 |
| M_H=0.05 no FC | 10 | 0 / 720000 | 0 | 0.185 | 0.706 | - | - | - | - |
| M_H=0.05 FC | 10 | 0 / 720000 | 0 | 0.185 | 0.706 | 62.3 | 11120.3 | 1120.7 | 136 |

## E. Theorem-6 clipping detail

- rho_f=1: open-coordinate steps clipped by the Theorem-6 cap on average 30.6 of 30.6 open coordinate-steps
- rho_f=10: open-coordinate steps clipped by the Theorem-6 cap on average 30.5 of 30.5 open coordinate-steps

## B. absgate threshold calibration

- calibration runs (gate held shut, seeds [100, 101], disjoint from evaluation): the |ghat| order statistic matching COHG's open rate; refit once on the realized |ghat| distribution of the pass-1 threshold, then FROZEN
- frozen threshold: 0.05806520209
- verification seed 100: realized coord-open rate 4.444e-04
- verification seed 101: realized coord-open rate 4.028e-04

## A. reproduction check: the SAME arm on GPU (`results/e2`)

- GPU `cohg_nogate` lr0.003 alpha=0.4: NMSE 5.9849+-18.9168, events 128.9 (n=10)
- CPU `sign-nogate alpha=0.4` (this study): NMSE 0.0031+-0.0011, events 105.0 (n=10)

| seed | GPU NMSE | GPU events | CPU NMSE | CPU events |
|---|---|---|---|---|
| 0 | 0.0015 | 33 | 0.0015 | 33 |
| 1 | 0.0029 | 111 | 0.0029 | 111 |
| 2 | 0.0015 | 91 | 0.0015 | 91 |
| 3 | 0.0027 | 178 | 0.0027 | 178 |
| 4 | 0.0032 | 107 | 0.0032 | 108 |
| 5 | 59.8231 | 298 | 0.0048 | 53 |
| 6 | 0.0030 | 64 | 0.0030 | 64 |
| 7 | 0.0045 | 97 | 0.0045 | 97 |
| 8 | 0.0030 | 163 | 0.0033 | 168 |
| 9 | 0.0034 | 147 | 0.0034 | 147 |

The two devices agree seed-for-seed to fp32 round-off on nine of ten seeds; on seed 5 a ~1e-7 relative perturbation is enough to flip the run between NMSE 0.005 and NMSE 59.8. The ungated arm sits ON the divergence boundary: its mean NMSE is decided by which side of a float a single seed lands on. That tail, not the mean, is what the certificate gate is buying.

## Headline answers

**(1) Does a matched-conservatism control (B/C/D) reach COHG's events-vs-NMSE point?** B YES, C/D NO. absgate reaches NMSE 0.0150+-0.0032 at 10.4 events vs COHG 0.0162+-0.0021 at 9.0 (paired p=0.40 on NMSE, p=0.44 on events): statistically indistinguishable, at a LOWER open rate (2.61e-04 vs 4.72e-04). This is a real negative result for the 'certification is what buys the operating point' reading. randgate (0.0410) and periodicgate (0.0453) both collapse to roughly the un-adapted fixed-lr 0.003 level (0.0522), so matching the RATE of conservatism is not sufficient -- WHICH coordinates open, and when, is the whole mechanism. The surviving distinction for absgate is procedural, not performance: its threshold had to be fitted offline on held-out seeds (two passes) and still drifted 38% in realized rate from the calibration seeds to the evaluation seeds, whereas COHG's threshold is computed online from the run's own certificate.

**(2) Does tuned ungated SIGN dominate gated COHG?** No -- but it beats it on NMSE at every alpha. alpha=0.4 gives 0.0031+-0.0011 vs COHG's 0.0162 (p=0.002), i.e. 5x lower error, while paying 105+-48 instability events against COHG's 9+-13 (p=0.002). It is a clean Pareto trade, not domination, and the ungated NMSE is not robust: the identical arm on GPU has one seed at NMSE 59.8 (arm-A reproduction check above).

**(3) Does enforcing the Theorem-6 step condition preserve performance?** No. t6clip caps EVERY open step (100% clipped at both rho_f), collapsing NMSE to 0.0507 (rho_f=1) and 0.0520 (rho_f=10) with 0 events -- the controller becomes provably safe and effectively non-adaptive (final LRs stay at the mis-set 0.003). Because |ghat| ~ 0.06 at the gate, the cap 2(1-1/c)|ghat|/rho_f ~ 0.06 is ~7x smaller than alpha=0.4, so the theorem's condition and the practical step size are simply not compatible at this scale.

**(4) Does fail-closed catch a misspecified M_H?** Yes as a DETECTOR, but there was nothing to catch. The monitor's closed-gate fraction rises monotonically with misspecification (7.8% -> 59% -> 93% of steps at M_H = 5 -> 0.5 -> 0.05) and the probe budget with it (+8% -> +60% -> +92% HVPs), so a wrong prior is loudly visible online. But the parallel exact-sensitivity audit found ZERO certificate violations in all 720000 checked coordinate-steps of EVERY arm, including M_H 100x too small: the M_H drift term is not what binds e_col in this regime (max ratio 0.71, p99 ~0.18), so under-specifying it does not break the bound here. Fail-closed also changed no run's outcome (NMSE, events and open rate are bit-identical to the no-FC arms) because COHG does ALL of its adaptation in the first ~50 steps, before the monitor has two probes to compare.
