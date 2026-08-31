# Round-4 follow-up: a DENSE EARLY PROBE, so a measurement exists before the adaptation window

`round4_adaptmh.md` and `round4_warmup.md` established the timing trap on E2 `mackey_drift`: every certified COHG gate opening happens at steps 1-15, while the first probe-to-probe drift observation `M_obs` cannot exist before step `2 * probe_every` = 20. Holding the gate until an observation exists therefore removes *all* adaptation. This study runs remedy (b) of `round4_warmup.md` section 9: make an observation exist EARLY.

`--probe-dense-until T` (new flag in `code/experiments/e2_timeseries.py`; default `0` = off = the legacy bit-identical path) fires the KW spectral probe at **every** step for `t <= T`, in addition to the `--probe-every` cadence, and reverts to `--probe-every` afterwards. Every step, not every 2 steps: on this config a probe costs ~150 HVPs and the dense window adds only `T - 1` = 19 of them, i.e. ~3% of the run's budget (measured below), so there was no reason to halve the resolution. With `T = 20` the first `M_obs` lands at **step 1**, before the first gate opening.

Arms (all `mackey_drift`, GRU, 12000 steps, lr0 0.003, alpha 0.4, c = 2, K 10, rank 4, gamma 0.9, probe-every 20, seeds 0-9, CPU, `--validate-cert`):

* **(i)** `cohg --adaptive-mh 1 --probe-dense-until 20` -- the measured envelope, gate free to open from step 1.
* **(ii)** (i) `+ --gate-warmup first-obs` -- the gate may only open after an `M_obs` exists, which now happens at step 1.
* **(iii)** `cohg --M-H 5 --fail-closed --probe-dense-until 20` -- what the dense schedule does to the FIXED-prior monitor.

References: `results/e2_adaptmh` (adaptive envelope, 20-step cadence, no warm-up), `results/e2_warmup` (warm-up `first-obs` on the 20-step cadence: 0 openings), `results/e2_controls` (fixed prior). Statistics are mean+-sd with ddof = 1; paired tests are exact sign-flip over the 10 seeds.

## 0. Default-path regression (`--probe-dense-until 0`)

| seed | losses identical | lam_hist identical | NMSE (new / e2_verify4) | events | HVPs | cert viol |
|---|---|---|---|---|---|---|
| 0 | True | True | 0.0148824551 / 0.0148824551 | 0 / 0 | 94588 / 94588 | 0 / 0 |
| 1 | True | True | 0.0184537453 / 0.0184537453 | 0 / 0 | 94588 / 94588 | 0 / 0 |

## 1. Arm summary

`open coord-steps` = `coord_open_frac * 6 * 12000` (6 parameter groups). `held` = steps the warm-up hold kept the gate shut.

| arm | n | NMSE | events | open coord-steps | coord-open rate | opening steps (first / last) | held | closed steps | probes | HVPs | HVPs / no-FC | cert viol / checked | cert max ratio | final envelope |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed prior M_H=5, no FC (reference) | 10 | 0.0162+-0.0021 | 9.0+-12.6 | 34.0+-8.2 | 4.722e-04 | n/a (not logged) | - | - | - | 94588 | 1.000 | 0 / 720000 | 0.8374 | 5 |
| fixed prior M_H=5, fail-closed (reference) | 10 | 0.0162+-0.0021 | 9.0+-12.6 | 34.0+-8.2 | 4.722e-04 | n/a (not logged) | - | 939.7 | - | 102602 | 1.085 | 0 / 720000 | 0.8374 | 5 |
| adaptive KAPPA=1, probe-every 20, no warm-up | 10 | 0.0162+-0.0021 | 9.0+-12.6 | 34.0+-8.2 | 4.722e-04 | 1.0 / 8.6 | - | 8.6 | 601.5 | 94816 | 1.002 | 0 / 720000 | 0.8374 | 29.92 |
| adaptive KAPPA=1 + warm-up first-obs (probe-every 20) | 10 | 0.0522+-0.0022 | 0.0+-0.0 | 0.0+-0.0 | 0.000e+00 | never opens | 20.0 | 25.7 | 601.6 | 94831 | 1.003 | 0 / 720000 | 0.8241 | 212.7 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 10 | 0.0250+-0.0079 | 3.4+-8.7 | 21.6+-11.2 | 3.000e-04 | 2.9 / 9.4 | 0.0 | 2.6 | 619.0 | 97476 | 1.031 | 0 / 720000 | 0.8215 | 2014 |
| **(ii) (i) + warm-up first-obs** | 10 | 0.0250+-0.0079 | 3.4+-8.7 | 21.6+-11.2 | 3.000e-04 | 2.9 / 9.4 | 1.0 | 2.6 | 619.0 | 97476 | 1.031 | 0 / 720000 | 0.8215 | 2014 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 10 | 0.0503+-0.0044 | 0.0+-0.0 | 1.0+-2.0 | 1.389e-05 | 6.3 / 6.3 | 0.0 | 9402.2 | 1103.3 | 172312 | 1.822 | 0 / 720000 | 0.8215 | 5 |

## 2. Openings kept: number and timing, per seed

The reference row block is the no-warm-up adaptive arm (`results/e2_adaptmh`); the warm-up arm on the 20-step cadence (`results/e2_warmup`) has 0 openings on every seed and is omitted from the per-seed listing.

| arm | seed | opening steps | open coord-steps | held steps | release step | closed steps | closure steps <= 20 |
|---|---|---|---|---|---|---|---|
| adaptive KAPPA=1, probe-every 20, no warm-up | 0 | 1, 2, 3, 4, 5, 6, 7, 8 | 33 | None | None | 1 | 1 |
| adaptive KAPPA=1, probe-every 20, no warm-up | 1 | 1, 2, 3, 4, 5, 6 | 26 | None | None | 20 | 1 |
| adaptive KAPPA=1, probe-every 20, no warm-up | 2 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 | 45 | None | None | 1 | 1 |
| adaptive KAPPA=1, probe-every 20, no warm-up | 3 | 1, 2, 3, 4, 5, 6 | 25 | None | None | 39 | 1 |
| adaptive KAPPA=1, probe-every 20, no warm-up | 4 | 1, 2, 3, 4, 5, 6, 7, 8 | 33 | None | None | 1 | 1 |
| adaptive KAPPA=1, probe-every 20, no warm-up | 5 | 1, 2, 3, 4, 5, 6, 7 | 31 | None | None | 20 | 1 |
| adaptive KAPPA=1, probe-every 20, no warm-up | 6 | 1, 2, 3, 4, 5, 6, 7, 8, 9 | 36 | None | None | 1 | 1 |
| adaptive KAPPA=1, probe-every 20, no warm-up | 7 | 1, 2, 3, 4, 5, 6, 7, 8 | 32 | None | None | 1 | 1 |
| adaptive KAPPA=1, probe-every 20, no warm-up | 8 | 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 | 51 | None | None | 1 | 1 |
| adaptive KAPPA=1, probe-every 20, no warm-up | 9 | 1, 2, 3, 4, 5, 6, 7 | 28 | None | None | 1 | 1 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 0 | 4, 5, 6, 7, 8, 9, 10 | 16 | 0 | None | 3 | 3 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 1 | 2, 4, 5, 6, 7 | 17 | 0 | None | 2 | 2 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 2 | 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 | 38 | 0 | None | 1 | 1 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 3 | 2, 4, 5, 6 | 15 | 0 | None | 2 | 2 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 4 | 2, 4, 5, 6, 7, 8, 9 | 22 | 0 | None | 2 | 2 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 5 | 4, 6, 7, 8 | 10 | 0 | None | 4 | 4 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 6 | 2, 4, 5, 6, 7, 8, 9, 10, 11 | 28 | 0 | None | 2 | 2 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 7 | 3, 5, 6, 7, 8, 9 | 15 | 0 | None | 3 | 3 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 8 | 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 | 43 | 0 | None | 4 | 4 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 9 | 3, 5, 6, 7 | 12 | 0 | None | 3 | 3 |
| **(ii) (i) + warm-up first-obs** | 0 | 4, 5, 6, 7, 8, 9, 10 | 16 | 1 | 1 | 3 | 3 |
| **(ii) (i) + warm-up first-obs** | 1 | 2, 4, 5, 6, 7 | 17 | 1 | 1 | 2 | 2 |
| **(ii) (i) + warm-up first-obs** | 2 | 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 | 38 | 1 | 1 | 1 | 1 |
| **(ii) (i) + warm-up first-obs** | 3 | 2, 4, 5, 6 | 15 | 1 | 1 | 2 | 2 |
| **(ii) (i) + warm-up first-obs** | 4 | 2, 4, 5, 6, 7, 8, 9 | 22 | 1 | 1 | 2 | 2 |
| **(ii) (i) + warm-up first-obs** | 5 | 4, 6, 7, 8 | 10 | 1 | 1 | 4 | 4 |
| **(ii) (i) + warm-up first-obs** | 6 | 2, 4, 5, 6, 7, 8, 9, 10, 11 | 28 | 1 | 1 | 2 | 2 |
| **(ii) (i) + warm-up first-obs** | 7 | 3, 5, 6, 7, 8, 9 | 15 | 1 | 1 | 3 | 3 |
| **(ii) (i) + warm-up first-obs** | 8 | 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 | 43 | 1 | 1 | 4 | 4 |
| **(ii) (i) + warm-up first-obs** | 9 | 3, 5, 6, 7 | 12 | 1 | 1 | 3 | 3 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 0 | (none) | 0 | 0 | None | 9575 | 19 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 1 | (none) | 0 | 0 | None | 9747 | 20 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 2 | 11 | 1 | 0 | None | 8985 | 19 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 3 | 2 | 6 | 0 | None | 8892 | 19 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 4 | (none) | 0 | 0 | None | 9826 | 20 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 5 | (none) | 0 | 0 | None | 9517 | 19 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 6 | (none) | 0 | 0 | None | 9525 | 19 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 7 | 6 | 3 | 0 | None | 8944 | 18 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 8 | (none) | 0 | 0 | None | 9243 | 20 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 9 | (none) | 0 | 0 | None | 9768 | 19 |

Openings kept relative to the no-warm-up adaptive arm, paired by seed (`open coord-steps`, and the set of opening STEPS):

| seed | (ref) no warm-up steps | (i) dense steps | (ii) dense+warmup steps | (iii) fixed-prior dense steps | coord-steps ref / (i) / (ii) / (iii) |
|---|---|---|---|---|---|
| 0 | 1, 2, 3, 4, 5, 6, 7, 8 | 4, 5, 6, 7, 8, 9, 10 | 4, 5, 6, 7, 8, 9, 10 | (none) | 33 / 16 / 16 / 0 |
| 1 | 1, 2, 3, 4, 5, 6 | 2, 4, 5, 6, 7 | 2, 4, 5, 6, 7 | (none) | 26 / 17 / 17 / 0 |
| 2 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 | 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 | 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 | 11 | 45 / 38 / 38 / 1 |
| 3 | 1, 2, 3, 4, 5, 6 | 2, 4, 5, 6 | 2, 4, 5, 6 | 2 | 25 / 15 / 15 / 6 |
| 4 | 1, 2, 3, 4, 5, 6, 7, 8 | 2, 4, 5, 6, 7, 8, 9 | 2, 4, 5, 6, 7, 8, 9 | (none) | 33 / 22 / 22 / 0 |
| 5 | 1, 2, 3, 4, 5, 6, 7 | 4, 6, 7, 8 | 4, 6, 7, 8 | (none) | 31 / 10 / 10 / 0 |
| 6 | 1, 2, 3, 4, 5, 6, 7, 8, 9 | 2, 4, 5, 6, 7, 8, 9, 10, 11 | 2, 4, 5, 6, 7, 8, 9, 10, 11 | (none) | 36 / 28 / 28 / 0 |
| 7 | 1, 2, 3, 4, 5, 6, 7, 8 | 3, 5, 6, 7, 8, 9 | 3, 5, 6, 7, 8, 9 | 6 | 32 / 15 / 15 / 3 |
| 8 | 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ... | 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 | 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 | (none) | 51 / 43 / 43 / 0 |
| 9 | 1, 2, 3, 4, 5, 6, 7 | 3, 5, 6, 7 | 3, 5, 6, 7 | (none) | 28 / 12 / 12 / 0 |

## 3. Direction and size of the lambda moves

`lam` is the per-group log learning rate, initialised at `log(0.003)` = -5.8091. `net dlam` is the mean over the 6 groups of `lam_final - lam_0` (positive = LR raised); `LR ratio` is `exp(net dlam)`.

| arm | net dlam | LR ratio | max dlam | min dlam | max abs dlam (traj) | groups moved | open coord-steps LR down / up |
|---|---|---|---|---|---|---|---|
| fixed prior M_H=5, no FC (reference) | 1.6105+-0.1666 | 5.0675 | +2.5127 | +0.4120 | 2.5127 | 6.0 / 6 | 0.0 / 0.0 |
| fixed prior M_H=5, fail-closed (reference) | 1.6105+-0.1666 | 5.0675 | +2.5127 | +0.4120 | 2.5127 | 6.0 / 6 | 0.0 / 0.0 |
| adaptive KAPPA=1, probe-every 20, no warm-up | 1.6105+-0.1666 | 5.0675 | +2.5127 | +0.4120 | 2.5127 | 6.0 / 6 | 1.4 / 32.6 |
| adaptive KAPPA=1 + warm-up first-obs (probe-every 20) | -0.0000+-0.0000 | 1.0000 | -0.0000 | -0.0000 | 0.0000 | 6.0 / 6 | 0.0 / 0.0 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 1.0146+-0.6004 | 3.3202 | +1.9704 | +0.0463 | 1.9704 | 6.0 / 6 | 0.0 / 21.6 |
| **(ii) (i) + warm-up first-obs** | 1.0146+-0.6004 | 3.3202 | +1.9704 | +0.0463 | 1.9704 | 6.0 / 6 | 0.0 / 21.6 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 0.0467+-0.1008 | 1.0529 | +0.0852 | +0.0231 | 0.0852 | 6.0 / 6 | 0.0 / 1.0 |

## 4. Retrospective consistency of every certified opening

For each gate opening at step `t`: the envelope in force at `t` and the NEXT probe's observed drift rate `M_obs`. The opening is *retrospectively consistent* iff that next observation does not exceed the envelope the opening was certified under. `cold-start` openings precede the run's first `M_obs` and can only be certified by the unverified floor -- with the dense schedule the first `M_obs` exists at step 1, so this column is the direct measure of what the dense probe bought.

| arm | seed | openings | first M_obs at | cold-start | with a following probe | consistent vs envelope in force | frac | consistent vs FINAL envelope | worst next-M_obs / envelope |
|---|---|---|---|---|---|---|---|---|---|
| adaptive KAPPA=1, probe-every 20, no warm-up | 0 | 8 | 20 | 8 | 8 | 0 | 0.0000 | 8 | 4.3535 |
| adaptive KAPPA=1, probe-every 20, no warm-up | 1 | 6 | 20 | 6 | 6 | 0 | 0.0000 | 6 | 2.3459 |
| adaptive KAPPA=1, probe-every 20, no warm-up | 2 | 12 | 20 | 12 | 12 | 0 | 0.0000 | 12 | 5.8048 |
| adaptive KAPPA=1, probe-every 20, no warm-up | 3 | 6 | 20 | 6 | 6 | 0 | 0.0000 | 6 | 1.6082 |
| adaptive KAPPA=1, probe-every 20, no warm-up | 4 | 8 | 20 | 8 | 8 | 0 | 0.0000 | 8 | 4.6807 |
| adaptive KAPPA=1, probe-every 20, no warm-up | 5 | 7 | 20 | 7 | 7 | 0 | 0.0000 | 7 | 3.1619 |
| adaptive KAPPA=1, probe-every 20, no warm-up | 6 | 9 | 20 | 9 | 9 | 0 | 0.0000 | 9 | 1.1858 |
| adaptive KAPPA=1, probe-every 20, no warm-up | 7 | 8 | 20 | 8 | 8 | 0 | 0.0000 | 8 | 3.0251 |
| adaptive KAPPA=1, probe-every 20, no warm-up | 8 | 13 | 20 | 13 | 13 | 0 | 0.0000 | 13 | 8.8076 |
| adaptive KAPPA=1, probe-every 20, no warm-up | 9 | 7 | 20 | 7 | 7 | 0 | 0.0000 | 7 | 2.9853 |
| adaptive KAPPA=1 + warm-up first-obs (probe-every 20) | 0 | 0 | 20 | 0 | 0 | 0 | - | 0 | 0.0000 |
| adaptive KAPPA=1 + warm-up first-obs (probe-every 20) | 1 | 0 | 20 | 0 | 0 | 0 | - | 0 | 0.0000 |
| adaptive KAPPA=1 + warm-up first-obs (probe-every 20) | 2 | 0 | 20 | 0 | 0 | 0 | - | 0 | 0.0000 |
| adaptive KAPPA=1 + warm-up first-obs (probe-every 20) | 3 | 0 | 20 | 0 | 0 | 0 | - | 0 | 0.0000 |
| adaptive KAPPA=1 + warm-up first-obs (probe-every 20) | 4 | 0 | 20 | 0 | 0 | 0 | - | 0 | 0.0000 |
| adaptive KAPPA=1 + warm-up first-obs (probe-every 20) | 5 | 0 | 20 | 0 | 0 | 0 | - | 0 | 0.0000 |
| adaptive KAPPA=1 + warm-up first-obs (probe-every 20) | 6 | 0 | 20 | 0 | 0 | 0 | - | 0 | 0.0000 |
| adaptive KAPPA=1 + warm-up first-obs (probe-every 20) | 7 | 0 | 20 | 0 | 0 | 0 | - | 0 | 0.0000 |
| adaptive KAPPA=1 + warm-up first-obs (probe-every 20) | 8 | 0 | 20 | 0 | 0 | 0 | - | 0 | 0.0000 |
| adaptive KAPPA=1 + warm-up first-obs (probe-every 20) | 9 | 0 | 20 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 0 | 7 | 1 | 0 | 7 | 7 | 1.0000 | 7 | 0.7088 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 1 | 5 | 1 | 0 | 5 | 4 | 0.8000 | 5 | 4.1328 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 2 | 11 | 1 | 0 | 11 | 11 | 1.0000 | 11 | 0.1244 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 3 | 4 | 1 | 0 | 4 | 3 | 0.7500 | 4 | 2.2444 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 4 | 7 | 1 | 0 | 7 | 6 | 0.8571 | 7 | 2.8571 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 5 | 4 | 1 | 0 | 4 | 3 | 0.7500 | 4 | 1.8714 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 6 | 9 | 1 | 0 | 9 | 8 | 0.8889 | 9 | 1.0490 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 7 | 6 | 1 | 0 | 6 | 5 | 0.8333 | 6 | 3.4690 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 8 | 11 | 1 | 0 | 11 | 11 | 1.0000 | 11 | 0.0227 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 9 | 4 | 1 | 0 | 4 | 3 | 0.7500 | 4 | 2.6535 |
| **(ii) (i) + warm-up first-obs** | 0 | 7 | 1 | 0 | 7 | 7 | 1.0000 | 7 | 0.7088 |
| **(ii) (i) + warm-up first-obs** | 1 | 5 | 1 | 0 | 5 | 4 | 0.8000 | 5 | 4.1328 |
| **(ii) (i) + warm-up first-obs** | 2 | 11 | 1 | 0 | 11 | 11 | 1.0000 | 11 | 0.1244 |
| **(ii) (i) + warm-up first-obs** | 3 | 4 | 1 | 0 | 4 | 3 | 0.7500 | 4 | 2.2444 |
| **(ii) (i) + warm-up first-obs** | 4 | 7 | 1 | 0 | 7 | 6 | 0.8571 | 7 | 2.8571 |
| **(ii) (i) + warm-up first-obs** | 5 | 4 | 1 | 0 | 4 | 3 | 0.7500 | 4 | 1.8714 |
| **(ii) (i) + warm-up first-obs** | 6 | 9 | 1 | 0 | 9 | 8 | 0.8889 | 9 | 1.0490 |
| **(ii) (i) + warm-up first-obs** | 7 | 6 | 1 | 0 | 6 | 5 | 0.8333 | 6 | 3.4690 |
| **(ii) (i) + warm-up first-obs** | 8 | 11 | 1 | 0 | 11 | 11 | 1.0000 | 11 | 0.0227 |
| **(ii) (i) + warm-up first-obs** | 9 | 4 | 1 | 0 | 4 | 3 | 0.7500 | 4 | 2.6535 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 0 | 0 | 1 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 1 | 0 | 1 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 2 | 1 | 1 | 0 | 1 | 0 | 0.0000 | 0 | 15.7558 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 3 | 1 | 1 | 0 | 1 | 0 | 0.0000 | 0 | 25.2696 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 4 | 0 | 1 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 5 | 0 | 1 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 6 | 0 | 1 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 7 | 1 | 1 | 0 | 1 | 0 | 0.0000 | 0 | 17.4920 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 8 | 0 | 1 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 9 | 0 | 1 | 0 | 0 | 0 | - | 0 | 0.0000 |

Totals over the 10 seeds:

| arm | openings | cold-start | with a following probe | consistent vs envelope in force | frac | consistent vs FINAL envelope | frac |
|---|---|---|---|---|---|---|---|
| adaptive KAPPA=1, probe-every 20, no warm-up | 84 | 84 | 84 | 0 | 0.0000 | 84 | 1.0000 |
| adaptive KAPPA=1 + warm-up first-obs (probe-every 20) | 0 | 0 | 0 | 0 | - | 0 | - |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 68 | 0 | 68 | 61 | 0.8971 | 68 | 1.0000 |
| **(ii) (i) + warm-up first-obs** | 68 | 0 | 68 | 61 | 0.8971 | 68 | 1.0000 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 3 | 0 | 3 | 0 | 0.0000 | 0 | 0.0000 |

## 5. Paired tests (exact sign-flip, 10 seeds)

Differences are `arm - reference`, paired by seed.

| arm | reference | d NMSE | p | d events | p | d open coord-steps | p | d HVPs | p |
|---|---|---|---|---|---|---|---|---|---|
| **(i) adaptive KAPPA=1 + dense probe t<=20** | adaptive KAPPA=1, probe-every 20, no warm-up | +0.008786 | 0.0137 | -5.6 | 0.1250 | -12.4 | 0.0020 | +2660 | 0.0020 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | adaptive KAPPA=1 + warm-up first-obs (probe-every 20) | -0.027215 | 0.0020 | +3.4 | 0.1250 | +21.6 | 0.0020 | +2645 | 0.0020 |
| **(ii) (i) + warm-up first-obs** | adaptive KAPPA=1, probe-every 20, no warm-up | +0.008786 | 0.0137 | -5.6 | 0.1250 | -12.4 | 0.0020 | +2660 | 0.0020 |
| **(ii) (i) + warm-up first-obs** | adaptive KAPPA=1 + warm-up first-obs (probe-every 20) | -0.027215 | 0.0020 | +3.4 | 0.1250 | +21.6 | 0.0020 | +2645 | 0.0020 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | adaptive KAPPA=1, probe-every 20, no warm-up | +0.034169 | 0.0020 | -9.0 | 0.0625 | -33.0 | 0.0020 | +77496 | 0.0020 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | adaptive KAPPA=1 + warm-up first-obs (probe-every 20) | -0.001832 | 0.2500 | +0.0 | 1.0000 | +1.0 | 0.2500 | +77481 | 0.0020 |
| **(ii) (i) + warm-up first-obs** | **(i) adaptive KAPPA=1 + dense probe t<=20** | +0.000000 | 1.0000 | +0.0 | 1.0000 | +0.0 | 1.0000 | +0 | 1.0000 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | **(i) adaptive KAPPA=1 + dense probe t<=20** | +0.025383 | 0.0020 | -3.4 | 0.1250 | -20.6 | 0.0020 | +74836 | 0.0020 |

## 6. Envelope trajectory over the first 40 steps (seed 0)

`open` marks steps at which at least one coordinate opened. The reference block is the no-warm-up 20-step-cadence adaptive arm, whose first `M_obs` arrives at step 20 -- after every opening.

**adaptive KAPPA=1, probe-every 20, no warm-up** (seed 0): 601 probes, first `M_obs` at step 20, envelope ends at 21.767, 8 openings (steps 1, 2, 3, 4, 5, 6, 7, 8), closed steps 1, HVPs 94740.

| probe step | M_obs | envelope after | monitor closed | gate opened this step |
|---|---|---|---|---|
| 0 | - | 5 |  |  |
| 20 | 21.767 | 21.767 | yes |  |
| 21 | 3.0437 | 21.767 |  |  |
| 40 | 1.242 | 21.767 |  |  |

****(i) adaptive KAPPA=1 + dense probe t<=20**** (seed 0): 619 probes, first `M_obs` at step 1, envelope ends at 114.45, 7 openings (steps 4, 5, 6, 7, 8, 9, 10), closed steps 3, HVPs 97476.

| probe step | M_obs | envelope after | monitor closed | gate opened this step |
|---|---|---|---|---|
| 0 | - | 5 |  |  |
| 1 | 12.31 | 12.31 | yes |  |
| 2 | 85.393 | 85.393 | yes |  |
| 3 | 114.45 | 114.45 | yes |  |
| 4 | 112.49 | 114.45 |  | yes |
| 5 | 75.391 | 114.45 |  | yes |
| 6 | 81.118 | 114.45 |  | yes |
| 7 | 0.13303 | 114.45 |  | yes |
| 8 | 29.523 | 114.45 |  | yes |
| 9 | 7.8329 | 114.45 |  | yes |
| 10 | 14.503 | 114.45 |  | yes |
| 11 | 14.113 | 114.45 |  |  |
| 12 | 0.92744 | 114.45 |  |  |
| 13 | 1.6803 | 114.45 |  |  |
| 14 | 20.303 | 114.45 |  |  |
| 15 | 17.457 | 114.45 |  |  |
| 16 | 14.151 | 114.45 |  |  |
| 17 | 1.6955 | 114.45 |  |  |
| 18 | 1.6897 | 114.45 |  |  |
| 19 | 17.911 | 114.45 |  |  |
| 20 | 9.0377 | 114.45 |  |  |
| 40 | 2.9134 | 114.45 |  |  |

****(ii) (i) + warm-up first-obs**** (seed 0): 619 probes, first `M_obs` at step 1, envelope ends at 114.45, 7 openings (steps 4, 5, 6, 7, 8, 9, 10), closed steps 3, HVPs 97476.

| probe step | M_obs | envelope after | monitor closed | gate opened this step |
|---|---|---|---|---|
| 0 | - | 5 |  |  |
| 1 | 12.31 | 12.31 | yes |  |
| 2 | 85.393 | 85.393 | yes |  |
| 3 | 114.45 | 114.45 | yes |  |
| 4 | 112.49 | 114.45 |  | yes |
| 5 | 75.391 | 114.45 |  | yes |
| 6 | 81.118 | 114.45 |  | yes |
| 7 | 0.13303 | 114.45 |  | yes |
| 8 | 29.523 | 114.45 |  | yes |
| 9 | 7.8329 | 114.45 |  | yes |
| 10 | 14.503 | 114.45 |  | yes |
| 11 | 14.113 | 114.45 |  |  |
| 12 | 0.92744 | 114.45 |  |  |
| 13 | 1.6803 | 114.45 |  |  |
| 14 | 20.303 | 114.45 |  |  |
| 15 | 17.457 | 114.45 |  |  |
| 16 | 14.151 | 114.45 |  |  |
| 17 | 1.6955 | 114.45 |  |  |
| 18 | 1.6897 | 114.45 |  |  |
| 19 | 17.911 | 114.45 |  |  |
| 20 | 9.0377 | 114.45 |  |  |
| 40 | 2.9134 | 114.45 |  |  |

****(iii) fixed prior M_H=5, FC + dense probe t<=20**** (seed 0): 1115 probes, first `M_obs` at step 1, envelope ends at 5, 0 openings (none), closed steps 9575, HVPs 174156.

| probe step | M_obs | envelope after | monitor closed | gate opened this step |
|---|---|---|---|---|
| 0 | - | 5 |  |  |
| 1 | 12.31 | 5 | yes |  |
| 2 | 85.393 | 5 | yes |  |
| 3 | 114.45 | 5 | yes |  |
| 4 | 112.49 | 5 | yes |  |
| 5 | 26.492 | 5 | yes |  |
| 6 | 5.6372 | 5 | yes |  |
| 7 | 97.068 | 5 | yes |  |
| 8 | 98.231 | 5 | yes |  |
| 9 | 49.777 | 5 | yes |  |
| 10 | 119.53 | 5 | yes |  |
| 11 | 178.38 | 5 | yes |  |
| 12 | 1.2672 | 5 |  |  |
| 13 | 22.468 | 5 | yes |  |
| 14 | 176.93 | 5 | yes |  |
| 15 | 91.343 | 5 | yes |  |
| 16 | 140.91 | 5 | yes |  |
| 17 | 12.474 | 5 | yes |  |
| 18 | 19.69 | 5 | yes |  |
| 19 | 230.07 | 5 | yes |  |
| 20 | 96.799 | 5 | yes |  |
| 21 | 27.022 | 5 | yes |  |
| 40 | 29.793 | 5 | yes |  |

## 7. How much of the envelope is short-interval inflation

`M_obs = |rho_probe - rho_prev| / (eta_max * D)` divides by the path length `D` traversed since the previous probe. A 1-step interval has a tiny `D`, so the SAME trajectory yields a much larger `M_obs` at 1-step resolution than at 20-step resolution -- the ratio is not drift, it is the finite-difference denominator plus the spectral probe's own randomization noise (`rho` is a KW upper estimate, not an exact eigenvalue, so `|rho_t - rho_{t-1}|` has a noise floor that does not shrink with `D`).

The comparison is WITHIN each run and over the SAME interval `[0, 20]`: `1-step max/median` are the dense observations at `t = 1..20`; `20-step [0,20]` recomputes `M_obs` from the raw probe record as `|rho(20) - rho(0)| / (eta_max(0) * sum of the 20 path lengths)`, i.e. exactly what the 20-step cadence would have reported over the same trajectory.

| arm | seed | 1-step max (t<=20) | 1-step median | 20-step [0,20] | inflation max / 20-step | inflation median / 20-step | median M_obs at t>20 (20-step cadence) | final envelope |
|---|---|---|---|---|---|---|---|---|
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 0 | 114.45 | 14.503 | 8.9162 | 12.8 | 1.6 | 0.12875 | 114.45 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 1 | 187.83 | 21.443 | 7.9567 | 23.6 | 2.7 | 0.29303 | 187.83 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 2 | 3087.1 | 3.902 | 30.766 | 100.3 | 0.1 | 0.12536 | 3087.1 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 3 | 126.35 | 14.855 | 3.5058 | 36.0 | 4.2 | 0.7455 | 126.35 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 4 | 210.42 | 14.593 | 24.198 | 8.7 | 0.6 | 0.28807 | 210.42 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 5 | 136.66 | 23.98 | 10.926 | 12.5 | 2.2 | 0.42137 | 136.66 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 6 | 273.4 | 13.658 | 13.584 | 20.1 | 1.0 | 2.1745 | 273.4 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 7 | 222.69 | 29.568 | 12.592 | 17.7 | 2.3 | 0.18173 | 222.69 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 8 | 15601 | 23.613 | 26.277 | 593.7 | 0.9 | 0.14421 | 15601 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 9 | 177.1 | 35.914 | 10.095 | 17.5 | 3.6 | 0.74109 | 177.1 |
| **(ii) (i) + warm-up first-obs** | 0 | 114.45 | 14.503 | 8.9162 | 12.8 | 1.6 | 0.12875 | 114.45 |
| **(ii) (i) + warm-up first-obs** | 1 | 187.83 | 21.443 | 7.9567 | 23.6 | 2.7 | 0.29303 | 187.83 |
| **(ii) (i) + warm-up first-obs** | 2 | 3087.1 | 3.902 | 30.766 | 100.3 | 0.1 | 0.12536 | 3087.1 |
| **(ii) (i) + warm-up first-obs** | 3 | 126.35 | 14.855 | 3.5058 | 36.0 | 4.2 | 0.7455 | 126.35 |
| **(ii) (i) + warm-up first-obs** | 4 | 210.42 | 14.593 | 24.198 | 8.7 | 0.6 | 0.28807 | 210.42 |
| **(ii) (i) + warm-up first-obs** | 5 | 136.66 | 23.98 | 10.926 | 12.5 | 2.2 | 0.42137 | 136.66 |
| **(ii) (i) + warm-up first-obs** | 6 | 273.4 | 13.658 | 13.584 | 20.1 | 1.0 | 2.1745 | 273.4 |
| **(ii) (i) + warm-up first-obs** | 7 | 222.69 | 29.568 | 12.592 | 17.7 | 2.3 | 0.18173 | 222.69 |
| **(ii) (i) + warm-up first-obs** | 8 | 15601 | 23.613 | 26.277 | 593.7 | 0.9 | 0.14421 | 15601 |
| **(ii) (i) + warm-up first-obs** | 9 | 177.1 | 35.914 | 10.095 | 17.5 | 3.6 | 0.74109 | 177.1 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 0 | 230.07 | 96.799 | 2.4689 | 93.2 | 39.2 | 19.596 | 5 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 1 | 164.35 | 45.449 | 15.806 | 10.4 | 2.9 | 18.288 | 5 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 2 | 3087.1 | 68.217 | 27.73 | 111.3 | 2.5 | 14.087 | 5 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 3 | 184.52 | 54.013 | 20.479 | 9.0 | 2.6 | 12.222 | 5 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 4 | 206.02 | 51.477 | 6.7172 | 30.7 | 7.7 | 20.25 | 5 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 5 | 188.43 | 52.081 | 0.51809 | 363.7 | 100.5 | 18.535 | 5 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 6 | 260.62 | 115.1 | 8.6704 | 30.1 | 13.3 | 17.573 | 5 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 7 | 123.94 | 60.864 | 1.7544 | 70.6 | 34.7 | 14.404 | 5 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 8 | 15601 | 117.62 | 30.265 | 515.5 | 3.9 | 17.527 | 5 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 9 | 268.83 | 66.743 | 8.9892 | 29.9 | 7.4 | 21.069 | 5 |

| arm | inflation of the MAX (mean+-sd) | inflation of the MEDIAN (mean+-sd) |
|---|---|---|
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 84.3+-181.0 | 1.9+-1.3 |
| **(ii) (i) + warm-up first-obs** | 84.3+-181.0 | 1.9+-1.3 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 126.4+-172.3 | 21.5+-30.8 |

The same quantity stated as an envelope: with `KAPPA = 1` the envelope is `max(5, max_s M_obs,s)`, so the dense window sets it to the 1-step max, whereas a 20-step-only schedule would have set it to the 20-step value.

| arm | final envelope (dense) | 20-step [0,20] M_obs | envelope of the 20-step cadence (`results/e2_adaptmh`) | deployed prior |
|---|---|---|---|---|
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 2013.71+-4860.93 | 14.882+-8.990 | 29.92+-19.75 | 5.0 |
| **(ii) (i) + warm-up first-obs** | 2013.71+-4860.93 | 14.882+-8.990 | 29.92+-19.75 | 5.0 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 5.00+-0.00 | 12.340+-10.760 | 29.92+-19.75 | 5.0 |

## 8. The probe cost

| arm | probes | forced re-probes | HVPs | HVPs / no-FC (94588) | d HVPs vs no-warm-up adaptive | wall s |
|---|---|---|---|---|---|---|
| fixed prior M_H=5, no FC (reference) | - | - | 94588 | 1.000 | -228 | 5102 |
| fixed prior M_H=5, fail-closed (reference) | - | 103.8 | 102602 | 1.085 | +7786 | 5404 |
| adaptive KAPPA=1, probe-every 20, no warm-up | 601.5 | 2.4 | 94816 | 1.002 | +0 | 7009 |
| adaptive KAPPA=1 + warm-up first-obs (probe-every 20) | 601.6 | 3.1 | 94831 | 1.003 | +15 | 6548 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 619.0 | 3.9 | 97476 | 1.031 | +2660 | 6116 |
| **(ii) (i) + warm-up first-obs** | 619.0 | 3.9 | 97476 | 1.031 | +2660 | 6106 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 1103.3 | 982.0 | 172312 | 1.822 | +77496 | 9148 |

## 9. What the dense schedule does to the monitor

| arm | final envelope | max M_obs | median M_obs | closed steps | closure frac | fail-closed events | forced re-probes | closed steps within t<=20 |
|---|---|---|---|---|---|---|---|---|
| fixed prior M_H=5, no FC (reference) | 5 | - | - | - | - | - | - | 0.0 |
| fixed prior M_H=5, fail-closed (reference) | 5 | 234.7 | 0.7694 | 939.7 | 0.0783 | 45.6 | 103.8 | 0.0 |
| adaptive KAPPA=1, probe-every 20, no warm-up | 29.92 | 29.92 | 0.6433 | 8.6 | 0.0007 | 1.3 | 2.4 | 1.0 |
| adaptive KAPPA=1 + warm-up first-obs (probe-every 20) | 212.7 | 212.7 | 8.025 | 25.7 | 0.0021 | 1.5 | 3.1 | 0.7 |
| **(i) adaptive KAPPA=1 + dense probe t<=20** | 2014 | 2014 | 0.5464 | 2.6 | 0.0002 | 1.7 | 3.9 | 2.6 |
| **(ii) (i) + warm-up first-obs** | 2014 | 2014 | 0.5464 | 2.6 | 0.0002 | 1.7 | 3.9 | 2.6 |
| **(iii) fixed prior M_H=5, FC + dense probe t<=20** | 5 | 2937 | 17.94 | 9402.2 | 0.7835 | 178.2 | 982.0 | 19.2 |

## 10. Plain answer

**Does an early measurement rescue certified adaptation under a MEASURED envelope? Yes -- most of it, and cheaply. With one probe per step over the first 20 steps, 68 of the reference run's 84 certified openings survive (81%), **0 of them are cold-start** (down from 84 of 84), **61 of 68 (89.7%) are retrospectively consistent with the envelope in force** (up from 0 of 84) and **68 of 68 (100%) under the final envelope**, for +2.8% HVPs and an NMSE of 0.0250+-0.0079 against 0.0162+-0.0021 -- recovering 76% of the gap the naive warm-up gave away. But the envelope those openings are certified under is largely a short-interval artefact: a median 199.1, i.e. 19x the drift the SAME window actually shows at 20-step resolution and 40x the deployed prior. Under a FIXED prior the same schedule destroys the method instead.**

**1. The measurement now exists before the gate needs it, and that is the whole point.** On the 20-step cadence the first `M_obs` arrives at step 20 while every opening is at steps 1-15, so 84 of 84 openings are cold-start and 0 of 84 are retrospectively consistent (`round4_adaptmh.md`). With `--probe-dense-until 20` the first `M_obs` arrives at **step 1 on all 10 seeds**, the first opening moves from step 1.0 to step 2.9, and the cold-start count goes to **0 of 68**. Retrospective consistency -- the next observed `M_obs` does not exceed the envelope the opening was certified under -- goes **0/84 -> 61/68 = 0.8971**. The 7 exceptions are steps where the very next 1-step observation overshot the envelope, by up to 4.13x; all 68 openings are consistent under the run's final envelope.

**2. Holding the gate until an observation exists becomes FREE.** Arm (ii) adds `--gate-warmup first-obs` on top of the dense schedule and is **bit-identical to arm (i) on all 10 seeds** (losses AND `lam_hist` element for element: True). The hold now lasts 1 step -- step 0, before any probe pair can exist -- and suppresses nothing, because the certificate keeps the gate shut at step 0 anyway. This is the claim `round4_warmup.md` could not make: on the 20-step cadence the same flag costs every opening and 3.2x the NMSE; with a dense early probe it costs exactly zero. The verified-envelope requirement is a **probe-schedule** problem, not a certificate problem.

**3. The inflated short-interval `M_obs` does blow the envelope up -- but it does not close the gate.** `M_obs = |rho_probe - rho_prev| / (eta_max * D)` divides by the path length, and a 1-step `D` is ~20x smaller than a 20-step one; on top of that `rho` is a randomized KW upper estimate whose step-to-step difference has a noise floor that does NOT shrink with `D`. Measured within the same run over the same interval [0, 20]: the largest 1-step observation is a median **18.9x** (mean 84.3+-181.0) the 20-step aggregate over that identical window, while the MEDIAN 1-step observation is only x1.9+-1.3 of it -- the inflation lives almost entirely in the maximum, which is exactly the statistic `M_H,t = max(5, KAPPA * max_s M_obs,s)` reads. The envelope ends at a median 199.1 (mean 2014+-4861; seed 8 reaches 1.56e+04) where the coarse 20-step measurement of the same window is 11.76 and the 20-step-cadence run settles at 29.92. **Only ~5% of the envelope's size is coarse-grained drift; the other ~95% is short-interval inflation.** It does not close the gate for long: the envelope overshoots so far that no later observation violates it -- closed steps 2.6 per run (closure fraction 0.0002), *below* the 20-step adaptive arm's 8.6, and all 2.6 of them fall inside the dense window while the envelope is still climbing (seed 0: closed at probes 1, 2, 3, then open from step 4 on).

**4. What it costs: 16 openings, 54% more NMSE than the unverified run, and 2.8% of the probe budget.** The inflated envelope is not free even when it never closes the gate, because it enters the certificate itself: `dH = M_H * D`, so a ~40x larger `M_H` inflates `beta_col` and with it the gate threshold `c * beta_col`. Openings 34.0+-8.2 -> 21.6+-11.2 coordinate-steps per seed (paired d = -12.4, p = 0.0020), 16 of the 84 opening steps lost, mean envelope in force at an opening 3142 against the deployed prior 5.0, and the learning rate ends 3.32x up instead of 5.07x. NMSE 0.0162+-0.0021 -> 0.0250+-0.0079 (paired d = +0.008786, exact sign-flip p = 0.0137) -- **1.54x worse than the unverified-floor run, but still 76% of the way from the frozen-LR baseline (0.0522, `results/e2` fixed lr 0.003) back to it**. Stability does not get worse: events 9.0+-12.6 -> 3.4+-8.7 (p = 0.1250). The probe cost is small and exactly the dense window: 601.5 -> 619.0 probes, 94816 -> 97476 HVPs = **1.031x** the no-monitor budget against 1.002x, +2.8% (p = 0.0020).

**5. The fixed prior does not survive the dense schedule.** Arm (iii) keeps `M_H = 5` and pays for the inflation instead of absorbing it: the 1-step observations exceed 5 at essentially every probe, so the monitor is violated on **19.2 of the first 20 steps** and on **9402 of 12000 steps overall (closure fraction 0.7835)**, with 178.2 fail-closed transitions and 982 forced re-probes -- HVPs 172312 = **1.822x** the no-monitor budget (paired +74836 vs arm (i), p = 0.0020). Only **3 openings survive across all 10 seeds** (7 of 10 seeds never open at all), lambda barely moves (LR ratio 1.053), and NMSE 0.0503+-0.0044 is statistically indistinguishable from the frozen-LR warm-up arm (0.0522+-0.0022, paired p = 0.2500). None of those 3 openings is retrospectively consistent, since a fixed 5.0 is below every 1-step observation. This is the honest control: the dense probe rescues certified adaptation **only when the envelope may be re-stated online**. Bolted onto a fixed prior it destroys the method and nearly doubles the probe bill.

**6. Validity is untouched, as everywhere in round 4.** **0 certificate violations in 720000 audited coordinate-steps in each of the three arms**, worst `|ghat_j - g_true_j| / beta_col_j` = 0.8215 (0.8374 on the fixed-prior reference). The default path is bit-identical: with `--probe-dense-until 0` (the default) seeds 0 and 1 reproduce `results/e2_verify4` element for element on the full 12000-entry loss trace and the sampled lambda trajectory, with the same HVP count 94588 and 0 violations.

**7. What this means for the claim.** `round4_warmup.md` concluded that requiring the envelope to be verified before use costs the method its entire advantage. That conclusion was an artefact of the probe SCHEDULE, not of the certificate: move one probe per step into the first 20 steps -- 19 extra probes, 2.8% of the run's HVPs -- and 68 of 84 certified openings come back with a measurement behind them, 90% of them retrospectively consistent, 100% consistent under the final envelope, and the `first-obs` hold becomes a no-op. Two caveats belong beside that in the paper. First, the envelope those openings are certified under is **~40x the deployed prior and ~19x the coarse-grained drift of the same window**, because `M_obs` at 1-step resolution is dominated by the finite-difference denominator and the spectral probe's own randomization noise. It is a valid ONLINE-ENFORCEABLE envelope -- never below any observation the run made, 0 violations against exact FMD -- but it is *not* a tight estimate of the Hessian drift rate, and quoting it as one would overstate the measurement: a `KAPPA * max` envelope built from 1-step differences buys auditability by being loose. Second, that looseness is paid for in the currency the method trades in: 16 of 84 openings and 1.54x the NMSE. The defensible statement is therefore *a dense early probe makes COHG's certified adaptation auditable against a measured envelope for ~3% probe overhead and ~54% NMSE over the unverified-floor run -- three quarters of the way back from the frozen-LR collapse -- provided the envelope is re-stated online; under a fixed prior the same schedule closes the gate 78% of the time and costs 1.82x the probe budget.*

### Reproduction

```
python code/experiments/launch_r4_denseprobe.py           # job list
RDQ_WORKERS=32 python code/experiments/r4_denseprobe_queue.py
python results/reanalysis/_round4_denseprobe.py           # this file
```
Outputs: `results/e2_denseprobe/` (30 arm runs) and `results/e2_denseprobe/verify/` (2 default-path regression runs).

### Raw numbers

* Openings kept, (i) dense vs the no-warm-up reference: 21.6+-11.2 vs 34.0+-8.2 open coordinate-steps per seed (68 vs 84 opening steps over the 10 seeds); (ii) dense + warm-up: 21.6+-11.2 (68 steps); (iii) fixed prior + dense: 1.0+-2.0 (3 steps). The 20-step-cadence warm-up arm keeps 0.0+-0.0.
* Cold-start openings (before any `M_obs` exists): reference 84 of 84; (i) 0 of 68; (ii) 0 of 68; (iii) 0 of 3.
* Retrospective consistency vs the envelope in force: reference 0 of 84; (i) 61 of 68; (ii) 61 of 68; (iii) 0 of 3. Under the FINAL envelope: 84 / 68 / 68 / 0.
* NMSE: reference 0.0162+-0.0021; warm-up (20-step cadence) 0.0522+-0.0022; (i) 0.0250+-0.0079; (ii) 0.0250+-0.0079; (iii) 0.0503+-0.0044.
* HVPs: reference 94816 (1.002x the no-monitor budget); (i) 97476 (1.031x, +2.8% vs reference); (ii) 97476 (1.031x); (iii) 172312 (1.822x).
* Certificate violations: (i) 0 / 720000; (ii) 0 / 720000; (iii) 0 / 720000.
* Short-interval inflation of `M_obs` on the same interval [0,20]: max x84.3+-181.0, median x1.9+-1.3 (arm (i)).

