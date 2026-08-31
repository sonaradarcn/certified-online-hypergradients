# Round-4 follow-up: holding the gate shut until the drift envelope is VERIFIED

`round4_adaptmh.md` established that under the online-enforced envelope `M_H,t = max(5, KAPPA * max_{s<=t} M_obs,s)` **every** COHG gate opening on E2 `mackey_drift` happens at steps 1-15, before the first probe-to-probe drift observation `M_obs` exists (the first probe pair completes at step `2 * probe_every` = 20). Every certified opening is therefore certified under the *unverified floor*. This study asks what is left if that is forbidden.

`--gate-warmup MODE` (new flag in `code/experiments/e2_timeseries.py`; default `off` is the legacy bit-identical path):

* `first-obs` -- no coordinate may open before the first `M_obs` has been recorded (step 20 on this config).
* `stable-env` -- additionally, the most recent probe must not have RAISED the envelope; a raise re-arms the hold until a probe passes without raising.

Config: `mackey_drift`, GRU, 12000 steps, lr0 0.003, alpha 0.4, c = 2, K 10, rank 4, gamma 0.9, probe-every 20, seeds 0-9, CPU, `--validate-cert` on every arm. Statistics are mean+-sd with ddof = 1; paired tests are exact sign-flip over the 10 seeds.

## 0. Default-path regression (`--gate-warmup off`)

| seed | losses identical | lam_hist identical | NMSE (new / e2_verify4) | events | HVPs | cert viol |
|---|---|---|---|---|---|---|
| 0 | True | True | 0.0148824551 / 0.0148824551 | 0 / 0 | 94588 / 94588 | 0 / 0 |
| 1 | True | True | 0.0184537453 / 0.0184537453 | 0 / 0 | 94588 / 94588 | 0 / 0 |

## 1. Arm summary

`open coord-steps` = `coord_open_frac * 6 * 12000` (6 parameter groups). `held` = steps the warm-up hold kept the gate shut; `suppressed` = open coordinate-steps the certificate gate WOULD have taken during the hold (recorded read-only, no effect on the run).

| arm | n | NMSE | events | open coord-steps | coord-open rate | opening steps (first / last) | held | suppressed | closed steps | HVPs | HVPs / no-FC | cert viol / checked | cert max ratio | final envelope |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed prior M_H=5, no FC, no warm-up | 10 | 0.0162+-0.0021 | 9.0+-12.6 | 34.0+-8.2 | 4.722e-04 | n/a (not logged) | - | - | - | 94588 | 1.000 | 0 / 720000 | 0.8374 | 5 |
| fixed prior M_H=5, fail-closed, no warm-up | 10 | 0.0162+-0.0021 | 9.0+-12.6 | 34.0+-8.2 | 4.722e-04 | n/a (not logged) | - | - | 939.7 | 102602 | 1.085 | 0 / 720000 | 0.8374 | 5 |
| adaptive envelope KAPPA=1, no warm-up | 10 | 0.0162+-0.0021 | 9.0+-12.6 | 34.0+-8.2 | 4.722e-04 | 1.0 / 8.6 | - | - | 8.6 | 94816 | 1.002 | 0 / 720000 | 0.8374 | 29.92 |
| **adaptive KAPPA=1 + warm-up first-obs** | 10 | 0.0522+-0.0022 | 0.0+-0.0 | 0.0+-0.0 | 0.000e+00 | never opens | 20.0 | 30.5 | 25.7 | 94831 | 1.003 | 0 / 720000 | 0.8241 | 212.7 |
| **adaptive KAPPA=1 + warm-up stable-env** | 10 | 0.0522+-0.0022 | 0.0+-0.0 | 0.0+-0.0 | 0.000e+00 | never opens | 20.0 | 30.5 | 25.7 | 94831 | 1.003 | 0 / 720000 | 0.8241 | 212.7 |
| **fixed prior M_H=5, FC + warm-up first-obs** | 10 | 0.0522+-0.0022 | 0.0+-0.0 | 0.0+-0.0 | 0.000e+00 | never opens | 20.0 | 30.5 | 9570.4 | 170540 | 1.803 | 0 / 720000 | 0.8241 | 5 |

## 2. Openings: number and timing, per seed

`suppressed` counts the openings the certificate gate would have taken during the warm-up hold; the opening steps listed are the certified openings that actually happened.

| arm | seed | opening steps | open coord-steps | held steps | release step | suppressed steps / coord-steps | closed steps |
|---|---|---|---|---|---|---|---|
| adaptive envelope KAPPA=1, no warm-up | 0 | 1, 2, 3, 4, 5, 6, 7, 8 | 33 | None | None | None / None | 1 |
| adaptive envelope KAPPA=1, no warm-up | 1 | 1, 2, 3, 4, 5, 6 | 26 | None | None | None / None | 20 |
| adaptive envelope KAPPA=1, no warm-up | 2 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 | 45 | None | None | None / None | 1 |
| adaptive envelope KAPPA=1, no warm-up | 3 | 1, 2, 3, 4, 5, 6 | 25 | None | None | None / None | 39 |
| adaptive envelope KAPPA=1, no warm-up | 4 | 1, 2, 3, 4, 5, 6, 7, 8 | 33 | None | None | None / None | 1 |
| adaptive envelope KAPPA=1, no warm-up | 5 | 1, 2, 3, 4, 5, 6, 7 | 31 | None | None | None / None | 20 |
| adaptive envelope KAPPA=1, no warm-up | 6 | 1, 2, 3, 4, 5, 6, 7, 8, 9 | 36 | None | None | None / None | 1 |
| adaptive envelope KAPPA=1, no warm-up | 7 | 1, 2, 3, 4, 5, 6, 7, 8 | 32 | None | None | None / None | 1 |
| adaptive envelope KAPPA=1, no warm-up | 8 | 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 | 51 | None | None | None / None | 1 |
| adaptive envelope KAPPA=1, no warm-up | 9 | 1, 2, 3, 4, 5, 6, 7 | 28 | None | None | None / None | 1 |
| **adaptive KAPPA=1 + warm-up first-obs** | 0 | (none) | 0 | 20 | 20 | 8 / 27 | 20 |
| **adaptive KAPPA=1 + warm-up first-obs** | 1 | (none) | 0 | 20 | 20 | 6 / 24 | 20 |
| **adaptive KAPPA=1 + warm-up first-obs** | 2 | (none) | 0 | 20 | 20 | 12 / 39 | 58 |
| **adaptive KAPPA=1 + warm-up first-obs** | 3 | (none) | 0 | 20 | 20 | 6 / 22 | 20 |
| **adaptive KAPPA=1 + warm-up first-obs** | 4 | (none) | 0 | 20 | 20 | 7 / 28 | 20 |
| **adaptive KAPPA=1 + warm-up first-obs** | 5 | (none) | 0 | 20 | 20 | 7 / 28 | 39 |
| **adaptive KAPPA=1 + warm-up first-obs** | 6 | (none) | 0 | 20 | 20 | 11 / 38 | 21 |
| **adaptive KAPPA=1 + warm-up first-obs** | 7 | (none) | 0 | 20 | 20 | 7 / 26 | 19 |
| **adaptive KAPPA=1 + warm-up first-obs** | 8 | (none) | 0 | 20 | 20 | 14 / 48 | 20 |
| **adaptive KAPPA=1 + warm-up first-obs** | 9 | (none) | 0 | 20 | 20 | 6 / 25 | 20 |
| **adaptive KAPPA=1 + warm-up stable-env** | 0 | (none) | 0 | 20 | 20 | 8 / 27 | 20 |
| **adaptive KAPPA=1 + warm-up stable-env** | 1 | (none) | 0 | 20 | 40 | 6 / 24 | 20 |
| **adaptive KAPPA=1 + warm-up stable-env** | 2 | (none) | 0 | 20 | 40 | 12 / 39 | 58 |
| **adaptive KAPPA=1 + warm-up stable-env** | 3 | (none) | 0 | 20 | 21 | 6 / 22 | 20 |
| **adaptive KAPPA=1 + warm-up stable-env** | 4 | (none) | 0 | 20 | 40 | 7 / 28 | 20 |
| **adaptive KAPPA=1 + warm-up stable-env** | 5 | (none) | 0 | 20 | 20 | 7 / 28 | 39 |
| **adaptive KAPPA=1 + warm-up stable-env** | 6 | (none) | 0 | 20 | 21 | 11 / 38 | 21 |
| **adaptive KAPPA=1 + warm-up stable-env** | 7 | (none) | 0 | 20 | 20 | 7 / 26 | 19 |
| **adaptive KAPPA=1 + warm-up stable-env** | 8 | (none) | 0 | 20 | 40 | 14 / 48 | 20 |
| **adaptive KAPPA=1 + warm-up stable-env** | 9 | (none) | 0 | 20 | 40 | 6 / 25 | 20 |
| **fixed prior M_H=5, FC + warm-up first-obs** | 0 | (none) | 0 | 20 | 20 | 8 / 27 | 9537 |
| **fixed prior M_H=5, FC + warm-up first-obs** | 1 | (none) | 0 | 20 | 20 | 6 / 24 | 9748 |
| **fixed prior M_H=5, FC + warm-up first-obs** | 2 | (none) | 0 | 20 | 20 | 12 / 39 | 9424 |
| **fixed prior M_H=5, FC + warm-up first-obs** | 3 | (none) | 0 | 20 | 20 | 6 / 22 | 9628 |
| **fixed prior M_H=5, FC + warm-up first-obs** | 4 | (none) | 0 | 20 | 20 | 7 / 28 | 9807 |
| **fixed prior M_H=5, FC + warm-up first-obs** | 5 | (none) | 0 | 20 | 20 | 7 / 28 | 9479 |
| **fixed prior M_H=5, FC + warm-up first-obs** | 6 | (none) | 0 | 20 | 20 | 11 / 38 | 9508 |
| **fixed prior M_H=5, FC + warm-up first-obs** | 7 | (none) | 0 | 20 | 20 | 7 / 26 | 9636 |
| **fixed prior M_H=5, FC + warm-up first-obs** | 8 | (none) | 0 | 20 | 20 | 14 / 48 | 9205 |
| **fixed prior M_H=5, FC + warm-up first-obs** | 9 | (none) | 0 | 20 | 20 | 6 / 25 | 9732 |

## 3. Direction and size of the lambda moves

`lam` is the per-group log learning rate, initialised at `log(0.003)` = -5.8091. `net dlam` is the mean over the 6 groups of `lam_final - lam_0` (positive = LR raised); `LR ratio` is `exp(net dlam)`; `max abs dlam (traj)` is over the whole sampled trajectory and all groups. `LR down / up` counts open coordinate-steps by the direction of the step taken.

| arm | net dlam | LR ratio | max dlam | min dlam | max abs dlam (traj) | groups moved | open coord-steps LR down / up |
|---|---|---|---|---|---|---|---|
| fixed prior M_H=5, no FC, no warm-up | 1.6105+-0.1666 | 5.0675 | +2.5127 | +0.4120 | 2.5127 | 6.0 / 6 | 0.0 / 0.0 |
| fixed prior M_H=5, fail-closed, no warm-up | 1.6105+-0.1666 | 5.0675 | +2.5127 | +0.4120 | 2.5127 | 6.0 / 6 | 0.0 / 0.0 |
| adaptive envelope KAPPA=1, no warm-up | 1.6105+-0.1666 | 5.0675 | +2.5127 | +0.4120 | 2.5127 | 6.0 / 6 | 1.4 / 32.6 |
| **adaptive KAPPA=1 + warm-up first-obs** | -0.0000+-0.0000 | 1.0000 | -0.0000 | -0.0000 | 0.0000 | 6.0 / 6 | 0.0 / 0.0 |
| **adaptive KAPPA=1 + warm-up stable-env** | -0.0000+-0.0000 | 1.0000 | -0.0000 | -0.0000 | 0.0000 | 6.0 / 6 | 0.0 / 0.0 |
| **fixed prior M_H=5, FC + warm-up first-obs** | -0.0000+-0.0000 | 1.0000 | -0.0000 | -0.0000 | 0.0000 | 6.0 / 6 | 0.0 / 0.0 |

## 4. Retrospective consistency of every certified opening

For each gate opening at step `t`: the envelope in force at `t` and the NEXT probe's observed drift rate `M_obs`. The opening is *retrospectively consistent* iff that next observation does not exceed the envelope the opening was certified under. `cold-start` openings precede the run's first `M_obs` and can only be certified by the unverified floor. The FINAL envelope is the one the run ends with (the fixed prior 5.0 on the non-adaptive arms).

| arm | seed | openings | cold-start | with a following probe | consistent vs envelope in force | frac | consistent vs FINAL envelope | worst next-M_obs / envelope |
|---|---|---|---|---|---|---|---|---|
| adaptive envelope KAPPA=1, no warm-up | 0 | 8 | 8 | 8 | 0 | 0.0000 | 8 | 4.3535 |
| adaptive envelope KAPPA=1, no warm-up | 1 | 6 | 6 | 6 | 0 | 0.0000 | 6 | 2.3459 |
| adaptive envelope KAPPA=1, no warm-up | 2 | 12 | 12 | 12 | 0 | 0.0000 | 12 | 5.8048 |
| adaptive envelope KAPPA=1, no warm-up | 3 | 6 | 6 | 6 | 0 | 0.0000 | 6 | 1.6082 |
| adaptive envelope KAPPA=1, no warm-up | 4 | 8 | 8 | 8 | 0 | 0.0000 | 8 | 4.6807 |
| adaptive envelope KAPPA=1, no warm-up | 5 | 7 | 7 | 7 | 0 | 0.0000 | 7 | 3.1619 |
| adaptive envelope KAPPA=1, no warm-up | 6 | 9 | 9 | 9 | 0 | 0.0000 | 9 | 1.1858 |
| adaptive envelope KAPPA=1, no warm-up | 7 | 8 | 8 | 8 | 0 | 0.0000 | 8 | 3.0251 |
| adaptive envelope KAPPA=1, no warm-up | 8 | 13 | 13 | 13 | 0 | 0.0000 | 13 | 8.8076 |
| adaptive envelope KAPPA=1, no warm-up | 9 | 7 | 7 | 7 | 0 | 0.0000 | 7 | 2.9853 |
| **adaptive KAPPA=1 + warm-up first-obs** | 0 | 0 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **adaptive KAPPA=1 + warm-up first-obs** | 1 | 0 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **adaptive KAPPA=1 + warm-up first-obs** | 2 | 0 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **adaptive KAPPA=1 + warm-up first-obs** | 3 | 0 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **adaptive KAPPA=1 + warm-up first-obs** | 4 | 0 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **adaptive KAPPA=1 + warm-up first-obs** | 5 | 0 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **adaptive KAPPA=1 + warm-up first-obs** | 6 | 0 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **adaptive KAPPA=1 + warm-up first-obs** | 7 | 0 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **adaptive KAPPA=1 + warm-up first-obs** | 8 | 0 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **adaptive KAPPA=1 + warm-up first-obs** | 9 | 0 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **adaptive KAPPA=1 + warm-up stable-env** | 0 | 0 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **adaptive KAPPA=1 + warm-up stable-env** | 1 | 0 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **adaptive KAPPA=1 + warm-up stable-env** | 2 | 0 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **adaptive KAPPA=1 + warm-up stable-env** | 3 | 0 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **adaptive KAPPA=1 + warm-up stable-env** | 4 | 0 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **adaptive KAPPA=1 + warm-up stable-env** | 5 | 0 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **adaptive KAPPA=1 + warm-up stable-env** | 6 | 0 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **adaptive KAPPA=1 + warm-up stable-env** | 7 | 0 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **adaptive KAPPA=1 + warm-up stable-env** | 8 | 0 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **adaptive KAPPA=1 + warm-up stable-env** | 9 | 0 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **fixed prior M_H=5, FC + warm-up first-obs** | 0 | 0 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **fixed prior M_H=5, FC + warm-up first-obs** | 1 | 0 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **fixed prior M_H=5, FC + warm-up first-obs** | 2 | 0 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **fixed prior M_H=5, FC + warm-up first-obs** | 3 | 0 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **fixed prior M_H=5, FC + warm-up first-obs** | 4 | 0 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **fixed prior M_H=5, FC + warm-up first-obs** | 5 | 0 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **fixed prior M_H=5, FC + warm-up first-obs** | 6 | 0 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **fixed prior M_H=5, FC + warm-up first-obs** | 7 | 0 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **fixed prior M_H=5, FC + warm-up first-obs** | 8 | 0 | 0 | 0 | 0 | - | 0 | 0.0000 |
| **fixed prior M_H=5, FC + warm-up first-obs** | 9 | 0 | 0 | 0 | 0 | - | 0 | 0.0000 |

Totals over the 10 seeds:

| arm | openings | cold-start | with a following probe | consistent vs envelope in force | frac | consistent vs FINAL envelope | frac |
|---|---|---|---|---|---|---|---|
| adaptive envelope KAPPA=1, no warm-up | 84 | 84 | 84 | 0 | 0.0000 | 84 | 1.0000 |
| **adaptive KAPPA=1 + warm-up first-obs** | 0 | 0 | 0 | 0 | - | 0 | - |
| **adaptive KAPPA=1 + warm-up stable-env** | 0 | 0 | 0 | 0 | - | 0 | - |
| **fixed prior M_H=5, FC + warm-up first-obs** | 0 | 0 | 0 | 0 | - | 0 | - |

## 5. Paired tests (exact sign-flip, 10 seeds)

Differences are `arm - reference`, paired by seed.

| arm | reference | d NMSE | p | d events | p | d open coord-steps | p | d HVPs | p |
|---|---|---|---|---|---|---|---|---|---|
| **adaptive KAPPA=1 + warm-up first-obs** | adaptive envelope KAPPA=1, no warm-up | +0.036001 | 0.0020 | -9.0 | 0.0625 | -34.0 | 0.0020 | +15 | 1.0000 |
| **adaptive KAPPA=1 + warm-up first-obs** | fixed prior M_H=5, fail-closed, no warm-up | +0.036001 | 0.0020 | -9.0 | 0.0625 | -34.0 | 0.0020 | -7770 | 0.0156 |
| **adaptive KAPPA=1 + warm-up stable-env** | adaptive envelope KAPPA=1, no warm-up | +0.036001 | 0.0020 | -9.0 | 0.0625 | -34.0 | 0.0020 | +15 | 1.0000 |
| **adaptive KAPPA=1 + warm-up stable-env** | fixed prior M_H=5, fail-closed, no warm-up | +0.036001 | 0.0020 | -9.0 | 0.0625 | -34.0 | 0.0020 | -7770 | 0.0156 |
| **fixed prior M_H=5, FC + warm-up first-obs** | adaptive envelope KAPPA=1, no warm-up | +0.036001 | 0.0020 | -9.0 | 0.0625 | -34.0 | 0.0020 | +75724 | 0.0020 |
| **fixed prior M_H=5, FC + warm-up first-obs** | fixed prior M_H=5, fail-closed, no warm-up | +0.036001 | 0.0020 | -9.0 | 0.0625 | -34.0 | 0.0020 | +67939 | 0.0020 |
| **adaptive KAPPA=1 + warm-up stable-env** | **adaptive KAPPA=1 + warm-up first-obs** | +0.000000 | 1.0000 | +0.0 | 1.0000 | +0.0 | 1.0000 | +0 | 1.0000 |

## 6. What the monitor sees after the hold (seed 0)

**adaptive KAPPA=1 + warm-up first-obs**: 601 probes, 600 with an `M_obs`; max M_obs 146.53; final envelope 146.53; hold released at step 20; closed steps 20; openings 0 (first -, last -).

| probe step | M_obs | envelope after |
|---|---|---|
| 0 | - | 5 |
| 20 | 2.4795 | 5 |
| 40 | 27.209 | 27.209 |
| 41 | 146.53 | 146.53 |
| 60 | 11.192 | 146.53 |
| 80 | 0.3716 | 146.53 |
| 100 | 3.564 | 146.53 |
| 120 | 7.0437 | 146.53 |
| 140 | 11.292 | 146.53 |
| 160 | 0.66451 | 146.53 |
| 180 | 7.6143 | 146.53 |
| 200 | 6.9487 | 146.53 |
| 220 | 0.67789 | 146.53 |
| 240 | 1.7607 | 146.53 |
| ... | | |

**adaptive KAPPA=1 + warm-up stable-env**: 601 probes, 600 with an `M_obs`; max M_obs 146.53; final envelope 146.53; hold released at step 20; closed steps 20; openings 0 (first -, last -).

| probe step | M_obs | envelope after |
|---|---|---|
| 0 | - | 5 |
| 20 | 2.4795 | 5 |
| 40 | 27.209 | 27.209 |
| 41 | 146.53 | 146.53 |
| 60 | 11.192 | 146.53 |
| 80 | 0.3716 | 146.53 |
| 100 | 3.564 | 146.53 |
| 120 | 7.0437 | 146.53 |
| 140 | 11.292 | 146.53 |
| 160 | 0.66451 | 146.53 |
| 180 | 7.6143 | 146.53 |
| 200 | 6.9487 | 146.53 |
| 220 | 0.67789 | 146.53 |
| 240 | 1.7607 | 146.53 |
| ... | | |

**fixed prior M_H=5, FC + warm-up first-obs**: 1095 probes, 1094 with an `M_obs`; max M_obs 1273.7; final envelope 5; hold released at step 20; closed steps 9537; openings 0 (first -, last -).

| probe step | M_obs | envelope after |
|---|---|---|
| 0 | - | 5 |
| 20 | 2.4795 | 5 |
| 40 | 27.209 | 5 |
| 41 | 146.53 | 5 |
| 60 | 11.192 | 5 |
| 61 | 147.78 | 5 |
| 80 | 7.5151 | 5 |
| 81 | 68.916 | 5 |
| 100 | 0.87592 | 5 |
| 120 | 7.0703 | 5 |
| 121 | 58.04 | 5 |
| 140 | 7.6505 | 5 |
| 141 | 352.28 | 5 |
| 160 | 17.111 | 5 |
| ... | | |

## 7. Where the warm-up arms land: the fixed-LR-0.003 baseline

Once the gate never opens, COHG runs at its initial learning rate for the whole stream. The comparison below is against the `fixed` arm at lr = 0.003 on the same config (`results/e2`). The per-step losses are not bit-identical because the COHG path builds the HVP graph and reads the loss off the oracle (a different fp32 rounding of the same arithmetic), but they agree to fp32 round-off.

| arm | NMSE | events | max rel. NMSE diff vs fixed lr0.003 | max rel. per-step loss diff |
|---|---|---|---|---|
| fixed lr = 0.003 (reference) | 0.0522+-0.0022 | 0.0+-0.0 | - | - |
| **adaptive KAPPA=1 + warm-up first-obs** | 0.0522+-0.0022 | 0.0+-0.0 | 1.667e-05 | 2.889e-03 |
| **adaptive KAPPA=1 + warm-up stable-env** | 0.0522+-0.0022 | 0.0+-0.0 | 1.667e-05 | 2.889e-03 |
| **fixed prior M_H=5, FC + warm-up first-obs** | 0.0522+-0.0022 | 0.0+-0.0 | 1.667e-05 | 2.889e-03 |
| adaptive envelope KAPPA=1, no warm-up | 0.0162+-0.0021 | 9.0+-12.6 | 7.631e-01 | 9.411e-01 |

## 8. What the hold does to the monitor itself

`M_obs = |rho_probe - rho_prev| / (eta_max * D)` is deflated by the learning rate twice over (`eta_max` explicitly, and the path length `D` grows with `eta`), so freezing lambda at lr0 makes the SAME stream look far more non-stationary to the monitor. The clean comparison is between the two adaptive-envelope arms, which both take 601 probes on the scheduled 20-step cadence: median `M_obs` on seed 0 goes 0.1253 (no warm-up, LR raised ~5x) -> 9.030 (warm-up, LR frozen at 0.003), a 72x inflation. `max M_obs` is NOT comparable across arms with different probe schedules: a forced 1-step re-probe divides by a tiny `D`, which is why the fixed-prior fail-closed arm reports 234.7 on the trajectory where the adaptive arm reports 29.92 -- same run, 103.8 forced re-probes vs 2.4.

| arm | final envelope | max M_obs | median M_obs (seed 0) | closed steps | closure frac | fail-closed events | forced re-probes | HVPs | HVPs / no-FC |
|---|---|---|---|---|---|---|---|---|---|
| fixed prior M_H=5, no FC, no warm-up | 5 | - | - | - | - | - | - | 94588 | 1.000 |
| fixed prior M_H=5, fail-closed, no warm-up | 5 | 234.7 | 0.1253 | 939.7 | 0.0783 | 45.6 | 103.8 | 102602 | 1.085 |
| adaptive envelope KAPPA=1, no warm-up | 29.92 | 29.92 | 0.1253 | 8.6 | 0.0007 | 1.3 | 2.4 | 94816 | 1.002 |
| **adaptive KAPPA=1 + warm-up first-obs** | 212.7 | 212.7 | 9.03 | 25.7 | 0.0021 | 1.5 | 3.1 | 94831 | 1.003 |
| **adaptive KAPPA=1 + warm-up stable-env** | 212.7 | 212.7 | 9.03 | 25.7 | 0.0021 | 1.5 | 3.1 | 94831 | 1.003 |
| **fixed prior M_H=5, FC + warm-up first-obs** | 5 | 1488 | 19.57 | 9570.4 | 0.7975 | 167.5 | 979.9 | 170540 | 1.803 |

## 9. Plain answer

**Does insisting on a verified envelope keep any certified adaptation? No. On this configuration it removes all of it, and the cost is COHG's entire advantage over its own initial learning rate: NMSE 0.0162+-0.0021 -> 0.0522+-0.0022 (x3.23, paired d = +0.036001, exact sign-flip p = 0.0020), with the fixed-prior variant additionally paying 1.80x the hypergradient-vector-product budget.**

**1. Nothing survives the hold.** Over 10 seeds x 12000 steps, `first-obs`, `stable-env` and the fixed-prior `first-obs` reference all record **0 gate openings, 0 open coordinate-steps, and a lambda that never leaves its initial value** (`max |dlam|` over the whole trajectory = 0.0000 in all three arms). The reason is timing, not severity: without the hold the certified openings occupy steps 1-15 only (last opening per seed: 8, 6, 12, 6, 8, 7, 9, 8, 15, 7) and the gate never opens again in the remaining ~11985 steps, while the first `M_obs` cannot exist before step `2 * probe_every` = 20. The certified-adaptation window closes 5 steps before the earliest possible measurement. The hold itself is short -- 20 steps, released at step 20 (`first-obs`) or 20-40 (`stable-env`, when the first probe raises the envelope) -- and the two modes are *bit-identical* on all 10 seeds, because after step 20 there is nothing left to hold back.

**2. The suppressed openings are exactly the ones the paper reports.** In the no-warm-up run **100% of the 34.0+-8.2 open coordinate-steps per seed (84 open steps over the 10 seeds) fall in steps 1-15**, i.e. entirely inside the hold -- so the hold does not remove *most* of COHG's certified adaptation on this configuration, it removes all of it. On the held trajectory itself the read-only counter says the gate would have opened 30.5+-8.3 coordinate-steps during the 20-step hold; that counterfactual differs slightly from 34.0 because once the first opening is suppressed lambda stops moving, so the later would-have-opened steps are evaluated on a different trajectory.

**3. The cost is the whole method.** With lambda frozen the run is the fixed-LR-0.003 baseline to fp32 round-off (section 7: NMSE agrees with `results/e2` fixed lr0.003 to 1.7e-05 relative). NMSE goes 0.0162+-0.0021 -> 0.0522+-0.0022, i.e. **x3.23 worse**, p = 0.0020 on both the adaptive and the fixed-prior no-warm-up reference; open coordinate-steps -34.0 (p = 0.0020). The one thing that improves is instability: events 9.0+-12.6 -> 0.0+-0.0 (p = 0.0625), because the LR never rises. Certificate validity is untouched and was never at issue: **0 violations in 720000 audited coordinate-steps in every warm-up arm**, worst |ghat - g_true| / beta ratio 0.8241 (vs 0.8374 without the hold).

**4. A second, less obvious cost: the monitor gets much more expensive once the LR stops rising.** `M_obs` is deflated by the learning rate (explicitly through `eta_max`, and again through the path length `D`), so the SAME stream looks far more non-stationary when lambda is frozen at lr0 than when COHG has raised it ~5x. At the same 601-probe schedule the seed-0 median `M_obs` rises 0.1253 -> 9.030 (72x) and the max 21.77 -> 146.5. The envelope the adaptive arm ends at therefore rises 29.92 -> 212.7, and for the FIXED prior `M_H = 5` the monitor is then violated essentially all the time: median `M_obs` on seed 0 19.57, **closure fraction 0.7975 (9570 of 12000 steps closed)**, 167.5 fail-closed transitions, 979.9 forced re-probes, HVPs 170540 = **1.803x** the no-monitor budget (vs 1.085x without the hold; paired d = +67939 HVPs vs the fixed-prior no-warm-up arm, p = 0.0020). The adaptive envelope absorbs this (1.003x), which is the one place where the online-enforced envelope of round4_adaptmh clearly pays for itself.

**5. What this means for the claim.** The honest reading is that on E2 `mackey_drift` the certificate's cold start is *load-bearing*: every certified update COHG makes is taken under a drift envelope that no measurement has yet had the chance to support, and the measurement that arrives first (step 20) exceeds the deployed floor on all ten seeds. `round4_adaptmh.md` showed the envelope can be re-stated online for free; this run shows that requiring it to be *verified before use* is not free -- it costs the method. The two defensible positions are therefore (a) state the prior as an assumption of the theorem and report, as here, that the whole gain arrives in the first 15 steps under that assumption, or (b) shorten `probe_every` (or add a step-0 probe pair) so a measurement exists before the adaptation window opens -- which this experiment does not test, and which would change the probe budget rather than the certificate. What is NOT defensible is claiming the openings are backed by the drift diagnostic: 0 of 84 are.

