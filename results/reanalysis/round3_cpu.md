# Round-3 CPU experiments (review P6 / P10 / P1 / P2 + R2 minor)

Config shared by every E2 arm below: `mackey_drift`, mis-set init `lr0=0.003`, 12000 steps, `gamma 0.9` (B3 sweeps it), `kw-eps 0.1`, `probe-every 20`, `K 10`, `rank 4`, gate factor `c=2`, `alpha (--meta-lr) 0.4`, seeds 0-9, **device CPU** -- identical to `launch_e2_controls.py`, so every number is directly comparable to `results/e2_controls/SUMMARY.md`.

All three grids are complete (10/10 seeds for B1 and B2, 10 seeds x 4 gammas for B3). Statistics are `ddof=1` sample std; every paired test is an EXACT two-sided sign-flip permutation test enumerating all `2^10 = 1024` sign assignments of the per-seed paired differences (so the smallest attainable p is `2/1024 = 0.00195`, printed as 0.0020).

Regime switches of the `mackey_drift` stream land at **steps 4004 and 8007**. Derivation: `mackey_glass_drift(n=25000)` switches tau at series indices `seg = 25000//3 = 8333` and `2*seg = 16666`; `OrderedWindowStream` at step `t` draws window starts from a 500-wide recency band ending at `center(t) = int(t/(T-1) * (n_win - 1))` with `n_win = 25000 - 20 - 1 = 24979` and `T = 12000`; the first `t` with `center(t) >= 8333` is 4004 and with `center(t) >= 16666` is 8007. (These are the steps at which post-switch samples first enter the batch, not the nominal `T/3` markers 4000/8000.)

## 0. Reproduction check (`results/e2_verify`: the patched script is additive)

`launch_round3.py --part verify` re-runs the frozen COHG reference arm (`cohg`, `M_H=5`, `--validate-cert`, no fail-closed) on the PATCHED `e2_timeseries.py` for seeds 0 and 1. Its only purpose is to prove that the round-3 additions (`madgate`, `ogd_doubling`, `--validate-full`, the forced-re-probe bookkeeping) left the legacy code path bit-identical.

| seed | ref NMSE (`e2_controls`) | re-run NMSE (`e2_verify`) | events | coord-open | HVPs | cert viol | cert max ratio | full `losses` list | full `lam_hist` |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.014882455106279844 | 0.014882455106279844 | 0 / 0 | 4.583333e-04 / 4.583333e-04 | 94588 / 94588 | 0 / 0 | 0.750368 / 0.750368 | identical | identical |
| 1 | 0.018453745308905131 | 0.018453745308905131 | 0 / 0 | 3.611111e-04 / 3.611111e-04 | 94588 / 94588 | 0 / 0 | 0.584560 / 0.584560 | identical | identical |

**BIT-IDENTICAL** on both seeds -- not merely equal summary statistics: the complete 12000-entry `losses` array and the complete `lam_hist` match element for element. The patch is behaviour-preserving.

Second, independent instance of the same check: the B3 `gamma=0.9` arm in `results/e2_gamma` is the reference config plus `--validate-full`, and it reproduces all ten reference seeds exactly (NMSE, events, coord-open, HVPs) -- confirming that the extra `gamma=1` validation recursion is a pure observer.

## B1. Online calibration-free threshold gate (`madgate`, review P6)

Rule: coordinate `j` opens iff `|ghat_j| > c * MAD_t(|ghat_j|)`, where `MAD_t` is the running median-absolute-deviation of `|ghat_j|` over the last 200 steps (strictly past values -- the current step is appended after the test), the gate is held shut for the first 50 steps while the window fills, and `c = 2` is the SAME constant COHG uses. No certificate, no spectral probe, no calibration seeds, no held-out data -- everything is read off the run's own hypergradient stream. The estimator, the `alpha = 0.4` sign step and the box `[log 1e-5, log 1]` are unchanged.

| arm | n | NMSE mean+-std | NMSE med | events mean+-std | coord-open rate | steps-with-open | lambda window | HVPs | HVPs/step | wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| COHG (certificate gate) | 10 | 0.0162+-0.0021 | 0.0156 | 9.0+-12.6 | 4.72e-04 | 7.00e-04 | 50-50 | 94588 | 7.882 | 5102 |
| B absgate (offline-calibrated const threshold) | 10 | 0.0150+-0.0032 | 0.0141 | 10.4+-13.2 | 2.61e-04 | 1.27e-03 | 100-150 | 11992 | 0.999 | 609 |
| **B1 madgate (online, calibration-free)** | 10 | **0.0070+-0.0016** | 0.0076 | **94.2+-45.9** | **3.458e-01** | 8.451e-01 | 50-11950 | 11992 | 0.999 | 627 |
| A sign-nogate alpha=0.4 (no gate at all) | 10 | 0.0031+-0.0011 | 0.0031 | 105.0+-48.2 | 1.00e+00 (ungated) | 1.00e+00 | 50-11950 | 11992 | 0.999 | 665 |
| fixed lr=0.003 (un-adapted; GPU) | 10 | 0.0522+-0.0022 | 0.0525 | 0.0+-0.0 | - | - | never | 0 | 0 | 394 |

`lambda window` = median over seeds of the first and last sampled step (`lam_hist`, every 50 steps) at which lambda actually moved. For madgate it is 50-11950 on **every one of the ten seeds** -- lambda is still moving at the last sample. For COHG it is 50-50 on every seed.

Per-seed madgate NMSE: 0.00497 0.00782 0.00420 0.00624 0.00729 0.00809 0.00784 0.00790 0.00584 0.00950.
Per-seed madgate events: 33 109 88 160 103 28 63 117 161 80.
Per-seed madgate coord-open: 3.503e-01 3.486e-01 3.509e-01 3.469e-01 3.445e-01 3.410e-01 3.380e-01 3.324e-01 3.470e-01 3.582e-01 (a tight band -- this is not a seed accident).

### Why the rate is 730x COHG's

- Mean MAD base over the run: **4.0077e-04** (per-seed 2.453e-04 to 6.140e-04), so the realized threshold is `c * MAD = 8.0153e-04`.
- absgate's offline-frozen constant threshold, fitted to reproduce COHG's open rate, is **0.05807** -- **72.4x larger** than madgate's threshold.
- For scale, the top-400 `|ghat|` order statistics dumped by the absgate calibration run (seed 0) span 0.00704 to 0.115.

A median-absolute-deviation measures the *temporal dispersion* of `|ghat_j|`, which is small because `|ghat_j|` is a slowly-varying quantity; the certificate's `beta_col_j` measures the *estimation error* of `ghat_j`, which is two orders of magnitude larger. The two are not on the same scale, so the same constant `c=2` produces a threshold ~72x too permissive and the gate degenerates: 34.6% of coordinate-steps open, at least one coordinate opens on 84.5% of steps, and lambda random-walks over the whole box for 12000 steps (final per-group LRs run from the lower wall 1e-5 to the upper wall 1.0).

### Paired tests (exact sign-flip permutation, n=10; negative delta = the first arm is BETTER)

| comparison | delta NMSE | p | delta events | p |
|---|---|---|---|---|
| madgate - COHG | -0.00920 | 0.0020 | +85.2 | 0.0020 |
| madgate - absgate | -0.00807 | 0.0020 | +83.8 | 0.0020 |
| madgate - sign-nogate(alpha=0.4) | +0.00390 | 0.0020 | -10.8 | 0.1406 |

madgate differs from COHG on BOTH axes at the smallest attainable p; it differs from the ungated sign arm on NMSE but **not** on instability events (p=0.14).

## B2. Prospective doubling schedule for the projected-gradient controller (review P10)

`ogd_doubling` keeps COHG's certificate gate and the magnitude-aware projected-OGD step (`mode='ogd'`), but replaces the fixed `alpha=0.4` with the theory-prescribed prospective schedule `alpha_tau = D / (G_k sqrt(tau))`, `D = lam_max - lam_min = 11.5129` (the box width), `tau = 1..2^k` inside doubling epoch `k` (epoch `k` spans steps `[2^k - 1, 2^(k+1) - 1)`), and `G_k` the per-coordinate running max of `|ghat_j| + beta_j` over all steps strictly BEFORE the epoch began (a coordinate with no history yet bootstraps `G` from its current observation). The comparison arm `cohg_ogd` is the identical controller with `alpha` fixed at 0.4, re-run on the SAME device (CPU).

| arm | n | NMSE mean+-std | NMSE med | events mean+-std | coord-open rate | steps-with-open | lambda window | HVPs | HVPs/step |
|---|---|---|---|---|---|---|---|---|---|
| cohg_ogd fixed alpha=0.4 (CPU, same device) | 10 | 0.0516+-0.0019 | 0.0518 | 0.0+-0.0 | 4.236e-04 | 7.00e-04 | 50-50 | 94588 | 7.882 |
| cohg_ogd fixed alpha=0.4 (GPU, `results/e2`) | 10 | 0.0516+-0.0019 | 0.0518 | 0.0+-0.0 | 4.236e-04 | 7.00e-04 | 50-50 | 94588 | 7.882 |
| **B2 ogd_doubling (prospective schedule)** | 10 | **1.6992+-0.7692** | 2.0569 | **24.8+-78.4** | 1.653e-04 | 1.833e-04 | 50-50 | 94588 | 7.882 |
| COHG sign step (certificate gate, alpha=0.4) | 10 | 0.0162+-0.0021 | 0.0156 | 9.0+-12.6 | 4.72e-04 | 7.00e-04 | 50-50 | 94588 | 7.882 |
| fixed lr=0.003 (un-adapted; GPU) | 10 | 0.0522+-0.0022 | 0.0525 | 0.0+-0.0 | - | - | never | 0 | 0 |

(The CPU and GPU `cohg_ogd` arms agree to ~1e-8 relative on every seed but are not bit-identical; they round to the same 4 decimals.)

Per-seed doubling NMSE: 1.2223 2.4961 0.8861 2.2286 2.0921 2.0876 1.5677 2.0262 **0.0572** 2.3276. Per-seed events: 0 0 0 0 0 0 0 0 **248** 0.

**The realized `alpha_t` schedule, in one sentence:** it starts at `D/1e-12 = 1.15e13` on step 0 (where `ghat` and `beta` are still exactly zero so `G_k` hits its `1e-12` clamp), then decays as `1/sqrt(tau)` inside each doubling epoch while `G_k` ratchets upward at every epoch boundary, so that by step 50 the per-coordinate `alpha` already spans 1.19e-13 to 2.71e+05 across seeds (median range [0.047, 0.43]) and by step 11950 it spans 0 to 1.43e+03 (median range [0.0035, 0.032]) -- i.e. it is never near the fixed 0.4 at the moment it matters and its spread across seeds and coordinates covers eighteen orders of magnitude.

Sampled every 50 steps, pooled over the 10 seeds:

| step | alpha_min: median [min, max] over seeds | alpha_max: median [min, max] over seeds |
|---|---|---|
| 0 | 1.151e+13 [1.15e+13, 1.15e+13] | 1.151e+13 [1.15e+13, 1.15e+13] |
| 50 | 0.0473 [1.19e-13, 4.22e+04] | 0.432 [5.33e-10, 2.71e+05] |
| 200 | 0.0246 [0, 1.45e+03] | 0.225 [0, 7.29e+04] |
| 1000 | 0.00956 [0, 3.39e+02] | 0.0873 [0, 7.42e+03] |
| 6000 | 0.00485 [0, 1.56e+02] | 0.0443 [0, 3.06e+03] |
| 11950 | 0.00345 [0, 7.58e+01] | 0.0315 [0, 1.43e+03] |

### Mechanism: the first certified step is fatal

Every doubling seed does all of its lambda movement in the first 50 steps (lambda window 50-50) with the step size still at 1e13-1e5, and lands on a **wall of the lambda box**:

- 9 of 10 seeds put ALL six group LRs on the LOWER wall `1e-5` (300x below the already mis-set init 0.003) and stay there: the model barely learns, giving NMSE 0.89-2.50 with 0 instability events (a frozen tiny LR cannot blow up).
- seed 8 puts two of six groups on the UPPER wall `1.0`, giving NMSE 0.0572 with 248 events -- the entire events mean and its 78.4 std come from this single seed, which is why the events test is p=1.0000 (nine of ten paired differences are exactly zero).

By contrast `cohg_ogd` at fixed `alpha=0.4` moves the LRs by <8% (final per-group LRs 0.00300-0.00324) and simply stays near the mis-set init.

| comparison | delta NMSE | p | delta events | p |
|---|---|---|---|---|
| doubling - fixed alpha (both CPU) | +1.64756 | 0.0020 | +24.8 | 1.0000 |
| doubling - COHG sign step | +1.68299 | 0.0020 | +15.8 | 1.0000 |

### B2 addendum (round-35 review, R1 minor): where the first certified step actually falls

The review asked whether the first accepted update really uses the no-history bootstrap, since `ghat = beta = 0` at `t=0` means no coordinate can open there and any later step has a prior step behind it.  Re-run of the same ten configs under the same launcher arguments with the per-step `(G_k, alpha, open_mask)` logged (instrumented copy of `e2_timeseries.py`, deleted after the measurement; identical seeds, stream and driver flags, verified against the shipped `ogd_alpha_log[0] = 1.1512925e13` and against each seed's stored `gate_open_frac * steps`):

| seed | first open | epoch k | tau | pre-epoch Gmax | Gk (abs ghat at t=1) min..max | alpha_1 min..max | coords opened | all accepted opens in the run |
|---|---|---|---|---|---|---|---|---|
| 0 | t=1 | 1 | 1 | 0 on all 6 | 1.23e-05 .. 7.93e-04 | 1.45e+04 .. 9.36e+05 | 6/6 | 1, 2 |
| 1 | t=1 | 1 | 1 | 0 on all 6 | 2.77e-04 .. 7.08e-03 | 1.63e+03 .. 4.15e+04 | 6/6 | 1, 2 |
| 2 | t=1 | 1 | 1 | 0 on all 6 | 1.47e-09 .. 7.89e-06 | 1.46e+06 .. 7.82e+09 | 6/6 | 1, 2, 3, 4 |
| 3 | t=1 | 1 | 1 | 0 on all 6 | 2.22e-04 .. 7.70e-03 | 1.50e+03 .. 5.19e+04 | 6/6 | 1, 2 |
| 4 | t=1 | 1 | 1 | 0 on all 6 | 1.65e-05 .. 3.18e-03 | 3.62e+03 .. 6.96e+05 | 6/6 | 1, 2 |
| 5 | t=1 | 1 | 1 | 0 on all 6 | 5.78e-05 .. 4.32e-03 | 2.67e+03 .. 1.99e+05 | 6/6 | 1, 2 |
| 6 | t=1 | 1 | 1 | 0 on all 6 | 2.95e-07 .. 1.01e-03 | 1.15e+04 .. 3.91e+07 | 6/6 | 1, 2 |
| 7 | t=1 | 1 | 1 | 0 on all 6 | 3.33e-05 .. 3.61e-03 | 3.19e+03 .. 3.46e+05 | 6/6 | 1, 2 |
| 8 | t=1 | 1 | 1 | 0 on all 6 | 1.13e-10 .. 2.52e-06 | 4.57e+06 .. 1.02e+11 | 6/6 | 1, 2 |
| 9 | t=1 | 1 | 1 | 0 on all 6 | 1.38e-04 .. 5.15e-03 | 2.24e+03 .. 8.34e+04 | 6/6 | 1, 2 |

Findings, all ten seeds identical in structure:

1. **Nothing opens at `t=0`.**  `ghat = beta = 0` there, so `|ghat_j| > c * beta_j` is false and the `1e13` step size of `ogd_alpha_log[0]` is never applied to anything.  It is a logged artefact of the `1e-12` clamp, not the step that does the damage.
2. **The first accepted opening is `t=1` on every seed, and all six coordinates open at once.**  `t=1` is the first step of doubling epoch `k=1` (epoch 1 spans `[2^1-1, 2^2-1) = [1,3)`), so `tau=1`.
3. **The bootstrap branch does fire there.**  `ogd_Gmax` is updated only *after* the step size is fixed, and its only prior contribution is step 0's `|ghat|+beta = 0`, so the running maximum carried into epoch 1 is **exactly zero on all six coordinates** and `torch.where(ogd_Gmax > 0, ogd_Gmax, cur_G)` takes the current observation.  The manuscript's "bootstrap value" wording is therefore correct as written; what was missing was why a step with a predecessor still has an empty history.
4. **`beta` is still exactly zero at `t=1` as well**, so `G_{1,j} = |ghat_{1,j}| + beta_{1,j} = |ghat_{1,j}|` (1.13e-10 to 7.70e-03 across seeds and coordinates) and `alpha_{1,j} = D / |ghat_{1,j}|` (1.50e+03 to 1.02e+11, i.e. four to eleven orders of magnitude above the calibrated 0.4).
5. **The step is exactly the box width.**  `CoordGatedController.maybe_update` in `mode="ogd"` takes `step = meta_lr * ghat`, so `alpha_{1,j} * ghat_{1,j} = +-D = +-11.5129` identically, independent of the seed and of how small `|ghat|` happens to be.  Every open coordinate is therefore thrown from `lam_0 = -5.809` to a clamp of the box in that single step, which is why the arm's outcome is decided at `t=1` rather than accumulated.
6. **The gate then shuts for the rest of the run.**  The only further accepted openings are `t=2` on every seed and `t=3, t=4` on seed 2, matching each seed's stored `gate_open_frac * steps` exactly (2 steps, 4 on seed 2).  Nothing opens over the remaining ~11,995 steps.

So the "first 50 steps" bracket quoted from the 50-step `lam_hist` grid is really steps 1-2 (1-4 on one seed), and the causal statement in `sections/gating.tex` and around Table `tab:doubling` is now written against these numbers.

## B3. Discounted vs full-horizon hypergradient sign agreement (review P1)

Each run carries a SECOND exact FMD recursion at `gamma = 1` (the full-horizon sensitivity `S_t`, m HVPs/step, not charged to the method's HVP budget) alongside COHG, plus the `--validate-cert` exact DISCOUNTED recursion. Per step and coordinate the run logs `sign(ghat_t,j)` (what the controller acts on) against `sign(g_full_t,j)`, and separately `sign(g_disc_exact_t,j)` against `sign(g_full_t,j)` -- which separates the discounting bias from the sketch/lazy estimation error. JSON keys: `full_finite`, `full_agree`, `full_open`, `full_open_agree`, `full_nz`, `full_nz_agree`, `full_disc_agree`, `full_disc_nz`, `full_disc_nz_agree`, `full_n_coord` (=6).

Agreement rates below are pooled coordinate-step counts (they agree with the mean-of-per-seed-rates to <=0.0004). "all"/"post"/"else" are restricted to coordinate-steps where BOTH signs are nonzero (`full_nz`); the gate-OPEN column uses `full_open_agree / full_open`, and an open coordinate necessarily has `sign(ghat) != 0`.

| gamma | n | NMSE mean+-std | events mean+-std | coord-open | S_full finite frac | sign agree (all) | sign agree (gate OPEN) | agree, 200 steps after a switch | agree elsewhere | EXACT-discounted vs full (all) | EXACT-disc, post-switch | cert viol |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.8 | 10 | 0.0260+-0.0032 | 5.0+-10.2 | 3.083e-04 | 1.000 | 0.7588 | **0.9955** (221/222) | 0.7312 | 0.7597 | 0.6740 | 0.6633 | 0/720000 |
| 0.9 | 10 | 0.0162+-0.0021 | 9.0+-12.6 | 4.722e-04 | 1.000 | 0.7370 | **1.0000** (340/340) | 0.6724 | 0.7392 | 0.6493 | 0.6143 | 0/720000 |
| 0.95 | 10 | 0.0123+-0.0019 | 10.7+-13.3 | 5.653e-04 | 1.000 | 0.7637 | **1.0000** (407/407) | 0.7090 | 0.7656 | 0.6506 | 0.6254 | 0/720000 |
| 0.99 | 10 | 0.0095+-0.0018 | 11.7+-14.6 | 6.597e-04 | 1.000 | 0.8818 | **1.0000** (475/475) | 0.8512 | 0.8828 | 0.7482 | 0.7171 | 0/720000 |

- **Finite full-horizon sensitivity: 1.000 at every gamma, every seed, every one of the 12000 steps.** The `gamma=1` recursion never overflowed (`full_blown` never fired), so no step is excluded from any rate above and the comparison is exhaustive.
- Denominators: `all` uses 719940 = 720000 - 60 coordinate-steps; the 60 dropped are step `t=0` on each of the 10 seeds x 6 coordinates, where `ghat` is still exactly zero. `post` uses exactly 24000 = 2 windows x 200 steps x 6 coords x 10 seeds.
- Pooled over all four gammas: **1443 of 1444 gate-OPEN coordinate-steps carry the full-horizon sign (0.99931)**. The single exception is `gamma=0.8`, seed 5, step 2391.
- Certificate audit on the gamma arms: 0 violations out of 720000 checked coordinate-steps at every gamma; worst `|ghat-g_true|/beta_col` ratio 0.833 / 0.837 / 0.845 / 0.917 for gamma 0.8 / 0.9 / 0.95 / 0.99.
- Paired NMSE tests against `gamma=0.9` (n=10, exact): `0.8` +0.00981 (p=0.0020), `0.95` -0.00387 (p=0.0020), `0.99` -0.00665 (p=0.0020). Events: -4.0 (p=0.1250), +1.7 (p=0.5000), +2.7 (p=0.1250). Larger gamma monotonically lowers NMSE and raises the open rate; the events differences are not significant.

### The post-switch window contains no gate-open steps

Last step at which ANY coordinate opened, per seed:

| gamma | last open step, seeds 0-9 |
|---|---|
| 0.8 | 7, 4, **2556**, 4, 6, **2391**, 8, 6, **2910**, 5 |
| 0.9 | 8, 6, 12, 6, 8, 7, 9, 8, 15, 7 |
| 0.95 | 9, 7, 12, 7, 8, 8, 11, 8, 14, 8 |
| 0.99 | 9, 7, 12, 8, 9, 8, 11, 9, 14, 8 |

Open coordinate-steps falling inside the two 200-step post-switch windows ([4004, 4204) and [8007, 8207)): **0, at every gamma.** COHG's adaptation is over by step ~15 (by step ~2910 at gamma=0.8), thousands of steps before the first regime switch, so "does gate-open agreement degrade after a switch" has an empty sample here -- the `open_post` rate is undefined, not low. The post-switch degradation reported in the table is measured over ALL coordinate-steps, gate-shut included.

## B4. Drift-prior (M_H) misspecification with exact ground truth (review P2)

E1 teacher-student, tier `kw_drift`, `gamma 0.9`, `K 10`, `r 4`, 1000 steps, seeds 0-4, fp64, exact `ExactFMD` ground truth. A violation is `e_t < ||S_t - S_hat_t||_F` on ANY step. `M_H*` is calibrated as the LARGEST probe-to-probe observed drift rate `M_obs = |rho_probe - rho_prev| / (eta_max * D)` on 3 calibration seeds (100-102) disjoint from the evaluation seeds: **M_H* = 2.276** (pooled median 0.278, p99 1.91). The legacy E1/E2 prior of 5.0 is therefore already ~2.2x conservative here.

| M_H / M_H* | M_H | fail-closed | n | violation rate | worst true err / bound (max over seeds) | closure fraction | probes / nominal | valid rate |
|---|---|---|---|---|---|---|---|---|
| 1 | 2.276 | no | 5 | 0.0000 | 0.709 | 0.000 | 1.000 | 1.0000 |
| 1 | 2.276 | yes | 5 | 0.0000 | 0.709 | 0.049 | 1.054 | 1.0000 |
| 0.3 | 0.6828 | no | 5 | 0.0000 | 0.73 | 0.000 | 1.000 | 1.0000 |
| 0.3 | 0.6828 | yes | 5 | 0.0000 | 0.73 | 0.385 | 1.456 | 1.0000 |
| 0.1 | 0.2276 | no | 5 | 0.0000 | 0.735 | 0.000 | 1.000 | 1.0000 |
| 0.1 | 0.2276 | yes | 5 | 0.0000 | 0.735 | 0.758 | 1.800 | 1.0000 |
| 0.03 | 0.06828 | no | 5 | 0.0000 | 0.738 | 0.000 | 1.000 | 1.0000 |
| 0.03 | 0.06828 | yes | 5 | 0.0000 | 0.738 | 0.919 | 1.926 | 1.0000 |
| 0.01 | 0.02276 | no | 5 | 0.0000 | 0.738 | 0.000 | 1.000 | 1.0000 |
| 0.01 | 0.02276 | yes | 5 | 0.0000 | 0.738 | 0.967 | 1.968 | 1.0000 |

### The E2 points already measured (second series in Fig. 10)

E2 GRU, `mackey_drift`, per-coordinate certificate audit `|ghat_j - g_true_j| > beta_col_j` against a parallel exact DISCOUNTED FMD, 10 seeds, 12000 steps. `M_H*` for E2 is the largest observed drift rate of the M_H=5 fail-closed arm: **761.5**, so the paper's M_H=5 is already 0.007x M_H* -- i.e. every E2 point is on the UNDER-specified side.

| M_H / M_H* | M_H | fail-closed | n | violation rate | worst \|ghat-g\|/beta (max over seeds) | closure fraction | HVPs / no-FC baseline |
|---|---|---|---|---|---|---|---|
| 6.57e-03 | 5 | no | 10 | 0.0000 | 0.837 | 0.000 | 1.000 |
| 6.57e-03 | 5 | yes | 10 | 0.0000 | 0.837 | 0.078 | 1.085 |
| 6.57e-04 | 0.5 | no | 10 | 0.0000 | 0.838 | 0.000 | 1.000 |
| 6.57e-04 | 0.5 | yes | 10 | 0.0000 | 0.838 | 0.590 | 1.595 |
| 6.57e-05 | 0.05 | no | 10 | 0.0000 | 0.838 | 0.000 | 1.000 |
| 6.57e-05 | 0.05 | yes | 10 | 0.0000 | 0.838 | 0.927 | 1.920 |

Figure: `paper/main/figs/fig10_misspec.pdf` (PNG in `results/figures/fig10_misspec.png`).

**Reading.** Over a 100x sweep of M_H below its calibrated value, on BOTH problems and with exact ground truth, the certificate is never violated: violation rate is 0.0000 at every point and the worst true-error/bound ratio stays flat at 0.71-0.74 (E1) and 0.84 (E2) -- the bound is never even approached within 15%. What DOES move, monotonically and steeply, is the fail-closed monitor: closure fraction rises 0.05 -> 0.39 -> 0.76 -> 0.92 -> 0.97 across M_H/M_H* = 1 -> 0.3 -> 0.1 -> 0.03 -> 0.01 on E1 (0.08 -> 0.59 -> 0.93 on E2), and the probe budget with it (1.05x -> 1.97x HVPs on E1, 1.08x -> 1.92x on E2). So: (i) the M_H drift term is NOT what binds the certificate in either regime; (ii) the fail-closed monitor is a correct and loud DETECTOR of a wrong prior, at a probe cost that saturates at 2x; (iii) M_H is a soft knob here, not a load-bearing assumption, and the monitor's value is that a badly wrong M_H announces itself online instead of silently invalidating the certificate.

## B5. Device-stratified sensitivity (R2 minor)

Full tables in `results/reanalysis/device_sensitivity.md`. Summary:

| arm | n | mean \|dNMSE\| | median rel | max rel | mean \|d ev\| | regime flips (ev>30 or NMSE>1) | divergence flips (NMSE>1) | gate-rate differs | GPU mean | CPU mean | GPU median | CPU median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cohg | 10 | 1.131e-03 | 3.78e-05 | 2.16e-01 | 0.00 | 0 | 0 | 4 | 0.0150 | 0.0162 | 0.0147 | 0.0156 |
| cohg_nogate | 10 | 5.982e+00 | 1.40e-07 | 1.00e+00 | 25.10 | 0 | 1 | 0 | 5.9849 | 0.0031 | 0.0030 | 0.0031 |

## Plain answers

**(1) Does the online calibration-free MAD gate match COHG, and at what HVP cost?** **No -- it does not reproduce COHG's operating point on either axis, and the HVP saving is real but beside the point.** madgate lands at NMSE 0.0070+-0.0016 with 94.2+-45.9 instability events, against COHG's 0.0162+-0.0021 at 9.0+-12.6 (paired dNMSE -0.00920, p=0.0020; d events +85.2, p=0.0020 -- the smallest p an exact 10-seed sign-flip test can produce, i.e. all ten seeds move the same way on both axes). It is 2.3x better on NMSE and 10x worse on stability; that is a different point on the frontier, not COHG's point. Where it *does* land is essentially on the UNGATED sign arm: madgate vs `sign-nogate(alpha=0.4)` is +0.00390 on NMSE and -10.8 on events with p=0.14 -- statistically indistinguishable in instability. The reason is a scale mismatch, not a tuning accident: a running MAD of `|ghat_j|` measures temporal dispersion (mean base 4.008e-04, so `c*MAD = 8.02e-04`) while COHG's `c*beta_col` measures estimation error (absgate's rate-matched constant surrogate is 0.05807, **72.4x larger**). With `c=2` applied to the wrong scale the gate opens on 34.6% of coordinate-steps and 84.5% of steps -- versus COHG's 0.047% and 0.07% -- and lambda keeps moving to step 11950 on every seed instead of freezing at step 50, ending with per-group LRs scattered from the 1e-5 wall to the 1.0 wall. The HVP cost it avoids is genuine: 11992 HVPs (0.999/step) vs COHG's 94588 (7.882/step), a **7.89x** saving, because it never runs the spectral probe -- but it buys nothing here, since absgate obtains the same 7.89x saving AND matches COHG's NMSE/events (0.0150+-0.0032, 10.4+-13.2, p=0.40/0.44 in round 2). The honest conclusion for the P6 reply: the *cheap* part of absgate (no spectral probe) transfers; the *calibration-free* part does not, because no self-referential statistic of the `|ghat|` stream reconstructs the certificate's scale. Making the gate calibration-free by using a MAD threshold does not yield a certificate-free COHG -- it yields the ungated method.

**(2) Does the prospective doubling schedule help or hurt versus a fixed alpha?** **It hurts, catastrophically and unambiguously.** `ogd_doubling` lands at NMSE 1.6992+-0.7692 (median 2.0569) against the same-device fixed-alpha `cohg_ogd` at 0.0516+-0.0019 with 0.0+-0.0 events: paired dNMSE +1.64756, p=0.0020 (all ten seeds worse). It is 33x worse than the fixed-alpha controller and 32x worse than doing nothing at all (fixed lr=0.003 gives 0.0522). The events comparison is +24.8 at p=1.0000 and is meaningless: nine of ten seeds have exactly 0 events and the entire mean comes from seed 8. The mechanism is fully visible in the schedule and in the final LRs: `alpha_tau = D/(G_k sqrt(tau))` with `D = 11.5129` starts at 1.15e13 on step 0 (both `ghat` and `beta` are still zero, so `G_k` hits its 1e-12 clamp) and is still spread over 1.19e-13 to 2.71e+05 across seeds at step 50 -- precisely the window in which the certificate gate opens (all ten seeds do all of their lambda movement inside the first 50 steps). The first certified coordinate therefore gets a step 4-13 orders of magnitude too large and is thrown onto a wall of the lambda box, where the gate then shuts and it stays: 9 of 10 seeds park ALL six group LRs on the lower wall 1e-5 (300x below the already mis-set init) and under-fit into NMSE 0.89-2.50 with zero events, while seed 8 parks two groups on the upper wall 1.0 and takes 248 events at NMSE 0.0572. The schedule's asymptotic `1/sqrt(tau)` decay is irrelevant because by the time alpha reaches the 0.4 neighbourhood (median range [0.0035, 0.032] at step 11950, and only 5 of 10 seeds ever have `alpha_max > 0.4` at step 50) the gate has been shut for thousands of steps. This is the same conclusion the round-2 Theorem-6 clipping control (`t6clip`) reached from the opposite direction: there the theory-prescribed step was ~7x too SMALL and froze the controller, here it is orders of magnitude too LARGE and slams it into the constraint set. Worst-case-optimal step-size theory and this operating point are not compatible; the fixed alpha is doing real work and should be presented as a tuned hyperparameter, not derived from regret bounds.

**(3) Does the discounted sign track the full-horizon sign where the gate opens, and does agreement degrade after switches or at small gamma?** **Where the gate opens: yes, essentially perfectly -- 1443 of 1444 gate-open coordinate-steps pooled over all four gammas carry the full-horizon sign (0.99931), with 340/340, 407/407 and 475/475 exact at gamma 0.9, 0.95 and 0.99 and a single exception at gamma=0.8 (221/222, seed 5, step 2391).** This is the empirically important result: the certificate only guarantees that the sign of the DISCOUNTED hypergradient is correct, yet on the coordinate-steps the controller actually acts on, the discounted sign is also the `gamma=1` sign. Away from the gate the picture is much weaker -- overall agreement is only 0.7370 (gamma=0.9), 0.7588 (0.8), 0.7637 (0.95) and 0.8818 (0.99) over 719940 coordinate-steps, so the gate is selecting exactly the ~0.05% of coordinate-steps where the discounting surrogate is trustworthy. Small gamma does NOT monotonically degrade agreement: the ordering is 0.99 (0.8818) >> 0.95 (0.7637) > 0.8 (0.7588) > 0.9 (0.7370), so only gamma=0.99 is clearly better and 0.8 vs 0.9 goes the "wrong" way; the clean monotone trends are in NMSE (0.0260 -> 0.0162 -> 0.0123 -> 0.0095) and open rate (3.08e-04 -> 4.72e-04 -> 5.65e-04 -> 6.60e-04), both improving with gamma, each paired difference vs gamma=0.9 significant at p=0.0020. Isolating the discounting bias from the estimator error, the EXACT discounted hypergradient agrees with full-horizon LESS often than the sketched estimate does (0.6493 vs 0.7370 at gamma=0.9), so the residual disagreement is a property of the discount, not of the sketch. After a regime switch, agreement does degrade, mildly and consistently: in the 200 steps following steps 4004 and 8007 it falls to 0.6724 vs 0.7392 elsewhere at gamma=0.9 (-6.7 points), 0.7312 vs 0.7597 at 0.8, 0.7090 vs 0.7656 at 0.95, 0.8512 vs 0.8828 at 0.99, over exactly 24000 post-switch coordinate-steps per gamma; the exact-discounted series degrades the same way (0.6143 vs 0.6505 at gamma=0.9). **Important caveat: this post-switch degradation cannot be attributed to the gate, because the gate never opens there.** COHG's last open step is between 4 and 15 on every seed at gamma >= 0.9 (and at most 2910 at gamma=0.8), so zero of the 24000 post-switch coordinate-steps are gate-open at any gamma -- the post-vs-else comparison is measured over gate-SHUT steps, and the question "does the certified sign go wrong after a switch" has no sample in this design. Finally, the full-horizon reference is sound throughout: `S_t` at gamma=1 stayed numerically finite on 100.0% of steps in all 40 runs, and the certificate audit found 0 violations in 720000 checked coordinate-steps at each gamma (worst ratio 0.833-0.917), so none of the above is contaminated by an overflowed or invalid reference.
