# Round 4: does the offline-calibrated static gate threshold transfer to other domains?

The E2 control `absgate` opens coordinate j iff `|ghat_j| > T` with the CONSTANT `T = 0.05806520209` (printed as 0.05807), fitted offline on `mackey_drift` to reproduce COHG's measured coordinate-open rate (`results/e2_controls/absgate_threshold.json`, calibration seeds 100/101, disjoint from the evaluation seeds).  On that stream it matches COHG's NMSE at 1 HVP/step instead of 7.9 (`results/e2_controls/SUMMARY.md`).  This round transfers that number AS IS -- no recalibration, no per-domain refit, not even a rescale -- to three other settings and asks whether it still works.

Every `absgate` arm is COHG's estimator and COHG's PURE-SIGN step of size alpha; only the gate rule differs, and the certificate is never read, so no spectral probe is paid.  That makes the arm cost-matched to `cohg_nogate` and the ONLY difference from COHG the gate.

Statistics: ddof=1 sample std; every paired comparison is an EXACT two-sided sign-flip permutation test enumerating all 2^n sign assignments of the per-seed paired differences.  Degradation metrics are the unified ones from `results/reanalysis/_reanalyze.py` (spike = finite loss above 10x the running median of the last 500 finite losses once 100 are banked; plus every non-finite loss; max-excess = max finite loss / median loss; worst-window = worst trailing 100-step mean), computed from the raw per-step loss traces.

## Status of this round (this file is regenerated as runs land)

| artefact | have | want | state |
|---|---|---|---|
| E2 lorenz_drift (absgate/COHG/cohg_nogate x seeds 0-9, CPU) | 30 | 30 | **complete** |
| E2 mackey_drift CALIBRATION-stream COHG reference (gate stats) | 3 | 3 | **complete** |
| E2 reproduction check (patched driver vs results/e2_controls) | 1 | 1 | **complete** |
| E4 GPT-2 absgate (seeds 0-2) | 3 | 3 | **complete** |
| E4 GPT-2 COHG reference re-run (repro check + gate stats) | 1 | 1 | **complete** |
| E3 Split-CIFAR-100 absgate (ewc10/ewc1000 x seeds 0-9) | 20 | 20 | **complete** |
| E3 COHG reference re-runs (repro check + gate stats) | 2 | 2 | **complete** |

Sections whose runs have not landed are explicitly marked NOT YET AVAILABLE rather than being silently omitted.

## 0. Reproduction checks: the drivers are additive

`absgate` and `--log-gate-stats` were added to `e3_continual.py` and `e4_gpt2_tta.py` (and `--log-gate-stats` alone to `e2_timeseries.py`) as new branches; the default path is untouched and the new JSON keys appear only when the corresponding flag is set.

**E4 (deterministic on this machine).** Re-running the stored `results/e4_v2/gpt2_cohg_r0_lr0.001_s0` config on the PATCHED `e4_gpt2_tta.py`:

| quantity | stored | re-run |
|---|---|---|
| online PPL | 20.6782339831342 | 20.6782339831342 |
| events | 0 | 0 |
| HVPs | 4100 | 4100 |
| full 2999-entry `losses` list | **identical element for element** | |

Differing shared keys: **none**.  Keys present only in the re-run (additive): ['domain_order', 'tokens_per_domain', 'mean_logloss', 'legacy_hold', 'held_bound', 'gate_open_steps', 'gate_stats'].

**E2 (deterministic on CPU).** Re-running the stored `results/e2_controls/mackey_drift_absgate_lr0.003_a0.4_s0` config on the PATCHED `e2_timeseries.py`:

| quantity | stored | re-run |
|---|---|---|
| NMSE | 0.0135047243411915 | 0.0135047243411915 |
| events | 3 | 3 |
| HVPs | 11992 | 11992 |
| full 12000-entry `losses` list | **identical** | |
| full `lam_hist` | **identical** | |

Differing shared keys: **none**.  Keys present only in the re-run: ['seg_bounds', 'seg_stats', 'scale_shift', 'lam_every', 'gate_open_steps', 'gate_open_env', 'gate_open_events', 'gate_open_events_truncated', 'coord_open_counts', 'n_probes', 'adaptive_mh', 'adapt_probe_log', 'adapt_mh_final', 'adapt_mh_raises', 'adapt_mobs_max', 'madgate_window', 'madgate_warmup', 'madgate_mean_mad', 'ogd_doubling_D', 'ogd_alpha_log', 'ogd_final_G', 'full_finite', 'full_agree', 'full_open', 'full_open_agree', 'full_nz', 'full_nz_agree', 'full_disc_agree', 'full_disc_nz', 'full_disc_nz_agree', 'full_n_coord'] (the shipped run predates later additive keys).

**E3 is NOT run-to-run reproducible on this machine, and was not before this round.** `e3_continual.py` trains a ResNet-18(GN) with cuDNN convolutions whose backward pass uses non-deterministic atomics, so two runs of the SAME config with the SAME seed diverge at the 1e-5 level within one step and then separate.  Evidence that this is a pre-existing property of the driver and not of this round's patch: `results/e3` and `results/e3_traced` hold the same configs run twice on this machine by earlier rounds, and they disagree -- including for the `fixed` arm, which runs no COHG code at all.

| arm | pairs | max \|delta avg_acc\| | mean \|delta avg_acc\| |
|---|---|---|---|
| fixed | 20 | 0.2508 | 0.0428 |
| hd | 20 | 0.0289 | 0.0087 |
| cohg | 20 | 0.2962 | 0.0277 |
| cohg_nogate | 20 | 0.1818 | 0.0502 |

What CAN be checked, and was: with the old and the patched script run back to back on the same card (2 tasks x 1 epoch, `cohg ewc10 s0`), step 0 of the loss trace is BIT-IDENTICAL, the divergence starts at step 1 at a relative 1e-6 and grows, and the complete `lam_hist` (the controller trajectory, sampled every 25 steps) is IDENTICAL for the whole run.  A CPU re-run of the same short config -- deterministic -- is reported below.

Full-length E3 COHG re-run of this round versus the stored `results/e3_traced` run of the same config:

| ewc0 | stored avg_acc | re-run avg_acc | stored events | re-run events | first differing loss step |
|---|---|---|---|---|---|
| 10 | 0.3292 | 0.3524 | 0 | 0 | 1 |
| 1000 | 0.3190 | 0.3387 | 2 | 0 | 1 |

## 1. E2 `lorenz_drift` -- the nearest domain (same driver, same 13k-param GRU, different attractor)

Config, identical for every arm and identical to the `mackey_drift` study the constant was calibrated on except for the dataset: mis-set init `lr0=0.003` (10x too low -- the same mis-set-low operating point `launch_e2.py` uses for `lorenz_drift`), 12000 steps, alpha (`--meta-lr`) 0.4, gamma 0.9, `kw-eps` 0.1, `probe-every` 20, K 10, rank 4, gate factor c=2, M_H 5, seeds 0-9, device CPU.

| arm | n | NMSE | events (in-run) | coord-open rate | steps-with-open | HVPs | wall (s) | unified events | non-finite | spikes | max-excess | worst-window |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| absgate T=0.05807 TRANSFERRED (CPU) | 10 | 0.0175+-0.0108 | 8.4+-16.7 | 2.986e-04+-1.312e-04 | 1.550e-03+-5.432e-04 | 11992+-0 | 898+-152 | 8.4+-16.7 | 0.0+-0.0 | 8.4+-16.7 | 326+-342 | 0.3372+-0.03706 |
| COHG certificate gate (CPU) | 10 | 0.0200+-0.0084 | 5.1+-11.4 | 4.250e-04+-1.070e-04 | 7.750e-04+-1.715e-04 | 94588+-0 | 5651+-849 | 5.1+-11.4 | 0.0+-0.0 | 5.1+-11.4 | 129+-105 | 0.2013+-0.0775 |
| cohg_nogate ungated sign step alpha=0.4 (CPU) | 10 | 0.0028+-0.0015 | 97.0+-53.2 | 0.000e+00+-0.000e+00 | 1.000e+00+-0.000e+00 | 11992+-0 | 799+-180 | 97.0+-53.2 | 0.0+-0.0 | 97.0+-53.2 | 7.23e+03+-5.95e+03 | 0.07379+-0.03553 |
| COHG certificate gate (GPU, results/e2) | 10 | 0.0194+-0.0087 | 6.8+-13.4 | 4.514e-04+-1.136e-04 | 8.500e-04+-2.108e-04 | 94588+-0 | 3578+-365 | 6.8+-13.4 | 0.0+-0.0 | 6.8+-13.4 | 134+-107 | 0.1966+-0.08119 |
| cohg_nogate (GPU, results/e2) | 10 | 0.0028+-0.0016 | 94.9+-52.5 | 0.000e+00+-0.000e+00 | 1.000e+00+-0.000e+00 | 11992+-0 | 536+-73 | 94.9+-52.5 | 0.0+-0.0 | 94.9+-52.5 | 7.32e+03+-5.94e+03 | 0.07379+-0.03553 |
| fixed lr=0.003 (mis-set init, un-adapted; GPU) | 10 | 0.0476+-0.0028 | 0.0+-0.0 | - | - | 0+-0 | 392+-85 | 0.0+-0.0 | 0.0+-0.0 | 0.0+-0.0 | 47.5+-26.4 | 0.4433+-0.06625 |

The fixed-LR family on the same stream (`results/e2`, GPU) -- the NMSE/instability frontier every adaptive arm is judged against:

| fixed lr | n | NMSE mean+-std | events mean+-std |
|---|---|---|---|
| 0.003  <- mis-set init of every adaptive arm | 10 | 0.04758+-0.002783 | 0.0+-0.0 |
| 0.01 | 10 | 0.02006+-0.0009796 | 3.6+-9.0 |
| 0.03 | 10 | 0.008627+-0.0003814 | 17.1+-17.1 |
| 0.1 | 10 | 0.003056+-0.0001454 | 43.6+-33.5 |
| 0.3 | 10 | 0.001826+-0.0001609 | 187.3+-59.0 |
| 0.6 | 10 | 5.802e+33+-1.835e+34 | 1362.3+-3049.2 |
| 1 | 10 | 4.378e+35+-8.964e+35 | 4069.8+-4629.5 |

`lambda window` (median over seeds of the first and last sampled step at which lambda actually moved): absgate 50-200, cohg 50-50, nogate 50-11950

### Paired tests vs the certificate gate and vs the ungated step
(exact sign-flip permutation, n=10 -> smallest attainable p = 0.0020; negative delta = the FIRST arm is better)

| comparison | metric | delta (mean paired diff) | p | n |
|---|---|---|---|---|
| absgate - COHG | NMSE | -0.00254 | 0.5508 | 10 |
| absgate - COHG | unified events | +3.3 | 0.6250 | 10 |
| absgate - COHG | worst-window | +0.1358 | 0.0039 | 10 |
| absgate - COHG | max-excess | +196.7 | 0.0176 | 10 |
| absgate - cohg_nogate | NMSE | +0.01468 | 0.0020 | 10 |
| absgate - cohg_nogate | unified events | -88.6 | 0.0020 | 10 |
| absgate - fixed lr0.003 | NMSE | -0.03012 | 0.0039 | 10 |
| COHG - fixed lr0.003 | NMSE | -0.02757 | 0.0020 | 10 |

## 2. E3 Split-CIFAR-100 continual learning (task-IL, ResNet-18 GN, 10 tasks x 2 epochs, lr0 0.05, alpha 0.4, seeds 0-9)

lambda here is 6 per-stage log-LRs plus one log-EWC-strength coordinate; the constant threshold is applied to all seven, as a transferred rule has no way to know that one of them is not a learning rate.

### 2.1  EWC operating point ewc0 = 10

| arm | n | final avg acc | BWT | events (in-run) | coord-open rate | steps-with-open | HVPs | wall (s) | unified events | non-finite | spikes | max-excess | worst-window |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| absgate T=0.05807 TRANSFERRED | 10 | 0.2819+-0.0758 | -0.0598+-0.0400 | 0.0+-0.0 | 3.838e-02+-1.302e-02 | 2.386e-01+-8.550e-02 | 3336+-0 | 2477+-265 | 0.0+-0.0 | 0.0+-0.0 | 0.0+-0.0 | 4.78+-1.01 | 2.493+-0.09869 |
| COHG certificate gate | 10 | 0.3754+-0.0268 | -0.0790+-0.0288 | 0.0+-0.0 | 3.289e-04+-5.714e-20 | 6.579e-04+-1.143e-19 | 24904+-0 | 10034+-1071 | 0.0+-0.0 | 0.0+-0.0 | 0.0+-0.0 | 5.92+-0.902 | 2.716+-0.08025 |
| cohg_nogate ungated sign step alpha=0.4 | 10 | 0.2264+-0.0685 | -0.0763+-0.0200 | 31.7+-100.2 | 0.000e+00+-0.000e+00 | 1.000e+00+-0.000e+00 | 3201+-307 | 3510+-1374 | 100.6+-147.6 | 31.7+-100.2 | 68.9+-105.0 | 2.22e+36+-7.02e+36 | inf |
| hd (Baydin et al.) | 10 | 0.3672+-0.0226 | -0.1185+-0.0209 | 0.0+-0.0 | - | - | 0+-0 | 5574+-6946 | 0.0+-0.0 | 0.0+-0.0 | 0.0+-0.0 | 6.96+-1.26 | 2.748+-0.07145 |
| fixed lambda | 10 | 0.3815+-0.0211 | -0.0688+-0.0242 | 0.0+-0.0 | - | - | 0+-0 | 5521+-6583 | 0.0+-0.0 | 0.0+-0.0 | 0.0+-0.0 | 7.19+-1.4 | 2.904+-0.1051 |

absgate final per-group log-LRs span [-11.51, -4.20] (eta in [1e-05, 0.0151]); the box is [log 1e-5, log 1] = [-11.51, 0.00] and the init is log 0.05 = -3.00.

COHG final per-group log-LRs span [-3.40, -2.60] (eta in [0.0335, 0.0746]).

| comparison | metric | delta (mean paired diff) | p | n |
|---|---|---|---|---|
| absgate - COHG | avg acc | -0.0935 | 0.0059 | 10 |
| absgate - COHG | BWT | +0.0192 | 0.2598 | 10 |
| absgate - COHG | unified events | +0.0 | 1.0000 | 10 |
| absgate - COHG | worst-window | -0.2235 | 0.0020 | 10 |
| absgate - cohg_nogate | avg acc | +0.0555 | 0.1855 | 10 |
| absgate - cohg_nogate | unified events | -100.6 | 0.0312 | 10 |
| absgate - fixed | avg acc | -0.0996 | 0.0020 | 10 |
| COHG - fixed | avg acc | -0.0061 | 0.3906 | 10 |

### 2.2  EWC operating point ewc0 = 1000

| arm | n | final avg acc | BWT | events (in-run) | coord-open rate | steps-with-open | HVPs | wall (s) | unified events | non-finite | spikes | max-excess | worst-window |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| absgate T=0.05807 TRANSFERRED | 10 | 0.2786+-0.0852 | -0.0543+-0.0354 | 0.0+-0.0 | 3.769e-02+-1.356e-02 | 2.305e-01+-8.718e-02 | 3336+-0 | 2758+-437 | 0.0+-0.0 | 0.0+-0.0 | 0.0+-0.0 | 4.75+-0.99 | 2.497+-0.09647 |
| COHG certificate gate | 10 | 0.3497+-0.0929 | -0.0852+-0.0773 | 1.9+-2.7 | 4.605e-04+-1.699e-04 | 7.895e-04+-1.699e-04 | 24482+-399 | 9553+-612 | 198.5+-188.7 | 1.9+-2.7 | 196.6+-186.7 | 7.24e+37+-7.18e+37 | 3.407e+37+-9.989e+37 |
| cohg_nogate ungated sign step alpha=0.4 | 10 | 0.2048+-0.0743 | -0.0896+-0.0439 | 45.6+-83.4 | 0.000e+00+-0.000e+00 | 1.000e+00+-0.000e+00 | 3233+-236 | 3645+-1174 | 77.4+-105.0 | 45.6+-83.4 | 31.8+-56.6 | 1.37e+36+-3.21e+36 | inf |
| hd (Baydin et al.) | 10 | 0.3796+-0.0221 | -0.0967+-0.0170 | 0.0+-0.0 | - | - | 0+-0 | 2212+-644 | 27.5+-87.0 | 0.0+-0.0 | 27.5+-87.0 | 3.06e+31+-9.68e+31 | 4.104e+30+-1.298e+31 |
| fixed lambda | 10 | 0.3652+-0.0525 | -0.0714+-0.0434 | 1.2+-1.9 | - | - | 0+-0 | 2180+-575 | 109.2+-141.0 | 1.2+-1.9 | 108.0+-139.4 | 4.85e+37+-6.37e+37 | 1.959e+37+-5.612e+37 |

absgate final per-group log-LRs span [-11.51, -4.20] (eta in [1e-05, 0.0151]); the box is [log 1e-5, log 1] = [-11.51, 0.00] and the init is log 0.05 = -3.00.

COHG final per-group log-LRs span [-9.63, -2.60] (eta in [6.55e-05, 0.0746]).

| comparison | metric | delta (mean paired diff) | p | n |
|---|---|---|---|---|
| absgate - COHG | avg acc | -0.0712 | 0.1309 | 10 |
| absgate - COHG | BWT | +0.0310 | 0.3477 | 10 |
| absgate - COHG | unified events | -198.5 | 0.0312 | 10 |
| absgate - COHG | worst-window | -3.407e+37 | 0.0312 | 10 |
| absgate - cohg_nogate | avg acc | +0.0738 | 0.0488 | 10 |
| absgate - cohg_nogate | unified events | -77.4 | 0.0039 | 10 |
| absgate - fixed | avg acc | -0.0867 | 0.0137 | 10 |
| COHG - fixed | avg acc | -0.0155 | 0.5664 | 10 |

## 3. E4 GPT-2 124M streaming test-time adaptation (wiki -> news -> code, 512k tokens/domain, 2999 steps, lr0 1e-3, alpha 0.4, seeds 0-2)

| arm | n | online PPL | events (in-run) | coord-open rate | steps-with-open | HVPs | peak GiB | wall (s) | unified events | non-finite | spikes | max-excess | worst-window |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| absgate T=0.05807 TRANSFERRED (rank 0) | 3 | 26.573+-8.008 | 0.0+-0.0 | 4.844e-02+-2.023e-02 | 2.032e-01+-4.442e-02 | 900+-0 | 16.9+-0.0 | 8344+-1186 | 0.0+-0.0 | 0.0+-0.0 | 0.0+-0.0 | 2.08+-0.716 | 4.503+-1.26 |
| COHG certificate gate (cohg_r0, the e4_v2 reference) | 8 | 20.675+-0.039 | 0.0+-0.0 | 3.334e-04+-5.795e-20 | 3.334e-04+-5.795e-20 | 4100+-0 | 17.0+-0.0 | 15993+-2106 | 0.0+-0.0 | 0.0+-0.0 | 0.0+-0.0 | 1.4+-0.036 | 3.779+-0.008525 |
| COHG certificate gate (cohg_r0, the e4_v2 reference), seeds 0-2 (the paired subset) | 3 | 20.689+-0.016 | 0.0+-0.0 | 3.334e-04+-6.639e-20 | 3.334e-04+-6.639e-20 | 4100+-0 | 17.0+-0.0 | 15300+-145 | 0.0+-0.0 | 0.0+-0.0 | 0.0+-0.0 | 1.36+-0.0133 | 3.778+-0.01216 |
| COHG certificate gate (rank 4) | 3 | 20.689+-0.016 | 0.0+-0.0 | 3.334e-04+-6.639e-20 | 3.334e-04+-6.639e-20 | 4692+-0 | 18.8+-0.0 | 25339+-591 | 0.0+-0.0 | 0.0+-0.0 | 0.0+-0.0 | 1.36+-0.0133 | 3.778+-0.01216 |
| cohg_nogate ungated sign step alpha=0.4 | 8 | 23.770+-6.235 | 0.0+-0.0 | 0.000e+00+-0.000e+00 | 1.000e+00+-0.000e+00 | 1492+-0 | 18.8+-0.0 | 23571+-6341 | 0.5+-1.4 | 0.0+-0.0 | 0.5+-1.4 | 6.09+-10.5 | 4.952+-1.765 |
| fixed lr=1e-3 | 8 | 21.092+-0.031 | 0.0+-0.0 | - | - | 0+-0 | 6.0+-0.0 | 1477+-26 | 0.0+-0.0 | 0.0+-0.0 | 0.0+-0.0 | 1.38+-0.0258 | 3.788+-0.008692 |

Paired tests use the three seeds absgate was run on (n=3 -> the smallest attainable two-sided p is 2/8 = 0.2500, so these tests can only ever be suggestive; the effect sizes are the informative column).

| comparison | metric | delta (mean paired diff) | p | n |
|---|---|---|---|---|
| absgate - cohg_r0 | online PPL | +5.8843 | 0.2500 | 3 |
| absgate - cohg_r0 | unified events | +0.0 | 1.0000 | 3 |
| absgate - cohg_r0 | worst-window | +0.7253 | 0.2500 | 3 |
| absgate - cohg_nogate | online PPL | +0.5630 | 1.0000 | 3 |
| absgate - fixed | online PPL | +5.4780 | 0.2500 | 3 |

absgate final per-block log-LRs span [-13.82, -2.30] (eta in [1e-06, 0.1]); the box is [log 1e-6, log 0.1] = [-13.82, -2.30] and the init is log 1e-3 = -6.91.

## 4. Where |ghat| actually lives, and where the two thresholds live

Pooled over every (step, coordinate) pair of a run, then the median over seeds.  `c*beta` is COHG's REALIZED certificate threshold `gate_factor * beta_col_j` (c = 2), available only on arms that actually compute the certificate.  `frac |ghat|>T` is the coordinate-open rate the TRANSFERRED constant produces; `frac |ghat|>c*beta` is the rate COHG's own gate produces on the same run.

Read the two rate columns carefully: they are not the same kind of number.  On a COHG row `frac |ghat|>T` is a COUNTERFACTUAL -- what the transferred constant WOULD have opened had it been applied to the trajectory COHG actually followed -- while `frac |ghat|>c*beta` is the rate COHG's own gate realized there.  On an absgate row `frac |ghat|>T` IS the realized rate and there is no certificate to compare against.  The two differ because the arms follow different lambda trajectories: on the calibration stream the constant's counterfactual rate on COHG's trajectory is 1.722e-03, while absgate's own realized rate there is 2.61e-04 (`results/e2_controls/SUMMARY.md`).

| run source | n | med \|ghat\| | p90 | p99 | max | med c*beta | p90 | p99 | max | frac \|ghat\|>T | frac \|ghat\|>c*beta |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E2 mackey_drift (CALIBRATION STREAM), COHG | 3 | 5.860e-06 | 1.641e-04 | 1.310e-03 | 1.548e-01 | 4.566e-04 | 6.112e-03 | 9.543e-02 | 1.949e+00 | 1.722e-03 | 4.583e-04 |
| E2 lorenz_drift, COHG | 10 | 1.093e-05 | 1.764e-04 | 3.261e-03 | 2.299e-01 | 5.286e-03 | 4.265e-02 | 1.279e+02 | 4.981e+03 | 1.750e-03 | 4.028e-04 |
| E2 lorenz_drift, absgate | 10 | 3.830e-06 | 1.271e-04 | 2.862e-03 | 2.370e-01 | - | - | - | - | 2.361e-04 | - |
| E3 ewc10, COHG | 1 | 3.513e-02 | 1.558e-01 | 3.824e-01 | 4.560e+00 | 2.848e+152 | 7.328e+279 | 1.493e+306 | 1.701e+308 | 3.502e-01 | 3.289e-04 |
| E3 ewc10, absgate | 10 | 2.829e-03 | 3.033e-02 | 1.433e-01 | 4.366e+00 | - | - | - | - | 4.248e-02 | - |
| E3 ewc1000, COHG | 1 | 4.436e-02 | 1.780e-01 | 4.621e-01 | 4.560e+00 | 7.274e+148 | 9.132e+275 | 3.783e+305 | 1.518e+308 | 4.094e-01 | 3.289e-04 |
| E3 ewc1000, absgate | 10 | 4.678e-03 | 3.010e-02 | 1.363e-01 | 4.367e+00 | - | - | - | - | 3.990e-02 | - |
| E4 GPT-2, COHG (cohg_r0) | 1 | 5.751e-03 | 2.987e-02 | 1.442e-01 | 1.740e+00 | 1.396e+160 | 5.138e+269 | 9.170e+296 | 1.520e+308 | 4.018e-02 | 3.334e-04 |
| E4 GPT-2, absgate | 3 | 5.428e-03 | 2.628e-02 | 2.125e-01 | 6.880e+00 | - | - | - | - | 4.168e-02 | - |

### Scale mismatch factor

| domain | median c*beta (COHG) | T = 0.05807 | T / median(c*beta) | T / median \|ghat\| | realized open rate: T | realized open rate: certificate |
|---|---|---|---|---|---|---|
| E2 mackey_drift (CALIBRATION STREAM), COHG | 4.566e-04 | 0.05807 | 127 | 9.91e+03 | 1.722e-03 | 4.583e-04 |
| E2 lorenz_drift, COHG | 5.286e-03 | 0.05807 | 11 | 5.31e+03 | 1.750e-03 | 4.028e-04 |
| E3 ewc10, COHG | 2.848e+152 | 0.05807 | 2.04e-154 | 1.65 | 3.502e-01 | 3.289e-04 |
| E3 ewc1000, COHG | 7.274e+148 | 0.05807 | 7.98e-151 | 1.31 | 4.094e-01 | 3.289e-04 |
| E4 GPT-2, COHG (cohg_r0) | 1.396e+160 | 0.05807 | 4.16e-162 | 10.1 | 4.018e-02 | 3.334e-04 |

## 5. Plain answer

### E2 `lorenz_drift` -- the transfer SURVIVES on the headline metric and DEGRADES the tail

Moving the constant from `mackey_drift` to `lorenz_drift` -- a different attractor, same driver, same 13k-param GRU, same mis-set init -- the transferred gate is statistically indistinguishable from the certificate gate on what the paper reports:

- NMSE 0.01746 vs 0.02001, paired delta -0.00254, **p = 0.5508** (not distinguishable), at 11 992 HVPs against 94 588 -- 7.9x cheaper.
- unified events, paired delta +3.3, **p = 0.6250** (not distinguishable).
- realized coordinate-open rate 2.986e-04 vs 4.250e-04 -- within a factor 1.42 of COHG's without any refit.

But the degradation metrics that the headline NMSE averages away do separate them:

- worst trailing-100-step window: paired delta +0.1358, **p = 0.0039**.
- max-excess (max finite loss / median loss): paired delta +196.7, **p = 0.0176**.

So on the nearest domain the answer is: the static threshold still works well enough to reproduce the accuracy claim, but it is measurably worse in the tail.  It survives because the `|ghat|` scale of `lorenz_drift` happens to resemble that of `mackey_drift`; nothing in the rule guaranteed that, and the next section shows what happens when it does not.

### E3 Split-CIFAR-100 -- the transfer FAILS at both EWC operating points

| ewc0 | absgate acc | COHG acc | fixed acc | absgate coord-open | COHG coord-open | absgate-COHG (p) | absgate-fixed (p) | absgate-nogate (p) | COHG-fixed (p) |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.2819+-0.0758 | 0.3754+-0.0268 | 0.3815+-0.0211 | 3.838e-02 | 3.289e-04 | -0.0935 (0.0059) | -0.0996 (0.0020) | +0.0555 (0.1855) | -0.0061 (0.3906) |
| 1000 | 0.2786+-0.0852 | 0.3497+-0.0929 | 0.3652+-0.0525 | 3.769e-02 | 4.605e-04 | -0.0712 (0.1309) | -0.0867 (0.0137) | +0.0738 (0.0488) | -0.0155 (0.5664) |

The transferred constant opens 82-117x more often than the certificate gate and loses accuracy at both operating points, but the strength of the evidence differs and should be stated as it is.  At ewc0=10 it costs 9.35 points against COHG (p=0.0059) and 9.96 against not adapting at all (p=0.0020, the smallest p attainable at n=10).  At ewc0=1000 it costs 8.67 points against `fixed` (p=0.0137) but its 7.12-point deficit against COHG is NOT significant (p=0.1309) -- COHG's own spread is large at that operating point (sd 0.093), so the comparison against the un-adapted baseline is the informative one there.

Against the fully ungated sign step the transferred gate is not reliably better at ewc0=10 (+0.0555, p=0.1855) and is modestly better at ewc0=1000 (+0.0738, p=0.0488).  So it is not worthless -- it still shuts most of the time -- but on this domain it lands much closer to having no gate than to having the certificate gate.

Two mechanism details specific to E3.  (i) The constant drags five or six of the six per-stage log-LRs down from the init -3.00 to between -5.8 and the -11.51 floor, while COHG leaves them at -2.6/-3.4.  (ii) lambda here contains one coordinate that is NOT a learning rate -- the log EWC strength -- and the transferred constant never opens it, whereas COHG spends its single gate opening exactly there (2.30 -> 1.90 at ewc0=10, 6.91 -> 6.51 at ewc0=1000).  One scalar threshold applied to a lambda vector whose coordinates are not commensurate cannot distinguish the coordinate that matters; the certificate, being per-coordinate and scale-aware, can.

### E4 GPT-2 -- the transfer FAILS

- online PPL 26.573 vs the certificate gate's 20.689 on the same seeds (+28.4%), and worse than doing nothing at all (`fixed` 21.09).
- realized coordinate-open rate 4.844e-02 vs 3.334e-04 -- the transferred constant is 145x more permissive here.

Per-seed, because the outcome is NOT uniform across seeds -- this matters for how the claim is worded:

| seed | online PPL | coord-open rate | PPL of last 500 steps | blocks pinned at the 1e-6 FLOOR | blocks pinned at the 0.1 CEILING |
|---|---|---|---|---|---|
| 0 | 35.819 | 7.119e-02 | 166.3 | 2 of 6 | 1 of 6 |
| 1 | 21.805 | 3.246e-02 | 7.8 | 3 of 6 | 0 of 6 |
| 2 | 22.096 | 4.168e-02 | 8.2 | 2 of 6 | 0 of 6 |

What is CONSISTENT across seeds is the over-permissiveness and the freezing: the transferred constant opens 100-200x more often than the certificate gate, and ends with most blocks pinned at the 1e-6 floor, so the model stops adapting.  What is SEED-DEPENDENT is whether it additionally drives a block to the 0.1 CEILING and blows the loss up late in the stream (seed 0 does: last-500 PPL 166.3 against the reference's 6.9; seed 1 does not).  The claim to make is therefore 'the transferred threshold is uncalibrated here and is sometimes catastrophic', not 'it always blows up'.

In no seed does any loss go non-finite, so the in-run `events` counter reads 0 throughout and misses all of this; only the unified worst-window / max-excess metrics see it.

### What the certificate provides that the constant does not

On GPT-2 the reference run shows COHG opening its gate on **exactly one step of 2999** (`gate_open_steps = [1]`), because the realized threshold `c*beta_j` climbs past the entire range of `|ghat|` within the first handful of steps (median `c*beta` 1.4e+160 against `|ghat|` max 1.74) and diverges from there.

That is the honest shape of the result, and it should be written that way: on this domain COHG does not out-adapt the constant, it REFUSES to adapt.  Its PPL (20.68) is essentially the un-adapted `fixed` baseline (21.09), bought with a single certified step.  The transferred constant has no notion of whether the estimate is trustworthy, keeps stepping for all 2999 steps, and destroys the run.

### One column that must NOT be read as a win for the constant

On E3 and E4 several DEGRADATION metrics come out lower for absgate than for COHG -- E3 ewc10 worst-window -0.2235 (p=0.0020), E3 ewc1000 unified events -198.5 (p=0.0312), E4 max-excess 2.08 vs COHG's 1.36.  This is not the transferred gate being safer.  It is an artefact of HOW it fails: by slamming the learning rates to the 1e-6 / 1e-5 floor it produces a model that barely updates, and a model that barely updates has a very smooth loss trace.  Spike counts and worst-window are defined relative to a run's own running median, so a frozen learner scores well on them while losing 9-10 accuracy points and 5.9 PPL.  The degradation metrics are meaningful for comparing arms that are all still learning (that is how they are used on E2); on the two domains where the transferred constant freezes the learner they must be read alongside the accuracy/PPL column, not instead of it.

A related null result worth stating: on E4 absgate is not better than the fully UNGATED sign step either (PPL +0.563, p=1.0000), and on E3 ewc10 it is not better either (p=0.1855).  Outside its calibration stream the transferred threshold buys little or nothing over having no gate at all.

So the certificate's contribution is not a better threshold value -- it is a per-domain, per-step, calibration-free answer to *whether the hypergradient may be acted on at all*.  A constant fitted on one stream encodes the scale of that stream and nothing else: it cannot become more conservative when the estimator degrades, and it cannot become more permissive when the estimator is sharp.

