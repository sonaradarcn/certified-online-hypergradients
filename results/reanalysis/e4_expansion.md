# E4 (GPT-2 test-time adaptation): seed expansion + alternate domain order

Generated from `results/e4_v2/` (standard order **wiki -> news -> code**) and
`results/e4_orders/` (alternate order **code -> news -> wiki**, files prefixed
`gpt2order_cnw_`). Analysis only; no paper text was edited.

## 0. Conventions, definitions, and what is / is not included

* **Loading.** Identical to `code/experiments/make_paper_figures_nc.py` and
  `results/reanalysis/_fig8_gpt2_lambda.py`: keys `online_ppl`, `losses`
  (per-step log-loss, length 2999), `lam_hist` (150 rows,
  `[t, lam_0..lam_5]`, sampled every 20 steps, `t = 0..2980`),
  `gate_open_frac`, `events`, `drift_steps = [1000, 2000]`, `domain_order`,
  `peak_mem_gb`, `wall_s`. All runs: `lr0 = 1e-3`, `steps = 2999`,
  `tokens_per_domain = 512000`, 3 domains with boundaries at t = 1000, 2000.
* **Dispersion.** All `std` are sample standard deviations, `ddof = 1`.
* **Worst-window PPL.** Trailing-100-step mean of the per-step log-loss,
  evaluated at every 20th step (window ends t = 99, 119, ..., 2979), maximum
  over windows, then `exp(.)`. Non-finite losses are dropped inside a window;
  no run in this set has any non-finite loss, so this is a plain mean.
  (The every-step variant used in `unified_metrics_e4.csv` differs by
  < 0.35 PPL for the stable arms and preserves every ordering and every
  degraded/not-degraded classification below.)
* **Max-excess.** `max(finite log-loss) / median(finite log-loss)`, in
  log-loss space, matching `_reanalyze.py`.
* **Degraded run (definition used throughout).** A run is *degraded* if
  **either** (i) its final `online_ppl` exceeds the mean `online_ppl` of the
  fixed-lr baseline **of the same domain order** by more than 2.0 PPL,
  **or** (ii) its worst-window PPL exceeds 2x the *median worst-window PPL of
  the gated arm* (`cohg_r0`) of the same order. Criterion (i) catches
  whole-stream damage, (ii) catches transient blow-ups that the stream average
  washes out. Thresholds are stated with each table.
* **Paired test.** Exact two-sided sign-flip (randomization) test on the
  per-seed paired difference in `online_ppl`, enumerating all `2^n` sign
  assignments; p = fraction of assignments whose `|mean|` is >= the observed
  `|mean|`. With n = 8 the smallest attainable p is `2/256 = 0.0078125`; with
  n = 3 it is `2/8 = 0.25`.
* **Seed coverage (updated).** `results/e4_v2/gpt2_cohg_nogate_lr0.001_s7.json`
  has since landed and is now included. All four standard-order arms therefore
  have **n = 8** (seeds 0-7). All four alternate-order arms have **n = 3**
  (seeds 0-2). Every `cohg_nogate` number in Sections 1 and 3 below was
  recomputed at n = 8; the superseded n = 7 values are noted inline where the
  change is material.
* **Two worst-window variants.** The every-20th-step variant defined above is
  what Sections 1-3 tabulate. The paper's Table `tab:e4` uses the *every-step*
  variant of `unified_metrics` (`max_e mean(loss[e-100:e])` over all
  `e = 100..2999`, then `exp(.)`), which is what the paper's own definition
  states; Section 1.6 lists that variant at n = 8 so the two documents agree
  number for number. The variants differ by < 0.35 PPL on the stable arms and
  produce identical degraded/not-degraded classifications.

---

## 1. Standard stream (wiki -> news -> code), expanded seeds

### 1.1 Online PPL

| arm | n | seeds | mean +- std | median | min | max | worst seed |
|---|---|---|---|---|---|---|---|
| fixed lr 1e-3 | 8 | 0-7 | **21.0919 +- 0.0306** | 21.0945 | 21.0499 | 21.1473 | s4 (21.1473) |
| hd ml2 | 8 | 0-7 | **20.1687 +- 0.0506** | 20.1785 | 20.0973 | 20.2391 | s7 (20.2391) |
| cohg_r0 (gated) | 8 | 0-7 | **20.6753 +- 0.0386** | 20.6795 | 20.6193 | 20.7281 | s7 (20.7281) |
| cohg_nogate | 8 | 0-7 | **23.7696 +- 6.2355** | 20.8995 | 19.7730 | 38.0265 | s1 (38.0265) |

Per-seed online PPL:

| arm | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 |
|---|---|---|---|---|---|---|---|---|
| fixed | 21.0950 | 21.0940 | 21.0960 | 21.0549 | 21.1473 | 21.0499 | 21.1088 | 21.0891 |
| hd ml2 | 20.1970 | 20.1173 | 20.2161 | 20.1260 | 20.1702 | 20.0973 | 20.1868 | 20.2391 |
| cohg_r0 | 20.6782 | 20.6808 | 20.7071 | 20.6193 | 20.6673 | 20.6223 | 20.6992 | 20.7281 |
| cohg_nogate | 19.7730 | 38.0265 | 20.2305 | 26.7212 | 23.7196 | 20.6291 | 21.1698 | 19.8868 |

### 1.2 Worst-window PPL and max-excess

| arm | n | worst-window PPL mean +- std | median | min | max | max-excess mean / median / max |
|---|---|---|---|---|---|---|
| fixed lr 1e-3 | 8 | 43.9267 +- 0.4334 | 44.2082 | 43.2643 | 44.2762 | 1.3754 / 1.3757 / 1.4051 |
| hd ml2 | 8 | 42.4572 +- 0.4701 | 42.7566 | 41.7172 | 42.8302 | 1.3787 / 1.3796 / 1.4072 |
| cohg_r0 (gated) | 8 | 43.4879 +- 0.4159 | 43.7619 | 42.8374 | 43.8112 | 1.3967 / 1.3987 / 1.4642 |
| cohg_nogate | 8 | 911.4373 +- 2232.8925 | 63.8770 | 47.8863 | 6423.2978 | 6.0873 / 2.5952 / 31.9522 |

Per-seed worst-window PPL:

| arm | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 |
|---|---|---|---|---|---|---|---|---|
| fixed | 44.2521 | 43.2643 | 44.2762 | 43.5586 | 44.2153 | 43.4156 | 44.2010 | 44.2308 |
| hd ml2 | 42.8196 | 41.7172 | 42.8302 | 42.0446 | 42.7247 | 41.9418 | 42.7911 | 42.7885 |
| cohg_r0 | 43.7911 | 42.8374 | 43.8112 | 43.1291 | 43.7916 | 43.0188 | 43.7705 | 43.7532 |
| cohg_nogate | 47.9985 | 516.3483 | 60.0944 | 6423.2978 | 60.2233 | 68.1191 | 67.5308 | 47.8863 |

### 1.3 Degraded runs (standard order)

Thresholds: fixed-baseline mean = 21.0919 -> **PPL threshold 23.0919**;
gated (`cohg_r0`) median worst-window = 43.7619 -> **worst-window threshold 87.5237**.

| arm | degraded / n | fraction | which seeds (criterion triggered) |
|---|---|---|---|
| fixed lr 1e-3 | 0 / 8 | 0.0% | - |
| hd ml2 | 0 / 8 | 0.0% | - |
| cohg_r0 (gated) | 0 / 8 | **0.0%** | - |
| cohg_nogate | 3 / 8 | **37.5%** | s1 (both), s3 (both), s4 (PPL only) |

Removing the gate is the only change that produces degraded runs. Seed 3 is the
clearest transient case: online PPL 26.72 but a worst-window PPL of 6423, i.e. a
100-step stretch at ~300x the ambient perplexity that the stream average hides.

### 1.4 Paired test, cohg_r0 vs fixed (standard order)

Common seeds: 0,1,2,3,4,5,6,7 (**n = 8**). Paired differences
`cohg_r0 - fixed`:

| s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 |
|---|---|---|---|---|---|---|---|
| -0.4167 | -0.4132 | -0.3889 | -0.4356 | -0.4800 | -0.4276 | -0.4097 | -0.3610 |

mean difference **-0.4166** PPL (std 0.0347), negative on **8/8** seeds.
Exact two-sided sign-flip over all 256 sign assignments:
**p = 0.0078125** (the smallest value attainable at n = 8).

For reference, same test on the same 8 seeds:

* `cohg_r0 - hd_ml2` = **+0.5066** (positive on 8/8), p = 0.0078125 -- hypergradient
  descent with meta-lr 2 remains better than the certified gated arm on this stream.
* `cohg_nogate - fixed` (n = 8 common seeds) = **+2.6777** (std 6.2358),
  p = 0.296875 -- the ungated arm's mean penalty is not significant at n = 8
  because its damage is concentrated in a minority of seeds (sign pattern
  4 negative / 4 positive); its variance and worst-case, not its mean, are
  what the gate controls. Per-seed differences:
  [-1.3219, +16.9325, -0.8655, +5.6664, +2.5723, -0.4208, +0.0610, -1.2023].

### 1.5 Paper's 3-seed numbers vs the expanded means

The paper's E4 numbers reproduce exactly on seeds 0-2, and the expanded means
move very little for the three stable arms.

| arm | paper (3 seeds, s0-2) | recomputed s0-2 | expanded | shift |
|---|---|---|---|---|
| cohg_r0 | 20.69 +- 0.02 | 20.6887 +- 0.0160 | **20.6753 +- 0.0386** (n=8) | -0.013, std 2.4x wider |
| cohg_nogate | 26.0 +- 10.4 | 26.0100 +- 10.4091 | **23.7696 +- 6.2355** (n=8) | -2.240, std 0.60x |
| hd ml2 | 20.18 +- 0.05 | 20.1768 +- 0.0524 | **20.1687 +- 0.0506** (n=8) | -0.008, std unchanged |
| fixed | 21.10 | 21.0950 +- 0.0010 | **21.0919 +- 0.0306** (n=8) | -0.003, std 30x wider |

Reading: **every ranking in the paper survives the seed expansion**
(hd2 < cohg_r0 < fixed < nogate on the mean), and the cohg_r0-vs-fixed gap is
now backed by 8/8 seeds at p = 0.0078 rather than 3/3. The one number that
should be restated is the ungated arm: at 3 seeds it read 26.0 +- 10.4, at 8
seeds it reads 23.8 +- 6.2, and its median (20.90) is *below* the fixed
baseline mean. The honest characterization is *heavy-tailed, not uniformly
worse*: 5/8 ungated seeds land within 1.4 PPL of fixed (19.77, 19.89, 20.23,
20.63, 21.17) and 3/8 are degraded, with the worst at 38.0 PPL and a
worst-window of 6423. The 3-seed std of the fixed arm
(0.0010) was also an accident of seeds 0-2; the true seed spread is ~0.03.

### 1.6 Every-step worst-window variant (the variant the paper tabulates)

Same traces, `max_e mean(loss[e-100:e])` over all `e = 100..2999`, then
`exp(.)`; `ddof = 1` over seeds. These are the values that appear in the
paper's Table `tab:e4` and in the alternate-order table.

| arm | n | worst-window PPL (every-step) | per-seed |
|---|---|---|---|
| fixed lr 1e-3 | 8 | 44.1590 +- 0.3833 | 44.430, 43.510, 44.467, 43.880, 44.367, 43.753, 44.469, 44.392 |
| hd ml2 | 8 | 42.7865 +- 0.3811 | 43.058, 42.124, 43.035, 42.510, 43.001, 42.415, 43.158, 42.991 |
| cohg_r0 (gated) | 8 | 43.7543 +- 0.3719 | 44.005, 43.105, 44.038, 43.486, 43.978, 43.388, 44.083, 43.951 |
| cohg_nogate | 8 | 951.7768 +- 2310.6180 | 48.143, 610.251, 60.252, 6650.266, 61.012, 68.818, 67.531, 47.941 |
| fixed lr 3e-3 | 3 | 44.2670 +- 0.6816 | 45.024, 43.700, 44.077 |
| hd ml0.2 | 3 | 44.2270 +- 0.5762 | 44.535, 43.562, 44.584 |
| hd ml20 | 3 | 39.4017 +- 1.2957 | 38.973, 40.858, 38.374 |
| hd ml200 (transferred) | 3 | non-finite (3/3) | overflow |
| cohg r=4 | 3 | 43.7160 +- 0.5303 | 44.005, 43.105, 44.038 |

Alternate order (n = 3 each): fixed 42.6207 +- 0.6052; hd ml2 42.0955 +- 0.5783;
cohg_r0 42.7974 +- 0.5184; cohg_nogate 167787.07 +- 289743.25
(per-seed 50.041, 957.816, 502353.338).

Degraded classification under this variant is unchanged in both orders
(gated 0/8 and 0/3; ungated 3/8 = s1, s3, s4 and 2/3 = s1, s2), with
thresholds 23.0919 / 87.9290 (standard order) and 23.0986 / 85.5879
(alternate order).

Max-excess at n = 8, standard order: fixed 1.3754 +- 0.0258,
hd ml2 1.3787 +- 0.0253, cohg_r0 1.3967 +- 0.0360,
cohg_nogate 6.0873 +- 10.4894 (per-seed 1.414, 2.862, 3.181, 31.952,
1.465, 3.915, 1.581, 2.329).

Cost at n = 8, standard order: fixed 0.41 h / 6.05 GB, hd ml2 0.60 h /
6.51 GB, cohg_r0 4.44 h / 16.96 GB, cohg_nogate 6.55 h / 18.80 GB,
cohg r=4 7.04 h / 18.82 GB (n = 3). The gated arm costs 10.8x the wall-clock
and 2.80x the peak memory of fixed.

---

## 2. Alternate order code -> news -> wiki (n = 3 per arm)

Domain 1 = code (t < 1000), domain 2 = news (1000 <= t < 2000),
domain 3 = wiki (t >= 2000).

### 2.1 Online PPL

| arm | n | mean +- std | median | min | max | worst seed |
|---|---|---|---|---|---|---|
| fixed lr 1e-3 | 3 | **21.0986 +- 0.0017** | 21.0981 | 21.0972 | 21.1005 | s2 (21.1005) |
| hd ml2 | 3 | **20.3614 +- 0.0844** | 20.4038 | 20.2642 | 20.4162 | s0 (20.4162) |
| cohg_r0 (gated) | 3 | **21.4029 +- 0.4303** | 21.6388 | 20.9062 | 21.6637 | s2 (21.6637) |
| cohg_nogate | 3 | **392.1805 +- 615.2561** | 53.3558 | 20.8171 | 1102.3685 | s2 (1102.3685) |

Per-seed: fixed [21.0981, 21.0972, 21.1005]; hd2 [20.4162, 20.2642, 20.4038];
cohg_r0 [20.9062, 21.6388, 21.6637]; nogate [20.8171, 53.3558, 1102.3685].

### 2.2 Worst-window PPL and max-excess

| arm | worst-window PPL mean +- std | median | min | max | max-excess mean / median / max |
|---|---|---|---|---|---|
| fixed lr 1e-3 | 42.6212 +- 0.6052 | 42.9498 | 41.9228 | 42.9910 | 1.3460 / 1.3450 / 1.3519 |
| hd ml2 | 42.0955 +- 0.5783 | 42.3604 | 41.4323 | 42.4939 | 1.3615 / 1.3591 / 1.3708 |
| cohg_r0 (gated) | 42.7974 +- 0.5184 | 42.7940 | 42.2808 | 43.3175 | 1.3401 / 1.3422 / 1.3433 |
| cohg_nogate | 141585.05 +- 244414.92 | 894.4465 | 50.0407 | 423810.67 | 9.5427 / 4.8278 / 22.4139 |

Per-seed worst-window: fixed [42.9910, 41.9228, 42.9498];
hd2 [42.4939, 41.4323, 42.3604]; cohg_r0 [42.7940, 42.2808, 43.3175];
nogate [50.0407, 894.4465, 423810.6746].

### 2.3 Degraded runs (alternate order)

Thresholds: fixed-baseline mean = 21.0986 -> **PPL threshold 23.0986**;
gated median worst-window = 42.7940 -> **worst-window threshold 85.5879**.

| arm | degraded / n | fraction | which seeds |
|---|---|---|---|
| fixed lr 1e-3 | 0 / 3 | 0.0% | - |
| hd ml2 | 0 / 3 | 0.0% | - |
| cohg_r0 (gated) | 0 / 3 | **0.0%** | - |
| cohg_nogate | 2 / 3 | **66.7%** | s1 (both), s2 (both) |

Paired test, `cohg_r0 - fixed`, common seeds 0,1,2 (n = 3):
differences [-0.1919, +0.5415, +0.5632], mean **+0.3043**,
exact two-sided sign-flip **p = 0.5** (4/8) -- **not significant, and the sign
is reversed relative to the standard order**. (`cohg_r0 - hd_ml2` = +1.0415,
p = 0.25; `cohg_nogate - fixed` = +371.08, p = 0.5.)

### 2.4 KEY QUESTION -- when does the gate open under code -> news -> wiki?

**Answer, stated plainly: the gate does not open at or after a domain switch.
In all three seeds it opens exactly once, in the very first sampling interval
(0, 20] -- inside domain 1 (code), 980 steps before the first boundary -- and
never again. Zero openings at or after t = 1000, and zero at or after t = 2000.**

Per-seed detail (`gate_open_frac x steps` gives the exact number of accepted
meta-updates; `lam_hist` is sampled every 20 steps, so a change between samples
localises the event to the half-open interval `(t_prev, t_cur]`):

| seed | gate_open_frac | exact #opens / 2999 steps | open interval | domain | coords moved | delta lambda |
|---|---|---|---|---|---|---|
| 0 | 3.3344448e-04 | **1** | (0, 20] | D1 = code | all 6 | emb -0.4, h0-2 +0.4, h3-5 -0.4, h6-8 +0.4, h9-11 +0.4, ln_f +0.4 |
| 1 | 3.3344448e-04 | **1** | (0, 20] | D1 = code | all 6 | emb -0.4, h0-2 -0.4, h3-5 -0.4, h6-8 -0.4, h9-11 -0.4, ln_f +0.4 |
| 2 | 3.3344448e-04 | **1** | (0, 20] | D1 = code | all 6 | emb -0.4, h0-2 -0.4, h3-5 -0.4, h6-8 -0.4, h9-11 -0.4, ln_f +0.4 |

Open counts by domain, all three seeds: **D1 (code) = 1, D2 (news) = 0,
D3 (wiki) = 0.** Step indices of the openings: **{20} for s0, {20} for s1,
{20} for s2** (upper bound of the localising interval; the accepted update is
somewhere in steps 1-20).

Resulting lambda (start `log 1e-3 = -6.9078` for all coords, meta-lr 0.4):

* s0 -> `[-7.3078, -6.5078, -7.3078, -6.5078, -6.5078, -6.5078]`
* s1 -> `[-7.3078, -7.3078, -7.3078, -7.3078, -7.3078, -6.5078]`
* s2 -> `[-7.3078, -7.3078, -7.3078, -7.3078, -7.3078, -6.5078]`

and then constant for the remaining 2979 steps.

**Contrast with the standard order** (`results/e4_v2`, all 8 seeds): also
exactly **1** open per run, also in `(0, 20]`, also zero opens at the
boundaries -- but there the accepted step is **+0.4 on all six coordinates in
all 8 seeds** (lambda -6.9078 -> -6.5078 everywhere, i.e. LR raised ~1.49x).
Under code-first the same certified rule *lowers* the LR on 5 of the 6
coordinates in seeds 1 and 2, and on 2 of 6 (emb, h3-5) in seed 0. So the direction of the single certified
adaptation is order-dependent, while its timing (once, at stream start) is not.

**Implication that should be stated in the paper.** On GPT-2 the certified gate
accepts 1 meta-update in 2999 steps in *both* orders. COHG here is
operationally "fixed lambda, shifted once within the first 20 steps"; the entire
online-PPL difference against the fixed baseline is attributable to that single
early step, and the gate provides no boundary-triggered re-adaptation. That is
consistent with the safety claim (0/8 and 0/3 degraded runs) but it is not
evidence of drift-responsive adaptation, and the current fig8 annotation
("only gate opening of the whole run") should be read as a general property of
this regime rather than a seed-0 curiosity.

### 2.5 Ungated arm under code -> news -> wiki: lambda range, saturation, degradation

Gate forced open every step (`gate_open_frac = 1.0`, 2999/2999 accepted).
Clamps are `[log 1e-6, log 1e-1] = [-13.8155, -2.3026]`, span 11.5129.

| seed | online PPL | per-coord lambda range (max-min) | fraction of lam_hist samples at lower clamp | at upper clamp | degraded? |
|---|---|---|---|---|---|
| 0 | 20.8171 | [10.000, 11.113, 11.513, 11.513, 11.513, 11.513] | 0.4067 | 0.1022 | no |
| 1 | 53.3558 | [11.513, 11.513, 11.513, 10.713, 11.513, 11.513] | 0.4644 | 0.0811 | **yes** |
| 2 | 1102.3685 | [11.513, 11.513, 11.513, 11.513, 11.513, 11.200] | 0.3733 | 0.0933 | **yes** |

Every coordinate in every seed touches the **lower** clamp exactly
(`min = -13.8155` for all 6 coords in all 3 seeds), and 4-5 of 6 coordinates per
seed touch the **upper** clamp exactly (the remaining ones stop 0.31-1.51 short);
the traversed range equals the full admissible span (11.5129) for 14 of 18
coordinate-seed pairs and exceeds 10.7 for all 18. Roughly
**37-46% of samples sit pinned at the lower clamp and 8-10% at the upper clamp**
-- the ungated meta-optimizer bang-bangs between "no learning" and "10x the
safe LR" rather than settling.

Lambda at the boundaries (nearest sample) shows no coherent structure, e.g.
s2 at t = 1000 `[-13.816, -13.816, -11.816, -13.816, -11.816, -7.416]` and at
t = 2000 `[-4.303, -3.503, -10.216, -11.816, -8.616, -13.416]`.

**Yes, seeds degrade: 2 of 3.** Seed 2 is catastrophic (online PPL 1102.4, peak
per-step log-loss 172.6 at step 364, worst-window PPL 4.2e5,
per-domain PPL [114.9 / 4435.7 / 2630.1]); seed 1 is intermediate (53.4;
per-domain [70.6 / 61.2 / 35.1]); seed 0 is benign (20.8, per-domain
[7.20 / 37.62 / 33.31], slightly *better* than fixed). No run produced a
non-finite loss, so `events = 0` everywhere and the damage is entirely
finite-but-large loss, which is why `worst-window` and not `events` is the
metric that exposes it.

---

## 3. Cross-order comparison

### 3.1 Headline table

| arm | wiki->news->code: PPL (n) | code->news->wiki: PPL (n=3) | wiki-first: worst-window | code-first: worst-window |
|---|---|---|---|---|
| fixed lr 1e-3 | 21.0919 +- 0.0306 (8) | 21.0986 +- 0.0017 (3) | 43.9267 +- 0.4334 | 42.6212 +- 0.6052 |
| hd ml2 | **20.1687 +- 0.0506** (8) | **20.3614 +- 0.0844** (3) | 42.4572 +- 0.4701 | 42.0955 +- 0.5783 |
| cohg_r0 (gated) | 20.6753 +- 0.0386 (8) | 21.4029 +- 0.4303 (3) | 43.4879 +- 0.4159 | 42.7974 +- 0.5184 |
| cohg_nogate (ungated) | 23.7696 +- 6.2355 (8) | 392.1805 +- 615.2561 (3) | 911.44 +- 2232.89 | 141585.05 +- 244414.92 |

### 3.2 Gate effect, by order

| quantity | wiki -> news -> code | code -> news -> wiki |
|---|---|---|
| gated - fixed (mean paired diff) | **-0.4166** (n=8, p = 0.0078) | **+0.3043** (n=3, p = 0.50, n.s.) |
| gated - hd2 (mean paired diff) | +0.5066 (n=8, p = 0.0078) | +1.0415 (n=3, p = 0.25) |
| ungated - fixed (mean paired diff) | +2.6777 (n=8, p = 0.297) | +371.08 (n=3, p = 0.50) |
| gate openings per run (cohg_r0) | 1 / 2999, all seeds, in (0,20] | 1 / 2999, all seeds, in (0,20] |
| direction of the single accepted step | +0.4 on all 6 coords, 8/8 seeds | -0.4 on 5 of 6 coords in s1,s2; -0.4 on 2 of 6 in s0 |
| degraded runs, gated | 0 / 8 | 0 / 3 |
| degraded runs, ungated | 3 / 8 (37.5%) | 2 / 3 (66.7%) |
| ungated worst-seed PPL | 38.03 (1.80x fixed) | 1102.37 (52.2x fixed) |
| ungated worst-seed worst-window PPL | 6423 | 423811 |

### 3.3 Per-domain PPL (mean over seeds; context for the order effect)

| arm | wiki->news->code (D1 wiki / D2 news / D3 code) | code->news->wiki (D1 code / D2 news / D3 wiki) |
|---|---|---|
| fixed | 34.04 / 37.32 / 7.38 | 7.34 / 37.15 / 34.45 |
| hd ml2 | 33.72 / 35.08 / 6.93 | 7.29 / 36.19 / 32.00 |
| cohg_r0 | 33.77 / 36.21 / 7.22 | 7.47 / 37.87 / 34.67 |
| cohg_nogate | 32.37 / 36.79 / 13.29 | 64.24 / 1511.51 / 899.50 |

The domains themselves are near order-invariant under fixed lr (code ~7.3-7.4,
news ~37.1-37.3, wiki ~34.0-34.5), so the arm-level order effects below are
about the *adapters*, not about the data.

### 3.4 Does the gate's benefit / behaviour change with order? -- yes for benefit, no for behaviour

* **Behaviour is order-invariant.** In both orders the certified gate accepts
  exactly one meta-update per run, in the first 20 steps, and zero thereafter;
  `gate_open_frac` is bit-identical (3.3344448e-04) across all 11 gated runs.
  The gate never fires at a domain boundary in either order.
* **Benefit is not order-invariant, and it flips sign.** Wiki-first, gating
  beats fixed by 0.417 PPL on 8/8 seeds (p = 0.0078). Code-first, gating is
  0.304 PPL *worse* than fixed on average (2/3 seeds worse, p = 0.50), and
  the seed spread widens 11x against the 8-seed
  wiki-first std (0.0386 -> 0.4303), or 27x against the like-for-like 3-seed
  wiki-first std (0.0160 -> 0.4303). The mechanism is visible in
  the accepted step: wiki-first it always raises the LR (+0.4 on all six
  coordinates, which helps on this stream), code-first it usually lowers it,
  and lowering the LR from an already well-tuned 1e-3 buys nothing on
  whole-stream PPL. Because the whole arm rests on one accepted step, the arm's
  outcome inherits that step's variance directly.
* **Safety is order-invariant and is the claim that holds.** 0/8 and 0/3
  degraded gated runs versus 3/8 and 2/3 degraded ungated runs; the gated arm's
  worst-case worst-window PPL is below the fixed baseline's wiki-first
  (43.81 vs 44.28) and above it by only 0.33 code-first (43.32 vs 42.99), while the
  ungated arm reaches 6.4e3 and 4.2e5 respectively. Notably the *ungated*
  failure is markedly worse code-first (worst PPL 1102 vs 38), so the alternate
  order strengthens, not weakens, the case for the gate as a safety mechanism
  even as it removes the accuracy advantage.
* **hd ml2 stays the best accuracy arm in both orders** (20.17 wiki-first,
  20.36 code-first) with 0 degraded runs in this GPT-2 setting -- the
  order sweep does not change that, and any claim of accuracy superiority for
  COHG on E4 should continue to be avoided.
* **Cost, for completeness** (unchanged by order): fixed 0.41-0.42 h /
  6.05 GB, hd2 0.59-0.60 h / 6.51 GB, cohg_r0 4.44-4.97 h / 16.96 GB,
  cohg_nogate 6.03-6.71 h / 18.80 GB per run. The gated arm pays ~11x the
  wall-clock and ~2.8x the peak memory of fixed for one accepted meta-update.

---

## 4. Figure

`paper/main/figs/fig9_gpt2_order.pdf` (+ `results/figures/fig9_gpt2_order.png`
preview at 300 dpi), generated by `results/reanalysis/_fig9_gpt2_order.py`.
Single column 3.3 in wide, 8 pt base font, Okabe-Ito palette, no titles,
Type-42 fonts.

* **(a)** cohg_r0 lambda trajectories, all 3 code-first seeds overlaid (seed 0
  bold, seeds 1-2 thin), six per-coordinate curves offset vertically by 0.018
  for visibility, dashed boundaries at t = 1000 / 2000, dotted marker plus
  annotation at the single gate opening (t <= 20). The panel shows the flat
  post-step trajectories and the empty boundaries.
* **(b)** the worst ungated seed (cohg_nogate s2, online PPL 1102.4) with the
  clamp lines at `log 1e-6` and `log 1e-1`, showing full-range bang-bang
  saturation across all three domains.
* Domain labels along the top read `code / news / wiki` (vs `domain 1/2/3` in
  fig8), so the two figures can sit side by side.

## 5. Files produced

* `results/reanalysis/e4_expansion.md` (this file)
* `results/reanalysis/_fig9_gpt2_order.py` (figure generator, reproducible)
* `paper/main/figs/fig9_gpt2_order.pdf`
* `results/figures/fig9_gpt2_order.png`
