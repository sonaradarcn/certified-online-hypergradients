# Round-3 GPU reanalysis: mis-set initialization, reverse-order n = 8, traced E3

Analysis only -- no paper text and no result artifact was edited. Every number below is recomputed from the raw run JSONs by `results/reanalysis/_round3_gpu.py`, which reuses the metric code of `_reanalyze.py` verbatim.

**Conventions.** `ddof = 1` for every std. **Unified spike rule**, identical in every regime: maintain a deque of the last 500 *finite* losses; once it holds >= 100 entries a step counts as a spike iff `loss_t > 10 x median(window)`; every non-finite loss is additionally an event; `unified events = spikes + non-finite`. **max-excess** = `max(finite loss) / median(finite loss)`, in loss space. **worst-window** = `max_e mean(loss[e-100 .. e-1])` over all `e = 100..n`, averaging finite entries only; a window containing *no* finite entry contributes `+inf` (this is what produces the `inf` cells in section 3 -- a separate finite-restricted variant is tabulated alongside so the arms remain comparable). On GPT-2 worst-window is reported as `exp(.)` = perplexity; on CIFAR-100 it stays in raw cross-entropy. **Paired tests** are exact two-sided sign-flip (randomization) tests enumerating all `2^n` sign assignments; `p` floors are `2/2^n` = 0.25 (n=3), 0.0078125 (n=8), 0.001953125 (n=10).

Three completed GPU result sets are covered, all launched by `code/experiments/launch_r3_chain.py` on the local 2x RTX 3080:

| set | directory | runs | phase |
|---|---|---|---|
| mis-set initialization | `results/e4_misset/` | 9 | p2_misset (review P7) |
| reverse-order expansion | `results/e4_orders/` (+5 new) | 17 total | p2_misset |
| traced E3 | `results/e3_traced/` | 80 | p3_e3traced (review P5/P9) |

### The three plain answers up front

1. **Mis-set init (P7).** *The certified gate does adapt when adaptation is genuinely needed, and it does stop when it stops -- but it stops far too early.* From lr0 = 1e-4 it opens the gate **twice** (steps 1 and 2, all three seeds) versus **once** from the well-set 1e-3, and every accepted move raises the LR. It then never opens again for 2997 steps and never at a domain boundary. The LR ends at only 1.49x-2.06x the mis-set init against the 10x needed, recovering 23.2% of the mis-set PPL penalty (24.888 -> 24.006 against 21.092 well-set). Safety is untouched: 0/3 degraded, 0 spikes, 0 non-finite, max-excess indistinguishable from the fixed baseline, while the ungated ablation from the same start degrades 1/3 with a worst-window PPL of 3.7e6.
2. **Reverse order at n = 8.** *The "+0.30 n.s. code-first penalty" does not change in kind.* The paired test is arithmetically unchanged (n = 3, +0.3043 PPL, p = 0.5) because `fixed` gained no seeds; the unpaired n = 8 view softens the gap to +0.2837 PPL, still smaller than the gated arm's own seed std (0.3881) and still unsupported. Gate timing for the five new seeds is now **exact** and confirms the finding: `gate_open_steps = [1]`, exactly one opening, well before step 20, zero at either boundary, in every one of the five. Degraded runs: **0 / 8**.
3. **Traced E3.** *Both earlier claims hold, in opposite regimes.* At **ewc10** the gate separation is real and the trace metrics strengthen it: `cohg` is not merely as safe as `fixed` and `hd` but measurably safer (lower max-excess on 9/10 and 10/10 seeds respectively), while the ungated ablation blows up in 6/10 seeds. At **ewc1000** HD's dominance holds and hardens: `hd` triggers 0/10 and blows up 1/10, `cohg` triggers **6/10** and blows up 6/10 -- the worst trigger rate of any arm anywhere in E3. **Caveat:** the traced set's ewc1000 block does not reproduce the canonical `results/e3` seed for seed (max |delta acc| 0.296); the ewc10 block does (max 0.041).

---

## 1. E4 mis-set initialization (`results/e4_misset/`, lr0 = 1e-4, seeds 0-2)

Review item P7, *"adaptation when it is actually needed"*: the stream and every other flag are the `e4_v2` standard (wiki -> news -> code, 2999 steps, boundaries t = 1000 / 2000, 512k tokens per domain, `meta_lr = 0.4`, `K = 20`, `gamma = 0.9`, `probe_every = 100`, `kw_eps = 0.15`), and the *only* change is that the initial learning rate is set to **1e-4, i.e. 10x below the well-set default 1e-3 and 30x below the post-hoc-best 3e-3**. All 9 runs carry `legacy_hold = False`, `held_bound = vector_prop10` (the corrected Proposition-10 vector bound), so the whole block is post-fix provenance.

`lam_hist` is 150 rows `[t, lam_0..lam_5]` sampled every 20 steps (t = 0, 20, ..., 2980), and both COHG arms additionally store `gate_open_steps`, so gate timing here is **exact**, not interval-localised. Lambda init = `log 1e-4 = -9.2103` on all six LR groups (emb, h0-2, h3-5, h6-8, h9-11, ln_f); clamps `[log 1e-6, log 1e-1] = [-13.8155, -2.3026]`. The certified step is `lam_j <- lam_j - meta_lr * s_j * sign(ghat_j)` with `s_j = min(1, (|ghat_j| - beta_j)/|ghat_j|) <= 1`, so an accepted move is *at most* 0.40 in lambda (a factor 1.4918 in LR) per coordinate.

### 1.1 Online PPL, worst-window, max-excess, events

| arm | n | online PPL mean +- std | median | min | max | worst-window PPL | max-excess | spikes | non-finite | unified events | events (stored) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed lr 1e-4 | 3 | **24.8876 +- 0.0137** | 24.8856 | 24.8749 | 24.9021 | 50.71 +- 0.32 | 1.3006 +- 0.0115 | 0.00 +- 0.00 | 0.00 +- 0.00 | 0.00 +- 0.00 | 0.00 +- 0.00 |
| cohg_r0 (gated) lr0 1e-4 | 3 | **24.0057 +- 0.0203** | 24.0116 | 23.9831 | 24.0224 | 48.09 +- 0.33 | 1.3089 +- 0.0112 | 0.00 +- 0.00 | 0.00 +- 0.00 | 0.00 +- 0.00 | 0.00 +- 0.00 |
| cohg_nogate lr0 1e-4 | 3 | **28.8490 +- 12.4080** | 22.8352 | 20.5938 | 43.1179 | 1.24e+06 +- 2.14e+06 | 15.0130 +- 22.4540 | 5.67 +- 9.81 | 0.00 +- 0.00 | 5.67 +- 9.81 | 0.00 +- 0.00 |

Per-seed online PPL / worst-window PPL / max-excess / spikes:

| arm | s0 | s1 | s2 |
|---|---|---|---|
| fixed lr 1e-4 | 24.8856 / 50.91 / 1.2916 / 0 | 24.8749 / 50.34 / 1.3136 / 0 | 24.9021 / 50.87 / 1.2966 / 0 |
| cohg_r0 (gated) lr0 1e-4 | 24.0116 / 48.32 / 1.2994 / 0 | 23.9831 / 47.71 / 1.3212 / 0 | 24.0224 / 48.24 / 1.3061 / 0 |
| cohg_nogate lr0 1e-4 | 22.8352 / 62.74 / 2.5126 / 0 | 43.1179 / 3.71e+06 / 40.9352 / 17 | 20.5938 / 45.13 / 1.5911 / 0 |

### 1.2 Where the mis-set arms land relative to the well-set reference points

Reference rows recomputed from `results/e4_v2/` through the identical code path.

| operating point | n | online PPL | worst-window PPL (mean) | max-excess (mean) |
|---|---|---|---|---|
| fixed lr **1e-4** (mis-set, 10x too small) | 3 | **24.8876 +- 0.0137** | 50.71 | 1.3006 |
| **cohg_r0 from lr0 = 1e-4** (gated, must adapt) | 3 | **24.0057 +- 0.0203** | 48.09 | 1.3089 |
| cohg_nogate from lr0 = 1e-4 | 3 | **28.8490 +- 12.4080** | 1.24e+06 | 15.0130 |
| fixed lr 1e-3 (well-set default) | 8 | 21.0919 +- 0.0306 | 44.16 | 1.3754 |
| cohg_r0 from lr0 = 1e-3 (well-set) | 8 | 20.6753 +- 0.0386 | 43.75 | 1.3967 |
| hd ml2 from lr0 = 1e-3 | 8 | 20.1687 +- 0.0506 | 42.79 | 1.3787 |
| fixed lr 3e-3 (**post-hoc best** grid point) | 3 | 19.8396 +- 0.0269 | 44.27 | 1.6540 |

Paired sign-flip tests inside the mis-set block (common seeds 0-2, n = 3, p floor 0.25):

| contrast | per-seed diffs (s0, s1, s2) | mean diff | sign pattern | p (exact) |
|---|---|---|---|---|
| `cohg_r0` - `fixed` | -0.8740, -0.8918, -0.8798 | **-0.8819** | 3 neg / 0 pos | 0.25 |
| `cohg_nogate` - `fixed` | -2.0504, +18.2430, -4.3083 | **+3.9614** | 2 neg / 1 pos | 1 |
| `cohg_nogate` - `cohg_r0` | -1.1764, +19.1348, -3.4286 | **+4.8433** | 2 neg / 1 pos | 1 |

**Recovery accounting.** Mis-setting the LR costs `fixed 1e-4 - fixed 1e-3` = **+3.7957 PPL** (24.8876 vs 21.0919). The certified gate recovers **0.8819 PPL** of that, i.e. **23.2%** of the distance back to the well-set fixed baseline; it still finishes **+2.9138 PPL** above well-set fixed 1e-3 and **+4.1661 PPL** above the post-hoc-best 3e-3 point. For scale, the gate's benefit at the *well-set* start was only -0.4166 PPL, so mis-setting doubles the gate's payoff in absolute PPL while leaving ~77% of the mis-set penalty on the table.

### 1.3 Gate behaviour of `cohg_r0` at lr0 = 1e-4 -- the key question

| seed | `gate_open_frac` | exact #opens / 2999 | `gate_open_steps` | `coord_open_frac` | accepted coord-moves (of 2999 x 6) | lambda end | implied end LR |
|---|---|---|---|---|---|---|---|
| 0 | 0.000666888963 | **2** | [1, 2] | 0.000389018562 | **7** | [-8.5131, -8.8103, -8.8103, -8.8103, -8.8103, -8.8103] | [2.008e-04, 1.492e-04, 1.492e-04, 1.492e-04, 1.492e-04, 1.492e-04] |
| 1 | 0.000666888963 | **2** | [1, 2] | 0.000389018562 | **7** | [-8.4899, -8.8103, -8.8103, -8.8103, -8.8103, -8.8103] | [2.055e-04, 1.492e-04, 1.492e-04, 1.492e-04, 1.492e-04, 1.492e-04] |
| 2 | 0.000666888963 | **2** | [1, 2] | 0.000389018562 | **7** | [-8.5080, -8.8103, -8.8103, -8.8103, -8.8103, -8.8103] | [2.018e-04, 1.492e-04, 1.492e-04, 1.492e-04, 1.492e-04, 1.492e-04] |

Compare the well-set arm: `results/e4_v2/gpt2_cohg_r0_lr0.001_s*` has `gate_open_frac = 3.3344448e-04` = **1** open and **6** accepted coordinate-moves in all 8 seeds. Starting 10x too low, the gate opens **twice** and accepts **seven** coordinate-moves -- it does respond to the larger error, but by exactly one extra coordinate-move.

Per-coordinate lambda displacement from init (`log 1e-4 = -9.2103`) and the resulting LR multiplier:

| seed | quantity | emb | h0-2 | h3-5 | h6-8 | h9-11 | ln_f |
|---|---|---|---|---|---|---|---|
| 0 | delta lambda | +0.6972 | +0.4000 | +0.4000 | +0.4000 | +0.4000 | +0.4000 |
| 0 | LR multiplier | 2.008x | 1.492x | 1.492x | 1.492x | 1.492x | 1.492x |
| 1 | delta lambda | +0.7205 | +0.4000 | +0.4000 | +0.4000 | +0.4000 | +0.4000 |
| 1 | LR multiplier | 2.055x | 1.492x | 1.492x | 1.492x | 1.492x | 1.492x |
| 2 | delta lambda | +0.7023 | +0.4000 | +0.4000 | +0.4000 | +0.4000 | +0.4000 |
| 2 | LR multiplier | 2.018x | 1.492x | 1.492x | 1.492x | 1.492x | 1.492x |

The structure is identical in all three seeds and decomposes exactly: **open 1 (step 1) moves all six coordinates by the full +0.400; open 2 (step 2) moves only `emb`, and only by ~+0.30** (the trust factor `s_j` < 1 there). 6 + 1 = the 7 accepted coordinate-moves in the table above.

Trajectory check over all 150 `lam_hist` samples (t = 0, 20, ..., 2980):

- **seed 0**: lambda changes in **1** of 149 sample gaps ([(0, 20)]); bit-constant from the t = 20 sample through the t = 2980 sample: **True**. Exact opens `[1, 2]` -- both inside the first three steps of the stream. Opens at t >= 1000 (news boundary): **0**. Opens at t >= 2000 (code boundary): **0**.
- **seed 1**: lambda changes in **1** of 149 sample gaps ([(0, 20)]); bit-constant from the t = 20 sample through the t = 2980 sample: **True**. Exact opens `[1, 2]` -- both inside the first three steps of the stream. Opens at t >= 1000 (news boundary): **0**. Opens at t >= 2000 (code boundary): **0**.
- **seed 2**: lambda changes in **1** of 149 sample gaps ([(0, 20)]); bit-constant from the t = 20 sample through the t = 2980 sample: **True**. Exact opens `[1, 2]` -- both inside the first three steps of the stream. Opens at t >= 1000 (news boundary): **0**. Opens at t >= 2000 (code boundary): **0**.

Highest lambda any coordinate reaches in any seed: **-8.4899** (LR = 2.055e-04, i.e. 2.06x the init). Targets: `log 1e-3 = -6.9078`, `log 3e-3 = -5.8091`. Remaining climb to 1e-3: **1.5821** in lambda (4.87x more LR), which at <= 0.40 per accepted move needs **>= 4 further gate openings**; to reach the post-hoc-best 3e-3 it needs >= 7. It makes **zero**.

### 1.4 `cohg_nogate` at lr0 = 1e-4: lambda range, clamp saturation, degradation

The gate is forced open at every step (`gate_open_frac = 1.0`, 2999/2999). `coord_open_frac = 0.0` here means the certified controller is **never consulted** (`CoordGatedController.maybe_update` is bypassed, so its counters stay at zero) -- it does *not* mean zero coordinates were certified. Clamps `[-13.8155, -2.3026]`, admissible span 11.5129.

| seed | online PPL | worst-window PPL | max-excess | spikes | per-coord lambda range (max-min) | frac of lam samples at lower clamp | at upper clamp | lambda end | peak per-step log-loss (step) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 22.8352 | 62.74 | 2.5126 | 0 | [10.000, 11.513, 11.513, 11.513, 11.513, 11.513] | 0.4267 | 0.0867 | [-13.816, -13.816, -13.816, -11.103, -13.016, -11.103] | 8.4499 (t=2205) |
| 1 | 43.1179 | 3.71e+06 | 40.9352 | 17 | [10.800, 11.200, 11.513, 11.513, 11.513, 11.513] | 0.3833 | 0.1022 | [-13.416, -13.016, -2.303, -2.703, -2.303, -2.703] | 142.6159 (t=2748) |
| 2 | 20.5938 | 45.13 | 1.5911 | 0 | [9.200, 11.200, 11.513, 11.513, 11.513, 11.513] | 0.4011 | 0.0856 | [-13.816, -13.816, -13.816, -13.816, -13.416, -13.016] | 5.3147 (t=2237) |

**18 / 18** coordinate-seed pairs touch the lower clamp exactly and **13 / 18** touch the upper clamp exactly; 38-43% of all lambda samples sit pinned at the lower clamp and 9-10% at the upper one. Started 10x *below* the well-set LR -- the one situation in which climbing is unambiguously the right move -- the ungated meta-optimizer still does not climb and settle: it bang-bangs across the full admissible span, exactly as it does from the well-set start.

Per-domain PPL (D1 wiki t < 1000, D2 news 1000 <= t < 2000, D3 code t >= 2000):

| arm | seed | D1 wiki | D2 news | D3 code |
|---|---|---|---|---|
| fixed lr 1e-4 | 0 | 38.009 | 46.047 | 8.796 |
| fixed lr 1e-4 | 1 | 38.103 | 45.769 | 8.817 |
| fixed lr 1e-4 | 2 | 37.999 | 46.108 | 8.805 |
| cohg_r0 (gated) lr0 1e-4 | 0 | 37.186 | 43.993 | 8.454 |
| cohg_r0 (gated) lr0 1e-4 | 1 | 37.260 | 43.701 | 8.463 |
| cohg_r0 (gated) lr0 1e-4 | 2 | 37.177 | 44.036 | 8.459 |
| cohg_nogate lr0 1e-4 | 0 | 31.247 | 40.462 | 9.410 |
| cohg_nogate lr0 1e-4 | 1 | 32.097 | 39.944 | 62.547 |
| cohg_nogate lr0 1e-4 | 2 | 32.216 | 35.860 | 7.553 |

**Degraded-run classification** (definition of `e4_expansion.md` section 0, anchored on the fixed baseline of the *same configuration*, i.e. fixed lr 1e-4): PPL threshold = 24.8876 + 2.0 = **26.8876**; worst-window threshold = 2 x median worst-window of `cohg_r0` = 2 x 48.24 = **96.49**.

| arm | degraded / n | which seeds (criterion) |
|---|---|---|
| fixed lr 1e-4 | **0 / 3** | - |
| cohg_r0 (gated) lr0 1e-4 | **0 / 3** | - |
| cohg_nogate lr0 1e-4 | **1 / 3** | s1 (PPL+worst-window) |

Cost:

| arm | wall-clock h | peak GB | HVPs |
|---|---|---|---|
| fixed lr 1e-4 | 0.68 | 6.05 | 0 |
| cohg_r0 (gated) lr0 1e-4 | 3.22 | 17.43 | 4100 |
| cohg_nogate lr0 1e-4 | 6.88 | 18.80 | 1492 |

### 1.5 Plain answer to review P7

**Does the certified gate adapt when adaptation is genuinely needed? Partly -- it moves in the right direction, immediately, and it stops; but it moves far too little, and it stops long before the adaptation is finished.**

1. *It notices, and it notices fast.* Started 10x too low, `cohg_r0` opens the gate at steps **1 and 2** in all three seeds, and every accepted coordinate-move **raises** the LR (`+0.400` on all six groups at open 1, a further `+0.30` on `emb` at open 2). Under the well-set start the same rule opens **once** and accepts six moves; mis-set, it opens **twice** and accepts seven. So the gate is measurably more active when the initialization is worse -- but the extra activity amounts to a single extra coordinate-move.
2. *It stops -- and stays stopped.* After step 2 the gate never opens again for the remaining 2997 steps; lambda is bit-constant from the t = 20 sample to the t = 2980 sample in every seed. There are **zero** openings at or after either domain boundary (t = 1000, t = 2000). So the answer to "does it stop when done" is an unambiguous *yes* -- the gate does not chatter and does not drift.
3. *But it stops before it is done.* The final LR is only **1.49x-2.06x** the mis-set init (2.0e-4 on `emb`, 1.49e-4 elsewhere), against the 10x needed to reach 1e-3 and the 30x needed to reach the post-hoc best 3e-3. Reaching 1e-3 would require at least 4 more openings; it makes none. Online PPL lands at **24.006 +- 0.020**, versus 24.888 +- 0.014 for fixed 1e-4, 21.092 +- 0.031 for well-set fixed 1e-3 and 19.840 +- 0.027 for the post-hoc best. The gate recovers **23.2%** of the mis-set penalty and leaves 77% of it standing.
4. *The safety claim survives the stress test intact.* 0/3 degraded gated runs, 0 spikes, 0 non-finite losses, max-excess 1.309 +- 0.011 -- statistically indistinguishable from the fixed baseline's 1.301 +- 0.012, and its worst-window (48.09) is *below* the fixed baseline's (50.71). Removing the gate from the same starting point produces 1/3 degraded runs, a seed at 43.1 PPL with a worst-window PPL of 3.7e6 and a single-step log-loss of 142.6, and 5.7 +- 9.8 spikes per run.
5. *Honest framing for the paper.* This block is the strongest available evidence that COHG's gate is not merely inert -- it does fire more, and in the correct direction, when the LR is genuinely mis-set, and it beats the mis-set fixed baseline on **3/3 seeds** with a mean gap of -0.882 PPL (p = 0.25 only because n = 3 floors it there; the effect is 43x the seed std). But it is *not* evidence of full recovery: a method that certifiably adapts would have to keep opening, and this one certifies its way to a stop after two steps. The correct claim is "certified partial recovery from a mis-set LR, at zero cost in stability", not "the gate finds the right LR".

---

## 2. Reverse-order stream (code -> news -> wiki) refreshed at n = 8

`results/e4_orders/`, files `gpt2order_cnw_*`. `cohg_r0` now covers seeds 0-7; `fixed`, `hd ml2` and `cohg_nogate` are still at seeds 0-2. D1 = code (t < 1000), D2 = news, D3 = wiki (t >= 2000).

### 2.0 Provenance of the eight `cohg_r0` seeds (stated up front, as required)

| seed | `legacy_hold` | `held_bound` | `gate_open_steps` | `gate_open_frac` | `coord_open_frac` | events | online PPL | wall h |
|---|---|---|---|---|---|---|---|---|
| 0 | *(key absent)* | *(key absent)* | *(key absent)* | 0.000333444481 | 0.000333444481 | 0 | 20.9062 | 5.06 |
| 1 | *(key absent)* | *(key absent)* | *(key absent)* | 0.000333444481 | 0.000333444481 | 0 | 21.6388 | 4.64 |
| 2 | *(key absent)* | *(key absent)* | *(key absent)* | 0.000333444481 | 0.000333444481 | 0 | 21.6637 | 5.21 |
| 3 | False | vector_prop10 | [1] | 0.000333444481 | 0.000333444481 | 0 | 21.6223 | 3.09 |
| 4 | False | vector_prop10 | [1] | 0.000333444481 | 0.000333444481 | 0 | 21.6452 | 3.43 |
| 5 | False | vector_prop10 | [1] | 0.000333444481 | 0.000333444481 | 0 | 21.6098 | 3.65 |
| 6 | False | vector_prop10 | [1] | 0.000333444481 | 0.000333444481 | 0 | 21.2949 | 3.00 |
| 7 | False | vector_prop10 | [1] | 0.000333444481 | 0.000333444481 | 0 | 20.6776 | 3.20 |

Seeds **0-2** were produced by the pre-fix driver revision and carry **no** `legacy_hold` / `held_bound` / `gate_open_steps` keys. Seeds **3-7** were launched by `launch_r3_chain.py` phase `p2_misset` *after* the held-bound correction (review P4: the full vector-valued Proposition-10 drift-hold, `dh.probe(..., eta_vec=eta)` / `dh.bounds(eta)`, replacing the scalar path) and record `legacy_hold = False`, `held_bound = vector_prop10`. **The two sub-blocks are therefore not identical provenance and every n = 8 aggregate below must carry that caveat.**

What is checkable is that the correction did not change the gate's realized behaviour on this stream: `gate_open_frac` and `coord_open_frac` are **bit-identical** (3.3344448e-04, i.e. exactly one open and six accepted coordinate-moves in 2999 steps) across all eight seeds, old and new, and every seed has `events = 0`. The new seeds also run ~1.5x faster in wall-clock (3.0-3.7 h vs 4.6-5.2 h), consistent with a less contended card rather than a different amount of work (`hvp_total = 4100` in all eight).

### 2.1 The refreshed `cohg_r0` row at n = 8

| quantity | n = 3 (seeds 0-2, as published) | **n = 8 (seeds 0-7)** |
|---|---|---|
| online PPL mean +- std | 21.4029 +- 0.4303 | **21.3823 +- 0.3881** |
| online PPL median | 21.6388 | **21.6161** |
| online PPL min / max | 20.9062 / 21.6637 | **20.6776 / 21.6637** |
| worst-window PPL mean +- std | 42.7974 +- 0.5184 | **42.7709 +- 0.4091** |
| worst-window median / min / max | 42.7940 / 42.2808 / 43.3175 | **42.6622 / 42.2808 / 43.3175** |
| max-excess mean +- std | 1.3401 +- 0.0047 | **1.3584 +- 0.0186** |
| spikes / non-finite / stored events (totals) | 0 / 0 / 0 | **0 / 0 / 0** |
| **degraded runs** | 0 / 3 | **0 / 8** |

Per-seed detail, all eight seeds:

| seed | online PPL | worst-window PPL | max-excess | max finite log-loss | median log-loss | spikes | non-finite | degraded? |
|---|---|---|---|---|---|---|---|---|
| 0 | 20.9062 | 42.7940 | 1.3422 | 4.5462 | 3.3871 | 0 | 0 | no |
| 1 | 21.6388 | 42.2808 | 1.3433 | 4.5918 | 3.4184 | 0 | 0 | no |
| 2 | 21.6637 | 43.3175 | 1.3347 | 4.5683 | 3.4227 | 0 | 0 | no |
| 3 | 21.6223 | 42.5305 | 1.3758 | 4.7192 | 3.4302 | 0 | 0 | no |
| 4 | 21.6452 | 43.2010 | 1.3801 | 4.7316 | 3.4283 | 0 | 0 | no |
| 5 | 21.6098 | 42.4352 | 1.3736 | 4.7005 | 3.4220 | 0 | 0 | no |
| 6 | 21.2949 | 43.1744 | 1.3720 | 4.6744 | 3.4071 | 0 | 0 | no |
| 7 | 20.6776 | 42.4335 | 1.3455 | 4.5426 | 3.3761 | 0 | 0 | no |

**Degraded-run definition** (e4_expansion.md section 0): degraded iff online PPL > (same-order fixed mean + 2.0) **or** worst-window PPL > 2 x (median worst-window of the gated arm of the same order). The fixed mean is unchanged at 21.0986 -> **PPL threshold 23.0986**. The gated arm's own median worst-window moves from 42.7940 (n = 3) to 42.6622 (n = 8), so its worst-window threshold moves from **85.5879** to **85.3245**. **No classification changes under either threshold.**

| arm | n | degraded / n (n=8 threshold) | degraded / n (n=3 threshold) | which seeds |
|---|---|---|---|---|
| fixed lr 1e-3 | 3 | **0 / 3** | 0 / 3 | - |
| hd ml2 | 3 | **0 / 3** | 0 / 3 | - |
| cohg_r0 (gated) | 8 | **0 / 8** | 0 / 8 | - |
| cohg_nogate | 3 | **2 / 3** | 2 / 3 | s1 (PPL+WW), s2 (PPL+WW) |

### 2.2 Paired test on common seeds -- does the "+0.30 n.s. code-first penalty" change?

**No.** `fixed` exists only for seeds 0-2 on this stream, so the *paired* contrast is still n = 3 and is arithmetically **unchanged** by the refresh -- adding seeds to one arm cannot alter a paired statistic whose pairs did not change.

| contrast | n | per-seed diffs (s0, s1, s2) | mean diff | sign | p (exact) |
|---|---|---|---|---|---|
| `cohg_r0` - `fixed` | 3 | -0.1919, +0.5415, +0.5632 | **+0.3043** | 2 pos / 1 neg | 0.5 |
| `cohg_r0` - `hd ml2` | 3 | +0.4900, +1.3746, +1.2599 | +1.0415 | 3 pos | 0.25 |
| `cohg_nogate` - `fixed` | 3 | -0.2810, +32.2586, +1081.2681 | +371.0819 | 2 pos / 1 neg | 0.5 |

### 2.3 The n = 8 summary, reported separately (not a paired test)

| quantity | value |
|---|---|
| `cohg_r0` n = 8 mean +- std | **21.3823 +- 0.3881** |
| `fixed` n = 3 mean +- std | 21.0986 +- 0.0017 |
| unpaired difference of means | **+0.2837 PPL** |
| same difference at n = 3 (published) | +0.3043 PPL |
| shift caused by the five new seeds | -0.0206 PPL |
| `cohg_r0` std, n = 3 -> n = 8 | 0.4303 -> 0.3881 |

The five new seeds (21.6223, 21.6452, 21.6098, 21.2949, 20.6776) land inside the existing n = 3 range [20.9062, 21.6637] except s7 = 20.6776, which is a new minimum and is *below* the fixed baseline. So the direction of the finding is unchanged (`cohg_r0` still costs ~+0.28 PPL against fixed under code-first, versus -0.42 PPL under wiki-first) but the magnitude softens from +0.3043 to **+0.2837** and the arm's dispersion stays large (0.3881, versus 0.0386 for the same arm under wiki-first at n = 8 -- a 10x wider seed spread). Two of eight seeds (s0, s7) beat fixed; six do not. **The sign flip relative to the standard order survives at n = 8, and it remains statistically unsupported.**

### 2.4 Gate-open timing, including the five new seeds

Seeds 3-7 store `gate_open_steps` explicitly, so their timing is exact. Seeds 0-2 lack the key and are localised from `lam_hist` to the half-open interval `(t_prev, t_cur]`.

| seed | exact #opens / 2999 | open step(s) | before step 20? | opens at t >= 1000 | opens at t >= 2000 | lambda end | coords raised / lowered |
|---|---|---|---|---|---|---|---|
| 0 | **1** | (0, 20] (localised) | **True** | 0 | 0 | [-7.3078, -6.5078, -7.3078, -6.5078, -6.5078, -6.5078] | 4 / 2 |
| 1 | **1** | (0, 20] (localised) | **True** | 0 | 0 | [-7.3078, -7.3078, -7.3078, -7.3078, -7.3078, -6.5078] | 1 / 5 |
| 2 | **1** | (0, 20] (localised) | **True** | 0 | 0 | [-7.3078, -7.3078, -7.3078, -7.3078, -7.3078, -6.5078] | 1 / 5 |
| 3 | **1** | `[1]` (exact) | **True** | 0 | 0 | [-7.3078, -7.3078, -7.3078, -7.3078, -7.3078, -6.5078] | 1 / 5 |
| 4 | **1** | `[1]` (exact) | **True** | 0 | 0 | [-7.3078, -7.3078, -7.3078, -7.3078, -7.3078, -6.5078] | 1 / 5 |
| 5 | **1** | `[1]` (exact) | **True** | 0 | 0 | [-7.3078, -7.3078, -7.3078, -7.3078, -7.3078, -6.5078] | 1 / 5 |
| 6 | **1** | `[1]` (exact) | **True** | 0 | 0 | [-7.3078, -7.3078, -7.3078, -6.5078, -6.5078, -6.5078] | 3 / 3 |
| 7 | **1** | `[1]` (exact) | **True** | 0 | 0 | [-7.3078, -6.5078, -6.5078, -6.5078, -6.5078, -6.5078] | 5 / 1 |

Per-coordinate delta lambda (init `log 1e-3 = -6.9078`, meta_lr 0.4):

| seed | emb | h0-2 | h3-5 | h6-8 | h9-11 | ln_f | # lowered |
|---|---|---|---|---|---|---|---|
| 0 | -0.4000 | +0.4000 | -0.4000 | +0.4000 | +0.4000 | +0.4000 | 2 |
| 1 | -0.4000 | -0.4000 | -0.4000 | -0.4000 | -0.4000 | +0.4000 | 5 |
| 2 | -0.4000 | -0.4000 | -0.4000 | -0.4000 | -0.4000 | +0.4000 | 5 |
| 3 | -0.4000 | -0.4000 | -0.4000 | -0.4000 | -0.4000 | +0.4000 | 5 |
| 4 | -0.4000 | -0.4000 | -0.4000 | -0.4000 | -0.4000 | +0.4000 | 5 |
| 5 | -0.4000 | -0.4000 | -0.4000 | -0.4000 | -0.4000 | +0.4000 | 5 |
| 6 | -0.4000 | -0.4000 | -0.4000 | +0.4000 | +0.4000 | +0.4000 | 3 |
| 7 | -0.4000 | +0.4000 | +0.4000 | +0.4000 | +0.4000 | +0.4000 | 1 |

Lambda is bit-constant from the t = 20 sample onwards in **all eight** seeds: **True**. So: exactly one accepted meta-update per run, at step 1 (verified exactly for s3-s7, localised to (0, 20] for s0-s2), then 2979 steps of frozen lambda. **Zero** openings at either domain boundary in any of the eight seeds.

The five new seeds also reproduce the *direction* finding: s3, s4, s5 lower five of six coordinates (identical pattern to s1 and s2), s6 lowers three, s7 lowers only `emb`. Across all eight seeds `emb` is lowered **8/8** times and `ln_f` is raised **8/8** times; the four middle blocks split. The published claim -- "under code-first the same certified rule usually *lowers* the LR, whereas wiki-first it always raises it" -- holds at n = 8, with the sharper statement that the embedding group is lowered in every single seed of this order and in no seed of the standard order.

### 2.5 Plain answer

**The "+0.30 n.s. code-first penalty" does not change in kind, only in size, and the gate's timing finding is now exact rather than inferred.** The paired test is untouched at +0.3043 PPL, p = 0.5 (the fixed arm gained no seeds, so the pairs are literally the same three). The unpaired n = 8 view softens the penalty to **+0.2837 PPL** with a std of 0.3881, i.e. the penalty is smaller than one seed-to-seed standard deviation of the gated arm and remains not significant by any available test. Meanwhile the safety side strengthens: **0 / 8 degraded gated runs**, 0 spikes and 0 non-finite losses across all eight seeds, max-excess 1.3584 +- 0.0186, worst-window max 43.32 against the fixed baseline's 42.99 -- versus 2/3 degraded ungated runs with a worst-window of 4.2e5. And the timing answer is now certain for the new seeds: **exactly 1 opening, at step 1, in every one of the five**, well before step 20 and 999 steps before the first domain boundary.

---

## 3. E3 traced re-run (`results/e3_traced/`, 80 runs, lr0 = 0.05, seeds 0-9)

{`cohg`, `cohg_nogate`, `hd`, `fixed`} x {ewc0 = 10, ewc0 = 1000} x seeds 0-9, all launched with `--log-losses` in the retained-holdout condition (E3 default), so the per-step loss trace that `results/e3` never stored is now available and spikes / max-excess / worst-window become computable for E3 for the first time. Hyperparameters were verified identical to the canonical set seed by seed: `meta_lr` (fixed 0.1, hd 0.02, cohg 0.4, cohg_nogate 0.4), `rank = 4`, `K = 10`, `gamma = 0.9`, `lr0 = 0.05`, matching `ewc0`. Traces are 3040 steps (10 tasks x 304 steps).

E3's recovery rule differs from E4's and matters for the censored analysis: on a non-finite loss the driver does **not** restore a checkpoint. It halves the LR scale *and* the EWC strength (`lam <- clamp(lam - log 2)`) and resets the estimator state, then continues.

### 3.1 (a) Verification of the traced re-run against the untraced canonical `results/e3`

| arm | ewc | n matched | max abs delta avg_acc | mean abs delta avg_acc | max abs delta BWT | seeds with delta events != 0 | **flag: seeds with \|delta acc\| > 0.02** |
|---|---|---|---|---|---|---|---|
| fixed | 10 | 10 | **0.0160** | 0.0090 | 0.0217 | none | none |
| hd | 10 | 10 | **0.0196** | 0.0089 | 0.0282 | none | none |
| **cohg (gated)** | 10 | 10 | **0.0408** | 0.0117 | 0.0369 | none | **2: s2, s5** |
| cohg_nogate | 10 | 10 | **0.0903** | 0.0392 | 0.0823 | s2, s6 | **8: s0, s1, s3, s4, s5, s6, s7, s8** |
| fixed | 1000 | 10 | **0.2508** | 0.0767 | 0.2056 | s0, s1, s3, s5, s6, s8 | **6: s0, s1, s2, s3, s5, s8** |
| hd | 1000 | 10 | **0.0289** | 0.0085 | 0.0317 | none | **1: s9** |
| **cohg (gated)** | 1000 | 10 | **0.2962** | 0.0438 | 0.2318 | s0, s1, s2, s3, s5, s6, s9 | **4: s0, s3, s8, s9** |
| cohg_nogate | 1000 | 10 | **0.1818** | 0.0613 | 0.0817 | s0, s4, s5, s9 | **8: s0, s1, s2, s3, s4, s5, s8, s9** |

**This is the headline verification result and it is not a clean pass.** The four *stable* configurations reproduce well: `fixed`/`hd`/`cohg` at ewc = 10 and `hd` at ewc = 1000 agree to max |delta acc| = 0.016 / 0.020 / 0.041 / 0.029 with mean |delta| <= 0.012, and 0 event-count differences. The four *unstable* configurations do not: `fixed` @ ewc1000 differs by up to **0.251** accuracy, `cohg` @ ewc1000 by up to **0.296**, `cohg_nogate` by up to 0.090 (ewc10) and 0.182 (ewc1000), with event counts swinging by hundreds in both directions.

Per-seed rows where |delta avg_acc| > 0.02:

| arm | ewc | seed | canonical avg_acc | traced avg_acc | delta | events can / tr |
|---|---|---|---|---|---|---|
| **cohg (gated)** | 10 | 2 | 0.3849 | 0.3441 | **-0.0408** | 0 / 0 |
| **cohg (gated)** | 10 | 5 | 0.3939 | 0.4157 | **+0.0218** | 0 / 0 |
| cohg_nogate | 10 | 0 | 0.3103 | 0.2873 | **-0.0230** | 0 / 0 |
| cohg_nogate | 10 | 1 | 0.1883 | 0.2283 | **+0.0400** | 0 / 0 |
| cohg_nogate | 10 | 3 | 0.2087 | 0.1184 | **-0.0903** | 0 / 0 |
| cohg_nogate | 10 | 4 | 0.2387 | 0.2134 | **-0.0253** | 0 / 0 |
| cohg_nogate | 10 | 5 | 0.1584 | 0.2150 | **+0.0566** | 0 / 0 |
| cohg_nogate | 10 | 6 | 0.1764 | 0.2600 | **+0.0836** | 163 / 0 |
| cohg_nogate | 10 | 7 | 0.2637 | 0.2877 | **+0.0240** | 0 / 0 |
| cohg_nogate | 10 | 8 | 0.2469 | 0.2871 | **+0.0402** | 0 / 0 |
| fixed | 1000 | 0 | 0.1000 | 0.3508 | **+0.2508** | 854 / 0 |
| fixed | 1000 | 1 | 0.3700 | 0.3975 | **+0.0275** | 72 / 0 |
| fixed | 1000 | 2 | 0.3468 | 0.3965 | **+0.0497** | 0 / 0 |
| fixed | 1000 | 3 | 0.1000 | 0.2270 | **+0.1270** | 170 / 2 |
| fixed | 1000 | 5 | 0.1000 | 0.3472 | **+0.2472** | 576 / 6 |
| fixed | 1000 | 8 | 0.3273 | 0.3770 | **+0.0497** | 1 / 2 |
| hd | 1000 | 9 | 0.4020 | 0.3731 | **-0.0289** | 0 / 0 |
| **cohg (gated)** | 1000 | 0 | 0.3402 | 0.3190 | **-0.0212** | 0 / 2 |
| **cohg (gated)** | 1000 | 3 | 0.1000 | 0.3962 | **+0.2962** | 401 / 0 |
| **cohg (gated)** | 1000 | 8 | 0.3030 | 0.3391 | **+0.0361** | 2 / 2 |
| **cohg (gated)** | 1000 | 9 | 0.3744 | 0.4042 | **+0.0298** | 3 / 2 |
| cohg_nogate | 1000 | 0 | 0.2818 | 0.1000 | **-0.1818** | 0 / 190 |
| cohg_nogate | 1000 | 1 | 0.2040 | 0.2491 | **+0.0451** | 0 / 0 |
| cohg_nogate | 1000 | 2 | 0.3008 | 0.2613 | **-0.0395** | 0 / 0 |
| cohg_nogate | 1000 | 3 | 0.2058 | 0.2467 | **+0.0409** | 0 / 0 |
| cohg_nogate | 1000 | 4 | 0.2271 | 0.1002 | **-0.1269** | 0 / 210 |
| cohg_nogate | 1000 | 5 | 0.1925 | 0.1080 | **-0.0845** | 0 / 56 |
| cohg_nogate | 1000 | 8 | 0.2327 | 0.2008 | **-0.0319** | 0 / 0 |
| cohg_nogate | 1000 | 9 | 0.2620 | 0.2228 | **-0.0392** | 2 / 0 |

**Diagnosis.** What is verifiable from the artifacts: every stored hyperparameter matches seed by seed (`method`, `seed`, `lr0`, `ewc0`, `meta_lr`, `rank`, `K`, `gamma`), `hvp_total` matches exactly (so the two sets do the same amount of work), and the flag sets differ only by `--log-losses`, which is write-only. The execution environments were nevertheless not identical: mean wall-clock per arm differs by 1.14x-7.05x (traced/canonical), largest on `hd` @ ewc10 (791 s -> 5574 s), consistent with the traced phase running two E3 jobs per card (`SLOTS_PER_GPU = 2` in `launch_r3_chain.py`) rather than with extra computation. No device identifier is stored in either set of JSONs and no run log survives for the canonical set, so the hardware attribution cannot be closed from the artifacts; what can be said is that the two runs used different execution conditions and therefore different floating-point reduction orders. The mechanism is the one already quantified in `results/reanalysis/device_sensitivity.md`: a float-reassociation perturbation of order 1e-7 leaves stable trajectories intact (there, |dNMSE|/NMSE ~ 1e-7 on 5 of 10 seeds and identical event counts on 10/10) but is amplified without bound once a run enters the divergent regime -- the same study shows a `cohg_nogate` seed moving from NMSE 59.8 to 0.0048 and an event count from 298 to 53 under nothing but a device change on identical code and seed. E3 @ ewc1000 is exactly that divergent regime.

**Consequence, which must be stated wherever E3 numbers are used.** At ewc = 10 the two sets are interchangeable and the traced set can be quoted as canonical. At ewc = 1000 they are **not** interchangeable: the outcome of an individual seed is device-determined, and any ewc1000 claim must be phrased over the distribution (e.g. "k of 10 seeds blow up"), never over a specific seed's accuracy. Notably the traced set is the *more favourable* of the two for `fixed` (+0.075 mean acc, 3 canonical collapses at 0.1000 disappear) and for `cohg` (+0.037, the canonical s3 collapse disappears), and the *less* favourable for `cohg_nogate` (-0.044).

Arm-level means, canonical vs traced:

| arm | ewc | canonical avg_acc (n=10) | traced avg_acc (n=10) | delta means | canonical BWT | traced BWT | delta means | canonical collapses | traced collapses |
|---|---|---|---|---|---|---|---|---|---|
| fixed | 10 | 0.3787 +- 0.0191 | 0.3815 +- 0.0211 | **+0.0028** | -0.0733 +- 0.0225 | -0.0688 +- 0.0242 | **+0.0044** | 0/10 | 0/10 |
| hd | 10 | 0.3660 +- 0.0227 | 0.3672 +- 0.0226 | **+0.0012** | -0.1220 +- 0.0208 | -0.1185 +- 0.0209 | **+0.0034** | 0/10 | 0/10 |
| **cohg (gated)** | 10 | 0.3803 +- 0.0179 | 0.3754 +- 0.0268 | **-0.0049** | -0.0711 +- 0.0165 | -0.0790 +- 0.0288 | **-0.0079** | 0/10 | 0/10 |
| cohg_nogate | 10 | 0.2167 +- 0.0627 | 0.2264 +- 0.0685 | **+0.0097** | -0.0954 +- 0.0411 | -0.0763 +- 0.0200 | **+0.0192** | 1/10 | 2/10 |
| fixed | 1000 | 0.2900 +- 0.1329 | 0.3652 +- 0.0525 | **+0.0752** | -0.1233 +- 0.1114 | -0.0714 +- 0.0434 | **+0.0519** | 3/10 | 0/10 |
| hd | 1000 | 0.3805 +- 0.0219 | 0.3796 +- 0.0221 | **-0.0009** | -0.0950 +- 0.0171 | -0.0967 +- 0.0170 | **-0.0017** | 0/10 | 0/10 |
| **cohg (gated)** | 1000 | 0.3124 +- 0.1150 | 0.3497 +- 0.0929 | **+0.0374** | -0.1148 +- 0.0895 | -0.0852 +- 0.0773 | **+0.0296** | 2/10 | 1/10 |
| cohg_nogate | 1000 | 0.2489 +- 0.0413 | 0.2048 +- 0.0743 | **-0.0441** | -0.0724 +- 0.0170 | -0.0896 +- 0.0439 | **-0.0172** | 0/10 | 3/10 |

### 3.2 (b) Unified trace-level degradation metrics, per arm

Now computable from the stored `losses`. Worst-window is in raw cross-entropy. Losses in the divergent arms reach ~1e38 (float32 overflow threshold) while still being *finite*, so arithmetic means of max-excess and worst-window are dominated by one or two seeds and are reported only for completeness; **the median, the maximum, and the blow-up count are the readable summaries.** A run is called a *blow-up* here if its maximum finite loss exceeds 100 (ambient cross-entropy on this task is ~1.7-2.1).

| arm | ewc | n | spikes mean +- std | non-finite mean +- std | unified events | blow-ups (max finite loss > 100) | max-excess median | max-excess max (seed) | worst-window median | worst-window max (seed) | median loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed | 10 | 10 | 0.00 +- 0.00 | 0.00 +- 0.00 | 0.00 +- 0.00 | **0 / 10** | 6.867 | 10.462 (s3) | 2.887 | 3.106 (s3) | 1.708 +- 0.022 |
| hd | 10 | 10 | 0.00 +- 0.00 | 0.00 +- 0.00 | 0.00 +- 0.00 | **0 / 10** | 6.773 | 9.814 (s3) | 2.737 | 2.897 (s3) | 1.641 +- 0.023 |
| **cohg (gated)** | 10 | 10 | 0.00 +- 0.00 | 0.00 +- 0.00 | 0.00 +- 0.00 | **0 / 10** | 5.722 | 7.413 (s8) | 2.688 | 2.916 (s3) | 1.703 +- 0.025 |
| cohg_nogate | 10 | 10 | 68.90 +- 104.97 | 31.70 +- 100.24 | 100.60 +- 147.58 | **6 / 10** | 8.52e+08 | 2.22e+37 (s2) | 3.23e+07 | **inf** (s2) | 2.067 +- 0.123 |
| fixed | 1000 | 10 | 108.00 +- 139.45 | 1.20 +- 1.93 | 109.20 +- 141.02 | **4 / 10** | 7.507 | 1.37e+38 (s6) | 2.990 | 1.79e+38 (s5) | 1.786 +- 0.065 |
| hd | 1000 | 10 | 27.50 +- 86.96 | 0.00 +- 0.00 | 27.50 +- 86.96 | **1 / 10** | 6.675 | 3.06e+32 (s8) | 2.752 | 4.10e+31 (s8) | 1.676 +- 0.024 |
| **cohg (gated)** | 1000 | 10 | 196.60 +- 186.65 | 1.90 +- 2.69 | 198.50 +- 188.72 | **6 / 10** | 6.53e+37 | 1.81e+38 (s8) | 2.54e+36 | 3.18e+38 (s5) | 1.793 +- 0.093 |
| cohg_nogate | 1000 | 10 | 31.80 +- 56.56 | 45.60 +- 83.36 | 77.40 +- 104.97 | **8 / 10** | 2.89e+05 | 1.00e+37 (s5) | 10221.430 | **inf** (s0) | 2.080 +- 0.110 |

`worst-window = inf` arises when some 100-step window contains **no** finite loss at all. The finite-restricted variant (max over windows holding at least one finite entry) and the count of all-non-finite windows:

| arm | ewc | worst-window (strict) median / max | worst-window (finite-restricted) median / max | all-non-finite windows, total over 10 seeds |
|---|---|---|---|---|
| fixed | 10 | 2.887 / 3.106 | 2.887 / 3.106 | 0 |
| hd | 10 | 2.737 / 2.897 | 2.737 / 2.897 | 0 |
| **cohg (gated)** | 10 | 2.688 / 2.916 | 2.688 / 2.916 | 0 |
| cohg_nogate | 10 | 3.23e+07 / **inf** | 3.23e+07 / 5.11e+37 | 119 |
| fixed | 1000 | 2.990 / 1.79e+38 | 2.990 / 1.79e+38 | 0 |
| hd | 1000 | 2.752 / 4.10e+31 | 2.752 / 4.10e+31 | 0 |
| **cohg (gated)** | 1000 | 2.54e+36 / 3.18e+38 | 2.54e+36 / 3.18e+38 | 0 |
| cohg_nogate | 1000 | 10221.430 / **inf** | 10221.430 / 6.94e+36 | 202 |

The clean, comparable view -- restricted to the arms/seeds that never blow up, plus the blow-up counts:

| arm | ewc | non-blow-up seeds | max-excess over those (mean +- std) | worst-window over those (mean +- std) | blow-up seeds |
|---|---|---|---|---|---|
| fixed | 10 | 10 / 10 | 7.189 +- 1.402 | 2.904 +- 0.105 | - |
| hd | 10 | 10 / 10 | 6.965 +- 1.261 | 2.748 +- 0.071 | - |
| **cohg (gated)** | 10 | 10 / 10 | 5.921 +- 0.902 | 2.716 +- 0.080 | - |
| cohg_nogate | 10 | 4 / 10 | 4.909 +- 0.745 | 2.455 +- 0.034 | s1, s2, s3, s4, s5, s8 |
| fixed | 1000 | 6 / 10 | 6.498 +- 0.973 | 2.873 +- 0.099 | s3, s5, s6, s8 |
| hd | 1000 | 9 / 10 | 6.757 +- 1.266 | 2.750 +- 0.073 | s8 |
| **cohg (gated)** | 1000 | 4 / 10 | 5.493 +- 0.988 | 2.744 +- 0.123 | s0, s1, s5, s6, s8, s9 |
| cohg_nogate | 1000 | 2 / 10 | 23.920 +- 28.349 | 2.904 +- 0.571 | s0, s2, s3, s4, s5, s7, s8, s9 |

Per-seed detail, ewc = 10 (max-excess / worst-window / spikes / non-finite):

| arm | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed | 6.78<br>2.81<br>0 / 0 | 8.55<br>2.99<br>0 / 0 | 5.99<br>2.75<br>0 / 0 | 10.46<br>3.11<br>0 / 0 | 6.96<br>2.99<br>0 / 0 | 7.17<br>2.90<br>0 / 0 | 6.61<br>2.95<br>0 / 0 | 6.35<br>2.88<br>0 / 0 | 7.38<br>2.82<br>0 / 0 | 5.65<br>2.84<br>0 / 0 |
| hd | 7.01<br>2.67<br>0 / 0 | 7.86<br>2.81<br>0 / 0 | 6.11<br>2.69<br>0 / 0 | 9.81<br>2.90<br>0 / 0 | 6.54<br>2.78<br>0 / 0 | 7.13<br>2.78<br>0 / 0 | 5.87<br>2.75<br>0 / 0 | 5.94<br>2.72<br>0 / 0 | 7.70<br>2.67<br>0 / 0 | 5.68<br>2.71<br>0 / 0 |
| **cohg (gated)** | 6.76<br>2.63<br>0 / 0 | 6.14<br>2.77<br>0 / 0 | 5.10<br>2.66<br>0 / 0 | 7.05<br>2.92<br>0 / 0 | 5.59<br>2.73<br>0 / 0 | 5.86<br>2.69<br>0 / 0 | 5.09<br>2.73<br>0 / 0 | 4.75<br>2.67<br>0 / 0 | 7.41<br>2.69<br>0 / 0 | 5.46<br>2.68<br>0 / 0 |
| cohg_nogate | 5.82<br>2.43<br>0 / 0 | 4.49e+13<br>2.63e+12<br>39 / 0 | 2.22e+37<br>**inf**<br>79 / 317 | 3.46e+18<br>1.57e+17<br>263 / 0 | 1.70e+09<br>6.45e+07<br>39 / 0 | 1.27e+30<br>4.31e+28<br>261 / 0 | 5.04<br>2.50<br>0 / 0 | 4.01<br>2.42<br>0 / 0 | 3296.31<br>205.84<br>8 / 0 | 4.77<br>2.46<br>0 / 0 |

Per-seed detail, ewc = 1000 (max-excess / worst-window / spikes / non-finite):

| arm | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed | 6.56<br>2.82<br>0 / 0 | 8.23<br>2.99<br>0 / 0 | 5.76<br>2.75<br>0 / 0 | 9.17e+37<br>3.96e+36<br>265 / 2 | 6.79<br>2.99<br>0 / 0 | 1.30e+38<br>1.79e+38<br>275 / 6 | 1.37e+38<br>6.76e+36<br>271 / 2 | 6.16<br>2.87<br>0 / 0 | 1.26e+38<br>6.00e+36<br>269 / 2 | 5.50<br>2.82<br>0 / 0 |
| hd | 6.90<br>2.67<br>0 / 0 | 7.71<br>2.81<br>0 / 0 | 6.08<br>2.66<br>0 / 0 | 9.59<br>2.89<br>0 / 0 | 6.45<br>2.78<br>0 / 0 | 6.92<br>2.75<br>0 / 0 | 5.76<br>2.75<br>0 / 0 | 5.86<br>2.72<br>0 / 0 | 3.06e+32<br>4.10e+31<br>275 / 0 | 5.55<br>2.71<br>0 / 0 |
| **cohg (gated)** | 6.00e+37<br>2.04e+36<br>424 / 2 | 7.06e+37<br>3.04e+36<br>289 / 2 | 5.03<br>2.65<br>0 / 0 | 6.87<br>2.92<br>0 / 0 | 5.48<br>2.74<br>0 / 0 | 1.60e+38<br>3.18e+38<br>465 / 9 | 1.33e+38<br>7.43e+36<br>335 / 2 | 4.59<br>2.67<br>0 / 0 | 1.81e+38<br>3.99e+36<br>173 / 2 | 1.20e+38<br>5.95e+36<br>280 / 2 |
| cohg_nogate | 3.37e+35<br>**inf**<br>20 / 190 | 43.97<br>3.31<br>2 / 0 | 85.09<br>4.98<br>3 / 0 | 472.62<br>36.73<br>10 / 0 | 3.34e+36<br>**inf**<br>20 / 210 | 1.00e+37<br>5.21e+35<br>186 / 56 | 3.87<br>2.50<br>0 / 0 | 66.01<br>4.50<br>3 / 0 | 6.64e+05<br>20406.13<br>19 / 0 | 5.77e+05<br>21608.87<br>55 / 0 |

### 3.3 (c) Censored analysis: time to first non-finite trigger

A *trigger* is the first non-finite loss, i.e. exactly the step at which the E3 driver halves lambda and the EWC strength. Metrics are computed on the trace strictly before that step; never-triggering runs are right-censored at the full 3040-step horizon.

| arm | ewc | frac triggering | mean steps to 1st trigger (triggering seeds) | min / max | mean steps survived (all, censored) | frac of horizon survived |
|---|---|---|---|---|---|---|
| fixed | 10 | **0.00** (0/10) | - | - | 3040.0 +- 0.0 | 1.0000 |
| hd | 10 | **0.00** (0/10) | - | - | 3040.0 +- 0.0 | 1.0000 |
| **cohg (gated)** | 10 | **0.00** (0/10) | - | - | 3040.0 +- 0.0 | 1.0000 |
| cohg_nogate | 10 | **0.10** (1/10) | 733.0 | 733 / 733 | 2809.3 +- 729.5 | 0.9241 |
| fixed | 1000 | **0.40** (4/10) | 1793.0 +- 850.3 | 791 / 2532 | 2541.2 +- 809.7 | 0.8359 |
| hd | 1000 | **0.00** (0/10) | - | - | 3040.0 +- 0.0 | 1.0000 |
| **cohg (gated)** | 1000 | **0.60** (6/10) | 1817.0 +- 875.9 | 653 / 2932 | 2306.2 +- 908.3 | 0.7586 |
| cohg_nogate | 1000 | **0.30** (3/10) | 1773.3 +- 693.6 | 1160 / 2526 | 2660.0 +- 693.7 | 0.8750 |

Per-seed first-trigger step (triggering seeds only; task boundaries every 304 steps):

- `cohg_nogate` @ ewc 10: s2@t=733 (task 3)
- `fixed` @ ewc 1000: s3@t=2532 (task 9), s5@t=2465 (task 9), s6@t=791 (task 3), s8@t=1384 (task 5)
- `cohg` @ ewc 1000: s0@t=2193 (task 8), s1@t=1000 (task 4), s5@t=2455 (task 9), s6@t=2932 (task 10), s8@t=653 (task 3), s9@t=1669 (task 6)
- `cohg_nogate` @ ewc 1000: s0@t=1634 (task 6), s4@t=2526 (task 9), s5@t=1160 (task 4)

Performance conditional on the trigger status:

| arm | ewc | avg_acc, all 10 | avg_acc, **never-triggering** | avg_acc, triggering | pre-trigger mean loss (median over seeds) | pre-trigger max-excess (median) | pre-trigger worst-window (median) | pre-trigger blow-ups |
|---|---|---|---|---|---|---|---|---|
| fixed | 10 | 0.3815 +- 0.0211 | 0.3815 +- 0.0211 | - | 1.741 | 6.867 | 2.887 | 0 / 10 |
| hd | 10 | 0.3672 +- 0.0226 | 0.3672 +- 0.0226 | - | 1.673 | 6.773 | 2.737 | 0 / 10 |
| **cohg (gated)** | 10 | 0.3754 +- 0.0268 | 0.3754 +- 0.0268 | - | 1.727 | 5.722 | 2.688 | 0 / 10 |
| cohg_nogate | 10 | 0.2264 +- 0.0685 | 0.2406 +- 0.0550 | 0.0990 | 1.06e+06 | 8.52e+08 | 3.23e+07 | 6 / 10 |
| fixed | 1000 | 0.3652 +- 0.0525 | 0.3823 +- 0.0185 | 0.3396 +- 0.0789 | 1.814 | 7.507 | 2.990 | 4 / 10 |
| hd | 1000 | 0.3796 +- 0.0221 | 0.3796 +- 0.0221 | - | 1.710 | 6.675 | 2.752 | 1 / 10 |
| **cohg (gated)** | 1000 | 0.3497 +- 0.0929 | 0.3872 +- 0.0209 | 0.3248 +- 0.1158 | 5.20e+34 | 5.50e+37 | 1.19e+36 | 6 / 10 |
| cohg_nogate | 1000 | 0.2048 +- 0.0743 | 0.2485 +- 0.0290 | 0.1027 +- 0.0046 | 338.223 | 2.89e+05 | 10221.430 | 8 / 10 |

Read-out of the censored block:

* At **ewc = 10** three of four arms never trigger at all (`fixed`, `hd`, `cohg`: 0/10 each, full 3040-step survival, zero spikes, zero non-finite). Only `cohg_nogate` triggers (1/10, s2 @ t = 733), and its damage is far larger than one trigger suggests: **6** of its 10 seeds blow up with finite losses up to 5.1e37, and its pre-trigger statistics are already destroyed. The certificate gate is the difference between an arm that never leaves the safe regime and one that leaves it in 6 of 10 seeds.
* At **ewc = 1000** the ordering inverts on this metric. `hd` triggers **0/10** and blows up in only **1/10** seeds (s8). `fixed` triggers 4/10 and blows up 4/10. `cohg_nogate` triggers only 3/10 but blows up **8/10** -- the two counts come apart because most of its damage is finite-but-enormous loss rather than overflow to `inf`. **`cohg` (gated) triggers 6/10 -- the worst of the four -- and blows up 6/10**, with first triggers spread from t = 653 to t = 2932 (mean 1817 +- 876), i.e. the gate does not delay the onset either.
* Conditioning on survival is informative: every arm's never-triggering seeds are healthy and near-identical (`fixed` 0.3823 +- 0.0185, `hd` 0.3796 +- 0.0221, `cohg` 0.3872 +- 0.0209), so the ewc1000 accuracy spread is entirely a composition effect -- *which* seeds fell over, not how well the survivors did. `cohg`'s 4 surviving seeds are the best-performing survivor set of the four arms; its problem at ewc1000 is exclusively that 6 of 10 seeds enter the divergent regime.

### 3.4 Accuracy / BWT / gate activity beside the trace metrics

| arm | ewc | avg_acc | BWT | collapse rate (avg_acc<0.15) | `gate_open_frac` | `coord_open_frac` | blow-ups | non-finite triggers | spikes (median) |
|---|---|---|---|---|---|---|---|---|---|
| fixed | 10 | 0.3815 +- 0.0211 | -0.0688 +- 0.0242 | 0.00 | - | - | 0/10 | 0/10 | 0.0 |
| hd | 10 | 0.3672 +- 0.0226 | -0.1185 +- 0.0209 | 0.00 | - | - | 0/10 | 0/10 | 0.0 |
| **cohg (gated)** | 10 | 0.3754 +- 0.0268 | -0.0790 +- 0.0288 | 0.00 | 0.00066 +- 0.00000 | 0.00033 +- 0.00000 | 0/10 | 0/10 | 0.0 |
| cohg_nogate | 10 | 0.2264 +- 0.0685 | -0.0763 +- 0.0200 | 0.20 | 1.00000 +- 0.00000 | 0.00000 +- 0.00000 | 6/10 | 1/10 | 23.5 |
| fixed | 1000 | 0.3652 +- 0.0525 | -0.0714 +- 0.0434 | 0.00 | - | - | 4/10 | 4/10 | 0.0 |
| hd | 1000 | 0.3796 +- 0.0221 | -0.0967 +- 0.0170 | 0.00 | - | - | 1/10 | 0/10 | 0.0 |
| **cohg (gated)** | 1000 | 0.3497 +- 0.0929 | -0.0852 +- 0.0773 | 0.10 | 0.00079 +- 0.00017 | 0.00046 +- 0.00017 | 6/10 | 6/10 | 226.5 |
| cohg_nogate | 1000 | 0.2048 +- 0.0743 | -0.0896 +- 0.0439 | 0.30 | 1.00000 +- 0.00000 | 0.00000 +- 0.00000 | 8/10 | 3/10 | 14.5 |

(`coord_open_frac = 0` for `cohg_nogate` means the certified controller is never consulted, not that zero coordinates were certified.)

### 3.5 Paired sign-flip tests on the traced runs (common seeds 0-9, n = 10)

Accuracy and BWT are tested directly. max-excess and worst-window span 38 orders of magnitude, so they are tested on **log10** (a monotone transform: the sign pattern, and hence the exact sign-flip test's evidence, is about the ordering, and log10 keeps the mean statistic from being decided by a single 1e38 seed). Seeds with `worst-window = inf` are excluded from the worst-window test only; the excluded count is shown.

| ewc | contrast | mean d avg_acc | p | mean d BWT | p | mean d log10 max-excess | p | mean d log10 worst-window (n used) | p |
|---|---|---|---|---|---|---|---|---|---|
| 10 | `cohg` - `fixed` | **-0.0061** | 0.3906 | -0.0101 | 0.08984 | **-0.0821** | 0.005859 | **-0.0290** (n=10) | 0.001953 |
| 10 | `cohg` - `hd` | **+0.0082** | 0.4395 | +0.0395 | 0.009766 | **-0.0691** | 0.001953 | **-0.0051** (n=10) | 0.01367 |
| 10 | `cohg` - `cohg_nogate` | **+0.1489** | 0.001953 | -0.0027 | 0.8516 | **-10.7462** | 0.03125 | **-7.3356** (n=9) | 0.0625 |
| 10 | `hd` - `fixed` | **-0.0143** | 0.04883 | -0.0497 | 0.001953 | **-0.0130** | 0.125 | **-0.0238** (n=10) | 0.001953 |
| 10 | `fixed` - `cohg_nogate` | **+0.1551** | 0.001953 | +0.0075 | 0.5234 | **-10.6641** | 0.03125 | **-7.3051** (n=9) | 0.0625 |
| 1000 | `cohg` - `fixed` | **-0.0155** | 0.5664 | -0.0138 | 0.5977 | **+7.4060** | 0.4434 | **+7.2085** (n=10) | 0.4141 |
| 1000 | `cohg` - `hd` | **-0.0299** | 0.4355 | +0.0115 | 0.6602 | **+19.1329** | 0.03125 | **+18.7728** (n=10) | 0.02734 |
| 1000 | `cohg` - `cohg_nogate` | **+0.1450** | 0.003906 | +0.0043 | 0.9023 | **+10.1938** | 0.2715 | **+17.2941** (n=8) | 0.0625 |
| 1000 | `hd` - `fixed` | **+0.0144** | 0.4668 | -0.0253 | 0.1074 | **-11.7269** | 0.04688 | **-11.5643** (n=10) | 0.001953 |
| 1000 | `fixed` - `cohg_nogate` | **+0.1604** | 0.003906 | +0.0182 | 0.4219 | **+2.7878** | 0.8965 | **+12.7610** (n=8) | 0.1875 |

### 3.6 (d) Plain answer: how do the four arms compare under trace-level metrics?

**The two EWC regimes give opposite verdicts, and both earlier claims survive -- one of them strengthened, the other narrowed.**

**At ewc = 10, the gate separation is real and the trace metrics make it stronger than the accuracy numbers did.** All of `fixed`, `hd` and `cohg` are perfectly clean: 0 spikes, 0 non-finite losses, 0 blow-ups, full 3040-step survival, 0/10 each. Their accuracies are statistically indistinguishable (`cohg` - `fixed` = -0.006, p = 0.39; `cohg` - `hd` = +0.008, p = 0.44). But on the degradation metrics `cohg` is the *best* of the three, not merely equal: max-excess 5.92 +- 0.90 versus 7.19 +- 1.40 (fixed) and 6.97 +- 1.26 (hd), and worst-window 2.716 +- 0.080 versus 2.904 +- 0.105 and 2.748 +- 0.071. The paired tests confirm it: `cohg` - `fixed` on log10 max-excess is negative on 9/10 seeds (p = 0.0059) and on log10 worst-window on **10/10** (p = 0.00195); `cohg` - `hd` on log10 max-excess is negative on **10/10** (p = 0.00195) and on log10 worst-window on 8/10 (p = 0.014). Against the ungated ablation the separation is categorical: `cohg_nogate` blows up in **6 of 10** seeds (s1-s5, s8; max finite loss up to 5.1e37), records 68.9 +- 105.0 spikes per run against `cohg`'s exact zero, triggers a non-finite recovery in s2, and loses 0.149 accuracy on 10/10 seeds (p = 0.00195). **At ewc10 the certificate gate is what keeps a method that would otherwise diverge in 6 of 10 seeds at exactly zero degradation events -- and it does so while being marginally *safer* than the fixed baseline, which the accuracy-only view could not show.**

**At ewc = 1000, HD's dominance over COHG holds and is reinforced by the trace metrics.** On accuracy alone the gap is small and insignificant: `cohg` 0.3497 +- 0.0929 versus `hd` 0.3796 +- 0.0221, paired difference -0.030, p = 0.44. On stability it is not small. `hd` never produces a non-finite loss (0/10 triggers) and blows up in a single seed (s8); `cohg` triggers in **6/10** seeds and blows up in **6/10**, the worst trigger rate of any arm in either regime, worse even than the ungated ablation (3/10 triggers, though 8/10 blow-ups) and than `fixed` (4/10 and 4/10). Paired on log10 max-excess, `cohg` - `hd` is positive on 6/10 seeds with p = 0.031; on log10 worst-window, positive on 7/10 with p = 0.027. And the conditional analysis says the gate is not buying a softer failure either: `cohg`'s first triggers span t = 653-2932 with mean 1817 +- 876, no later than `fixed`'s 1793 +- 850. **So the honest ewc1000 statement is: the certified gate does not protect against the EWC-1000 failure mode at all, HD does, and COHG is if anything the most trigger-prone arm there.** The single mitigating fact is the conditional-on-survival result: `cohg`'s 4 surviving seeds average 0.3872 +- 0.0209, the best survivor set of the four arms -- the ewc1000 problem is composition (how many seeds diverge), not quality (how the survivors do).

**A finding that cuts across both claims and should be stated in the paper.** At ewc = 1000 the *fixed* baseline itself blows up in 4/10 seeds and triggers 4/10. The ewc1000 instability is therefore a property of the **operating point**, not of adaptive LR control: it is the EWC penalty at strength 1000 that makes the loss surface divergent, and `hd`'s small, AdaGrad-normalised meta-steps happen to stay out of the divergent basin while both `cohg` variants and the fixed LR walk into it. Framing ewc1000 as "COHG fails where HD succeeds" is only half true; "ewc1000 is a divergent operating point in which only HD's step size is small enough to stay out, and COHG's certificate offers no protection there" is the supported statement.

**Caveat that limits (b), (c) and (d).** All of section 3 is computed on the traced set, whose ewc1000 block is device-divergent from the canonical set (section 3.1). Seed-level ewc1000 statements are not portable between the two sets; the ewc10 block is. Because the *ranking* of the arms at ewc1000 is the same in both sets (`hd` best and clean, `cohg` and `cohg_nogate` unstable, `fixed` intermediate), the qualitative conclusion above is robust to the device, but the specific counts (6/10, 4/10, 3/10, 1/10) are device-conditional and should be quoted as "on the traced set".

