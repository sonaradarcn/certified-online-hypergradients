# E3 no-retained-holdout condition (Split-CIFAR-100)

Generated from `results/e3_noholdout/*.json` (40 runs: 4 arms x 10 seeds, flags
`--no-holdout --log-losses`). The reference holdout condition is `results/e3/*.json`,
same lr0=0.05, EWC0=10, seeds 0-9.

In the no-holdout condition the hypergradient meta-objective is evaluated only on the
incoming batch's prequential loss; no stored examples from past tasks are used at any
point of the meta-update. Everything else (backbone, task order, lr0, EWC strength,
meta-lr, gate, recovery heuristic) is unchanged. The 128-example holdout is still carved
out of each task's training stream, so the data the learner trains on is bit-identical to
the default condition and both conditions log the same 3040 steps; under `--no-holdout`
those examples are simply never looked at again. EWC keeps its anchor and Fisher, which
belong to the learner rather than to the meta-objective.

The non-adaptive `fixed` arm therefore acts as a null control: the flag cannot touch it,
and any holdout / no-holdout difference it shows is GPU nondeterminism. It shows
-0.0027 accuracy at p = 0.5332, which sets the noise floor for reading the other rows.

Spike rule, max-excess and worst-window follow `results/reanalysis/_reanalyze.py`:
a window of the last 500 finite losses; once it holds >=100 entries a step is a spike if
`loss_t > 10 x median(window)`; every non-finite loss is also an event;
`unified events = spikes + non-finite`. `max-excess = max finite loss / median finite loss`;
`worst-window = max over t of mean(loss[t-99..t])`. Aggregates are mean +- std, ddof=1,
over the 10 seeds.


## 0. Sanity checks

| arm | n runs | `no_holdout` true | meta_lr (all runs) | expected | steps logged |
|---|---|---|---|---|---|
| COHG (gate on) | 10 | yes | 0.4 | 0.4 | 3040 |
| COHG w/o gate | 10 | yes | 0.4 | 0.4 | 3040 |
| HD | 10 | yes | 0.02 | 0.02 | 3040 |
| Fixed | 10 | yes | 0.1 | n/a | 3040 |

All 40 runs carry `no_holdout=true`.
Seeds present: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] for every arm; the holdout arms use the same seed set, so every
holdout / no-holdout comparison below is paired by seed.


## 1. Per-arm summary, no-holdout condition (n = 10)

| arm | avg_acc | BWT | non-finite events | collapse rate (avg_acc<0.15) | unified spikes | unified events | max-excess | worst-window |
|---|---|---|---|---|---|---|---|---|
| COHG (gate on) | 0.3804 +- 0.0198 | -0.0730 +- 0.0246 | 0.00 +- 0.00 | 0.00 (0/10) | 0.00 +- 0.00 | 0.00 +- 0.00 | 5.91 +- 0.89 | 2.718 +- 0.082 |
| COHG w/o gate | 0.2926 +- 0.0170 | -0.1371 +- 0.0232 | 0.00 +- 0.00 | 0.00 (0/10) | 0.00 +- 0.00 | 0.00 +- 0.00 | 5.22 +- 0.83 | 2.490 +- 0.072 |
| HD | 0.3618 +- 0.0247 | -0.1243 +- 0.0229 | 0.00 +- 0.00 | 0.00 (0/10) | 0.00 +- 0.00 | 0.00 +- 0.00 | 6.95 +- 1.23 | 2.751 +- 0.070 |
| Fixed | 0.3760 +- 0.0220 | -0.0757 +- 0.0255 | 0.00 +- 0.00 | 0.00 (0/10) | 0.00 +- 0.00 | 0.00 +- 0.00 | 7.19 +- 1.42 | 2.906 +- 0.104 |

COHG gate_open_frac = 0.000658 +- 0.000000 (min 0.000658, max 0.000658); coord_open_frac = 0.000329 +- 0.000000.
COHG w/o gate gate_open_frac = 1.000000 +- 0.000000 (gate disabled; open at every step by construction).


### Per-seed detail (no-holdout)

| arm | seed | avg_acc | BWT | non-finite | spikes | max-excess | worst-window | gate_open_frac |
|---|---|---|---|---|---|---|---|---|
| COHG (gate on) | 0 | 0.3477 | -0.1194 | 0 | 0 | 6.81 | 2.625 | 0.000658 |
| COHG (gate on) | 1 | 0.3818 | -0.0769 | 0 | 0 | 6.13 | 2.765 | 0.000658 |
| COHG (gate on) | 2 | 0.3877 | -0.0812 | 0 | 0 | 5.21 | 2.662 | 0.000658 |
| COHG (gate on) | 3 | 0.3918 | -0.0537 | 0 | 0 | 7.01 | 2.919 | 0.000658 |
| COHG (gate on) | 4 | 0.3487 | -0.0823 | 0 | 0 | 5.58 | 2.738 | 0.000658 |
| COHG (gate on) | 5 | 0.4074 | -0.0637 | 0 | 0 | 5.81 | 2.681 | 0.000658 |
| COHG (gate on) | 6 | 0.3974 | -0.0266 | 0 | 0 | 5.08 | 2.732 | 0.000658 |
| COHG (gate on) | 7 | 0.3934 | -0.0851 | 0 | 0 | 4.76 | 2.670 | 0.000658 |
| COHG (gate on) | 8 | 0.3762 | -0.0837 | 0 | 0 | 7.38 | 2.701 | 0.000658 |
| COHG (gate on) | 9 | 0.3715 | -0.0572 | 0 | 0 | 5.37 | 2.686 | 0.000658 |
| COHG w/o gate | 0 | 0.2771 | -0.1224 | 0 | 0 | 6.27 | 2.405 | 1.000000 |
| COHG w/o gate | 1 | 0.3198 | -0.1178 | 0 | 0 | 4.92 | 2.622 | 1.000000 |
| COHG w/o gate | 2 | 0.2772 | -0.1633 | 0 | 0 | 4.89 | 2.448 | 1.000000 |
| COHG w/o gate | 3 | 0.2847 | -0.1442 | 0 | 0 | 5.32 | 2.581 | 1.000000 |
| COHG w/o gate | 4 | 0.2744 | -0.1833 | 0 | 0 | 5.64 | 2.453 | 1.000000 |
| COHG w/o gate | 5 | 0.2977 | -0.1041 | 0 | 0 | 4.36 | 2.467 | 1.000000 |
| COHG w/o gate | 6 | 0.3056 | -0.1260 | 0 | 0 | 4.20 | 2.522 | 1.000000 |
| COHG w/o gate | 7 | 0.3031 | -0.1393 | 0 | 0 | 4.42 | 2.410 | 1.000000 |
| COHG w/o gate | 8 | 0.2748 | -0.1267 | 0 | 0 | 6.69 | 2.537 | 1.000000 |
| COHG w/o gate | 9 | 0.3119 | -0.1437 | 0 | 0 | 5.52 | 2.459 | 1.000000 |
| HD | 0 | 0.3469 | -0.1326 | 0 | 0 | 6.96 | 2.702 | n/a |
| HD | 1 | 0.3916 | -0.0963 | 0 | 0 | 7.82 | 2.811 | n/a |
| HD | 2 | 0.3795 | -0.1316 | 0 | 0 | 6.18 | 2.663 | n/a |
| HD | 3 | 0.3464 | -0.1242 | 0 | 0 | 9.74 | 2.896 | n/a |
| HD | 4 | 0.3176 | -0.1579 | 0 | 0 | 6.53 | 2.785 | n/a |
| HD | 5 | 0.3862 | -0.1098 | 0 | 0 | 7.07 | 2.779 | n/a |
| HD | 6 | 0.3639 | -0.1112 | 0 | 0 | 5.86 | 2.751 | n/a |
| HD | 7 | 0.3358 | -0.1624 | 0 | 0 | 5.92 | 2.728 | n/a |
| HD | 8 | 0.3640 | -0.1218 | 0 | 0 | 7.69 | 2.671 | n/a |
| HD | 9 | 0.3859 | -0.0952 | 0 | 0 | 5.68 | 2.722 | n/a |
| Fixed | 0 | 0.3525 | -0.0953 | 0 | 0 | 6.75 | 2.808 | n/a |
| Fixed | 1 | 0.3845 | -0.0793 | 0 | 0 | 8.56 | 2.989 | n/a |
| Fixed | 2 | 0.3801 | -0.0801 | 0 | 0 | 5.93 | 2.751 | n/a |
| Fixed | 3 | 0.3789 | -0.0796 | 0 | 0 | 10.52 | 3.106 | n/a |
| Fixed | 4 | 0.3590 | -0.0783 | 0 | 0 | 7.02 | 2.992 | n/a |
| Fixed | 5 | 0.4118 | -0.0562 | 0 | 0 | 7.20 | 2.888 | n/a |
| Fixed | 6 | 0.4014 | -0.0220 | 0 | 0 | 6.62 | 2.956 | n/a |
| Fixed | 7 | 0.3680 | -0.0896 | 0 | 0 | 6.36 | 2.870 | n/a |
| Fixed | 8 | 0.3391 | -0.1167 | 0 | 0 | 7.30 | 2.857 | n/a |
| Fixed | 9 | 0.3847 | -0.0594 | 0 | 0 | 5.66 | 2.845 | n/a |

## 2. Paired comparisons within the no-holdout condition

Exact two-sided sign-flip test over all 2^10 = 1024 sign assignments of the 10 paired
per-seed differences.

| comparison | metric | mean paired difference | p (exact) | direction |
|---|---|---|---|---|
| COHG (gate on) vs COHG w/o gate | avg_acc | +0.0877 | 0.0020 | COHG higher |
| COHG (gate on) vs Fixed | avg_acc | +0.0044 | 0.4375 | COHG higher |
| COHG (gate on) vs HD | avg_acc | +0.0186 | 0.0410 | COHG higher |
| COHG (gate on) vs HD | bwt | +0.0513 | 0.0020 | COHG higher |


## 3. Holdout vs no-holdout, paired by seed

| arm | metric | holdout | no-holdout | mean paired delta (no-holdout - holdout) | p (exact sign-flip) |
|---|---|---|---|---|---|
| COHG (gate on) | avg_acc | 0.3803 +- 0.0179 | 0.3804 +- 0.0198 | +0.0001 | 0.9707 |
| COHG (gate on) | bwt | -0.0711 +- 0.0165 | -0.0730 +- 0.0246 | -0.0019 | 0.6328 |
| COHG (gate on) | events | 0.00 +- 0.00 | 0.00 +- 0.00 | +0.0000 | 1.0000 |
| COHG w/o gate | avg_acc | 0.2167 +- 0.0627 | 0.2926 +- 0.0170 | +0.0759 | 0.0059 |
| COHG w/o gate | bwt | -0.0954 +- 0.0411 | -0.1371 +- 0.0232 | -0.0416 | 0.0312 |
| COHG w/o gate | events | 31.60 +- 66.66 | 0.00 +- 0.00 | -31.6000 | 0.5000 |
| HD | avg_acc | 0.3660 +- 0.0227 | 0.3618 +- 0.0247 | -0.0042 | 0.1914 |
| HD | bwt | -0.1220 +- 0.0208 | -0.1243 +- 0.0229 | -0.0023 | 0.5215 |
| HD | events | 0.00 +- 0.00 | 0.00 +- 0.00 | +0.0000 | 1.0000 |
| Fixed | avg_acc | 0.3787 +- 0.0191 | 0.3760 +- 0.0220 | -0.0027 | 0.5332 |
| Fixed | bwt | -0.0733 +- 0.0225 | -0.0757 +- 0.0255 | -0.0024 | 0.6562 |
| Fixed | events | 0.00 +- 0.00 | 0.00 +- 0.00 | +0.0000 | 1.0000 |

The `events` row for COHG w/o gate is driven by two of the ten holdout seeds (s2 with 153
non-finite steps, s6 with 163); the other eight holdout seeds and all ten no-holdout seeds
record zero. With only two nonzero paired differences the sign-flip p for that row is
uninformative (0.5000 is its floor here); the seed-level count is the meaningful statistic:
2/10 holdout seeds trigger recovery versus 0/10 no-holdout seeds.

Collapse rates: COHG (gate on) 0/10 holdout vs 0/10 no-holdout; COHG w/o gate 1/10 holdout vs 0/10 no-holdout; HD 0/10 holdout vs 0/10 no-holdout; Fixed 0/10 holdout vs 0/10 no-holdout.
Holdout gate_open_frac (COHG) = 0.000658 +- 0.000000; no-holdout = 0.000658 +- 0.000000.


### The same paired comparisons under the holdout condition, for reference

| comparison | metric | holdout mean diff | holdout p | no-holdout mean diff | no-holdout p |
|---|---|---|---|---|---|
| COHG (gate on) vs COHG w/o gate | avg_acc | +0.1635 | 0.0020 | +0.0877 | 0.0020 |
| COHG (gate on) vs Fixed | avg_acc | +0.0016 | 0.7695 | +0.0044 | 0.4375 |
| COHG (gate on) vs HD | avg_acc | +0.0142 | 0.0918 | +0.0186 | 0.0410 |
| COHG (gate on) vs HD | bwt | +0.0509 | 0.0020 | +0.0513 | 0.0020 |


## 4. What survives without retained past-task data

**Survives.**

1. COHG's accuracy is unchanged. 0.3804 +- 0.0198 without a holdout versus
   0.3803 +- 0.0179 with one; the paired delta is +0.0001 (p = 0.9707). Its BWT is also
   unchanged, -0.0730 +- 0.0246 versus -0.0711 +- 0.0165 (delta -0.0019, p = 0.6328).
   The 128-example retained holdout buys COHG nothing on this benchmark.
2. The gate-on / gate-off separation survives. COHG beats its ungated ablation by
   +0.0877 accuracy (p = 0.0020) without a holdout, the same p as the holdout condition's
   +0.1635. The gate is still the component that matters.
3. COHG's forgetting advantage over HD survives and in fact sharpens: +0.0513 BWT
   (p = 0.0020) without a holdout versus +0.0242 (p = 0.0898) with one, at matched or
   better accuracy (+0.0186, p = 0.0410, versus +0.0142, p = 0.0918).
4. COHG's near-zero gate duty cycle is identical in both conditions, 0.000658 of steps.
   The meta-objective change does not make the certificate fire more often.

**Does not survive, or was never there.**

1. COHG's accuracy edge over a well-tuned fixed schedule is absent in both conditions
   (+0.0044, p = 0.4375 without a holdout; +0.0016, p = 0.7695 with one). Removing the
   holdout neither creates nor destroys an edge that the paper never claimed.
2. The *magnitude* of the gate ablation gap shrinks by roughly half, from +0.1635 to
   +0.0877, and its dramatic failure mode disappears. Under the holdout objective the
   ungated ablation triggered non-finite recovery on 2/10 seeds (31.60 +- 66.66 events)
   and collapsed below 0.15 accuracy on 1/10; under the prequential-only objective it
   triggers on 0/10 seeds, collapses on 0/10, and gains +0.0759 accuracy (p = 0.0059).
   So the strongest version of the instability claim, that an ungated hypergradient can
   diverge outright, is a property of the holdout meta-objective specifically, not of
   ungated hypergradients in general. The weaker claim, that gating improves accuracy
   and reduces forgetting, holds in both.
3. No arm produces a single spike or non-finite loss in the no-holdout condition
   (0.00 +- 0.00 unified events across all 40 runs; max-excess 5.2 to 7.2, worst-window
   2.49 to 2.91). The prequential-only meta-objective is a uniformly milder regime, so
   the stability metrics cannot separate the arms here at all. Removing the holdout
   costs COHG none of its stability, but it also removes the setting in which COHG's
   stability advantage over the ungated ablation was visible.
4. HD's behaviour is essentially unchanged. Accuracy -0.0042 (p = 0.1914), BWT -0.0023
   (p = 0.5215). HD's larger forgetting relative to COHG is not a holdout artefact.

