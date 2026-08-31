# Reproducibility protocol

This document gives the environments the reported runs used, the released
result trees, the exact command behind every experiment, the seed protocol, and
the statistical conventions the paper applies. It is the companion to
[`README.md`](README.md), which describes the method and the repository layout.

## Environments

**Software.** All experiments used PyTorch 2.12.1 (CUDA 12.6) under Python 3.11,
in single precision, with one exception: the quadratic and teacher-student
exact-reference audit runs of `experiments/e1_certificate.py` use fp64
throughout, because they compare the certificate against an exact discounted
sensitivity matrix and the comparison is only meaningful at that precision.

The torch version is the one figure the paper records, and
`requirements.txt` pins it. `torchvision` is pinned to the release built against
that torch, which the pairing determines. Nothing else was written into the run
artifacts, so `numpy`, `matplotlib`, `transformers`, `modelscope`, `pyarrow`,
`certifi` and `pypdf` carry tested floors at the versions present in the
environment that ran the campaigns rather than pins that were never recorded.
The environment carried numpy 2.4.6, matplotlib 3.11.0, transformers 5.12.1,
modelscope 1.38.0, pyarrow 24.0.0.

`certifi` and `pypdf` are optional: both are imported inside `try` blocks and
their absence only costs a CA bundle for the dataset downloads and a page-size
check on one figure. `modelscope`, `transformers` and `pyarrow` are required for
E4 alone. `torchvision` is required for E3 alone.

**Hardware.** The local machine holds two NVIDIA GeForce RTX 3080 20 GB cards.
Rented servers with NVIDIA TITAN Xp cards took parts of the time-series grids.
The GPT-2 runs used the local cards only. The gating-control study
(`results/e2_controls/`) is the one campaign that ran on CPU: the
Hessian-vector products of the 13k-parameter GRU are kernel-launch bound at that
size, so CPU is faster there, and the campaign carries its own COHG reference
arm on the same device so every comparison inside it is within-device. The
round-4 CPU campaigns (`e2_shift`, `e2_adaptmh`, `e2_warmup`, `e2_denseprobe`,
`e2_lorenz_absgate`, `e2_gamma`) follow the same rule.

Each (method, dataset, seed, configuration) combination was run independently
and wrote one JSON.

**Device sensitivity is real and is reported.** The same code, same seeds and
same configuration produce different loss traces on GPU and on CPU, because
float reassociation perturbs the trajectory at order 1e-7 relative.
`results/reanalysis/device_sensitivity.md` gives the paired GPU/CPU comparison
for the drifting Mackey-Glass reference arm: 0 of 10 seeds have a bitwise
identical loss trace, 4 of 10 differ in the realized per-coordinate gate-open
rate, and 0 of 10 change regime. The paper therefore quotes COHG NMSE to two
significant figures in any cross-device comparison, and keeps within-device
control grids at full precision.

## Released result trees

`results/` holds the raw per-run artifacts, byte for byte as the runs wrote
them. Nothing inside a JSON has been trimmed, so every reported mean, standard
deviation and permutation test can be recomputed from this tree alone, for the
trees that are here. `README.md`, section "Released results", lists what is
present, what each tree backs, and the four trees that are held back for size
along with what they back.

`.gitattributes` normalises tracked text to LF everywhere except `results/`,
which is exempt so that the records are stored exactly as written rather than
re-line-ended by git.

`results/*/SUMMARY*.md` are the per-campaign summaries written by the
`analyze_*.py` scripts, and `results/reanalysis/` holds the post-hoc analyses,
their CSV outputs and the scripts that wrote them.

## Exact commands

Every campaign was launched by a `launch_*.py` script in `experiments/`. Each
one is a plain local loop that builds a list of configurations and calls
`python <driver>.py …` through `subprocess`, skipping any configuration whose
output JSON already exists, so a campaign is restart-safe and can be split by
seed across workers. The launcher is therefore the exact command, and the driver
flags it passes are visible in its source and stored in the artifact it writes.

**The released `results/` tree is already populated.** A launcher will report
every configuration as already done and do nothing. Delete the target directory,
or point the driver at a fresh one with `--out-dir`, before re-running.

### Structural probe (Fig. 2)

```bash
for a in mlp gru resnet transformer; do
  python experiments/e0_structure.py --arch $a --seed 0
done
```

### Certificate audit, fp64 exact reference (Fig. 3, Fig. `misspec`)

```bash
python experiments/e1_certificate.py --problem quad    --tier prior
python experiments/e1_certificate.py --problem quad    --tier oracle
python experiments/e1_certificate.py --problem teacher --tier prior
python experiments/e1_certificate.py --problem teacher --tier kw --gamma 0.90
python experiments/e1_certificate.py --problem teacher --tier kw --gamma 0.95
python experiments/e1_certificate.py --problem teacher --tier kw --gamma 1.00
# drift-prior misspecification sweep and the fail-closed monitor
python experiments/analyze_misspec.py
```

### Online time series

```bash
python experiments/launch_e2.py                       # main grid -> results/e2
python experiments/launch_cal.py                      # meta-step calibration -> results/e2_cal
python experiments/launch_m4.py                       # design-axis ablations -> results/m4
python experiments/launch_e2_controls.py --part calib # threshold calibration
python experiments/launch_e2_controls.py --part main  # gating controls -> results/e2_controls
```

A single driver invocation, for reference, is the shape every launcher emits:

```bash
python experiments/e2_timeseries.py --method cohg --dataset mackey_drift \
    --seed 0 --steps 12000 --lr 0.003 --meta-lr 0.4 --gamma 0.9 \
    --K 10 --rank 4 --kw-eps 0.1 --probe-every 20 --M-H 5 --out-dir results/smoke
```

### Continual learning

```bash
python experiments/launch_e3_cal.py   # domain meta-step calibration -> results/e3_cal
python experiments/launch_e3.py       # main grid -> results/e3
```

### GPT-2 124M streaming adaptation

```bash
python experiments/e4_prepare_data.py   # builds data/e4_stream_{wiki,news,code}.pt
python experiments/launch_e4v2.py       # main grid -> results/e4_v2
python experiments/launch_e4_expand.py  # seed and domain-order expansion -> results/e4_orders
```

### Round-3 additions

```bash
python experiments/launch_round3.py --part verify   # bit-identity check -> results/e2_verify
python experiments/launch_round3.py --part b1
python experiments/launch_round3.py --part b2       # -> results/e2_controls
python experiments/launch_round3.py --part b3       # gamma sweep -> results/e2_gamma
python experiments/launch_r3_chain.py               # GPU chain -> e4_fix, e4_misset, e4_orders
python experiments/r3_cpu_queue.py                  # traced E3 -> results/e3_traced
python experiments/analyze_round3.py
python results/reanalysis/_round3_gpu.py
```

### Round-3.5 same-seed verification

```bash
python experiments/launch_e4_verify_all.py     # -> results/e4_verify_all
python experiments/watch_e4_verify_all.py      # queue poller
python experiments/compare_e4_verify_all.py    # -> results/e4_verify_all/COMPARE_ALL.md
python results/reanalysis/_common_stability.py
```

### Round-4 additions

```bash
python experiments/launch_round4.py --part verify   # -> results/e2_verify4, results/e1_verify4
python experiments/launch_round4.py --part shift    # late amplitude shift -> results/e2_shift
python experiments/launch_round4.py --part adaptmh  # -> results/e2_adaptmh, results/e1_adaptmh
python experiments/r4_cpu_queue.py
python experiments/launch_r4_warmup.py              # -> results/e2_warmup
python experiments/launch_r4_denseprobe.py          # -> results/e2_denseprobe
python experiments/launch_r4_e2cpu.py               # -> results/e2_lorenz_absgate
python experiments/launch_r4_absgate.py             # -> results/e4_absgate, results/e3_absgate
python experiments/analyze_r4_absgate.py
python results/reanalysis/_round4_shift.py
python results/reanalysis/_round4_adaptmh.py
python results/reanalysis/_round4_warmup.py
python results/reanalysis/_round4_denseprobe.py
```

### Analyses, tables and figures

```bash
python experiments/analyze_e1.py            # -> results/e1/SUMMARY.md
python experiments/analyze_e2.py            # -> results/e2/SUMMARY.md      (needs results/e2)
python experiments/analyze_e3.py            # -> results/e3/SUMMARY.md
python experiments/analyze_e2_controls.py   # -> results/e2_controls/SUMMARY.md
python experiments/analyze_cal.py           # frozen meta-step sizes
python results/reanalysis/_reanalyze.py     # unified metrics + censored analysis
python results/reanalysis/_key_questions.py
python results/reanalysis/_e3_noholdout.py
python experiments/analyze_device_sensitivity.py
python results/reanalysis/_fig8_gpt2_lambda.py
python results/reanalysis/_fig9_gpt2_order.py
python experiments/gen_appendix_tables.py   # appendix tables (needs results/e2)
python experiments/make_paper_figures_nc.py # the paper's data figures
python experiments/make_fig10_misspec.py
python experiments/make_pareto.py
```

Order matters: runs first, then per-campaign analyses, then the reanalyses,
then the appendix tables and figures. `make_paper_figures_nc.py` writes
print-size PDFs into `paper/main/figs/`, which belongs to the manuscript and is
not part of this repository, and PNG previews into `results/figures/`, which is.

## Seed protocol

Seeds are integers passed with `--seed` and control the stream construction,
the class order and stream reshuffle in continual learning, and the model
initialization. Each seed is one independent run and one JSON.

- **Evaluation seeds** are 0 to 9 for the time-series and continual-learning
  regimes. GPT-2 uses eight seeds for the fixed 1e-3 baseline, HD at alpha=2 and
  the gated and ungated COHG arms, and three for the remaining rows.
- **Calibration seeds** are 100 and 101, disjoint from every evaluation seed.
  Meta-step sizes were calibrated on those seeds under a zero-instability
  minimum-loss rule with grid extension at the boundaries, then frozen. COHG
  uses one value, alpha = 0.4, in all three regimes, fixed before the transfer
  experiments. The frozen values are in `results/e2_cal/FROZEN_META_LR.json` and
  `results/e3_cal/`.
- **The offline-calibrated absolute gate threshold** was fitted in two passes on
  calibration seeds and frozen at 0.0581 before any transfer run. The artifact
  is `results/e2_controls/absgate_threshold.json`.
- **The late-shift pilot** selected the two amplitude factors on seed 0. Seed 0
  is therefore excluded from that study's evaluation, which runs on seeds 1 to 9.

## Statistical conventions

- Aggregates are seed means with the sample standard deviation (`ddof=1`).
- Significance is the two-sided paired exact permutation test on the per-seed
  differences of the primary metric, enumerating all 2^n sign assignments. The
  smallest attainable p-value is therefore 2^(1-n): 0.002 at ten seeds, 0.0039
  at nine, 0.0078 at eight, 0.25 at three.
- An instability event on the time series is a training loss above ten times its
  running median over the last 500 steps, or a non-finite loss. In the
  continual-learning and GPT-2 regimes only non-finite losses count, so "no
  events" there means no non-finite losses and finite excursions are read off
  the trace metrics instead.
- Trace metrics: max-excess is the largest finite loss over the run's median
  finite loss; worst-window is the worst trailing 100-step mean loss, reported on
  GPT-2 as its exponential and elsewhere in the regime's raw loss units. They are
  computed only where per-step losses were retained.
- Where an arm has one extreme seed, the paper reports a median alongside the
  mean and says so.

## Two reproduction results the paper reports

**Same-seed rerun: 14 of 14 bit-identical.** Every legacy gated E4 run that
remains a reported result was re-executed on its own seed under the corrected
vector-valued held bound, and compared field by field against the shipped
artifact. All 14 pairs agree on `gate_open_frac`, `coord_open_frac`, the number
and location of the gate openings, the whole logged lambda trajectory, the
per-step loss trace, `hvp_total` and `events`. The earlier scalar drift-hold
shortcut was therefore inert on the entire reported E4 grid. Eleven of the
reruns are in `results/e4_verify_all/`; the other three are the pre-existing
`results/e4_fix/` runs, reused rather than repeated. The read-out is
`results/e4_verify_all/COMPARE_ALL.md` and the narrative is
`results/reanalysis/p4_verification.md`. The additive-flag bit-identity checks
are `results/e2_verify/`, `results/e2_verify4/` and `results/e1_verify4/`.

**E3 does not reproduce across the canonical and traced sets at ewc = 1000.**
`results/e3_traced/` re-runs Split-CIFAR-100 with per-step loss traces retained,
at the same hyperparameters and the same number of HVPs as `results/e3/`. At
ewc = 10 the stable-arm comparisons reproduce across the two sets. At
ewc = 1000 they do not: the two sets differ in which seeds fail. In the
canonical set COHG has 2 of 10 collapses and 5 of 10 non-finite triggers and the
fixed configuration has 3 of 10 collapses and 5 of 10 triggers, while on the
traced set COHG and the fixed configuration each collapse on 1 of 10. At that
strength neither the counts nor the ranking of the fixed, COHG and gate-off arms
transfers between the sets, so the paper reports the canonical and traced counts
separately and does not merge them. The same float-reassociation sensitivity
documented above is the mechanism. The read-outs are
`results/reanalysis/e3_trigger_censored.md` and
`results/reanalysis/common_stability.md`.

## What is not published

The fleet dispatch and provisioning scripts are not in this repository. They
hard-code SSH hosts, ports, users and plaintext passwords for the rented
machines, plus machine-specific working directories. They carry no scientific
content: they copy code out and results back, and every driver they invoke is
here in full. `README.md` names them.

`tools/make_repro_bundle.py` is included as the record of how the internal
reproducibility bundle was assembled, including its own credential denylist and
pre-copy secret scan. It targets the original working tree, whose layout nests
the code one level deeper than this repository does, so it does not run here
unmodified.
