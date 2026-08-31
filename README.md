# Certified Online Hypergradients (COHG)

**Online hypergradient adaptation that reports an upper bound on its own bias, per coordinate, at every step, and gates the update on it.**

Hypergradient descent adapts hyperparameters such as per-layer learning rates
along a data stream, and the one-step truncation that makes it cheap introduces
a bias that cannot be observed at run time. COHG maintains a discounted
sensitivity estimate `Ŝ_t = D_t + U_t R_tᵀ`, an exact group-aligned block plus a
rank-`r` sketch of the cross-group residual, and alongside it a certificate
recursion `e_{t+1} = γ ρ̄_t e_t + ε_t` driven by certified spectral inputs. The
recursion yields a per-coordinate bias bound `β_{t,j}`. The controller opens
coordinate `j` only when the estimate clears that bound by a margin,
`|ĝ_{t,j}| > c β_{t,j}` with `c ≥ 2`, which certifies the sign of the discounted
hypergradient on every open coordinate, and freezes every other coordinate. A
fail-closed drift monitor compares an observed rate `M_obs` against the deployed
prior `M_H`, re-probes above `M_H/2` and holds every coordinate shut above
`M_H`. A projected-gradient variant of the same gate admits a regret bound
driven by the observable certificate path.

The guarantees are conditional and the paper is explicit about the conditions.
The bound is against a discounted sensitivity, not the full-horizon one; it
holds under an assumed drift envelope and a `1-2δ` probe event; and it is proved
in exact arithmetic, so the floating-point implementation is audited empirically
rather than covered by the proof.

This repository contains the reference implementation, the experiment runners
and analysis scripts that produce the paper's figures and tables, **and the raw
result records behind most of them** (`results/`, see below). The released trees
cover the continual-learning and GPT-2 tables, the gating-control study, the
certificate audit, the drift-prior misspecification sweep, the threshold-transfer
study, the online drift-envelope, warm-up and dense-probe studies, and every
same-seed verification. Four run trees are **not** released, held back for size;
the table below says exactly what they are and what they back. Their summaries
are here even where their raw JSONs are not.

> **Paper:** *Certified online hypergradients: hyperparameter adaptation with
> conditional anytime bias certificates* (under review). The citation will be
> finalized on publication.

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the full protocol: the
environments, the released result trees, the exact command for every experiment,
the seed protocol, and the statistical conventions.

## Requirements

- Python 3.11 (the paper runs used 3.11; see `REPRODUCIBILITY.md`, "Environments")
- PyTorch 2.12.1 (CUDA 12.6). The time-series and continual-learning runs fit a
  single GPU comfortably; GPT-2 124M with certification peaks at 17.0 GB and was
  measured on a 20 GB card. The unit tests and the time-series runs work on CPU
  with `--device cpu`, and the gating-control study was run on CPU on purpose
- NumPy, Matplotlib
- E3 only: torchvision. E4 only: transformers, modelscope, pyarrow

```bash
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu126   # CUDA 12.6 build
pip install -r requirements.txt
```

Installing `requirements.txt` on its own gives you `torch==2.12.1` from PyPI,
which is not the CUDA 12.6 build the paper ran on; take the first line above if
you want that build.

`torch==2.12.1` is the one version the paper records, and `torchvision==0.27.1`
is the release built against it. No other version was written into the run
artifacts, so every remaining line of `requirements.txt` is a tested floor at
the version present in the environment that ran the campaigns, not a pin that
was never recorded. `certifi` and `pypdf` are optional and guarded by
`try`/`except` at their call sites.

## Repository layout

```
cohg/                the library
  estimator.py       COHG estimator: exact aligned block D_t plus rank-r
                     residual sketch U_t R_t^T, with refresh and lazy updates
  certificate.py     the certificate recursion, its certified spectral inputs
                     (rho_bar, kappa_bar), the per-coordinate column bound, the
                     drift hold and the fail-closed monitor (incl. the online
                     adaptive-M_H envelope)
  controller.py      the coordinate gate |ghat_j| > c*beta_j and the two
                     controller variants: sign steps and projected gradient
  hvp.py             Hessian-vector products on flat parameters, double backward
  groups.py          the hyperparameter-to-parameter group assignment
  functional.py      flat-parameter wrapper bridging nn.Module to the closures
  fmd.py             exact discounted forward-mode differentiation, the
                     reference sensitivity the audits compare against
  problems.py        analytic quadratic and teacher-student testbeds for E1
  baselines.py       HD (per-group and scalar), HDM, TFMD
experiments/
  e0_structure.py    structural probe on four architectures     -> results/e0
  e1_certificate.py  fp64 exact-reference certificate audit     -> results/e1*
  e2_timeseries.py   the online time-series driver              -> results/e2*, m4
  e3_continual.py    Split-CIFAR-100 continual learning         -> results/e3*
  e4_gpt2_tta.py     GPT-2 124M streaming adaptation            -> results/e4*
  e4_prepare_data.py builds the wiki/news/code token stream into data/
  data.py            the streams: Mackey-Glass, Lorenz, sunspots, Santa Fe, PTB
  models.py          the learners, all double-backward safe (hand-rolled GRU,
                     manual-attention transformer, GroupNorm ResNet)
  _gatestats.py      the additive --log-gate-stats instrumentation, inert when
                     the flag is off, so every default path stays byte-identical
  launch_*.py        the campaign launchers (18). Each is a plain local loop
                     over the driver, restart-safe by skipping existing outputs
  r*_queue.py        the detached local CPU queues the round-3/4 launchers use
  watch_*.py         queue pollers
  analyze_*.py       per-campaign analyses; several write results/*/SUMMARY.md
  compare_e4_*.py    the same-seed verification comparisons
  make_*.py          the figure scripts; make_paper_figures_nc.py is the one
                     that produced the figures in the manuscript
  gen_appendix_tables.py   the appendix tables, ddof=1, from the raw artifacts
  campaign_fix1.py, adopt_orphan.py   campaign repair and queue-handover helpers
tests/               numerical checks: exact FMD against finite differences and
                     an independent dense recursion, the certified spectral
                     bound against dense ground truth, column-wise certificate
                     validity, and the discounted-certificate preview
tools/
  make_repro_bundle.py   the internal bundle assembler, with its credential
                     denylist and pre-copy secret scan. Kept as the record of
                     how the bundle was built; it targets the original working
                     tree, which nests the code one level deeper than this
                     repository, so it does not run here unmodified
  run_e2_controls.py     detaches the E2-controls launcher from the shell
results/             the released raw result records (see below)
```

The original working tree nested the library and scripts under `code/`. This
repository flattens that one level, and the only change made to the released
scripts is the corresponding one-level correction in how each resolves the
repository root. Nothing else in them was touched, and no result record was
modified in any way.

## Released results

Most run JSONs are a single flat record: the configuration fields the run was
launched with (`method`, `dataset`, `seed`, `steps`, `lr0`, `meta_lr`, `rank`,
`K`, `gamma`), the outcome (`nmse`, or `avg_acc` and `bwt`, or `online_ppl`),
`events`, `gate_open_frac`, `coord_open_frac`, `hvp_total` and `wall_s`. Runs
that retained per-step logs add `losses` and `lam_hist`; the E3 runs add
`acc_matrix`, the E4 runs add `peak_mem_gb` and `mh_observed`. Runs launched
without trace retention, which is most of `results/e2_controls/`, carry the
scalar fields alone, and every reader script handles both shapes.

Three groups differ from that schema:

- `results/e1/` files are sweep artifacts, not one file per run: each is a JSON
  list of records over `(problem, tier, r, K, seed, gamma)` carrying
  `valid_rate`, the tightness quartiles and `contractive_frac`. The 1,808 fp64
  exact-reference runs live inside those lists.
- `results/e0/` files carry `summary` and `snapshots` rather than a run record,
  because the structural probe measures the residual structure along a
  trajectory rather than an outcome.
- `results/e2_cal/FROZEN_META_LR.json`, `results/e3_cal/`'s frozen file and
  `results/e2_controls/absgate_threshold.json` are calibration artifacts: the
  frozen meta-step size per method and the frozen absolute gate threshold, with
  the calibration seeds and protocol recorded alongside.

| Tree | Contents | Backs |
|---|---|---|
| `results/e2_controls/` | 186 run JSON + `absgate_threshold.json` + `SUMMARY.md` | the gating-control study: rate-matched random and periodic gates, the offline-calibrated absolute threshold, the online MAD threshold, the ungated alpha ladder and the step-condition control. Also the drifting-stream half of the drift-prior misspecification figure |
| `results/e2_gamma/` | 40 JSON | the certificate discount sweep, gamma in {0.8, 0.9, 0.95, 1.0}, ten seeds each |
| `results/e2_cal/` | 92 run JSON + `FROZEN_META_LR.json` | the meta-step calibration on seeds 100 and 101 and the frozen per-method values |
| `results/e2_denseprobe/` | 32 JSON | whether probing every step until T manufactures an early enough drift observation to rescue certified adaptation under the warm-up hold |
| `results/e2_lorenz_absgate/` | 34 JSON | the frozen absolute threshold transferred, unrecalibrated, to the other drifting stream |
| `results/e2_warmup/` | 32 JSON | holding the gate shut until the drift envelope is verified, and the bit-identity check of the default path |
| `results/e2_adaptmh/`, `results/e1_adaptmh/` | 20 + 4 JSON | the online-enforceable drift envelope `M_H,t = max(M_H_floor, KAPPA * max M_obs)`, KAPPA in {1, 2} |
| `results/e3/` | 180 JSON | the Split-CIFAR-100 main comparison and the appendix's full E3 configuration table |
| `results/e3_traced/` | 80 JSON | the same runs with per-step loss traces retained, and the trigger-censored read-out |
| `results/e3_noholdout/` | 40 JSON | the control in which the meta-objective is reduced to the incoming batch's prequential loss |
| `results/e3_absgate/` | 22 JSON | the transferred absolute threshold on Split-CIFAR-100, both EWC strengths |
| `results/e3_cal/` | 16 run JSON + `FROZEN_META_LR_E3.json` | the domain meta-step calibration for the raw-scale baselines and the frozen values |
| `results/e4_v2/` | 47 JSON | the GPT-2 124M standard domain order (wiki-news-code) |
| `results/e4_orders/` | 17 JSON | the reversed domain order (code-news-wiki) |
| `results/e4_misset/`, `results/e4_fix/`, `results/e4/` | 9 JSON; 3 JSON + `COMPARE.md`; 27 JSON | the mis-set initialization block, the three-seed held-bound re-run with its seed-for-seed comparison, and the earlier E4 pass |
| `results/e4_absgate/` | 4 JSON | the transferred absolute threshold on the GPT-2 stream |
| `results/e4_verify_all/` | 11 JSON + `COMPARE_ALL.md` | the full-scope same-seed verification: 14 pairs, all bit-identical |
| `results/e2_verify/`, `results/e2_verify4/`, `results/e1_verify4/` | 2 + 2 + 2 JSON | bit-identity checks that each round's additive flags left the default path unchanged |
| `results/e1/` | 10 JSON + `SUMMARY.md` | the fp64 exact-reference certificate audit: validity, tightness and the contraction boundary |
| `results/e1_misspec/` | 11 JSON | the drift-prior misspecification sweep on the exact-reference audit |
| `results/e0/` | 4 JSON | the structural probe on MLP, GRU, ResNet and transformer |
| `results/reanalysis/` | 18 Markdown + 4 CSV + 11 scripts | the post-hoc analyses: unified metrics, censored analysis, device sensitivity, gate-open timing, the round-3, round-4 and cross-regime stability read-outs, and the scripts that wrote them |
| `results/figures/` | 17 PNG + 7 PDF | the figure files as regenerated from the trees above |
| `results/e2_smoke/`, `e2_smoke4/`, `e3_smoke/`, `e3_probe/`, `e3_prefreeze/`, `e3_unfair/`, `e4_smoke/`, `results/m0_sanity.json` | 82 JSON | smoke and pilot runs kept for completeness. No reported number depends on them |

That is **1,074 files and 179 MB** of released records: 35 released trees, one
root-level sanity artifact, and the summary of the one held-back tree that has
one.

**Not released.** Four run trees are held back, because the complete
`results/` tree is 685 MB and this repository holds its released records under a
200 MB budget:

- `results/e2/` (1,158 JSON, 328 MB) — the main online time-series grid: the
  fixed-lambda grid, the adaptive methods from three initial learning rates on
  four datasets over ten seeds. It backs the drifting Mackey-Glass table, the
  two appendix time-series configuration tables, the hyperparameter-trajectory
  figure and the ablation figure. Its per-run `losses` and `lam_hist` traces are
  what make it 283 KB per file. **`results/e2/SUMMARY.md` is here**, so every
  per-configuration mean, standard deviation and event count in that grid can be
  read without the raw traces.
- `results/e2_shift/` (206 JSON, 91 MB) — the late-amplitude-shift study, six
  scale factors by five arms by ten seeds plus a pilot. It backs the
  late-shift table. Its full narrative read-out, including every number in that
  table, is `results/reanalysis/round4_shift.md` and
  `round4_shift_s1-9.md`, both of which are here.
- `results/m4/` (190 JSON, 56 MB) — the design-axis ablation sweeps: rank,
  refresh period `K`, gate factor `c` and discount `gamma` on two streams. It
  backs the design-axis controls reported in the supplement.
- `results/e2_prebackoff/` (107 JSON, 31 MB) — superseded pre-amendment runs.
  No reported number depends on them and no released script reads them.

Everything else the paper reports is backed by a tree above. A script whose
input tree is held back will find no files and say so rather than producing a
partial result: `analyze_e2.py` and `gen_appendix_tables.py` need
`results/e2/`, and `results/reanalysis/_round4_shift.py` needs
`results/e2_shift/`.

> The campaigns were dispatched to rented GPU servers and a LAN box by a set of
> `remote_*.py`, `dispatch_*.py`, `fleet_audit.py` and `harvest_loop.py`
> scripts, and by their local resume and collection helpers. Those are **not
> published**: they hard-code SSH hosts, ports, users and plaintext passwords
> for the rented machines, plus machine-specific working directories. They carry
> no scientific content and would only mislead. Each is a thin wrapper around
> the `python experiments/…` commands listed below, which fully describe what
> was run. The raw dispatch logs under `results/logs/` and `results/fleet_logs/`
> are withheld for the same reason.

## Quick start

Numerical correctness first. None of these needs a GPU.

```bash
python tests/run_m0.py         # exact FMD vs an independent dense recursion and vs central differences
python tests/run_kw_check.py   # the certified spectral bound upper-bounds ||I - diag(eta) H||_2
python tests/run_v3_check.py   # column-wise certificate validity and coordinate-gate behaviour
python tests/run_f3_preview.py # discounting and the certificate on nonconvex problems
```

A first real run. Write it to a fresh directory: the drivers skip a run whose
output JSON already exists, and `results/` ships populated, so pointing this at
a released tree would do nothing.

```bash
python experiments/e2_timeseries.py --method cohg --dataset mackey_drift \
    --seed 0 --steps 12000 --lr 0.003 --meta-lr 0.4 --gamma 0.9 \
    --K 10 --rank 4 --kw-eps 0.1 --probe-every 20 --M-H 5 \
    --out-dir results/smoke --device cpu
```

## Reproducing the paper

Each block writes one JSON per run. A rerun skips a run whose output already
exists, so the launchers are restart-safe and can be split across workers by
seed. **Because the released `results/` tree is already populated, delete the
target directory or redirect it with `--out-dir` before re-running, or every run
will be skipped as already done.**

```bash
# Structural probe on four architectures -> Fig. 2
for a in mlp gru resnet transformer; do python experiments/e0_structure.py --arch $a --seed 0; done

# fp64 exact-reference certificate audit -> Fig. 3, and the misspecification figure
python experiments/e1_certificate.py --problem quad    --tier prior
python experiments/e1_certificate.py --problem quad    --tier oracle
python experiments/e1_certificate.py --problem teacher --tier prior
python experiments/e1_certificate.py --problem teacher --tier kw --gamma 0.90
python experiments/e1_certificate.py --problem teacher --tier kw --gamma 0.95
python experiments/e1_certificate.py --problem teacher --tier kw --gamma 1.00
```

```bash
# Online time series. launch_e2 is the main grid; launch_cal freezes the
# meta-step sizes it uses -> the drifting-stream table, the appendix
# time-series tables, the trajectory figure and the ablation figure
python experiments/launch_cal.py
python experiments/launch_e2.py

# Design-axis ablations: rank, K, gate factor c, discount gamma -> supplement
python experiments/launch_m4.py

# Gating controls: rate-matched random and periodic, absolute threshold,
# online MAD, the ungated alpha ladder, the step-condition control -> control table
python experiments/launch_e2_controls.py --part calib
python experiments/launch_e2_controls.py --part main
```

```bash
# Split-CIFAR-100 continual learning -> the continual-learning table and the
# per-task accuracy figure; --part-free, the launcher holds the whole matrix
python experiments/launch_e3_cal.py
python experiments/launch_e3.py

# The no-holdout control -> the appendix holdout table
# (same launcher, e3_continual.py --no-holdout; see launch_e3.py for the arms)
```

```bash
# GPT-2 124M streaming adaptation -> the GPT-2 table and the perplexity figure
python experiments/e4_prepare_data.py     # builds data/e4_stream_{wiki,news,code}.pt
python experiments/launch_e4v2.py         # standard order  -> results/e4_v2
python experiments/launch_e4_expand.py    # reversed order and seed expansion -> results/e4_orders
```

```bash
# Round-3: bit-identity check, the added control arms, the gamma sweep,
# the E4 held-bound fix, the mis-set E4 initialization and the traced E3 re-run
python experiments/launch_round3.py --part verify
python experiments/launch_round3.py --part b1
python experiments/launch_round3.py --part b2
python experiments/launch_round3.py --part b3
python experiments/launch_r3_chain.py
python experiments/r3_cpu_queue.py

# Round-3.5: the full-scope same-seed E4 verification, 14 pairs
python experiments/launch_e4_verify_all.py
python experiments/compare_e4_verify_all.py
```

```bash
# Round-4: the late amplitude shift, the online drift envelope, the warm-up and
# dense-probe controls, and the threshold-transfer study across three domains
python experiments/launch_round4.py --part verify
python experiments/launch_round4.py --part shift
python experiments/launch_round4.py --part adaptmh
python experiments/r4_cpu_queue.py
python experiments/launch_r4_warmup.py
python experiments/launch_r4_denseprobe.py
python experiments/launch_r4_e2cpu.py     # Lorenz threshold transfer
python experiments/launch_r4_absgate.py   # GPT-2 and Split-CIFAR-100 threshold transfer
```

## Figures and tables

All of these run on the released tree as cloned, except the two noted as needing
a held-back tree. None of them needs a GPU.

```bash
python experiments/analyze_e1.py             # results/e1/SUMMARY.md
python experiments/analyze_e3.py             # results/e3/SUMMARY.md, the continual-learning table
python experiments/analyze_e2_controls.py    # results/e2_controls/SUMMARY.md, the control table
python experiments/analyze_cal.py            # the frozen meta-step-size table
python experiments/analyze_misspec.py        # results/reanalysis/misspec_curves.md
python experiments/analyze_round3.py         # results/reanalysis/round3_cpu.md
python experiments/analyze_r4_absgate.py     # results/reanalysis/round4_absgate_transfer.md
python experiments/analyze_device_sensitivity.py   # results/reanalysis/device_sensitivity.md
python experiments/compare_e4_verify_all.py  # results/e4_verify_all/COMPARE_ALL.md
python results/reanalysis/_reanalyze.py      # unified metrics and the censored analysis
python results/reanalysis/_key_questions.py
python results/reanalysis/_e3_noholdout.py   # results/reanalysis/e3_noholdout.md
python results/reanalysis/_common_stability.py     # cross-regime stability, trigger-censored E3
python results/reanalysis/_fig8_gpt2_lambda.py     # GPT-2 step-size trajectories
python results/reanalysis/_fig9_gpt2_order.py      # the same, reversed domain order
python experiments/make_fig10_misspec.py           # the misspecification figure
python experiments/make_pareto.py                  # the safety-performance plane
python experiments/make_paper_figures_nc.py        # the paper's data figures

python experiments/analyze_e2.py             # needs results/e2/ (held back)
python experiments/gen_appendix_tables.py    # needs results/e2/ (held back)
```

`make_paper_figures_nc.py` writes print-size PDFs into `paper/main/figs/`, which
belongs to the manuscript and is not part of this repository, and PNG previews
into `results/figures/`, which is. `make_figures.py` and
`make_paper_figures.py` are earlier, plainer entry points kept for reference;
`make_paper_figures_nc.py` is the one that produced the figures in the
manuscript.

All aggregate statistics use the sample standard deviation (`ddof=1`), and
significance is the two-sided paired exact permutation test over all `2^n` sign
assignments of the per-seed differences.

## Data sources

Everything is downloaded on first use into `data/`, which is git-ignored. No
dataset is redistributed here.

- **Mackey-Glass and Lorenz** are generated deterministically from a seed. The
  nonstationary variants switch their generating parameters twice along the
  stream; the late-shift study additionally scales the middle third of the
  series, inputs and targets alike, by a factor `F`.
- **Sunspots** — monthly mean total sunspot number, `SN_m_tot_V2.0`, from
  [SILSO](https://www.sidc.be/SILSO/), Royal Observatory of Belgium.
- **Santa Fe laser** — far-infrared laser intensity, data set A of the Santa Fe
  time-series competition, retrieved from the reservoirpy and CHARC mirrors.
- **Penn Treebank** — the standard character/word files from the Zaremba LSTM
  repository mirror.
- **CIFAR-100** — downloaded by torchvision and split into ten tasks of ten
  classes, task-incremental with known task boundaries.
- **GPT-2 124M** weights come from the ModelScope mirror `AI-ModelScope/gpt2`,
  which is why `modelscope` is a hard requirement of `e4_gpt2_tta.py` rather
  than an optional one.
- **The GPT-2 stream** is three domains built by `e4_prepare_data.py`: WikiText-103
  raw for `wiki`, the AG-News train split for `news`, and the `.py` files of the
  NumPy v1.26.4 source tree for `code`, each truncated to 512k tokens so the
  domain boundaries fall at steps 1,000 and 2,000 of a 2,999-step stream.

## Notes

- Every driver skips a run whose output JSON already exists, which makes the
  campaigns restart-safe and splittable by seed across workers. It also means a
  launcher pointed at a populated tree does nothing.
- All new flags added across the revision rounds are additive and default to the
  previous behaviour. The claim is checked, not asserted:
  `results/e2_verify/`, `results/e2_verify4/` and `results/e1_verify4/` are
  bit-identity re-runs of the frozen reference arms on the patched drivers, and
  `_gatestats.py` is inert when `--log-gate-stats` is off.
- The same code, same seed and same configuration do not give the same loss
  trace on GPU and on CPU. `results/reanalysis/device_sensitivity.md` measures
  it: 0 of 10 seeds bitwise identical, 4 of 10 differing in the realized
  gate-open rate, 0 of 10 changing regime. The paper quotes COHG NMSE to two
  significant figures in cross-device comparisons for that reason, and the
  gating-control study carries its own same-device reference arm.
- The gating-control study ran on CPU deliberately: the Hessian-vector products
  of the 13k-parameter GRU are kernel-launch bound at that size.
- Certification is not free. It costs a measured 7.9 HVPs per step at the
  13k-parameter time-series scale, of which 1.0 is the estimator and the rest the
  certification probes, and at GPT-2 124M a factor of 10.8 in wall time and 2.8
  in peak memory against unadapted fixed-rate training.
- The certificate covers a discounted sensitivity, so a certified sign is a
  certified sign of the discounted hypergradient. Where the gate opens, that
  sign agreed with the full-horizon sign on 1,443 of 1,444 gate-open
  coordinate-steps; away from the gate the two agree on about three quarters of
  coordinate-steps.

## License

Code and result records: MIT (see [`LICENSE`](LICENSE)). No third-party dataset
is redistributed in this repository; each is downloaded from its own source
under its own terms, listed under "Data sources" above.

## Citation

```bibtex
@article{cohg,
  title  = {Certified online hypergradients: hyperparameter adaptation with
            conditional anytime bias certificates},
  author = {Junfei Yi and Yuxiang Wang},
  note   = {Under review},
  year   = {2026}
}
```
