"""Assemble the COHG reproducibility bundle.

Collects, into ``repro_bundle/`` at the repository root:

  * the code revision (``code/cohg``, ``code/experiments``, ``code/tools``),
    which contains every launch configuration and every analysis script;
  * the raw per-seed result artifacts (one JSON per (method, dataset, seed)
    configuration, loss traces included, byte-for-byte as written by the runs);
  * the per-campaign summaries (``results/*/SUMMARY*.md``) and the reanalysis
    outputs (``results/reanalysis/*.md``, ``*.csv``, ``*.py``);
  * the print-size paper figures;
  * ``MANIFEST.md``, which maps every table and figure of the paper to the
    script that regenerates it, and lists the round-3 and round-3.5/4
    additions;
  * ``SHA256SUMS.txt``, the SHA-256 of every bundled file in ``sha256sum -c``
    format.

Two files are deliberately withheld (``CREDENTIAL_FILES``): the fleet dispatch
scripts that embed SSH hosts, users and plaintext passwords.  Before anything is
copied, every small text member is scanned for credential-shaped lines and the
build aborts on a hit, so a secret cannot reach the bundle by accident.

JSON artifacts are copied intact: nothing inside them is trimmed, so every
reported number can be recomputed from the bundle alone.  If the assembled
bundle would exceed ``--max-gb`` (default 2.0), the GPT-2 directories, whose
per-run ``losses`` and ``lam_hist`` arrays dominate the byte count, are dropped
and the omission is recorded in the manifest.

Usage::

    python code/tools/make_repro_bundle.py            # build, report size
    python code/tools/make_repro_bundle.py --clean    # rebuild from scratch

The bundle is left unzipped.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import re
import shutil
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEST = os.path.join(ROOT, "repro_bundle")

# --- what goes in -----------------------------------------------------------

CODE_TREES = [
    os.path.join("code", "cohg"),
    os.path.join("code", "experiments"),
    os.path.join("code", "tools"),
    os.path.join("code", "tests"),
]

# directory names never copied, anywhere
SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", ".pytest_cache"}

# file patterns never copied, anywhere
SKIP_FILES = ["*.pyc", "*.pyo", "*.bak_prefix", "*.log", "texput.log", ".*"]

# Files withheld because they carry live fleet credentials (SSH hosts, users and
# plaintext passwords for the rented TITAN Xp machines).  They are dispatch
# plumbing only: no reported number depends on them, and the experiment drivers
# they invoke (``code/experiments/*.py``) are bundled in full, so every run can
# be reproduced without them.  Recorded in the manifest so the omission is
# explicit rather than silent.
CREDENTIAL_FILES = [
    os.path.join("code", "tools", "remote_192.py"),
    os.path.join("code", "tools", "remote_setup.py"),
]

# Patterns that must never appear in a bundled text file.  Checked over the
# small text members (code, markdown, csv) before anything is copied; a hit
# aborts the build rather than shipping the secret.
_SECRET_KEY = r"""['"]?%s['"]?\s*[=:]\s*['"][^'"]{4,}['"]"""
SECRET_PATTERNS = [
    re.compile(_SECRET_KEY % r"pass(word|wd)?", re.I),
    re.compile(_SECRET_KEY % r"api[_-]?key", re.I),
    re.compile(_SECRET_KEY % r"(access|secret|auth)[_-]?(key|token)", re.I),
    re.compile(_SECRET_KEY % r"secret", re.I),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"sshpass\s+[^\n]{0,80}?-p\s*['\"]?\S{6,}", re.I),
]

# Cheap substring prefilter: a line can only match one of the patterns above if
# it contains one of these, so the regexes never run on the millions of numeric
# lines in the result JSONs.
SECRET_TRIGGERS = ("pass", "key", "token", "secret", "auth", "sshpass",
                   "PRIVATE KEY")

# Extensions worth scanning for secrets (the result JSONs are machine-written
# run artifacts and are far too large to regex-scan usefully).
SECRET_SCAN_EXT = {".py", ".md", ".txt", ".csv", ".sh", ".cfg", ".toml", ".yaml",
                   ".yml", ".json"}
SECRET_SCAN_MAX_BYTES = 512 * 1024

# results: file patterns to keep
RESULT_KEEP = ["*.json", "*.md", "*.csv", "*.py", "*.pdf", "*.png"]

# results subtrees never copied (smoke tests and superseded pre-amendment runs
# are not referenced by any reported number)
RESULT_SKIP_DIRS = {"fleet_logs"}

# GPT-2 result directories, dropped only if the bundle would exceed --max-gb
GPT2_DIRS = ["e4", "e4_v2", "e4_orders", "e4_smoke", "e4_fix", "e4_misset"]

FIG_TREE = os.path.join("paper", "main", "figs")


def _skip_file(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in SKIP_FILES)


def _is_credential(rel: str) -> bool:
    rel = rel.replace("\\", "/")
    return any(rel == c.replace("\\", "/") for c in CREDENTIAL_FILES)


def scan_secrets(files):
    """Return [(rel, lineno, pattern)] for every secret-looking line."""
    hits = []
    for ap, rel, sz in files:
        if os.path.splitext(rel)[1].lower() not in SECRET_SCAN_EXT:
            continue
        if sz > SECRET_SCAN_MAX_BYTES:
            continue
        try:
            with open(ap, "r", encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    low = line.lower()
                    if not any(t.lower() in low for t in SECRET_TRIGGERS):
                        continue
                    for pat in SECRET_PATTERNS:
                        if pat.search(line):
                            hits.append((rel, i, pat.pattern))
        except OSError:
            continue
    return hits


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_checksums(dest, files):
    """Write SHA256SUMS.txt over every bundled member; return (path, digest).

    The trailing digest is the SHA-256 of the checksum file itself, which acts
    as a single fingerprint for the whole bundle and is quoted in the manifest.
    """
    lines = []
    for _ap, rel, _sz in sorted(files, key=lambda t: t[1].replace("\\", "/")):
        dst = os.path.join(dest, rel)
        lines.append("%s  %s" % (sha256(dst), rel.replace("\\", "/")))
    path = os.path.join(dest, "SHA256SUMS.txt")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return path, sha256(path)


def code_revision():
    """Content fingerprint of the bundled code trees (the repo is not under git).

    The first 12 hex digits of the SHA-256 over (relative path, size, sha256)
    of every bundled code file, plus the newest modification time in the trees.
    Two builds print the same revision iff the code is byte-identical.
    """
    h = hashlib.sha256()
    newest = 0.0
    rows = []
    for tree in CODE_TREES:
        for ap, rel, sz in plan_tree(tree):
            if _is_credential(rel):
                continue
            rows.append((rel.replace("\\", "/"), sz, ap))
    for rel, sz, ap in sorted(rows):
        h.update(("%s|%d|%s\n" % (rel, sz, sha256(ap))).encode("utf-8"))
        try:
            newest = max(newest, os.path.getmtime(ap))
        except OSError:
            pass
    stamp = datetime.fromtimestamp(newest, timezone.utc).strftime("%Y-%m-%d")
    return h.hexdigest()[:12], stamp


def _keep_result(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in RESULT_KEEP)


def plan_tree(src_rel, keep=None):
    """Return [(abs_src, rel_dest, size)] for one subtree of the repo."""
    out = []
    src = os.path.join(ROOT, src_rel)
    if not os.path.isdir(src):
        return out
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if _skip_file(fn):
                continue
            if keep is not None and not keep(fn):
                continue
            ap = os.path.join(dirpath, fn)
            rel = os.path.relpath(ap, ROOT)
            if _is_credential(rel):
                continue
            try:
                sz = os.path.getsize(ap)
            except OSError:
                continue
            out.append((ap, rel, sz))
    return out


def plan_results():
    out = []
    rdir = os.path.join(ROOT, "results")
    for entry in sorted(os.listdir(rdir)):
        ap = os.path.join(rdir, entry)
        if os.path.isdir(ap):
            if entry in RESULT_SKIP_DIRS:
                continue
            out.extend(plan_tree(os.path.join("results", entry),
                                 keep=_keep_result))
        elif _keep_result(entry) and not _skip_file(entry):
            out.append((ap, os.path.join("results", entry),
                        os.path.getsize(ap)))
    return out


def human(nbytes):
    return "%.1f MB" % (nbytes / (1024.0 ** 2))


# --- manifest ---------------------------------------------------------------

# (paper object, what it reports, script that regenerates it)
TABLE_MAP = [
    ("Table 1 (tab:claimscope)", "scope of the five guarantee claims",
     "no data; written from the theorem statements"),
    ("Table 2 (tab:taxonomy)", "positioning against prior work",
     "no data; written from the cited papers"),
    ("Table 3 (tab:notation)", "notation", "no data"),
    ("Table 4 (tab:cost)", "amortized per-step cost, wall time, peak memory",
     "code/experiments/analyze_e2.py (HVP totals) and "
     "code/experiments/launch_e4v2.py run artifacts in results/e4_v2/ "
     "(wall time, peak memory)"),
    ("Table 5 (tab:variants)", "certificate variants", "no data"),
    ("Table 6 (tab:e2)", "drifting Mackey-Glass, mis-set initialization",
     "code/experiments/analyze_e2.py over results/e2/"),
    ("Table 7 (tab:controls)", "RQ2b control study",
     "code/experiments/analyze_e2_controls.py over results/e2_controls/ "
     "(writes results/e2_controls/SUMMARY.md)"),
    ("Table 8 (tab:e3)", "Split-CIFAR-100 main comparison",
     "code/experiments/analyze_e3.py over results/e3/"),
    ("Table 9 (tab:e4)", "GPT-2 124M, standard domain order",
     "results/reanalysis/_reanalyze.py over results/e4_v2/ "
     "(writes results/reanalysis/unified_metrics_e4.csv)"),
    ("Table 10 (tab:e4orders)", "GPT-2 124M, reversed domain order",
     "results/reanalysis/_reanalyze.py over results/e4_orders/; "
     "narrative in results/reanalysis/e4_expansion.md"),
    ("Table 11 (tab:frozenml)", "frozen meta-step sizes per method and regime",
     "code/experiments/analyze_cal.py over results/e2_cal/ and "
     "results/e3_cal/"),
    ("Table 12 (tab:certparams)", "certification and controller parameters",
     "read from the launch configurations in code/experiments/launch_*.py "
     "and the args stored in each result JSON"),
    ("Table 13 (tab:e2stat)", "E2 stationary streams, all configurations",
     "code/experiments/gen_appendix_tables.py over results/e2/"),
    ("Table 14 (tab:e2drift)", "E2 drifting streams, all configurations",
     "code/experiments/gen_appendix_tables.py over results/e2/"),
    ("Table 15 (tab:e3full)", "E3 all configurations",
     "code/experiments/gen_appendix_tables.py over results/e3/"),
    ("Table 16 (tab:e3nh)", "E3 with and without the retained holdout",
     "results/reanalysis/_e3_noholdout.py over results/e3_noholdout/ and "
     "results/e3/ (writes results/reanalysis/e3_noholdout.md)"),
]

FIGURE_MAP = [
    ("Fig. 1 (fig:pipeline)", "the COHG loop",
     "TikZ source in paper/main/sections/fig_pipeline.tex; no data"),
    ("Fig. 2 (fig:structure), figs/fig1_e0_structure.pdf",
     "structural probe on four architectures",
     "code/experiments/e0_structure.py then "
     "code/experiments/make_paper_figures_nc.py"),
    ("Fig. 3 (fig:cert), figs/fig2_e1_certificate.pdf",
     "certificate validity and tightness",
     "code/experiments/e1_certificate.py, code/experiments/analyze_e1.py, "
     "then code/experiments/make_paper_figures_nc.py"),
    ("Fig. 4 (fig:pareto), figs/fig1b_pareto.pdf",
     "safety-performance plane",
     "code/experiments/make_pareto.py, restyled by "
     "code/experiments/make_paper_figures_nc.py"),
    ("Fig. 5 (fig:traj), figs/fig4_lambda_traj.pdf",
     "hyperparameter trajectories, gated and ungated",
     "code/experiments/make_paper_figures_nc.py over results/e2/"),
    ("Fig. 6 (fig:e3tasks), figs/fig5_e3_tasks.pdf",
     "Split-CIFAR-100 per-task accuracy",
     "code/experiments/make_paper_figures_nc.py over results/e3/"),
    ("Fig. 7 (fig:gpt2), figs/fig6_gpt2.pdf",
     "GPT-2 online perplexity along the stream",
     "code/experiments/make_paper_figures_nc.py over results/e4_v2/"),
    ("Fig. 8 (fig:gpt2lambda), figs/fig8_gpt2_lambda.pdf",
     "GPT-2 per-coordinate step-size trajectories",
     "results/reanalysis/_fig8_gpt2_lambda.py over results/e4_v2/ "
     "(also writes results/reanalysis/gate_open_timing.md)"),
    ("Fig. 9 (fig:gpt2order), figs/fig9_gpt2_order.pdf",
     "the same trajectories in the reversed domain order",
     "results/reanalysis/_fig9_gpt2_order.py over results/e4_orders/"),
    ("Fig. 10 (fig:ablations), figs/fig7_ablations.pdf",
     "ablation sweeps",
     "code/experiments/make_paper_figures_nc.py over results/e2/"),
    ("fig:misspec, figs/fig10_misspec.pdf",
     "curvature-bound misspecification: certificate validity and step-size "
     "behaviour across five settings of $M_H$",
     "code/experiments/make_fig10_misspec.py over results/e1_misspec/ and "
     "results/e2_controls/ (PNG preview in results/figures/"
     "fig10_misspec.png; narrative in results/reanalysis/misspec_curves.md "
     "via code/experiments/analyze_misspec.py)"),
]

# Round-3 additions: the campaigns and scripts added after the round-2 bundle.
# (path, what it holds / what it answers, produced by)
ROUND3_MAP = [
    ("results/e2_controls/ (madgate, ogd_doubling, cohg_ogd arms)",
     "calibration-free MAD threshold gate, doubling-trick OGD baseline and "
     "the OGD-driven COHG arm, alongside the round-2 control arms",
     "code/experiments/launch_round3.py --part b1/b2 (drivers: "
     "code/experiments/e2_timeseries.py); summarised by "
     "code/experiments/analyze_e2_controls.py"),
    ("results/e2_gamma/",
     "sweep of the certificate discount gamma in {0.8, 0.9, 0.95, 1.0}, "
     "ten seeds each, with --validate-full",
     "code/experiments/launch_round3.py --part b3"),
    ("results/e2_verify/",
     "bit-identity check: the frozen COHG reference arm re-run on the patched "
     "e2_timeseries.py, seeds 0-1",
     "code/experiments/launch_round3.py --part verify"),
    ("results/e1_misspec/",
     "certificate behaviour when the curvature bound M_H is misspecified",
     "code/experiments/e1_certificate.py under the misspecification grid; "
     "analysed by code/experiments/analyze_misspec.py"),
    ("results/e4_misset/",
     "GPT-2 124M from a mis-set initialization (lr0 = 1e-4), fixed / cohg_r0 "
     "/ cohg_nogate, seeds 0-2",
     "code/experiments/launch_r3_chain.py (driver: "
     "code/experiments/e4_gpt2_tta.py)"),
    ("results/e4_orders/ (seeds 3-7 of cohg_r0)",
     "reversed-domain-order expansion, cohg_r0 now covering seeds 0-7",
     "code/experiments/launch_r3_chain.py"),
    ("results/e4_fix/ (+ COMPARE.md)",
     "GPT-2 re-run of the well-set arm after the driver fix, seeds 0-2, and "
     "the seed-for-seed comparison against results/e4_v2/",
     "code/experiments/launch_r3_chain.py; comparison by "
     "code/experiments/compare_e4_fix.py"),
    ("results/e3_traced/",
     "Split-CIFAR-100 re-run with per-step loss traces retained (80 runs, "
     "lr0 = 0.05, seeds 0-9), i.e. results/e3 plus the losses it never stored",
     "code/experiments/e3_continual.py --log-losses --no-holdout, launched by "
     "code/experiments/r3_cpu_queue.py"),
    ("results/reanalysis/round3_cpu.md",
     "the CPU half of the round-3 analysis (E2 gates, gamma sweep, "
     "reproduction check)",
     "code/experiments/analyze_round3.py"),
    ("results/reanalysis/round3_gpu.md",
     "the GPU half (E4 mis-set init, reversed order, traced E3, "
     "misspecification)",
     "results/reanalysis/_round3_gpu.py"),
    ("results/reanalysis/misspec_curves.md",
     "curvature-bound misspecification narrative behind fig:misspec",
     "code/experiments/analyze_misspec.py"),
    ("results/figures/",
     "PNG previews of the round-3 figures, fig10_misspec.png included",
     "code/experiments/make_fig10_misspec.py"),
    ("code/experiments/launch_round3.py, launch_r3_chain.py, r3_cpu_queue.py",
     "the round-3 launchers: E2 CPU parts, the chained GPU campaigns and the "
     "CPU queue for the traced E3 re-run",
     "-"),
    ("code/experiments/e4_gpt2_tta.py (--legacy-hold), e3_continual.py "
     "(--no-holdout, --log-losses)",
     "the two drivers extended in round 3; both flags are additive and the "
     "legacy path is verified bit-identical (results/e2_verify/)",
     "-"),
]

# Round-3.5/4 additions: everything added after the round-3 bundle, in response
# to the round-3.5 review (P4 at full scope, the common stability table, the
# trigger-censored E3 read-out and the M_obs / M_H definitions).
# (path, what it holds / what it answers, produced by, guard)
# ``guard`` is a repo-relative path that must exist for the row to be written;
# ``None`` means the row is unconditional.  It exists for the artifacts that may
# still be in flight when the bundle is assembled.
ROUND35_MAP = [
    ("results/e4_verify_all/ (11 run JSONs + COMPARE_ALL.md)",
     "P4 same-seed verification at FULL scope: every legacy gated E4 run that "
     "is still a reported result (14 pairs: cohg_r0 seeds 0-7 and cohg(r=4) "
     "seeds 0-2 in the standard order, cohg_r0 seeds 0-2 in the reversed "
     "order), re-run on its own seed under the corrected vector-valued "
     "Proposition-10 held bound and compared field by field against the "
     "shipped artifact.  All 14 pairs are bit-identical, so the scalar "
     "drift-hold shortcut was inert on the whole reported E4 grid.  Eleven "
     "reruns live here; the remaining three are the pre-existing "
     "results/e4_fix/ runs, reused rather than repeated",
     "code/experiments/launch_e4_verify_all.py (driver: "
     "code/experiments/e4_gpt2_tta.py); compared by "
     "code/experiments/compare_e4_verify_all.py, which writes "
     "results/e4_verify_all/COMPARE_ALL.md; "
     "code/experiments/watch_e4_verify_all.py polls the queue",
     None),
    ("results/reanalysis/p4_verification.md",
     "narrative write-up of the P4 full-scope verification: scope argument "
     "for why only the gated arms can differ, and the read-out of the 14 "
     "pairs",
     "analysis document written against results/e4_verify_all/COMPARE_ALL.md",
     os.path.join("results", "reanalysis", "p4_verification.md")),
    ("results/reanalysis/common_stability.md",
     "one cross-regime stability table over E2 / E3 / E4 under a single rule "
     "set (spike count, non-finite incidence, max-excess ratio, worst "
     "trailing-100 window), so the three regimes are read off the same "
     "definitions; metric functions imported verbatim from _reanalyze.py",
     "results/reanalysis/_common_stability.py",
     None),
    ("results/reanalysis/e3_trigger_censored.md",
     "trigger-censored read-out of the traced E3 re-run: steps to the first "
     "divergence-recovery trigger, right-censored at the 3040-step horizon, "
     "and accuracy split by whether a run ever triggered -- separating "
     "triggering from blowing up, which come apart in the ungated arm",
     "results/reanalysis/_common_stability.py over results/e3_traced/",
     None),
    ("results/reanalysis/_common_stability.py",
     "the script behind both of the two documents above",
     "-",
     None),
    ("results/reanalysis/misspec_curves.md (Definitions section)",
     "the three curvature quantities kept apart: `M_obs`, the observed "
     "probe-to-probe drift rate recorded only by the fail-closed monitor "
     "(and how it differs from the base class's `mh_observed`); `M_H`, the "
     "DEPLOYED prior that is the denominator of the fig:misspec x-axis; and "
     "`M_obs^med / p99 / max`, stream properties rather than settings",
     "code/experiments/analyze_misspec.py",
     None),
    ("results/reanalysis/round3_cpu.md (B2 addendum)",
     "where the first certified step of the doubling-trick OGD arm actually "
     "falls: instrumented re-run of the same ten configs logging per-step "
     "(G_k, alpha, open_mask), showing nothing opens at t=0, the first "
     "accepted opening is t=1 with the empty-history bootstrap firing, and "
     "the step is exactly the box width",
     "code/experiments/analyze_round3.py; addendum measured with an "
     "instrumented copy of code/experiments/e2_timeseries.py, verified "
     "against the shipped ogd_alpha_log and gate_open_frac",
     None),
    ("paper/main/figs/fig10_misspec.pdf (regenerated)",
     "fig:misspec redrawn with both series on the M_H/M_H_deployed axis, "
     "consistent with the Definitions section above",
     "code/experiments/make_fig10_misspec.py",
     None),
]


def round35_rows():
    """ROUND35_MAP with the guarded rows that are not on disk dropped."""
    rows = []
    for a, b, c, guard in ROUND35_MAP:
        if guard is not None and not os.path.exists(os.path.join(ROOT, guard)):
            continue
        rows.append((a, b, c))
    return rows


# Round-4 additions: everything added after the round-3.5/4 bundle, answering
# three reviewer objections against the round-3/3.5 controls: (1) the E2
# fixed-prior/static-threshold controls were only ever exercised on the
# stream they were calibrated on; (2) the drift bound M_H is a fixed offline
# constant rather than something the run itself could audit; (3) the offline
# static gate threshold that matched COHG on its calibration stream was never
# tried on a different stream, architecture or domain.
# (path, what it holds / what it answers, produced by)
ROUND4_MAP = [
    ("results/e2_shift/ (206 run JSONs incl. results/e2_shift/pilot/)",
     "certified re-adaptation under a LATE amplitude shift: `--scale-shift F` "
     "multiplies the middle third of the mackey_drift stream (inputs and "
     "targets) by F in {0.1, 0.2, 1, 3, 5, 10}, fixed/hd/cohg/cohg_nogate/"
     "absgate arms, seeds 0-9, plus a 6-seed pilot",
     "code/experiments/launch_round4.py --part shift (driver: "
     "code/experiments/e2_timeseries.py --scale-shift); analysed by "
     "results/reanalysis/_round4_shift.py"),
    ("results/e2_adaptmh/",
     "E2 mackey_drift under the online-enforceable drift envelope "
     "`M_H,t = max(M_H_floor, KAPPA * max M_obs)`, KAPPA in {1, 2}, seeds 0-9",
     "code/experiments/launch_round4.py --part adaptmh (driver: "
     "code/experiments/e2_timeseries.py --adaptive-mh)"),
    ("results/e1_adaptmh/",
     "E1 teacher/kw_drift under the same online-enforceable envelope, "
     "KAPPA in {1, 2}, seeds 0-4",
     "code/experiments/launch_round4.py --part adaptmh (driver: "
     "code/experiments/e1_certificate.py --adaptive-mh); analysed together "
     "with results/e2_adaptmh/ by results/reanalysis/_round4_adaptmh.py"),
    ("results/e2_warmup/ (+ verify/)",
     "does holding the gate shut until the drift envelope is VERIFIED "
     "(`--gate-warmup first-obs|stable-env`) keep any certified adaptation on "
     "E2 mackey_drift; seeds 0-9 plus a bit-identity check of the default "
     "(`--gate-warmup off`) path",
     "code/experiments/launch_r4_warmup.py (driver: "
     "code/experiments/e2_timeseries.py --gate-warmup), dispatched by "
     "code/experiments/r4_warmup_queue.py; analysed by "
     "results/reanalysis/_round4_warmup.py"),
    ("results/e2_denseprobe/ (+ verify/)",
     "does probing at EVERY step until T (`--probe-dense-until T`) manufacture "
     "an early drift observation and rescue certified adaptation under the "
     "warm-up hold; seeds 0-9 over a T sweep",
     "code/experiments/launch_r4_denseprobe.py (driver: "
     "code/experiments/e2_timeseries.py --probe-dense-until), dispatched by "
     "code/experiments/r4_denseprobe_queue.py; analysed by "
     "results/reanalysis/_round4_denseprobe.py"),
    ("results/e2_verify4/",
     "bit-identity check: the frozen COHG reference arm (mackey_drift, "
     "lr0=0.003, M_H=5) re-run on the patched e2_timeseries.py, seeds 0-1, "
     "diffed against results/e2_verify/ -- all 54 legacy keys identical",
     "code/experiments/launch_round4.py --part verify"),
    ("results/e1_verify4/",
     "bit-identity check: teacher/kw_drift at the deployed prior, seeds 0-4, "
     "with and without --fail-closed, diffed against "
     "results/e1_misspec/teacher_kw_drift_x1_fc{0,1}.json -- all 29 shared "
     "keys identical",
     "code/experiments/launch_round4.py --part verify"),
    ("results/e2_lorenz_absgate/ (+ _verify/, _calib_ref/)",
     "transfer of the mackey_drift-calibrated static absgate threshold "
     "(T=0.05806520209, no recalibration) to the OTHER E2 drift stream, "
     "lorenz_drift, mis-set init lr0=0.003, seeds 0-9",
     "code/experiments/launch_r4_e2cpu.py (driver: "
     "code/experiments/e2_timeseries.py --method absgate)"),
    ("results/e3_absgate/ (+ _ref/)",
     "the same transferred threshold on Split-CIFAR-100 continual learning, "
     "ewc10/ewc1000 x seeds 0-9",
     "code/experiments/launch_r4_absgate.py --phase b_e3 (driver: "
     "code/experiments/e3_continual.py --method absgate --log-gate-stats)"),
    ("results/e4_absgate/ (+ _ref/)",
     "the same transferred threshold on GPT-2 124M streaming TTA, seeds 0-2",
     "code/experiments/launch_r4_absgate.py --phase a_e4 (driver: "
     "code/experiments/e4_gpt2_tta.py --method absgate --log-gate-stats)"),
    ("results/reanalysis/round4_shift.md",
     "narrative: COHG's certified response to the late shift is a refusal "
     "to move, not a re-adaptation, and the static threshold breaks in "
     "whichever direction the amplitude moved (203x over-open and 3/10 "
     "divergences at F=10, complete silence at F=0.2)",
     "results/reanalysis/_round4_shift.py over results/e2_shift/"),
    ("results/reanalysis/round4_adaptmh.md",
     "narrative: the online-enforced envelope is bit-identical in decisions "
     "to the fixed prior (0 openings lost) and costs 8%/5% fewer HVPs on "
     "E2/E1, but every certified opening on this config occurs before the "
     "first drift observation exists, so 0 of 84 openings are admissible "
     "under a strict retrospective-consistency reading (100% under the "
     "run's final envelope)",
     "results/reanalysis/_round4_adaptmh.py over results/e2_adaptmh/ and "
     "results/e1_adaptmh/"),
    ("results/reanalysis/round4_warmup.md",
     "narrative: requiring the envelope to be verified before use removes "
     "ALL of COHG's certified adaptation on this configuration (NMSE x3.23 "
     "worse, p=0.0020) because every opening falls at steps 1-15, five steps "
     "before the earliest possible measurement at step 20",
     "results/reanalysis/_round4_warmup.py over results/e2_warmup/"),
    ("results/reanalysis/round4_denseprobe.md",
     "narrative: whether probing every step until T manufactures an early "
     "enough drift observation to rescue adaptation under the warm-up hold",
     "results/reanalysis/_round4_denseprobe.py over results/e2_denseprobe/"),
    ("results/reanalysis/round4_absgate_transfer.md",
     "narrative: the offline-calibrated static threshold does not transfer "
     "outside its calibration stream -- on Lorenz it either over-opens or "
     "goes silent, on GPT-2 and Split-CIFAR-100 it slams learning rates to "
     "their floor and freezes the learner, while COHG's certificate rescales "
     "with the data and either matches or safely refuses",
     "code/experiments/analyze_r4_absgate.py over results/e2_lorenz_absgate/, "
     "results/e3_absgate/ and results/e4_absgate/"),
    ("code/experiments/launch_round4.py, r4_cpu_queue.py",
     "the round-4 CPU launcher (E2/E1 parts verify/shift/adaptmh) and its "
     "detached completion queue",
     "-"),
    ("code/experiments/launch_r4_warmup.py, r4_warmup_queue.py",
     "the --gate-warmup study launcher and its detached CPU queue",
     "-"),
    ("code/experiments/launch_r4_denseprobe.py, r4_denseprobe_queue.py",
     "the --probe-dense-until study launcher and its detached CPU queue",
     "-"),
    ("code/experiments/launch_r4_e2cpu.py",
     "the E2 Lorenz absgate-transfer CPU launcher",
     "-"),
    ("code/experiments/launch_r4_absgate.py",
     "the single detached, phase-ordered GPU queue for the E4 and E3 "
     "absgate-transfer studies",
     "-"),
    ("code/experiments/analyze_r4_absgate.py",
     "the absgate-transfer analysis script (writes round4_absgate_transfer.md)",
     "-"),
    ("code/experiments/_gatestats.py",
     "shared additive instrumentation (empirical |ghat_j| vs. certificate "
     "threshold c*beta_col_j distributions) behind --log-gate-stats in "
     "e2_timeseries.py / e3_continual.py / e4_gpt2_tta.py; inert when the "
     "flag is off, so every default path stays byte-identical",
     "-"),
    ("code/experiments/e2_timeseries.py (--scale-shift, --adaptive-mh, "
     "--gate-warmup, --probe-dense-until), e1_certificate.py (--adaptive-mh), "
     "e3_continual.py and e4_gpt2_tta.py (--method absgate, "
     "--log-gate-stats), code/experiments/data.py (apply_scale_shift), "
     "code/cohg/certificate.py (DriftHoldAdaptiveMH)",
     "the round-4 driver and library extensions; all new flags are additive "
     "and default to the legacy behaviour, verified bit-identical in "
     "results/e2_verify4/ and results/e1_verify4/",
     "-"),
]


NARRATIVE_MAP = [
    ("results/e1/SUMMARY.md", "code/experiments/analyze_e1.py"),
    ("results/e2/SUMMARY.md", "code/experiments/analyze_e2.py"),
    ("results/e2_controls/SUMMARY.md",
     "code/experiments/analyze_e2_controls.py"),
    ("results/reanalysis/unified_metrics.md (+ the three unified_metrics_*.csv)",
     "results/reanalysis/_reanalyze.py, then _key_questions.py"),
    ("results/reanalysis/censored_analysis.md (+ censored_e2_drift.csv)",
     "results/reanalysis/_reanalyze.py"),
    ("results/reanalysis/device_sensitivity.md",
     "code/experiments/analyze_device_sensitivity.py"),
    ("results/reanalysis/e3_noholdout.md",
     "results/reanalysis/_e3_noholdout.py"),
    ("results/reanalysis/gate_open_timing.md",
     "results/reanalysis/_fig8_gpt2_lambda.py"),
    ("results/reanalysis/e4_expansion.md",
     "analysis document written against results/e4_v2/ and "
     "results/e4_orders/; no generating script"),
    ("results/reanalysis/round3_cpu.md",
     "code/experiments/analyze_round3.py"),
    ("results/reanalysis/round3_gpu.md",
     "results/reanalysis/_round3_gpu.py"),
    ("results/reanalysis/misspec_curves.md",
     "code/experiments/analyze_misspec.py"),
    ("results/e4_fix/COMPARE.md",
     "code/experiments/compare_e4_fix.py"),
    ("results/e4_verify_all/COMPARE_ALL.md",
     "code/experiments/compare_e4_verify_all.py over results/e4_verify_all/, "
     "results/e4_fix/, results/e4_v2/ and results/e4_orders/"),
    ("results/reanalysis/common_stability.md",
     "results/reanalysis/_common_stability.py"),
    ("results/reanalysis/e3_trigger_censored.md",
     "results/reanalysis/_common_stability.py"),
]

# appended only when the file is on disk at build time
NARRATIVE_OPTIONAL = [
    (os.path.join("results", "reanalysis", "p4_verification.md"),
     ("results/reanalysis/p4_verification.md",
      "analysis document written against "
      "results/e4_verify_all/COMPARE_ALL.md; no generating script")),
]


def narrative_rows():
    rows = list(NARRATIVE_MAP)
    for guard, row in NARRATIVE_OPTIONAL:
        if os.path.exists(os.path.join(ROOT, guard)):
            rows.append(row)
    return rows


def write_manifest(dest, files, total, dropped, rev=None, revdate=None,
                   sums_digest=None):
    lines = []
    A = lines.append
    A("# COHG reproducibility bundle")
    A("")
    A("Assembled %s by `code/tools/make_repro_bundle.py`."
      % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    A("")
    A("Contents: %d files, %s." % (len(files), human(total)))
    if rev:
        A("")
        A("Code revision `%s` (newest source file %s). The revision is the "
          "content fingerprint of `code/` as bundled: it is stable across "
          "rebuilds and changes only when the code does." % (rev, revdate))
    if sums_digest:
        A("")
        A("`SHA256SUMS.txt` lists the SHA-256 of every file below, one per "
          "line, in `sha256sum -c` format. Its own SHA-256 is "
          "`%s`, which fingerprints the bundle as a whole." % sums_digest)
    A("")
    A("Every result JSON is the artifact written by the run that produced it, "
      "unmodified. One file is one (method, dataset, seed, configuration) run "
      "and carries its loss trace, so every reported mean, standard deviation "
      "(ddof=1) and permutation test can be recomputed from this bundle "
      "alone.")
    if dropped:
        A("")
        A("**Omission.** The bundle exceeded the size ceiling, so the "
          "following GPT-2 result directories were left out: %s. Everything "
          "else is present." % ", ".join(dropped))
    A("")
    A("## Layout")
    A("")
    A("| Path | What it holds |")
    A("| --- | --- |")
    A("| `code/cohg/` | estimator, certificate, controller, HVP oracle |")
    A("| `code/experiments/` | experiment drivers, launchers, analysis and "
      "figure scripts |")
    A("| `code/tools/` | dispatch, fleet and bundle tooling |")
    A("| `code/tests/` | unit tests |")
    A("| `results/<campaign>/` | raw per-run JSON artifacts and per-campaign "
      "`SUMMARY.md` |")
    A("| `results/reanalysis/` | post-hoc analyses, their CSV outputs and "
      "their scripts |")
    A("| `paper/main/figs/` | the print-size figures as they appear in the "
      "paper |")
    A("| `SHA256SUMS.txt` | SHA-256 of every file above |")
    A("")
    A("**Withheld.** %s are not included: they hold the SSH hosts, users and "
      "plaintext passwords of the rented machines the campaigns ran on. They "
      "are dispatch plumbing that copies code out and results back; no "
      "reported number depends on them, and the drivers they invoke are "
      "bundled in full."
      % ", ".join("`%s`" % c.replace("\\", "/") for c in CREDENTIAL_FILES))
    A("")
    A("## Round-3 additions")
    A("")
    A("Added after the round-2 bundle, in response to the round-3 review.")
    A("")
    A("| Path | What it holds | Produced by |")
    A("| --- | --- | --- |")
    for a, b, c in ROUND3_MAP:
        A("| `%s` | %s | %s |" % (a, b, c))
    A("")
    A("## Round-3.5/4 additions")
    A("")
    A("Added after the round-3 bundle, in response to the round-3.5 review: "
      "the P4 same-seed verification at full scope, the common cross-regime "
      "stability table, the trigger-censored E3 read-out, and the `M_obs` / "
      "`M_H` definitions behind fig:misspec.")
    A("")
    A("| Path | What it holds | Produced by |")
    A("| --- | --- | --- |")
    for a, b, c in round35_rows():
        A("| `%s` | %s | %s |" % (a, b, c))
    A("")
    A("## Round-4 additions")
    A("")
    A("Added after the round-3.5/4 bundle, answering three further reviewer "
      "objections: whether the E2 fixed-prior / static-threshold controls "
      "generalize off their calibration stream under a late amplitude shift, "
      "whether the drift bound `M_H` can be restated as an online-auditable "
      "quantity instead of a fixed offline constant, and whether the "
      "offline-calibrated static gate threshold transfers to a different "
      "stream, architecture or domain.")
    A("")
    A("| Path | What it holds | Produced by |")
    A("| --- | --- | --- |")
    for a, b, c in ROUND4_MAP:
        A("| `%s` | %s | %s |" % (a, b, c))
    A("")
    A("## Tables")
    A("")
    A("| Paper object | Reports | Regenerated by |")
    A("| --- | --- | --- |")
    for a, b, c in TABLE_MAP:
        A("| %s | %s | %s |" % (a, b, c))
    A("")
    A("## Figures")
    A("")
    A("| Paper object | Shows | Regenerated by |")
    A("| --- | --- | --- |")
    for a, b, c in FIGURE_MAP:
        A("| %s | %s | %s |" % (a, b, c))
    A("")
    A("## Summaries and reanalyses")
    A("")
    A("| Output | Regenerated by |")
    A("| --- | --- |")
    for a, b in narrative_rows():
        A("| `%s` | %s |" % (a, b))
    A("")
    A("## Order of regeneration")
    A("")
    A("1. Runs: `code/experiments/launch_*.py` write `results/<campaign>/`. "
      "A launcher skips a configuration whose artifact already exists, so a "
      "campaign can be resumed.")
    A("2. Per-campaign analyses: `analyze_e1.py`, `analyze_e2.py`, "
      "`analyze_e3.py`, `analyze_e2_controls.py`, `analyze_cal.py`.")
    A("3. Reanalyses: `results/reanalysis/_reanalyze.py`, then "
      "`_key_questions.py`, then `_e3_noholdout.py`, "
      "`analyze_device_sensitivity.py`, `_fig8_gpt2_lambda.py`, "
      "`_fig9_gpt2_order.py`.")
    A("4. Round-3 campaigns: `code/experiments/launch_round3.py` (E2 CPU "
      "parts `verify`, `b1`, `b2`, `b3`), `code/experiments/launch_r3_chain.py`"
      " (E4 GPU chain), `code/experiments/r3_cpu_queue.py` (traced E3), then "
      "`code/experiments/analyze_round3.py` and "
      "`results/reanalysis/_round3_gpu.py`, then "
      "`code/experiments/analyze_misspec.py` and "
      "`code/experiments/make_fig10_misspec.py`.")
    A("5. Round-3.5/4 verification and cross-regime read-outs: "
      "`code/experiments/launch_e4_verify_all.py` (the same-seed E4 reruns, "
      "monitored with `code/experiments/watch_e4_verify_all.py`), then "
      "`code/experiments/compare_e4_verify_all.py`, then "
      "`results/reanalysis/_common_stability.py`.")
    A("6. Round-4 campaigns: `code/experiments/launch_round4.py` (E2/E1 CPU "
      "parts `verify`, `shift`, `adaptmh`) with "
      "`code/experiments/r4_cpu_queue.py`; `code/experiments/"
      "launch_r4_warmup.py` with `r4_warmup_queue.py`; `code/experiments/"
      "launch_r4_denseprobe.py` with `r4_denseprobe_queue.py`; "
      "`code/experiments/launch_r4_e2cpu.py` (Lorenz absgate transfer) and "
      "`code/experiments/launch_r4_absgate.py` (E3/E4 absgate transfer, "
      "GPU); then `results/reanalysis/_round4_shift.py`, "
      "`_round4_adaptmh.py`, `_round4_warmup.py`, `_round4_denseprobe.py`, "
      "and `code/experiments/analyze_r4_absgate.py`.")
    A("7. Appendix tables: `code/experiments/gen_appendix_tables.py`.")
    A("8. Figures: `code/experiments/make_paper_figures_nc.py`.")
    A("")
    A("All aggregate statistics use the sample standard deviation "
      "(`ddof=1`). Significance is the two-sided paired exact permutation "
      "test over all $2^n$ sign assignments of the per-seed differences.")
    A("")
    with open(os.path.join(dest, "MANIFEST.md"), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))


# --- driver -----------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default=DEST)
    ap.add_argument("--max-gb", type=float, default=2.0,
                    help="drop the GPT-2 result directories above this size")
    ap.add_argument("--clean", action="store_true",
                    help="delete the destination first")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-secrets", action="store_true",
                    help="copy even if the secret scan finds a hit (never use "
                         "for a bundle that leaves this machine)")
    args = ap.parse_args(argv)

    files = []
    for tree in CODE_TREES:
        files.extend(plan_tree(tree))
    files.extend(plan_results())
    files.extend(plan_tree(FIG_TREE))

    total = sum(s for _, _, s in files)
    dropped = []
    ceiling = args.max_gb * (1024.0 ** 3)
    if total > ceiling:
        keep = []
        for ap_, rel, sz in files:
            parts = rel.replace("\\", "/").split("/")
            if len(parts) > 1 and parts[0] == "results" and parts[1] in GPT2_DIRS:
                if parts[1] not in dropped:
                    dropped.append(parts[1])
                continue
            keep.append((ap_, rel, sz))
        files = keep
        total = sum(s for _, _, s in files)

    print("planned: %d files, %s" % (len(files), human(total)))
    if dropped:
        print("dropped (over %.1f GB): %s" % (args.max_gb, ", ".join(dropped)))

    hits = scan_secrets(files)
    if hits:
        print("SECRET SCAN: %d suspect line(s)" % len(hits))
        for rel, ln, pat in hits[:20]:
            print("  %s:%d  /%s/" % (rel, ln, pat))
        if not args.allow_secrets:
            print("refusing to build; withhold the file (CREDENTIAL_FILES) or "
                  "pass --allow-secrets if it is a false positive")
            return 2
    else:
        print("secret scan: clean (%d text files checked)"
              % sum(1 for _a, r, s in files
                    if os.path.splitext(r)[1].lower() in SECRET_SCAN_EXT
                    and s <= SECRET_SCAN_MAX_BYTES))

    if args.dry_run:
        return 0

    if args.clean and os.path.isdir(args.dest):
        shutil.rmtree(args.dest)
    os.makedirs(args.dest, exist_ok=True)

    copied = 0
    for src, rel, _sz in files:
        dst = os.path.join(args.dest, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
        if copied % 500 == 0:
            print("  ... %d files" % copied)

    rev, revdate = code_revision()
    print("code revision %s (newest source file %s)" % (rev, revdate))

    _sums_path, sums_digest = write_checksums(args.dest, files)
    print("wrote SHA256SUMS.txt (%d entries, digest %s)"
          % (len(files), sums_digest[:16]))

    write_manifest(args.dest, files, total, dropped, rev=rev, revdate=revdate,
                   sums_digest=sums_digest)

    on_disk = 0
    nfiles = 0
    for dirpath, _dn, fns in os.walk(args.dest):
        for fn in fns:
            on_disk += os.path.getsize(os.path.join(dirpath, fn))
            nfiles += 1
    print("bundle: %s" % args.dest)
    print("wrote %d files, %s" % (nfiles, human(on_disk)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
