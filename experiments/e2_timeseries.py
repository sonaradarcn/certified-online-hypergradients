"""E2: online time-series meta-adaptation main comparison (tracker R020-R023).

Prequential protocol: at each step the incoming batch is (i) evaluated at the
current theta_t (this is the reported online loss AND the meta-objective
ell_t), then (ii) trained on with per-group SGD. Methods differ only in how
lambda (per-group log-LRs) evolves.

Methods:
    fixed          -- lambda frozen (grid over --lr gives the fixed family)
    hd / hd_scalar -- Baydin et al. 2018, per-group / scalar
    hdm            -- Gao et al. 2025 (null step + AdaGrad), per-group
    tfmd           -- truncated FMD (reset every K_trunc), compute-rich
    fmd            -- exact FMD oracle (gamma-discounted target for parity)
    cohg           -- full: sketch r, lazy K, discount gamma, KW tier, gate
    cohg_r0        -- ablation: no sketch
    cohg_nogate    -- ablation: updates always applied (no certificate gate);
                      already a PURE SIGN step of size --meta-lr (= alpha)
    t6clip         -- control E: COHG + Theorem-6 online step-size condition
    absgate        -- control B: gate on |ghat_j| > const threshold
    randgate       -- control C: gate opens i.i.d. with prob --randgate-p
    periodicgate   -- control D: gate opens every --periodic-every steps
    madgate        -- round-3 B1: calibration-free gate |ghat_j| > c * MAD_t,
                      MAD_t = running median-absolute-deviation of |ghat_j|
                      over the last --madgate-window steps (no certificate,
                      no calibration seeds, same c and same alpha-sign step)
    ogd_doubling   -- round-3 B2: cohg_ogd with the PROSPECTIVE projected-OGD
                      step size alpha_t = D / (G_k sqrt(tau)) under the
                      standard doubling trick (see below)

Matched-conservatism controls (B/C/D) use the SAME hypergradient estimator and
the same alpha-sign step as COHG; only the gate rule changes, and its rate is
matched to COHG's measured coord_open_frac.  They skip the spectral probe
(their gate never reads the certificate), which leaves the estimator recursion
identical (rho/kappa only feed the certificate, not S_hat).

Instability accounting: event when loss > 10x running median (window 500) or
non-finite; on non-finite loss, theta resets to last finite checkpoint and
lambda resets to init (event counted, run continues).

One process = one (method, dataset, seed) config -> results/e2/<...>.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cohg import (COHGEstimator, CoordGatedController, ExactFMD, GroupSpec,
                  T6ClipController)
from cohg.baselines import HDBaseline, HDMBaseline, TruncatedFMD
from cohg.certificate import (DriftHold, DriftHoldAdaptiveMH,
                             DriftHoldFailClosed, SpectralKW)
from cohg.functional import FlatModule
from cohg.hvp import HVPOracle

import data as D
import models as M
import _gatestats as GS

LAM_MIN, LAM_MAX = math.log(1e-5), math.log(1.0)

# controls B/C/D: same estimator + alpha-sign step, non-certificate gate.
# They never read the certificate, so the spectral probe is skipped.
CONTROL_METHODS = ("absgate", "randgate", "periodicgate", "madgate")
EST_METHODS = ("cohg", "cohg_r0", "cohg_nogate", "cohg_ogd", "ogd_doubling",
               "t6clip") + CONTROL_METHODS
# controllers that take the magnitude-aware projected-OGD step (mode="ogd")
OGD_METHODS = ("cohg_ogd", "ogd_doubling")


def build_stream(dataset: str, seed: int, device: str, total_steps: int,
                 scale_shift=None):
    """`scale_shift=F` multiplies the MIDDLE third of the series (inputs AND
    targets) by F -- the round-4 late amplitude shift.  Default None/1 leaves
    the series bit-identical to the legacy path."""
    n = 25_000
    if dataset == "mackey":
        series = D.mackey_glass(n, seed=seed)
    elif dataset == "lorenz":
        series = D.lorenz_x(n, seed=seed)
    elif dataset == "sunspot":
        series = D.sunspot_series(n)
    elif dataset == "santafe":
        series = D.santa_fe_laser(n)
    elif dataset == "mackey_drift":
        series = D.mackey_glass_drift(n, seed=seed)
        series = D.apply_scale_shift(series, scale_shift)
        return D.OrderedWindowStream(series, total_steps, window=20,
                                     batch_size=64, seed=seed + 1,
                                     device=device)
    elif dataset == "lorenz_drift":
        series = D.lorenz_x_drift(n, seed=seed)
        series = D.apply_scale_shift(series, scale_shift)
        return D.OrderedWindowStream(series, total_steps, window=20,
                                     batch_size=64, seed=seed + 1,
                                     device=device)
    else:
        raise ValueError(dataset)
    series = D.apply_scale_shift(series, scale_shift)
    return D.WindowStream(series, window=20, batch_size=64, seed=seed + 1,
                          device=device)


def segment_bounds(stream, total_steps: int):
    """Stream-step indices at which the 2nd / 3rd third of the SERIES first
    enters a batch.  For `mackey_drift` (n=25000, T=12000, window=20) these are
    the regime-switch steps 4004 and 8007 quoted in round3_cpu.md, and they are
    also where `--scale-shift` turns on and off.  None for shuffled streams."""
    if not isinstance(stream, D.OrderedWindowStream):
        return None
    seg = len(stream.x) // 3
    n_win, T = stream.n, total_steps
    out = []
    for target in (seg, 2 * seg):
        t = next((t for t in range(T)
                  if int(t / max(T - 1, 1) * (n_win - 1)) >= target), T)
        out.append(t)
    return out


def plain_grad(loss_fn, theta, batch):
    th = theta.detach().requires_grad_(True)
    loss = loss_fn(th, batch)
    (g,) = torch.autograd.grad(loss, th)
    return loss.detach(), g.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--dataset", required=True,
                    choices=["mackey", "lorenz", "sunspot", "santafe",
                             "mackey_drift", "lorenz_drift"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--lr", type=float, default=0.03, help="initial LR / fixed LR")
    ap.add_argument("--meta-lr", type=float, default=0.02)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--kw-eps", type=float, default=0.1)
    ap.add_argument("--kw-delta", type=float, default=0.01)
    ap.add_argument("--probe-every", type=int, default=20,
                    help="KW spectral probe cadence; A4 drift-hold (DriftHold) "
                    "certifies the gaps via observable path length")
    ap.add_argument("--M-H", type=float, default=5.0,
                    help="Hessian Lipschitz prior for DriftHold")
    ap.add_argument("--gate-factor", type=float, default=2.0,
                    help="gate threshold c (M4 ablation)")
    ap.add_argument("--tag", default="", help="suffix for the result filename")
    ap.add_argument("--K-trunc", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=None)
    # ---- e2_controls additions (certification vs generic conservatism) ----
    ap.add_argument("--rho-f", type=float, default=1.0,
                    help="control E (t6clip): prior meta-objective smoothness "
                    "in the Theorem-6 step condition")
    ap.add_argument("--absgate-threshold", type=float, default=None,
                    help="control B: constant |ghat_j| gate threshold; if "
                    "omitted the run is a CALIBRATION run (gate never opens, "
                    "the |ghat| upper order statistics are dumped)")
    ap.add_argument("--randgate-p", type=float, default=None,
                    help="control C: i.i.d. per-coordinate open probability")
    ap.add_argument("--periodic-every", type=int, default=None,
                    help="control D: open every N steps (all coordinates)")
    ap.add_argument("--fail-closed", action="store_true",
                    help="control F: fail-closed M_H monitor (DriftHold "
                    "subclass): re-probe at 0.5*M_H, close the gate above M_H")
    ap.add_argument("--validate-cert", action="store_true",
                    help="control F: run ExactFMD alongside and count "
                    "per-coordinate certificate violations "
                    "|ghat_j - g_true_j| > beta_col_j (costs m HVPs/step)")
    # ---- round-3 additions (B1 madgate / B2 doubling / B3 full horizon) ----
    ap.add_argument("--madgate-window", type=int, default=200,
                    help="B1 (madgate): running MAD window over |ghat_j|")
    ap.add_argument("--madgate-warmup", type=int, default=50,
                    help="B1 (madgate): steps with the gate held shut while "
                    "the MAD window fills")
    # ---- round-4 additions (scale shift / adaptive M_H envelope) ----
    ap.add_argument("--scale-shift", type=float, default=None,
                    help="round-4 E1: multiply the middle third of the series "
                    "(inputs AND targets) by this amplitude factor -- a LATE "
                    "distribution shift that changes the right learning rate. "
                    "Default None = legacy bit-identical path.")
    ap.add_argument("--adaptive-mh", type=float, default=None,
                    help="round-4 E2: online-enforceable drift envelope. At "
                    "every probe the envelope in force becomes "
                    "M_H,t = max(M_H_floor, KAPPA * max_{s<=t} M_obs,s) with "
                    "M_H_floor = --M-H (the deployed prior). Implies the "
                    "fail-closed monitor. Default None = fixed prior.")
    ap.add_argument("--gate-warmup", default="off",
                    choices=["off", "first-obs", "stable-env"],
                    help="round-4 E2 follow-up: hold the certificate gate "
                    "SHUT until the drift envelope is backed by a "
                    "measurement. 'first-obs': no coordinate may open before "
                    "the first probe-to-probe M_obs has been recorded (step "
                    "2*probe_every on the standard config). 'stable-env': "
                    "additionally require that the most recent probe did NOT "
                    "raise the envelope. 'off' (default) is the legacy "
                    "bit-identical path.")
    ap.add_argument("--probe-dense-until", type=int, default=0,
                    help="round-4 E2 follow-up (remedy b): DENSE early probe "
                    "schedule.  For t <= T the KW spectral probe fires at "
                    "EVERY step (in addition to the --probe-every cadence), "
                    "so a probe-to-probe drift observation M_obs exists from "
                    "step 1 instead of step 2*probe_every.  After T the "
                    "schedule reverts to --probe-every.  Default 0 = off = "
                    "the legacy bit-identical path.")
    ap.add_argument("--lam-every", type=int, default=50,
                    help="lam_hist sampling cadence (logging only)")
    ap.add_argument("--log-gate-stats", action="store_true",
                    help="round-4 additive instrumentation: pool |ghat_j| and "
                         "the certificate-scaled threshold c*beta_col_j over "
                         "every (step, coordinate) pair and write their "
                         "summary under the key 'gate_stats'.  Read-only.")
    ap.add_argument("--validate-full", action="store_true",
                    help="B3: run a second EXACT FMD at gamma=1 (full-horizon "
                    "sensitivity S_t) alongside and log per-step, "
                    "per-coordinate sign agreement of the estimated "
                    "hypergradient with the full-horizon one, split by gate "
                    "state (costs a further m HVPs/step, NOT charged to "
                    "hvp_total)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.adaptive_mh is not None:
        # the adaptive envelope is defined WITH the fail-closed monitor on
        args.fail_closed = True
    device = args.device
    stream = build_stream(args.dataset, args.seed, device, args.steps,
                          scale_shift=args.scale_shift)
    seg_bounds = segment_bounds(stream, args.steps)
    model = M.ManualGRU(n_in=1, n_hidden=64, n_out=1, n_layers=1).to(device)
    fm = FlatModule(model, lambda out, y: 0.5 * ((out - y) ** 2).mean())
    spec = fm.group_spec()          # per parameter tensor (m=6)
    p, m = spec.p, spec.m
    theta = fm.flat_params()
    lam = torch.full((m,), math.log(args.lr), device=device)
    lam_init = lam.clone()

    gstats = GS.GateStats(args.log_gate_stats, gate_factor=args.gate_factor)

    # ---- method state ----
    est = kw = ctrl = hd = hdm = tf = fmd = None
    fmd_ref = None
    rho_hold = (1.0, 0.0)
    if args.method in EST_METHODS:
        r = 0 if args.method == "cohg_r0" else args.rank
        est = COHGEstimator(spec, rank=r, refresh_every=args.K,
                            device=device, dtype=torch.float32,
                            discount=args.gamma)
        kw = SpectralKW(p, eps=args.kw_eps, delta=args.kw_delta,
                        seed=args.seed + 7)
        if args.adaptive_mh is not None:
            dh = DriftHoldAdaptiveMH(M_H_floor=args.M_H,
                                     kappa=args.adaptive_mh)
        elif args.fail_closed:
            dh = DriftHoldFailClosed(M_H=args.M_H)
        else:
            dh = DriftHold(M_H=args.M_H)
        if args.method == "t6clip":
            ctrl = T6ClipController(
                args.meta_lr, gate_factor=args.gate_factor, trust=1.0,
                lam_min=LAM_MIN, lam_max=LAM_MAX, mode="sign",
                rho_f=args.rho_f)
        else:
            ctrl = CoordGatedController(
                args.meta_lr, gate_factor=args.gate_factor, trust=1.0,
                lam_min=LAM_MIN, lam_max=LAM_MAX,
                mode="ogd" if args.method in OGD_METHODS else "sign")
        if args.method == "randgate":
            if args.randgate_p is None:
                raise ValueError("randgate requires --randgate-p")
            rg_gen = torch.Generator().manual_seed(args.seed + 991)
        if args.method == "periodicgate" and args.periodic_every is None:
            raise ValueError("periodicgate requires --periodic-every")
        if args.method == "madgate":
            mad_win = deque(maxlen=args.madgate_window)
        if args.method == "ogd_doubling":
            # prospective projected-OGD schedule (Zinkevich/doubling trick):
            #   alpha_tau = D / (G_k sqrt(tau)),  D = box width (lam range),
            #   tau = 1..2^k inside doubling epoch k, and G_k = running max of
            #   |ghat_j| + beta_j over ALL steps strictly before the epoch
            #   started (so the step size is prospective, never peeking at the
            #   current gradient).  Epoch k covers steps [2^k - 1, 2^(k+1) - 1).
            ogd_D = LAM_MAX - LAM_MIN
            ogd_Gmax = torch.zeros(m, device=device)      # running max so far
            ogd_Gk = None                                 # frozen at epoch start
            ogd_epoch = 0
            ogd_epoch_start = 0
            ogd_alpha_log = []
    elif args.method == "hd":
        hd = HDBaseline(spec, args.meta_lr, scalar=False,
                        lam_min=LAM_MIN, lam_max=LAM_MAX)
    elif args.method == "hd_scalar":
        hd = HDBaseline(spec, args.meta_lr, scalar=True,
                        lam_min=LAM_MIN, lam_max=LAM_MAX)
    elif args.method == "hdm":
        hdm = HDMBaseline(spec, meta_lr=args.meta_lr,
                          lam_min=LAM_MIN, lam_max=LAM_MAX)
    elif args.method == "tfmd":
        tf = TruncatedFMD(spec, args.meta_lr, K_trunc=args.K_trunc,
                          device=device, dtype=torch.float32,
                          lam_min=LAM_MIN, lam_max=LAM_MAX)
    elif args.method == "fmd":
        fmd = ExactFMD(spec, device=device, dtype=torch.float32,
                       discount=args.gamma)
    elif args.method != "fixed":
        raise ValueError(args.method)

    if args.validate_cert:
        if est is None:
            raise ValueError("--validate-cert needs an estimator method")
        fmd_ref = ExactFMD(spec, device=device, dtype=torch.float32,
                           discount=args.gamma)

    fmd_full = None
    if args.validate_full:
        if est is None:
            raise ValueError("--validate-full needs an estimator method")
        fmd_full = ExactFMD(spec, device=device, dtype=torch.float32,
                            discount=1.0)

    # ---- stream loop ----
    losses, lam_hist, events = [], [], 0
    med_win = deque(maxlen=500)
    ckpt = theta.clone()
    gate_opens = gate_total = 0
    hvp_total = 0
    y_var_acc, y_sq_acc, n_y = 0.0, 0.0, 0
    # round-4: per-segment target moments + loss moments (segments are the
    # thirds of the ORDERED stream; also the --scale-shift boundaries)
    n_seg = 3 if seg_bounds else 1
    seg_y_sum = [0.0] * n_seg
    seg_y_sq = [0.0] * n_seg
    seg_y_n = [0] * n_seg
    seg_loss_sum = [0.0] * n_seg
    seg_loss_n = [0] * n_seg
    seg_nonfinite = [0] * n_seg
    # round-4: dense gate-opening record
    gate_open_steps = []          # steps at which ANY coordinate opened
    gate_open_env = []            # drift envelope in force at those steps
    gate_open_events = []         # [t, j, sign] per open coordinate-step
    gate_open_ev_trunc = False
    GATE_EV_CAP = 20000
    coord_open_counts = None      # per-coordinate cumulative open count
    seg_coord_open = [0] * n_seg  # per-segment OPEN coordinate-steps
    seg_coord_total = [0] * n_seg  # per-segment total coordinate-steps
    seg_steps_open = [0] * n_seg  # per-segment steps with >=1 open
    seg_closed = [0] * n_seg      # per-segment fail-closed steps
    n_probes = 0
    adapt_probe_log = []          # [t, M_H_before, M_obs, M_H_after, closed]
    n_probe_log = 0
    # --gate-warmup: probe-level M_obs record (kept for EVERY monitored arm so
    # the fixed-prior reference can be audited the same way), plus the
    # warm-up hold accounting.  Pure instrumentation.
    probe_mobs_log = []           # [t, M_obs or None] at every probe
    # --probe-dense-until: raw probe record so an M_obs over ANY probe pair
    # can be recomputed offline (1-step vs 20-step differences on the SAME
    # run).  [t, rho, kappa, path_since_prev_probe, eta_max_at_prev_probe,
    #          M_H_in_force_after, closed_after]
    probe_raw_log = []
    n_mobs_log = 0
    last_probe_raised = False     # did the most recent probe raise M_H?
    warmup_held_steps = 0         # steps the warm-up hold kept the gate shut
    warmup_supp_steps = 0         # ... of which >=1 coordinate WOULD have opened
    warmup_supp_coord = 0         # suppressed OPEN coordinate-steps
    warmup_supp_log = []          # [t, n_coord_would_open] on those steps
    warmup_release_t = None       # first step at which the hold was lifted
    # control B/C/D per-coordinate gate accounting (mirrors ctrl's)
    ctl_open = ctl_total = 0
    ghat_abs_pool = []            # calibration dump for control B
    mad_stat_sum, mad_stat_n = 0.0, 0   # mean MAD threshold base (control B1)
    open_mask_log = None
    # --validate-cert accounting
    cert_checked = cert_viol = cert_viol_steps = 0
    cert_max_ratio = 0.0
    cert_max_excess = 0.0
    cert_ratio_hist = []
    fc_closed_steps = 0
    force_probe = False
    # --validate-full accounting (B3): per-step coordinate counts, so the
    # post-regime-switch windows can be cut offline
    full_finite = []          # bool: S_full (gamma=1) still numerically finite
    full_agree = []           # #coords with sign(ghat) == sign(g_full)
    full_open = []            # #coords the gate opened this step
    full_open_agree = []      # #coords both open AND sign-agreeing
    full_nz = []              # #coords with both signs nonzero
    full_nz_agree = []        # ... of which agreeing
    full_disc_agree = []      # sign(g_disc_exact) vs sign(g_full) (needs
                              # --validate-cert too); else empty
    full_disc_nz = []
    full_disc_nz_agree = []
    full_blown = False        # S_full overflowed -> stop paying its HVPs
    t0 = time.time()

    for t in range(args.steps):
        batch = stream.next()
        _, y = batch
        ys, yq, yn = float(y.sum()), float((y ** 2).sum()), y.numel()
        y_var_acc += ys
        y_sq_acc += yq
        n_y += yn
        if seg_bounds:
            si = 0 if t < seg_bounds[0] else (1 if t < seg_bounds[1] else 2)
        else:
            si = 0
        seg_y_sum[si] += ys
        seg_y_sq[si] += yq
        seg_y_n[si] += yn

        if args.method == "hdm":
            theta, lam, loss_val = hdm.step(fm.loss_fn, theta, batch, lam)
            eta = None
        else:
            # --probe-dense-until: probe at EVERY step while t <= T so a
            # probe-to-probe M_obs exists before the adaptation window.
            dense_probe = (args.probe_dense_until > 0
                           and t <= args.probe_dense_until)
            do_probe = (est is not None
                        and args.method not in CONTROL_METHODS
                        and args.method != "cohg_nogate"
                        and (t % args.probe_every == 0 or force_probe
                             or dense_probe))
            need_graph = (est is not None and
                          (t % args.K == 0 or do_probe or
                           # unchanged legacy schedule for the pre-existing
                           # arms (cohg_nogate builds the oracle on probe
                           # steps and ignores it) -- keeps old runs bit-exact
                           (args.method not in CONTROL_METHODS
                            and t % args.probe_every == 0))) or \
                         (tf is not None) or (fmd is not None) or \
                         (fmd_ref is not None) or \
                         (fmd_full is not None and not full_blown)
            if need_graph:
                oracle = HVPOracle(fm.loss_fn, theta, batch)
                loss, g = oracle.loss, oracle.grad
            else:
                oracle = None
                loss, g = plain_grad(fm.loss_fn, theta, batch)
            loss_val = float(loss)

            if est is not None:
                eta = spec.eta_vec(lam)
                if args.method == "cohg_nogate" or \
                        args.method in CONTROL_METHODS:
                    # no certificate gate -> certificate unused; skip spectral
                    # probes entirely (nominal inputs keep the estimator
                    # recursion identical; e_col becomes meaningless/ignored)
                    rho, kappa = 1.0, 0.0
                elif do_probe:
                    was_scheduled = (t % args.probe_every == 0
                                     or dense_probe)
                    _d_prev = float(dh.path)
                    _etamax_prev = dh.eta_max0
                    pre_hvp = oracle.n_hvp
                    rho, kappa = kw.bounds(oracle.hvp, eta)
                    hvp_total += oracle.n_hvp - pre_hvp
                    _nr_before = int(getattr(dh, "n_raises", 0))
                    dh.probe(rho, kappa, eta_vec=eta)
                    n_probes += 1
                    last_probe_raised = (
                        int(getattr(dh, "n_raises", 0)) > _nr_before)
                    _mo = getattr(dh, "m_obs", None)
                    if _mo is not None:
                        if len(_mo) > n_mobs_log:
                            n_mobs_log = len(_mo)
                            probe_mobs_log.append([t, float(_mo[-1])])
                        else:
                            probe_mobs_log.append([t, None])
                    probe_raw_log.append(
                        [t, float(rho), float(kappa), _d_prev,
                         (float(_etamax_prev) if _etamax_prev else None),
                         float(getattr(dh, "M_H", args.M_H)),
                         int(bool(getattr(dh, "closed", False)))])
                    if args.adaptive_mh is not None:
                        pl = dh.probe_log
                        if len(pl) > n_probe_log:
                            n_probe_log = len(pl)
                            adapt_probe_log.append([t] + list(pl[-1]))
                        else:
                            adapt_probe_log.append([t, dh.M_H, None, dh.M_H,
                                                    int(dh.closed)])
                    # a forced re-probe never forces another one: the probe
                    # budget is at most doubled
                    force_probe = bool(getattr(dh, "want_reprobe", False)
                                       and was_scheduled)
                    rho, kappa = dh.bounds(eta)
                else:
                    rho, kappa = dh.bounds(eta)
                ghat = est.hypergrad(g)
                gnorm_meta = float(torch.linalg.vector_norm(g))
                beta_col = est.beta_col(gnorm_meta)
                gstats.add(ghat,
                           None if (args.method == "cohg_nogate"
                                    or args.method in CONTROL_METHODS)
                           else beta_col)
                if fmd_ref is not None:
                    # certificate audit: |ghat_j - g_true_j| <= beta_col_j.
                    # Both recursions run in fp32, so the comparison carries a
                    # float tolerance (1e-4 relative to the hypergradient
                    # scale); without it the single early step where the
                    # certificate is still EXACTLY zero flags ~1e-10 round-off.
                    g_true = fmd_ref.hypergrad(g).to(ghat.dtype)
                    err = (ghat - g_true).abs().double().cpu()
                    bc = beta_col.double().cpu()
                    scale = torch.maximum(ghat.abs(), g_true.abs()).double().cpu()
                    tol = 1e-4 * scale + 1e-9
                    cert_checked += int(err.numel())
                    nv = int((err > bc + tol).sum())
                    cert_viol += nv
                    cert_viol_steps += int(nv > 0)
                    step_ratio = float((err / (bc + tol)).max())
                    cert_ratio_hist.append(step_ratio)
                    cert_max_ratio = max(cert_max_ratio, step_ratio)
                    cert_max_excess = max(cert_max_excess,
                                          float((err - bc).max()))
                g_true = None if fmd_ref is None else g_true
                if args.method == "ogd_doubling":
                    # ---- B2: prospective doubling step size --------------
                    # advance the epoch BEFORE using G (G_k must be frozen at
                    # the max observed strictly before the epoch began)
                    if t >= (2 ** (ogd_epoch + 1) - 1):
                        ogd_epoch += 1
                        ogd_epoch_start = 2 ** ogd_epoch - 1
                        ogd_Gk = None
                    bc_dev = beta_col.to(ogd_Gmax.dtype).to(ogd_Gmax.device)
                    if ogd_Gk is None:
                        # coordinates with no history yet (first epochs)
                        # bootstrap G from the current observation
                        cur_G = ghat.abs().to(ogd_Gmax.dtype) + bc_dev
                        ogd_Gk = torch.where(ogd_Gmax > 0, ogd_Gmax, cur_G)
                    tau = t - ogd_epoch_start + 1
                    Gk = ogd_Gk.clamp_min(1e-12)
                    ogd_alpha = ogd_D / (Gk * math.sqrt(tau))
                    ctrl.meta_lr = ogd_alpha
                    if t % 50 == 0:
                        ogd_alpha_log.append(
                            [t, float(ogd_alpha.min()), float(ogd_alpha.max())])
                    # update the running max AFTER the step size is fixed
                    ogd_Gmax = torch.maximum(
                        ogd_Gmax, ghat.abs().to(ogd_Gmax.dtype) + bc_dev)

                # ---- --gate-warmup: is the envelope backed by data yet? --
                warm_hold = False
                if args.gate_warmup != "off" and est is not None                         and args.method not in CONTROL_METHODS                         and args.method != "cohg_nogate":
                    _seen = len(getattr(dh, "m_obs", None) or ())
                    warm_hold = (_seen < 1)
                    if args.gate_warmup == "stable-env":
                        warm_hold = warm_hold or bool(last_probe_raised)
                    if not warm_hold and warmup_release_t is None:
                        warmup_release_t = t

                if args.method == "cohg_nogate":
                    # same normalized step, gate forced open (ablation)
                    mag = ghat.abs().clamp_min(1e-30)
                    lam = (lam - args.meta_lr * torch.sign(ghat)
                           ).clamp(LAM_MIN, LAM_MAX)
                    gate_opens += 1
                    open_mask_log = torch.ones_like(ghat, dtype=torch.bool)
                elif args.method in CONTROL_METHODS:
                    if args.method == "absgate":
                        # always dump the upper order statistics of |ghat|
                        # (threshold calibration / realized-rate refit)
                        ghat_abs_pool.extend(ghat.abs().double().cpu().tolist())
                        if len(ghat_abs_pool) > 20000:
                            ghat_abs_pool = sorted(
                                ghat_abs_pool, reverse=True)[:400]
                        if args.absgate_threshold is None:
                            # calibration run: gate never opens
                            open_mask = torch.zeros_like(ghat,
                                                         dtype=torch.bool)
                        else:
                            open_mask = ghat.abs() > args.absgate_threshold
                    elif args.method == "randgate":
                        open_mask = (torch.rand(spec.m, generator=rg_gen)
                                     < args.randgate_p).to(ghat.device)
                    elif args.method == "madgate":
                        # B1: calibration-free per-coordinate threshold.
                        # MAD_t(j) = median_w |x - median_w(x)| over the last
                        # `window` values of x = |ghat_j| (STRICTLY past: the
                        # current step is appended after the test), gate held
                        # shut for the first `warmup` steps.
                        amag = ghat.abs().detach().to(torch.float64).cpu()
                        if len(mad_win) >= 2 and t >= args.madgate_warmup:
                            W = torch.stack(list(mad_win))       # (w, m)
                            med = W.median(dim=0).values
                            mad = (W - med).abs().median(dim=0).values
                            mad_thr = args.gate_factor * mad
                            open_mask = (amag > mad_thr).to(ghat.device)
                            mad_stat_sum += float(mad.sum())
                            mad_stat_n += 1
                        else:
                            open_mask = torch.zeros_like(ghat,
                                                         dtype=torch.bool)
                        mad_win.append(amag)
                    else:  # periodicgate
                        open_mask = torch.full_like(
                            ghat, float(t % args.periodic_every == 0)
                        ).bool()
                    open_mask_log = open_mask
                    ctl_total += int(open_mask.numel())
                    ctl_open += int(open_mask.sum())
                    if bool(open_mask.any()):
                        step = args.meta_lr * torch.sign(ghat)
                        lam = (lam - torch.where(
                            open_mask, step, torch.zeros_like(step))
                        ).clamp(LAM_MIN, LAM_MAX)
                        gate_opens += 1
                elif args.fail_closed and dh.closed:
                    # fail-closed: prior violated -> no coordinate may move
                    fc_closed_steps += 1
                    seg_closed[si] += 1
                    ctrl.n_coord_total += int(ghat.numel())
                    ctrl.n_steps += 1
                    open_mask_log = torch.zeros_like(ghat, dtype=torch.bool)
                elif warm_hold:
                    # --gate-warmup: the envelope is not yet backed by a
                    # measurement -> no coordinate may move.  Count what the
                    # certificate gate WOULD have opened (read-only).
                    warmup_held_steps += 1
                    would = (ghat.abs() > args.gate_factor
                             * beta_col.to(ghat.dtype).to(ghat.device))
                    nw = int(would.sum())
                    if nw:
                        warmup_supp_steps += 1
                        warmup_supp_coord += nw
                        warmup_supp_log.append([t, nw])
                    ctrl.n_coord_total += int(ghat.numel())
                    ctrl.n_steps += 1
                    open_mask_log = torch.zeros_like(ghat, dtype=torch.bool)
                else:
                    open_mask_log = (ghat.abs() > args.gate_factor
                                     * beta_col.to(ghat.dtype).to(ghat.device))
                    lam, opened = ctrl.maybe_update(lam, ghat, beta_col)
                    gate_opens += int(opened)
                if fmd_full is not None:
                    # ---- B3: discounted-vs-full-horizon sign agreement ----
                    if full_blown:
                        full_finite.append(False)
                        for L_ in (full_agree, full_open_agree, full_nz,
                                   full_nz_agree, full_disc_agree,
                                   full_disc_nz, full_disc_nz_agree):
                            L_.append(0)
                        full_open.append(int(open_mask_log.sum()))
                    else:
                        g_fu = fmd_full.hypergrad(g).to(ghat.dtype)
                        ok = bool(torch.isfinite(g_fu).all())
                        full_finite.append(ok)
                        om = open_mask_log
                        full_open.append(int(om.sum()))
                        if ok:
                            sh, sf = torch.sign(ghat), torch.sign(g_fu)
                            ag = (sh == sf)
                            nz = (sh != 0) & (sf != 0)
                            full_agree.append(int(ag.sum()))
                            full_open_agree.append(int((ag & om).sum()))
                            full_nz.append(int(nz.sum()))
                            full_nz_agree.append(int((ag & nz).sum()))
                            if g_true is not None:
                                sd = torch.sign(g_true.to(ghat.dtype))
                                agd = (sd == sf)
                                nzd = (sd != 0) & (sf != 0)
                                full_disc_agree.append(int(agd.sum()))
                                full_disc_nz.append(int(nzd.sum()))
                                full_disc_nz_agree.append(int((agd & nzd).sum()))
                            else:
                                full_disc_agree.append(0)
                                full_disc_nz.append(0)
                                full_disc_nz_agree.append(0)
                        else:
                            for L_ in (full_agree, full_open_agree, full_nz,
                                       full_nz_agree, full_disc_agree,
                                       full_disc_nz, full_disc_nz_agree):
                                L_.append(0)
                # ---- round-4: dense gate-opening record --------------
                if open_mask_log is not None:
                    om = open_mask_log
                    if coord_open_counts is None:
                        coord_open_counts = [0] * int(om.numel())
                    seg_coord_total[si] += int(om.numel())
                    seg_coord_open[si] += int(om.sum())
                    if bool(om.any()):
                        seg_steps_open[si] += 1
                        gate_open_steps.append(t)
                        gate_open_env.append(float(getattr(dh, "M_H", args.M_H))
                                             if est is not None else None)
                        sg = torch.sign(ghat).tolist()
                        for j, o in enumerate(om.tolist()):
                            if o:
                                coord_open_counts[j] += 1
                                if len(gate_open_events) < GATE_EV_CAP:
                                    gate_open_events.append(
                                        [t, j, int(sg[j])])
                                else:
                                    gate_open_ev_trunc = True
                gate_total += 1
                pre = oracle.n_hvp if oracle is not None else 0
                est.step(oracle if t % args.K == 0 else None, lam, g, rho, kappa)
                if oracle is not None:
                    hvp_total += oracle.n_hvp - pre
                if fmd_ref is not None:
                    # validation-only HVPs: NOT charged to hvp_total
                    fmd_ref.step(oracle, lam)
                if fmd_full is not None and not full_blown:
                    # validation-only HVPs (gamma=1 recursion): NOT charged.
                    # On a non-contractive stream S_t can overflow; once it
                    # does it never recovers, so stop paying for it and mark
                    # the remaining steps non-finite.
                    fmd_full.step(oracle, lam)
                    if not bool(torch.isfinite(fmd_full.S).all()):
                        full_blown = True
            elif hd is not None:
                lam = hd.update(lam, g)
            elif tf is not None:
                lam = tf.step(oracle, lam, g)
                hvp_total += oracle.n_hvp
            elif fmd is not None:
                ghat = fmd.hypergrad(g).to(lam.dtype)
                lam = (lam - args.meta_lr * ghat).clamp(LAM_MIN, LAM_MAX)
                fmd.step(oracle, lam)
                hvp_total += oracle.n_hvp

            eta = spec.eta_vec(lam)
            step_vec = eta * g
            theta = theta - step_vec
            if est is not None:
                dh.step(float(torch.linalg.vector_norm(step_vec)))
            if oracle is not None:
                oracle.release()

        # ---- accounting & stability ----
        if not math.isfinite(loss_val):
            events += 1
            theta = ckpt.clone()
            # catastrophe backoff (uniform across ALL adaptive methods,
            # findings F15): halve the LR scale instead of resetting to the
            # possibly-catastrophic init — standard divergence recovery,
            # outside the certificate framework (like gradient clipping)
            lam = (lam - math.log(2.0)).clamp(LAM_MIN, LAM_MAX)
            if est is not None:
                est.reset_state()
            if fmd_ref is not None:
                fmd_ref.reset()
            if fmd_full is not None:
                fmd_full.reset()
                full_blown = False
            loss_val = float("nan")
        else:
            if len(med_win) >= 100:
                med = sorted(med_win)[len(med_win) // 2]
                if loss_val > 10.0 * med:
                    events += 1
            med_win.append(loss_val)
            if t % 200 == 0:
                ckpt = theta.clone()
        if math.isfinite(loss_val):
            seg_loss_sum[si] += loss_val
            seg_loss_n[si] += 1
        else:
            seg_nonfinite[si] += 1
        losses.append(loss_val)
        if t % args.lam_every == 0:
            lam_hist.append([t] + lam.tolist())

    # ---- summary ----
    y_mean = y_var_acc / n_y
    y_var = y_sq_acc / n_y - y_mean ** 2
    fin = [x for x in losses if math.isfinite(x)]
    nmse = (sum(fin) / len(fin)) * 2.0 / max(y_var, 1e-12)  # loss=0.5*MSE
    out = {
        "method": args.method, "dataset": args.dataset, "seed": args.seed,
        "steps": args.steps, "lr0": args.lr, "meta_lr": args.meta_lr,
        "rank": args.rank, "K": args.K, "gamma": args.gamma,
        "nmse": nmse, "events": events,
        "gate_open_frac": (gate_opens / gate_total) if gate_total else None,
        "coord_open_frac": ((ctl_open / ctl_total) if ctl_total
                            else (ctrl.gate_open_fraction if ctrl else None)),
        "hvp_total": hvp_total, "wall_s": time.time() - t0,
        "hdm_null_frac": (hdm.n_null / args.steps) if hdm else None,
        "losses": losses, "lam_hist": lam_hist,
    }
    # ---- round-4 extras (segment split / gate timing / envelope) ----
    seg_stats = []
    for i in range(n_seg):
        ny = max(seg_y_n[i], 1)
        mu = seg_y_sum[i] / ny
        var = seg_y_sq[i] / ny - mu ** 2
        ml = (seg_loss_sum[i] / seg_loss_n[i]) if seg_loss_n[i] else float("nan")
        seg_stats.append({
            "seg": i,
            "t_lo": 0 if i == 0 else seg_bounds[i - 1] if seg_bounds else 0,
            "t_hi": (seg_bounds[i] if (seg_bounds and i < 2) else args.steps),
            "y_var": var, "y_mean": mu, "n_y": seg_y_n[i],
            "mean_loss": ml, "n_finite": seg_loss_n[i],
            "n_nonfinite": seg_nonfinite[i],
            "nmse": (ml * 2.0 / max(var, 1e-12)
                     if ml == ml else float("nan")),
            "coord_open": seg_coord_open[i],
            "coord_total": seg_coord_total[i],
            "coord_open_frac": (seg_coord_open[i] / seg_coord_total[i]
                                if seg_coord_total[i] else None),
            "steps_open": seg_steps_open[i],
            "closed_steps": seg_closed[i],
        })
    out["seg_bounds"] = seg_bounds
    out["seg_stats"] = seg_stats
    out["scale_shift"] = args.scale_shift
    out["lam_every"] = args.lam_every
    out["gate_open_steps"] = gate_open_steps
    out["gate_open_env"] = gate_open_env
    out["gate_open_events"] = gate_open_events
    out["gate_open_events_truncated"] = gate_open_ev_trunc
    out["coord_open_counts"] = coord_open_counts
    out["n_probes"] = n_probes if est is not None else None
    out["adaptive_mh"] = args.adaptive_mh
    out["adapt_probe_log"] = adapt_probe_log if args.adaptive_mh else None
    out["adapt_mh_final"] = (float(dh.M_H) if args.adaptive_mh else None)
    out["adapt_mh_raises"] = (getattr(dh, "n_raises", None)
                              if args.adaptive_mh else None)
    out["adapt_mobs_max"] = (getattr(dh, "m_obs_max", None)
                             if args.adaptive_mh else None)
    # ---- --gate-warmup extras (None / empty on the legacy path) ----
    out["gate_warmup"] = args.gate_warmup
    out["probe_dense_until"] = args.probe_dense_until
    out["probe_raw_log"] = probe_raw_log if est is not None else None
    out["probe_mobs_log"] = probe_mobs_log if est is not None else None
    out["warmup_held_steps"] = warmup_held_steps
    out["warmup_suppressed_steps"] = warmup_supp_steps
    out["warmup_suppressed_coord"] = warmup_supp_coord
    out["warmup_suppressed_log"] = warmup_supp_log
    out["warmup_release_step"] = warmup_release_t
    # ---- e2_controls extras (None on the legacy arms) ----
    out["alpha"] = args.meta_lr
    out["M_H"] = args.M_H
    out["fail_closed"] = bool(args.fail_closed)
    out["rho_f"] = args.rho_f if args.method == "t6clip" else None
    out["absgate_threshold"] = args.absgate_threshold
    out["randgate_p"] = args.randgate_p
    out["periodic_every"] = args.periodic_every
    out["n_clipped"] = getattr(ctrl, "n_clipped", None) if ctrl else None
    out["failclosed_events"] = getattr(dh, "failclosed_events", None) \
        if (est is not None and args.fail_closed) else None
    out["failclosed_closed_steps"] = fc_closed_steps if args.fail_closed else None
    out["failclosed_reprobes"] = getattr(dh, "n_reprobe_req", None) \
        if (est is not None and args.fail_closed) else None
    if est is not None and args.fail_closed and getattr(dh, "m_obs", None):
        mo = sorted(dh.m_obs)
        n = len(mo)
        out["m_obs_stats"] = {
            "n": n, "mean": sum(mo) / n, "median": mo[n // 2],
            "p90": mo[min(n - 1, int(0.90 * n))],
            "p99": mo[min(n - 1, int(0.99 * n))], "max": mo[-1],
        }
    else:
        out["m_obs_stats"] = None
    if args.validate_cert:
        out["cert_checked"] = cert_checked
        out["cert_violations"] = cert_viol
        out["cert_violation_frac"] = cert_viol / max(cert_checked, 1)
        out["cert_violation_steps"] = cert_viol_steps
        out["cert_max_ratio"] = cert_max_ratio
        out["cert_max_excess"] = cert_max_excess
        cr = sorted(cert_ratio_hist)
        nc = max(len(cr), 1)
        out["cert_ratio_q"] = {
            "p50": cr[nc // 2] if cr else None,
            "p90": cr[min(nc - 1, int(0.90 * nc))] if cr else None,
            "p99": cr[min(nc - 1, int(0.99 * nc))] if cr else None,
            "max": cr[-1] if cr else None,
        }
    else:
        out["cert_checked"] = out["cert_violations"] = None
        out["cert_violation_frac"] = out["cert_violation_steps"] = None
        out["cert_max_ratio"] = out["cert_max_excess"] = None
        out["cert_ratio_q"] = None
    if ghat_abs_pool:
        out["ghat_abs_top"] = sorted(ghat_abs_pool, reverse=True)[:400]
    else:
        out["ghat_abs_top"] = None
    # ---- round-3 extras (None on every legacy arm) ----
    out["madgate_window"] = (args.madgate_window
                             if args.method == "madgate" else None)
    out["madgate_warmup"] = (args.madgate_warmup
                             if args.method == "madgate" else None)
    out["madgate_mean_mad"] = (mad_stat_sum / mad_stat_n
                               if mad_stat_n else None)
    out["ogd_doubling_D"] = (LAM_MAX - LAM_MIN
                             if args.method == "ogd_doubling" else None)
    out["ogd_alpha_log"] = (ogd_alpha_log
                            if args.method == "ogd_doubling" else None)
    _gsum = gstats.summary()
    if _gsum is not None:
        out["gate_stats"] = _gsum
    out["ogd_final_G"] = (ogd_Gmax.tolist()
                          if args.method == "ogd_doubling" else None)
    if args.validate_full:
        out["full_finite"] = [int(x) for x in full_finite]
        out["full_agree"] = full_agree
        out["full_open"] = full_open
        out["full_open_agree"] = full_open_agree
        out["full_nz"] = full_nz
        out["full_nz_agree"] = full_nz_agree
        out["full_disc_agree"] = full_disc_agree
        out["full_disc_nz"] = full_disc_nz
        out["full_disc_nz_agree"] = full_disc_nz_agree
        out["full_n_coord"] = int(spec.m)
    else:
        for k in ("full_finite", "full_agree", "full_open", "full_open_agree",
                  "full_nz", "full_nz_agree", "full_disc_agree",
                  "full_disc_nz", "full_disc_nz_agree", "full_n_coord"):
            out[k] = None
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(__file__), "..", "results", "e2")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{args.dataset}_{args.method}_lr{args.lr:g}{args.tag}_s{args.seed}"
    with open(os.path.join(out_dir, tag + ".json"), "w") as f:
        json.dump(out, f)
    print(f"[{tag}] nmse={nmse:.4f} events={events} "
          f"gate={out['gate_open_frac']} hvp={hvp_total} "
          f"wall={out['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
