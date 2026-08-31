"""E1: certificate validity & tightness (tracker R010, paper Fig. 2 + Table 2).

For each (problem, spectral tier, r, K, seed): run T inner SGD steps with a
frozen lambda, maintain ExactFMD ground truth alongside the COHG estimator,
and record per-step:
    e_t (certificate), true_err_t = ||S_t - S_hat_t||_F,
    rho_t, contractive flag, beta_t and true hypergradient bias.

Validity requires e_t >= true_err_t at every step (up to fp tolerance).

Problems:
    quad      -- StochasticQuadratic p=200 (oracle tier available)
    teacher   -- TinyMLP teacher-student (nonconvex, p~1.5K)
    mnistmlp  -- 784-32-32-10 MLP on MNIST (nonconvex, p~26K)

Usage:
    python e1_certificate.py --problem quad --tier oracle --seeds 10
    python e1_certificate.py --problem teacher --tier kw --seeds 10
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cohg import COHGEstimator, ExactFMD, GroupSpec
from cohg.certificate import (DriftHold, DriftHoldAdaptiveMH,
                              DriftHoldFailClosed, SpectralKW,
                              SpectralOracle, SpectralPrior)
from cohg.functional import FlatModule
from cohg.hvp import HVPOracle
from cohg.problems import StochasticQuadratic, TinyMLP

import data as D
import models as M


def build_problem(name: str, seed: int, device: str):
    """Returns (loss_fn, batches, theta0, spec, lam, L_prior, oracle_Q)."""
    if name == "quad":
        prob = StochasticQuadratic(p=200, seed=seed, noise=0.1, cond=50.0)
        spec = GroupSpec.from_sizes([70, 50, 50, 30])
        lam = torch.full((spec.m,), math.log(0.5))
        theta0 = torch.randn(prob.p, generator=torch.Generator().manual_seed(seed + 100),
                             dtype=torch.float64)
        batches = prob.batches(T=1000, seed=seed + 1)
        return prob.loss_fn, batches, theta0, spec, lam, prob.L_smooth, prob.Q
    if name == "teacher":
        prob = TinyMLP(n_in=8, n_hidden=24, n_out=2, seed=seed)
        spec = GroupSpec.from_sizes(prob.sizes)
        lam = torch.full((spec.m,), math.log(0.05))
        theta0 = prob.init_theta(seed=seed + 100)
        batches = prob.batches(T=1000, batch_size=64, seed=seed + 1)
        # L prior: crude but valid-in-practice envelope, calibrated in-run and
        # verified post-hoc (report violations if any); default conservative.
        return prob.loss_fn, batches, theta0, spec, lam, 25.0, None
    if name == "mnistmlp":
        torch.manual_seed(seed)
        model = M.DeepMLP(widths=(784, 32, 32, 10)).to(device)
        stream = D.ImageStream("mnist", batch_size=128, seed=seed + 1,
                               device=device)
        fm = FlatModule(model, lambda out, y:
                        torch.nn.functional.cross_entropy(out, y))
        spec = fm.group_spec()
        lam = torch.full((spec.m,), math.log(0.05), device=device)
        theta0 = fm.flat_params()
        batches = [stream.next() for _ in range(600)]
        return fm.loss_fn, batches, theta0, spec, lam, 50.0, None
    raise ValueError(name)


def run_one(problem: str, tier: str, r: int, K: int, seed: int, device: str,
            keep_traces: bool, gamma: float = 1.0, kw_eps: float = 0.05,
            M_H: float = 5.0, fail_closed: bool = False,
            adaptive_mh: float | None = None):
    dtype = torch.float64 if problem in ("quad", "teacher") else torch.float32
    torch.set_default_dtype(torch.float64 if dtype == torch.float64 else torch.float32)
    loss_fn, batches, theta0, spec, lam, L_prior, Q = build_problem(
        problem, seed, device)
    dev = theta0.device
    spec = spec.to(dev) if spec.g.device != dev else spec
    lam = lam.to(dev)

    if tier == "oracle":
        if Q is None:
            raise ValueError("oracle tier needs quadratic problem")
        sp = SpectralOracle(Q)
    elif tier == "prior":
        # for the quadratic the strong-convexity constant is known a priori
        mu = None
        if Q is not None:
            mu = float(torch.linalg.eigvalsh(Q).min())
        sp = SpectralPrior(L_prior, psd_mu=mu)
    elif tier == "kw":
        sp = SpectralKW(spec.p, eps=kw_eps, delta=1e-3, seed=seed + 7,
                        method="lanczos")
    elif tier == "kw_drift":
        # KW probes on refresh steps + A4 drift-hold in between (M_H prior)
        sp = SpectralKW(spec.p, eps=kw_eps, delta=1e-3, seed=seed + 7,
                        method="lanczos")
    else:
        raise ValueError(tier)
    if tier == "kw_drift":
        if adaptive_mh is not None:
            # round-4: online-enforceable envelope (implies fail-closed)
            dh = DriftHoldAdaptiveMH(M_H_floor=M_H, kappa=adaptive_mh)
            fail_closed = True
        elif fail_closed:
            dh = DriftHoldFailClosed(M_H=M_H)
        else:
            dh = DriftHold(M_H=M_H)
    else:
        dh = None

    fmd = ExactFMD(spec, device=dev, dtype=dtype, discount=gamma)
    est = COHGEstimator(spec, rank=r, refresh_every=K, device=dev, dtype=dtype,
                        discount=gamma)

    theta = theta0.clone()
    n_valid = n_steps = 0
    tight, contract, traces = [], [], []
    kw_hvp = 0
    # B4 (misspecification study) accounting
    n_probes = n_closed = 0
    force_probe = False
    adapt_probe_log = []
    n_probe_log = 0
    worst_ratio = 0.0           # max_t true_err_t / e_t  (>1 == violation)
    for t, batch in enumerate(batches):
        oracle = HVPOracle(loss_fn, theta, batch)
        eta = spec.eta_vec(lam)
        if tier == "oracle":
            rho, kappa = sp.bounds(eta)
        elif tier == "prior":
            rho, kappa = sp.bounds(eta)
        elif tier == "kw_drift":
            scheduled = (t % K == 0)
            if scheduled or force_probe:
                pre = oracle.n_hvp
                rho, kappa = sp.bounds(oracle.hvp, eta)
                kw_hvp += oracle.n_hvp - pre
                n_probes += 1
                dh.probe(rho, kappa, eta_vec=eta)
                if adaptive_mh is not None:
                    pl = dh.probe_log
                    if len(pl) > n_probe_log:
                        n_probe_log = len(pl)
                        adapt_probe_log.append([t] + list(pl[-1]))
                    else:
                        adapt_probe_log.append([t, dh.M_H, None, dh.M_H,
                                                int(dh.closed)])
                # a forced re-probe never forces another one (probe budget at
                # most doubled) -- same rule as e2_timeseries.py
                force_probe = bool(getattr(dh, "want_reprobe", False)
                                   and scheduled)
                rho, kappa = dh.bounds(eta)
            else:
                rho, kappa = dh.bounds(eta)
            if fail_closed and getattr(dh, "closed", False):
                n_closed += 1
        else:
            # KW spectral probe only on refresh steps; prior fallback between
            if t % K == 0:
                rho, kappa = sp.bounds(oracle.hvp, eta)
                kw_hvp += oracle.n_hvp
                rho_hold, kappa_hold = rho, kappa
            else:
                fallback = SpectralPrior(L_prior).bounds(eta)
                rho, kappa = fallback
        fmd.step(oracle, lam)
        est.step(oracle if t % K == 0 else None, lam, oracle.grad, rho, kappa)
        true_err = float(torch.linalg.matrix_norm(
            fmd.S - est.dense_shat().to(fmd.S.dtype), ord="fro"))
        tol = 1e-9 if dtype == torch.float64 else 1e-4
        ok = est.e >= true_err * (1 - tol) - 1e-12
        n_valid += int(ok)
        n_steps += 1
        if true_err > (1e-12 if dtype == torch.float64 else 1e-6):
            tight.append(est.e / true_err)
            worst_ratio = max(worst_ratio, true_err / max(est.e, 1e-300))
        contract.append(rho < 1.0)
        if keep_traces and (t % 5 == 0):
            g_meta = oracle.grad
            bias_true = float(torch.linalg.vector_norm(
                est.hypergrad(g_meta) - fmd.hypergrad(g_meta.to(fmd.S.dtype)).to(est.d.dtype)))
            traces.append({"t": t, "e": est.e, "true_err": true_err,
                           "rho": rho, "beta": est.beta(float(torch.linalg.vector_norm(g_meta))),
                           "bias_true": bias_true})
        step_vec = eta * oracle.grad
        theta = theta - step_vec
        if dh is not None:
            dh.step(float(torch.linalg.vector_norm(step_vec)))
        oracle.release()

    tq = torch.tensor(tight) if tight else torch.tensor([float("nan")])
    return {
        "problem": problem, "tier": tier, "r": r, "K": K, "seed": seed,
        "gamma": gamma, "kw_eps": kw_eps,
        "valid_rate": n_valid / n_steps,
        "tight_q25": float(tq.quantile(0.25)), "tight_med": float(tq.median()),
        "tight_q75": float(tq.quantile(0.75)), "tight_q95": float(tq.quantile(0.95)),
        "contractive_frac": sum(contract) / len(contract),
        "final_e": est.e, "est_hvp": est.hvp_count, "kw_hvp": kw_hvp,
        "traces": traces if keep_traces else None,
        # ---- B4 misspecification-study extras ----
        "M_H": M_H, "fail_closed": bool(fail_closed),
        "violation_rate": 1.0 - n_valid / n_steps,
        "n_violations": n_steps - n_valid,
        "worst_true_over_bound": worst_ratio,
        "n_probes": n_probes,
        "probe_budget": (n_steps + K - 1) // K,
        "probe_overhead": (n_probes / max((n_steps + K - 1) // K, 1)
                           if tier == "kw_drift" else None),
        "closed_frac": (n_closed / n_steps) if fail_closed else None,
        "failclosed_events": getattr(dh, "failclosed_events", None)
        if dh is not None else None,
        "reprobe_requests": getattr(dh, "n_reprobe_req", None)
        if dh is not None else None,
        "m_obs": (sorted(dh.m_obs) if (dh is not None
                                       and getattr(dh, "m_obs", None)) else None),
        # ---- round-4 adaptive-envelope extras ----
        "adaptive_mh": adaptive_mh,
        "M_H_floor": M_H if adaptive_mh is not None else None,
        "adapt_probe_log": adapt_probe_log if adaptive_mh is not None else None,
        "adapt_mh_final": (float(dh.M_H) if adaptive_mh is not None else None),
        "adapt_mh_raises": (getattr(dh, "n_raises", None)
                            if adaptive_mh is not None else None),
        "adapt_mobs_max": (getattr(dh, "m_obs_max", None)
                           if adaptive_mh is not None else None),
        "n_steps": n_steps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True,
                    choices=["quad", "teacher", "mnistmlp"])
    ap.add_argument("--tier", required=True,
                    choices=["oracle", "prior", "kw", "kw_drift"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--rs", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    ap.add_argument("--Ks", type=int, nargs="+", default=[1, 5, 10, 20])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--kw-eps", type=float, default=0.05)
    ap.add_argument("--tag", default="")
    ap.add_argument("--trace-config", default="r4_K5",
                    help="config whose per-step traces are kept for Fig. 2")
    # ---- B4 (drift-prior misspecification) additions ----
    ap.add_argument("--M-H", type=float, default=5.0,
                    help="Hessian-Lipschitz prior for the A4' drift hold "
                    "(kw_drift tier only); 5.0 reproduces the legacy runs")
    ap.add_argument("--fail-closed", action="store_true",
                    help="use the DriftHoldFailClosed monitor: forced "
                    "re-probes at M_obs > M_H/2 and a closed-gate flag at "
                    "M_obs > M_H")
    ap.add_argument("--adaptive-mh", type=float, default=None,
                    help="round-4: online-enforceable drift envelope "
                    "M_H,t = max(--M-H, KAPPA * max_{s<=t} M_obs,s), "
                    "re-stated at every probe (implies --fail-closed)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed-offset", type=int, default=0)
    args = ap.parse_args()

    device = args.device if args.problem == "mnistmlp" else "cpu"
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(__file__), "..", "results", "e1")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    t0 = time.time()
    seed_list = [args.seed_offset + i for i in range(args.seeds)]
    for r, K, seed in itertools.product(args.rs, args.Ks, seed_list):
        keep = (f"r{r}_K{K}" == args.trace_config) and seed == 0
        row = run_one(args.problem, args.tier, r, K, seed, device, keep,
                      gamma=args.gamma, kw_eps=args.kw_eps,
                      M_H=args.M_H, fail_closed=args.fail_closed,
                      adaptive_mh=args.adaptive_mh)
        rows.append(row)
        if seed == 0:
            print(f"[{args.problem}/{args.tier}{args.tag}] r={r} K={K} "
                  f"g={args.gamma}: valid={row['valid_rate']:.3f} "
                  f"med_tight={row['tight_med']:.3g} "
                  f"contract={row['contractive_frac']:.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    path = os.path.join(out_dir, f"{args.problem}_{args.tier}{args.tag}.json")
    with open(path, "w") as f:
        json.dump(rows, f)
    n_bad = sum(1 for x in rows if x["valid_rate"] < 1.0)
    print(f"DONE {len(rows)} runs -> {path} | configs with violations: {n_bad}")


if __name__ == "__main__":
    main()
