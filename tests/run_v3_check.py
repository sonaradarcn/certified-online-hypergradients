"""v3 validation: column-wise certificate + coordinate gate behavior.

(1) Column validity: e_col[j] >= ||(S~ - S_hat)[:, j]||_2 for every t, j
    (teacher problem, gamma=0.9, KW-Lanczos probes + DriftHold, K=5, r=4).
(2) Gate behavior at mis-set lambda_0 (10x low LR): coordinate gates should
    OPEN and lambda should move toward higher LR; at well-set lambda_0 the
    gates should stay mostly closed (F6 expectation).
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cohg import COHGEstimator, CoordGatedController, ExactFMD, GroupSpec
from cohg.certificate import DriftHold, SpectralKW
from cohg.hvp import HVPOracle
from cohg.problems import TinyMLP

torch.set_default_dtype(torch.float64)

T, K, R, GAMMA = 800, 5, 4, 0.9

for scenario, lr0 in [("well-set", 0.05), ("mis-set-low", 0.005)]:
    torch.manual_seed(0)
    prob = TinyMLP(n_in=8, n_hidden=24, n_out=2, seed=0)
    spec = GroupSpec.from_sizes(prob.sizes)
    lam = torch.full((spec.m,), math.log(lr0))
    lam0 = lam.clone()
    theta = prob.init_theta(seed=100)
    fmd = ExactFMD(spec, discount=GAMMA)
    est = COHGEstimator(spec, rank=R, refresh_every=K, discount=GAMMA)
    kw = SpectralKW(spec.p, eps=0.05, delta=1e-3, seed=7)
    dh = DriftHold(M_H=5.0)
    ctrl = CoordGatedController(meta_lr=0.05, gate_factor=2.0,
                                lam_min=math.log(1e-5), lam_max=0.0)
    col_viol = 0
    losses = []
    for t, batch in enumerate(prob.batches(T=T, batch_size=64, seed=1)):
        oracle = HVPOracle(prob.loss_fn, theta, batch)
        eta = spec.eta_vec(lam)
        if t % K == 0:
            rho, kappa = kw.bounds(oracle.hvp, eta)
            dh.probe(rho, kappa, eta_vec=eta)
        rho, kappa = dh.bounds(eta)
        # gate BEFORE stepping estimators (signal at time t)
        ghat = est.hypergrad(oracle.grad)
        beta_col = est.beta_col(float(torch.linalg.vector_norm(oracle.grad)))
        lam, _ = ctrl.maybe_update(lam, ghat, beta_col)
        eta = spec.eta_vec(lam)
        fmd.step(oracle, lam)
        est.step(oracle if t % K == 0 else None, lam, oracle.grad, rho, kappa)
        # column validity
        E = fmd.S - est.dense_shat()
        col_norms = torch.linalg.vector_norm(E, dim=0)
        col_viol += int((est.e_col < col_norms * (1 - 1e-9) - 1e-12).sum())
        losses.append(float(oracle.loss))
        step_vec = eta * oracle.grad
        theta = theta - step_vec
        dh.step(float(torch.linalg.vector_norm(step_vec)))
        oracle.release()
    dlam = (lam - lam0)
    print(f"[{scenario}] col violations: {col_viol}/{T * spec.m} | "
          f"coord open frac: {ctrl.gate_open_fraction:.3f} | "
          f"loss first/last50: {sum(losses[:50]) / 50:.4f} -> "
          f"{sum(losses[-50:]) / 50:.4f} | "
          f"dlam: {[round(float(x), 2) for x in dlam]}")
