"""F3 preview: does discounting rescue the certificate on nonconvex problems?

Teacher-student MLP, KW spectral tier (measured rho), gamma ladder.
For each gamma: validity vs discounted-FMD ground truth, tightness, e_t scale,
and the measured rho_t distribution (how non-contractive is the real
trajectory?). Also the gamma-ladder hypergradient discrepancy (the online
short-horizon-bias diagnostic proposed in findings F3).
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cohg import COHGEstimator, ExactFMD, GroupSpec
from cohg.certificate import SpectralKW
from cohg.hvp import HVPOracle
from cohg.problems import TinyMLP

torch.set_default_dtype(torch.float64)

T = 600
K = 5
R = 4
KW_EPS = 0.05
prob = TinyMLP(n_in=8, n_hidden=24, n_out=2, seed=0)
spec = GroupSpec.from_sizes(prob.sizes)
lam = torch.full((spec.m,), math.log(0.05))
gammas = [1.0, 0.99, 0.95, 0.9]

for gamma in gammas:
    torch.manual_seed(0)
    theta = prob.init_theta(seed=100)
    fmd = ExactFMD(spec, discount=gamma)
    est = COHGEstimator(spec, rank=R, refresh_every=K, discount=gamma)
    kw = SpectralKW(spec.p, eps=KW_EPS, delta=1e-3, seed=7)
    rho_hold = None
    n_valid = 0
    tights, rhos, gnorm_ratio = [], [], []
    for t, batch in enumerate(prob.batches(T=T, batch_size=64, seed=1)):
        oracle = HVPOracle(prob.loss_fn, theta, batch)
        eta = spec.eta_vec(lam)
        if t % K == 0:
            rho, kappa = kw.bounds(oracle.hvp, eta)
            rho_hold = (rho, kappa)
        rho, kappa = rho_hold
        rhos.append(rho)
        fmd.step(oracle, lam)
        est.step(oracle if t % K == 0 else None, lam, oracle.grad, rho, kappa)
        true_err = float(torch.linalg.matrix_norm(fmd.S - est.dense_shat(), ord="fro"))
        n_valid += int(est.e >= true_err * (1 - 1e-9) - 1e-12)
        if true_err > 1e-12:
            tights.append(est.e / true_err)
        # is beta_t small relative to the signal? (gate usefulness proxy)
        g_meta = oracle.grad
        ghat = est.hypergrad(g_meta)
        beta = est.beta(float(torch.linalg.vector_norm(g_meta)))
        gn = float(torch.linalg.vector_norm(ghat))
        gnorm_ratio.append(gn / max(beta, 1e-300))
        theta = theta - eta * oracle.grad
        oracle.release()
    tq = torch.tensor(tights)
    rq = torch.tensor(rhos)
    gq = torch.tensor(gnorm_ratio)
    print(f"gamma={gamma:4.2f} | valid {n_valid}/{T} | "
          f"tight med {tq.median():.3g} p90 {tq.quantile(0.9):.3g} | "
          f"final e {est.e:.3g} | rho med {rq.median():.3f} max {rq.max():.3f} | "
          f"gate signal ||ghat||/beta med {gq.median():.3g} p90 {gq.quantile(0.9):.3g}")
