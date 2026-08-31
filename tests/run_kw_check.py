"""Validate SpectralKW (Tier A) against dense ground truth.

Checks over random symmetric H (indefinite, deep-net-like spiked spectra) and
heterogeneous eta: certified rho must upper-bound the true ||I - diag(eta)H||_2
in every trial (validity), while staying within ~1/sqrt(1-eps) of it
(usefulness).
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cohg.certificate import SpectralKW

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

p = 400
n_trials = 200
viol, ratios = 0, []
for trial in range(n_trials):
    gen = torch.Generator().manual_seed(trial)
    Q, _ = torch.linalg.qr(torch.randn(p, p, generator=gen))
    # spiked deep-net-like spectrum: few outliers + near-zero bulk + negatives
    eigs = torch.cat([
        torch.tensor([10.0, 6.0, 3.0]),
        1.5 * torch.randn(20, generator=gen),          # indefinite shoulder
        0.01 * torch.randn(p - 23, generator=gen),     # near-zero bulk
    ])
    H = (Q * eigs) @ Q.T
    eta = torch.exp(torch.empty(p).uniform_(-4.0, -1.5, generator=gen))
    A = torch.eye(p) - eta.unsqueeze(1) * H
    true_norm = float(torch.linalg.matrix_norm(A, ord=2))

    kw = SpectralKW(p, eps=0.25, delta=1e-3, seed=1000 + trial)
    rho, kappa = kw.bounds(lambda v: H @ v, eta)
    if rho < true_norm:
        viol += 1
    ratios.append(rho / true_norm)

ratios = torch.tensor(ratios)
print(f"violations: {viol}/{n_trials} (KW failure budget delta=1e-3)")
print(f"overestimation ratio: median {ratios.median():.4f}, "
      f"p95 {ratios.quantile(0.95):.4f}, max {ratios.max():.4f}")
sys.exit(1 if viol > 0 else 0)
