"""Exact forward-mode differentiation (FMD) sensitivity recursion.

Reference/oracle estimator maintaining the full S_t = d theta_t / d lambda
(p x m), for SGD with per-group learning rates eta_j = exp(lambda_j):

    S_{t+1} = (I - diag(eta) H_t) S_t + B_t,
    B_t[:, j] = -1_{group j} * eta * g_t   (aligned; see groups.py)

Update must be called at theta_t BEFORE the inner parameter step, using the
same batch. Cost: m HVPs per step, O(pm) memory. Only feasible for small p*m;
serves as ground truth in E0/E1/E2 and for finite-difference validation.
"""

from __future__ import annotations

import torch

from .groups import GroupSpec
from .hvp import HVPOracle


class ExactFMD:
    """discount < 1 tracks the discounted sensitivity
    S~_t = sum_k gamma^{t-k} (prod_{j=k+1}^{t-1} A_j) B_k  via
    S~_{t+1} = gamma * A_t S~_t + B_t  (recency-weighted hypergradient credit;
    gamma = 1 recovers the full-horizon sensitivity)."""

    def __init__(self, spec: GroupSpec, device="cpu", dtype=torch.float64,
                 discount: float = 1.0):
        self.spec = spec
        self.S = torch.zeros(spec.p, spec.m, device=device, dtype=dtype)
        self.gamma = discount

    def reset(self):
        self.S.zero_()

    def step(self, oracle: HVPOracle, lam: torch.Tensor) -> None:
        """Advance S_t -> S_{t+1} using H_t, g_t at theta_t (pre-step)."""
        spec = self.spec
        eta = spec.eta_vec(lam).to(self.S.dtype)
        HS = oracle.hvp_mat(self.S.to(oracle.grad.dtype)).to(self.S.dtype)
        B = spec.aligned_to_matrix(-(eta * oracle.grad.to(self.S.dtype)))
        self.S = self.gamma * (self.S - eta.unsqueeze(1) * HS) + B

    def hypergrad(self, meta_grad: torch.Tensor) -> torch.Tensor:
        """g_lambda = S_t^T grad_theta ell_t(theta_t)  (m,)."""
        return self.S.T @ meta_grad.to(self.S.dtype)
