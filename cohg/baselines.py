"""Meta-adaptation baselines for E2/E3/E4.

All baselines adapt per-group log-learning-rates lambda (m,) for the inner
SGD  theta_{t+1} = theta_t - eta(lambda)*g_t, matching COHG's interface.

- HD (Baydin et al., ICLR 2018), per-group form: 1-step truncated
  hypergradient  h_j = -eta_j * <g_t, g_{t-1}>_{group j};  lambda -= beta*h.
  (Scalar variant: sum over all groups.) This is exactly COHG with
  S_hat ~ B_{t-1}, no certificate, no gate.

- HDM (Gao et al. 2025, arXiv:2502.11229), adapted to per-group diagonal
  preconditioner P = diag(eta) with:
    feedback   h_x(P) = [f(x - P g) - f(x)] / ||g||^2
    grad       d h/d eta_j = -<g(x_+), g(x)>_{group j} / ||g||^2
    null step  x-update skipped if minibatch loss increases (their Eq. 8)
    AdaGrad    online learner over eta (their recommended stabilizer)
    projection eta clipped to [eta_min, eta_max]  (their Pi_P)
  Adaptation to streaming: f is the current minibatch loss; the null-step
  check uses one extra forward pass on the same batch (stated in paper text).

- TruncatedFMD: forward-mode sensitivity with periodic reset every K_trunc
  steps (short-horizon truncation made explicit). Runs the full recursion
  every step (m HVPs/step) — deliberately compute-RICHER than COHG, so a COHG
  win cannot be attributed to budget. No certificate, no gate.

All operate on log-LR lambda; updates are gradient steps with meta_lr.
"""

from __future__ import annotations

import torch

from .fmd import ExactFMD
from .groups import GroupSpec
from .hvp import HVPOracle


class HDBaseline:
    def __init__(self, spec: GroupSpec, meta_lr: float, scalar: bool = False,
                 lam_min: float | None = None, lam_max: float | None = None):
        self.spec = spec
        self.meta_lr = meta_lr
        self.scalar = scalar
        self.lam_min, self.lam_max = lam_min, lam_max
        self.prev_grad: torch.Tensor | None = None

    def update(self, lam: torch.Tensor, grad_now: torch.Tensor) -> torch.Tensor:
        """Call with the CURRENT step's gradient g_t (before the theta step)."""
        if self.prev_grad is not None:
            eta = self.spec.eta_vec(lam)
            h = -self.spec.group_sums(grad_now * eta * self.prev_grad)
            if self.scalar:
                h = torch.full_like(h, float(h.sum()))
            lam = lam - self.meta_lr * h
            if self.lam_min is not None or self.lam_max is not None:
                lam = lam.clamp(min=self.lam_min, max=self.lam_max)
        self.prev_grad = grad_now.detach().clone()
        return lam


class HDMBaseline:
    """Per-group HDM with null step + AdaGrad (the paper's stabilized form)."""

    def __init__(self, spec: GroupSpec, meta_lr: float = 0.1,
                 lam_min: float = -9.0, lam_max: float = 0.0,
                 adagrad_eps: float = 1e-12):
        self.spec = spec
        self.meta_lr = meta_lr
        self.lam_min, self.lam_max = lam_min, lam_max
        self.G2 = None
        self.eps = adagrad_eps
        self.n_null = 0

    def step(self, loss_fn, theta: torch.Tensor, batch, lam: torch.Tensor):
        """One joint (theta, lambda) step. Returns (theta_next, lam_next, loss)."""
        theta_ = theta.detach().requires_grad_(True)
        loss = loss_fn(theta_, batch)
        (g,) = torch.autograd.grad(loss, theta_)
        g = g.detach()
        loss = loss.detach()
        eta = self.spec.eta_vec(lam)
        cand = theta - eta * g
        with torch.no_grad():
            loss_cand = loss_fn(cand, batch)
        # hypergradient of feedback w.r.t. eta, chain to lambda (eta=exp(lam))
        cand_ = cand.detach().requires_grad_(True)
        loss_c = loss_fn(cand_, batch)
        (g_plus,) = torch.autograd.grad(loss_c, cand_)
        gnorm2 = float(g @ g)
        if gnorm2 > 1e-30:
            dh_deta = -self.spec.group_sums(g_plus.detach() * g) / gnorm2
            dh_dlam = dh_deta * torch.exp(lam)
            if self.G2 is None:
                self.G2 = torch.zeros_like(dh_dlam)
            self.G2 = self.G2 + dh_dlam ** 2
            lam = lam - self.meta_lr * dh_dlam / (self.G2 + self.eps).sqrt()
            lam = lam.clamp(min=self.lam_min, max=self.lam_max)
        # null step (Eq. 8): keep theta if the candidate is worse
        if float(loss_cand) <= float(loss):
            theta_next = cand.detach()
            step_loss = float(loss)
        else:
            theta_next = theta.detach()
            self.n_null += 1
            step_loss = float(loss)
        return theta_next, lam, step_loss


class TruncatedFMD:
    """Exact FMD with sensitivity reset every K_trunc steps + plain meta-SGD."""

    def __init__(self, spec: GroupSpec, meta_lr: float, K_trunc: int,
                 device="cpu", dtype=torch.float32,
                 lam_min: float | None = None, lam_max: float | None = None):
        self.fmd = ExactFMD(spec, device=device, dtype=dtype)
        self.spec = spec
        self.meta_lr = meta_lr
        self.K = K_trunc
        self.t = 0
        self.lam_min, self.lam_max = lam_min, lam_max

    def step(self, oracle: HVPOracle, lam: torch.Tensor,
             meta_grad: torch.Tensor) -> torch.Tensor:
        """Advance S, produce truncated hypergradient, update lambda."""
        if self.t % self.K == 0:
            self.fmd.reset()
        ghat = self.fmd.hypergrad(meta_grad).to(lam.dtype)
        lam = lam - self.meta_lr * ghat
        if self.lam_min is not None or self.lam_max is not None:
            lam = lam.clamp(min=self.lam_min, max=self.lam_max)
        self.fmd.step(oracle, lam)
        self.t += 1
        return lam
