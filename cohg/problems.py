"""Testbed problems for M0/E1 validation.

All problems expose:
    loss_fn(theta_flat, batch) -> scalar   (inner training loss, differentiable)
    meta_loss_fn(theta_flat, batch) -> scalar   (outer/eval loss)
    batches(T, seed) -> deterministic batch sequence (for finite differences)
"""

from __future__ import annotations

import math

import torch


class StochasticQuadratic:
    """L_t(theta) = 0.5 (theta - c_t)^T Q (theta - c_t),  c_t = theta* + noise_t.

    Hessian is the constant Q (analytically known -> SpectralOracle), gradients
    are stochastic through c_t. True sensitivity recursion available densely.
    """

    def __init__(self, p: int, seed: int = 0, noise: float = 0.1,
                 cond: float = 20.0, device="cpu", dtype=torch.float64):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        A = torch.randn(p, p, generator=gen, dtype=dtype)
        Qraw, _ = torch.linalg.qr(A)
        eigs = torch.logspace(0, math.log10(cond), p, dtype=dtype)
        eigs = eigs / eigs.max()  # spectrum in (1/cond, 1]
        self.Q = (Qraw * eigs) @ Qraw.T
        self.Q = 0.5 * (self.Q + self.Q.T)
        self.theta_star = torch.randn(p, generator=gen, dtype=dtype)
        self.noise = noise
        self.p = p
        self.dtype = dtype
        self.device = device
        self.mu = float(eigs.min())
        self.L_smooth = float(eigs.max())

    def batches(self, T: int, seed: int = 1):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        return [self.theta_star + self.noise * torch.randn(
            self.p, generator=gen, dtype=self.dtype) for _ in range(T)]

    def loss_fn(self, theta: torch.Tensor, c_t: torch.Tensor) -> torch.Tensor:
        d = theta - c_t
        return 0.5 * d @ (self.Q @ d)

    meta_loss_fn = loss_fn

    def true_S_step(self, S: torch.Tensor, eta_vec: torch.Tensor,
                    grad: torch.Tensor, spec) -> torch.Tensor:
        """Dense ground-truth recursion, independent of autograd."""
        B = spec.aligned_to_matrix(-(eta_vec * grad))
        return S - eta_vec.unsqueeze(1) * (self.Q @ S) + B


class TinyMLP:
    """Two-layer tanh MLP regression on a fixed synthetic teacher.

    Nonconvex, autograd-driven Hessian: exercises HVPOracle and the FD check
    on a realistic (non-quadratic) loss. Flat parameter layout:
    [W1 (h x in), b1 (h), W2 (out x h), b2 (out)] with per-tensor groups.
    """

    def __init__(self, n_in: int = 8, n_hidden: int = 16, n_out: int = 2,
                 seed: int = 0, dtype=torch.float64):
        self.shapes = [(n_hidden, n_in), (n_hidden,), (n_out, n_hidden), (n_out,)]
        self.sizes = [s[0] * s[1] if len(s) == 2 else s[0] for s in self.shapes]
        self.p = sum(self.sizes)
        self.dtype = dtype
        gen = torch.Generator().manual_seed(seed)
        self.teacher = torch.randn(n_out, n_in, generator=gen, dtype=dtype)
        self.n_in, self.n_out = n_in, n_out

    def init_theta(self, seed: int = 2) -> torch.Tensor:
        gen = torch.Generator().manual_seed(seed)
        parts = []
        for shape, size in zip(self.shapes, self.sizes):
            fan_in = shape[1] if len(shape) == 2 else max(shape[0], 1)
            parts.append(torch.randn(size, generator=gen, dtype=self.dtype)
                         / math.sqrt(fan_in))
        return torch.cat(parts)

    def batches(self, T: int, batch_size: int = 32, seed: int = 1):
        gen = torch.Generator().manual_seed(seed)
        out = []
        for _ in range(T):
            x = torch.randn(batch_size, self.n_in, generator=gen, dtype=self.dtype)
            y = x @ self.teacher.T + 0.05 * torch.randn(
                batch_size, self.n_out, generator=gen, dtype=self.dtype)
            out.append((x, y))
        return out

    def _unpack(self, theta: torch.Tensor):
        parts, off = [], 0
        for shape, size in zip(self.shapes, self.sizes):
            parts.append(theta[off:off + size].view(*shape))
            off += size
        return parts

    def loss_fn(self, theta: torch.Tensor, batch) -> torch.Tensor:
        x, y = batch
        W1, b1, W2, b2 = self._unpack(theta)
        h = torch.tanh(x @ W1.T + b1)
        pred = h @ W2.T + b2
        return 0.5 * ((pred - y) ** 2).mean()

    meta_loss_fn = loss_fn
