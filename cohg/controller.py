"""Certificate-gated meta-update controller.

Update rule (proposal section 4.3): with hypergradient estimate ghat and bias
bound beta_t = e_t * ||grad ell_t||,

    if ||ghat|| > gate_factor * beta_t:
        lam <- lam - meta_lr * s * ghat,   s = min(1, trust * (||ghat|| - beta_t) / ||ghat||)
    else:
        freeze lam (fall back to fixed-hyperparameter behavior)

Rationale: ||ghat|| > 2*beta ensures <ghat, g_true> >= ||ghat|| (||ghat|| - beta) > 0,
i.e. the update is a certified descent direction for the meta-loss surrogate.
"""

from __future__ import annotations

import torch


class CoordGatedController:
    """Per-coordinate certificate gate + normalized (sign-like) trust step.

    Coordinate j moves iff |ghat_j| > gate_factor * beta_j — then the SIGN of
    ghat_j provably matches the true hypergradient coordinate (certified sign
    descent), and the step is scale-free:

        lam_j <- lam_j - meta_lr * s_j * sign(ghat_j),
        s_j = min(1, trust * (|ghat_j| - beta_j) / |ghat_j|).

    Scale-freeness makes the comparison with AdaGrad-normalized baselines
    (HDM) fair, and avoids the tiny-raw-hypergradient stall seen in E2 v1.
    """

    def __init__(self, meta_lr: float, gate_factor: float = 2.0,
                 trust: float = 1.0, lam_min: float | None = None,
                 lam_max: float | None = None, mode: str = "sign"):
        """mode: 'sign' (practical, scale-free; T3a safety only — no
        worst-case regret, PROOF_AUDIT #14) or 'ogd' (theorem T3-OGD:
        magnitude-aware gated projected OGD with certified-bias regret)."""
        self.meta_lr = meta_lr
        self.gate_factor = gate_factor
        self.trust = trust
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.mode = mode
        self.n_coord_open = 0
        self.n_coord_total = 0
        self.n_steps_any_open = 0
        self.n_steps = 0

    def maybe_update(self, lam: torch.Tensor, ghat: torch.Tensor,
                     beta_col: torch.Tensor) -> tuple[torch.Tensor, bool]:
        b = beta_col.to(ghat.dtype).to(ghat.device)
        mag = ghat.abs()
        open_mask = mag > self.gate_factor * b
        self.n_coord_total += int(open_mask.numel())
        self.n_coord_open += int(open_mask.sum())
        self.n_steps += 1
        if bool(open_mask.any()):
            if self.mode == "ogd":
                step = self.meta_lr * ghat
            else:
                s = ((mag - b) / mag.clamp_min(1e-30)).clamp(0.0, 1.0) * self.trust
                s = s.clamp(max=1.0)
                step = self.meta_lr * s * torch.sign(ghat)
            lam = lam - torch.where(open_mask, step, torch.zeros_like(step))
            if self.lam_min is not None or self.lam_max is not None:
                lam = lam.clamp(min=self.lam_min, max=self.lam_max)
            self.n_steps_any_open += 1
            return lam, True
        return lam, False

    @property
    def gate_open_fraction(self) -> float:
        return (self.n_coord_open / self.n_coord_total
                if self.n_coord_total else 0.0)


class T6ClipController(CoordGatedController):
    """COHG gate + Theorem-6 online-checkable step-size condition (control E).

    Identical per-coordinate certificate gate as CoordGatedController, but the
    step MAGNITUDE of every open coordinate is additionally capped at

        alpha_j <= 2 * (1 - 1/c) * |ghat_j| / rho_f,

    with c = gate_factor and rho_f a prior on the smoothness of the meta
    objective in lambda.  This is exactly the descent condition of Theorem 6;
    enforcing it online makes the *experimental* controller satisfy the
    theorem's hypothesis instead of only being covered by it in spirit.
    The cap is applied on top of the certified trust factor s_j, so the
    realized magnitude is min(meta_lr * s_j, 2 (1 - 1/c) |ghat_j| / rho_f)
    <= min(meta_lr, 2 (1 - 1/c) |ghat_j| / rho_f).
    """

    def __init__(self, meta_lr: float, gate_factor: float = 2.0,
                 trust: float = 1.0, lam_min: float | None = None,
                 lam_max: float | None = None, mode: str = "sign",
                 rho_f: float = 1.0):
        super().__init__(meta_lr, gate_factor=gate_factor, trust=trust,
                         lam_min=lam_min, lam_max=lam_max, mode=mode)
        self.rho_f = float(rho_f)
        self.n_clipped = 0

    def maybe_update(self, lam: torch.Tensor, ghat: torch.Tensor,
                     beta_col: torch.Tensor) -> tuple[torch.Tensor, bool]:
        b = beta_col.to(ghat.dtype).to(ghat.device)
        mag = ghat.abs()
        open_mask = mag > self.gate_factor * b
        self.n_coord_total += int(open_mask.numel())
        self.n_coord_open += int(open_mask.sum())
        self.n_steps += 1
        if not bool(open_mask.any()):
            return lam, False
        s = ((mag - b) / mag.clamp_min(1e-30)).clamp(0.0, 1.0) * self.trust
        s = s.clamp(max=1.0)
        alpha = self.meta_lr * s
        cap = 2.0 * (1.0 - 1.0 / self.gate_factor) * mag / self.rho_f
        self.n_clipped += int(((cap < alpha) & open_mask).sum())
        step = torch.minimum(alpha, cap) * torch.sign(ghat)
        lam = lam - torch.where(open_mask, step, torch.zeros_like(step))
        if self.lam_min is not None or self.lam_max is not None:
            lam = lam.clamp(min=self.lam_min, max=self.lam_max)
        self.n_steps_any_open += 1
        return lam, True


class GatedController:
    def __init__(self, meta_lr: float, gate_factor: float = 2.0,
                 trust: float = 1.0, lam_min: float | None = None,
                 lam_max: float | None = None):
        self.meta_lr = meta_lr
        self.gate_factor = gate_factor
        self.trust = trust
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.n_updates = 0
        self.n_frozen = 0

    def maybe_update(self, lam: torch.Tensor, ghat: torch.Tensor,
                     beta: float) -> tuple[torch.Tensor, bool]:
        gnorm = float(torch.linalg.vector_norm(ghat))
        if gnorm > self.gate_factor * beta and gnorm > 0.0:
            scale = min(1.0, self.trust * (gnorm - beta) / gnorm)
            lam = lam - self.meta_lr * scale * ghat
            if self.lam_min is not None or self.lam_max is not None:
                lam = lam.clamp(min=self.lam_min, max=self.lam_max)
            self.n_updates += 1
            return lam, True
        self.n_frozen += 1
        return lam, False

    @property
    def gate_open_fraction(self) -> float:
        total = self.n_updates + self.n_frozen
        return self.n_updates / total if total else 0.0
