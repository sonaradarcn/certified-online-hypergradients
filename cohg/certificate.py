"""Certified spectral bounds for the propagation operator A_t = I - diag(eta) H_t.

The certificate recursion e_{t+1} = rho_t * e_t + eta_err_t needs, per step,

    rho_t   >= ||I - diag(eta) H_t||_2      (propagation contraction/expansion)
    kappa_t >= ||diag(eta) H_t||_2          (staleness charge on lazy steps)

Tiers (selection is milestone M2; M0 ships the two ends):
- SpectralOracle: exact dense computation from a known Hessian (quadratic
  testbed only). Validates the recursion logic itself.
- SpectralPrior (Tier C): deterministic prior smoothness bound ||H||_2 <= L.
  Always valid, including indefinite H. With heterogeneous per-group eta the
  operator is non-symmetric; we use the safe bounds
      kappa <= eta_max * L,   rho <= 1 + eta_max * L  (never contractive), or
      rho <= max(|1 - eta_min * mu|, |1 - eta_max * L|) if H is known PSD with
      lambda_min >= mu (contractive regime available).
- Tier A (KW92 high-probability randomized bound via HVP on H^2) lands in M2.

NOTE (non-symmetry): for heterogeneous eta, I - diag(eta)H is similar to the
symmetric I - D^{1/2} H D^{1/2} via D^{1/2}; theory work (T1) will track the
error in the D^{-1}-weighted norm to avoid the sqrt(eta_max/eta_min) factor.
The oracle tier computes the exact unweighted 2-norm, so M0 validation is
unaffected by this refinement.
"""

from __future__ import annotations

import torch


class SpectralOracle:
    """Exact bounds from an explicitly known, constant Hessian Q (p x p)."""

    def __init__(self, Q: torch.Tensor):
        self.Q = Q
        self._eye = torch.eye(Q.shape[0], dtype=Q.dtype, device=Q.device)

    def bounds(self, eta_vec: torch.Tensor):
        DH = eta_vec.to(self.Q.dtype).unsqueeze(1) * self.Q
        rho = torch.linalg.matrix_norm(self._eye - DH, ord=2).item()
        kappa = torch.linalg.matrix_norm(DH, ord=2).item()
        return rho, kappa


class SpectralPrior:
    """Tier C: prior smoothness constant L >= ||H_t||_2 for all t.

    If psd_mu is given (H_t PSD with lambda_min >= mu > 0), the contractive
    bound applies; otherwise the safe indefinite bound rho = 1 + eta_max * L.
    """

    def __init__(self, L_smooth: float, psd_mu: float | None = None):
        self.L = float(L_smooth)
        self.mu = psd_mu

    def bounds(self, eta_vec: torch.Tensor):
        eta_max = float(eta_vec.max())
        kappa = eta_max * self.L
        if self.mu is not None:
            eta_min = float(eta_vec.min())
            rho = max(abs(1.0 - eta_min * self.mu), abs(1.0 - eta_max * self.L))
        else:
            rho = 1.0 + kappa
        return rho, kappa


class SpectralKW:
    """Tier A: high-probability upper bound on ||A||_2 via randomized power
    iteration on A^T A with the Kuczynski-Wozniakowski (1992) tail bound.

    For symmetric PSD M and uniform random start, k power steps give
        P[ lambda_hat_k < (1 - eps) * lambda_max ] <= sqrt(pi*p)/2 * (1-eps)^(k-1)
    (KW92; conservative power-method form — Lanczos is tighter, constants to be
    finalized against the paper in M2 theory pass). Inverting at confidence
    delta_t yields the certified bound  lambda_max <= lambda_hat_k / (1-eps).

    A v = v - eta*(H v),  A^T u = u - H(eta*u): one M-matvec = 2 HVPs.
    Per-step failure budget delta_t = 6*delta/(pi^2 (t+1)^2) keeps the anytime
    union bound sum <= delta.
    """

    def __init__(self, p: int, eps: float = 0.25, delta: float = 1e-3,
                 seed: int = 0, method: str = "lanczos"):
        self.p = p
        self.eps = eps
        self.delta = delta
        self.gen = torch.Generator().manual_seed(seed)
        self.t = 0
        self.method = method  # "power" | "lanczos"

    def _k_steps(self, delta_t: float) -> int:
        import math
        if self.method == "lanczos":
            # KW92 Lanczos tail (verified against Urschel 2020,
            # arXiv:2003.09362 eq.(5), quoting KW92):
            #   P[(l1 - l1_hat)/l1 > eps] <= 1.648 sqrt(p) exp(-sqrt(eps)(2k-1))
            # for symmetric PD targets, uniform-sphere start, exact arithmetic.
            c = 1.648 * math.sqrt(self.p)
            k = 0.5 * (math.log(max(c / delta_t, 1.0)) / math.sqrt(self.eps) + 1)
        else:
            c = math.sqrt(math.pi * self.p) / 2.0
            k = 1 + math.log(max(c / delta_t, 1.0)) / (-math.log1p(-self.eps))
        return max(4, int(math.ceil(k)))

    def bounds(self, hvp, eta_vec: torch.Tensor):
        """hvp: v -> H v closure at current theta. Returns (rho, kappa)."""
        import math
        delta_t = 6.0 * self.delta / (math.pi ** 2 * (self.t + 1) ** 2)
        k = self._k_steps(delta_t)
        eta = eta_vec

        def mv_AtA(v):
            av = v - eta * hvp(v)
            return av - hvp(eta * av)

        def mv_BtB(v):  # B = diag(eta) H
            bv = eta * hvp(v)
            return hvp(eta * bv)

        est = self._lanczos if self.method == "lanczos" else self._power
        rho2 = est(mv_AtA, eta.device, eta.dtype, k)
        kap2 = est(mv_BtB, eta.device, eta.dtype, k)
        rho = math.sqrt(max(rho2, 0.0) / (1.0 - self.eps))
        kappa = math.sqrt(max(kap2, 0.0) / (1.0 - self.eps))
        self.t += 1
        return rho, kappa

    def _power(self, mv, device, dtype, k: int) -> float:
        v = torch.randn(self.p, generator=self.gen).to(device=device, dtype=dtype)
        v = v / torch.linalg.vector_norm(v)
        lam = 0.0
        for _ in range(k):
            w = mv(v)
            lam = float(v @ w)
            n = float(torch.linalg.vector_norm(w))
            if n < 1e-30:
                return 0.0
            v = w / n
        return max(lam, 0.0)

    def _lanczos(self, mv, device, dtype, k: int) -> float:
        """k-step Lanczos with full reorthogonalization; max Ritz value.

        The orthogonal basis V (p x k) is the memory hog at scale
        (124M x 25 fp32 ~ 12GB): store it on CPU when p is large and
        stream the current iterate for reorthogonalization — exact
        arithmetic preserved, GPU peak stays O(p)."""
        v_device = device if self.p * k <= 2 * 10 ** 8 else "cpu"
        V = torch.empty(self.p, k, device=v_device, dtype=dtype)
        alphas, betas = [], []
        v = torch.randn(self.p, generator=self.gen).to(device=device, dtype=dtype)
        v = v / torch.linalg.vector_norm(v)
        V[:, 0] = v.to(v_device)
        v_prev = None
        beta = 0.0
        for j in range(k):
            w = mv(v)
            if j > 0:
                w = w - beta * v_prev
            alpha = float(v @ w)
            alphas.append(alpha)
            w = w - alpha * v
            # full reorthogonalization against the (possibly CPU-resident) basis
            Vj = V[:, : j + 1]
            if v_device == "cpu":
                w_cpu = w.to("cpu")
                coef = Vj.T @ w_cpu
                w = (w_cpu - Vj @ coef).to(device)
            else:
                w = w - Vj @ (Vj.T @ w)
            beta = float(torch.linalg.vector_norm(w))
            if beta < 1e-30 or j == k - 1:
                break
            betas.append(beta)
            v_prev = v
            v = w / beta
            V[:, j + 1] = v.to(v_device)
        Tm = torch.diag(torch.tensor(alphas, dtype=torch.float64))
        if betas:
            off = torch.tensor(betas[: len(alphas) - 1], dtype=torch.float64)
            Tm += torch.diag(off, 1) + torch.diag(off, -1)
        try:
            evals = torch.linalg.eigvalsh(Tm)
            lam_hat = float(evals.max())
        except Exception:
            # ill-conditioned tridiagonal: Gershgorin upper bound on T's
            # spectrum >= max Ritz value -> certificate stays a valid upper
            # bound (only looser); avoids LAPACK convergence failures
            rowsum = Tm.abs().sum(dim=1) - Tm.diagonal().abs()
            lam_hat = float((Tm.diagonal() + rowsum).max())
        return max(lam_hat, 0.0)


class DriftHold:
    """Certified spectral bounds BETWEEN probes under drift (A4', v2).

    Assumptions (theorem 1-γ-lazy v2; PROOF_AUDIT #7/#8):
      A4'a  per-sample Hessian Lipschitz: ||H_ξ(x)-H_ξ(y)|| <= M_H ||x-y||  ∀ξ
      A4'b  batch dispersion:            ||H_ξ(x)-H_ξ'(x)|| <= nu_H        ∀ξ,ξ'

    After a probe (rho_0, kappa_0) at (theta_0, eta_0), with observable path
    length D_t and eta drift Δη_t = ||eta_t - eta_0||_∞:

        dH    = M_H * D_t + nu_H
        Hbar  = kappa_0 / eta_min_0 + dH          (certified ||H_t|| bound)
        iota  = Δη_t * Hbar + eta_max_0 * dH
        rho_t <= rho_0 + iota,   kappa_t <= kappa_0 + iota.

    M_H, nu_H are priors calibrated online (diagnostics recorded) and stated
    as assumptions of the theorem.
    """

    def __init__(self, M_H: float, nu_H: float = 0.0):
        self.M_H = float(M_H)
        self.nu_H = float(nu_H)
        self.rho0 = None
        self.kappa0 = None
        self.eta_min0 = None
        self.eta_max0 = None
        self.eta0 = None
        self.path = 0.0
        self._last_probe_rho = None
        self.mh_observed = []   # |Δrho|/path between consecutive probes
        self.nu_observed = []   # |Δrho| at near-zero path (batch dispersion)

    def probe(self, rho: float, kappa: float, eta_vec=None):
        if self._last_probe_rho is not None:
            d = self.path
            if d > 1e-12:
                self.mh_observed.append(abs(rho - self._last_probe_rho) / d)
            else:
                self.nu_observed.append(abs(rho - self._last_probe_rho))
        self._last_probe_rho = rho
        self.rho0, self.kappa0 = rho, kappa
        if eta_vec is not None:
            self.eta0 = eta_vec.detach().clone()
            self.eta_min0 = float(eta_vec.min())
            self.eta_max0 = float(eta_vec.max())
        self.path = 0.0

    def step(self, step_norm: float):
        """Record ||theta_{t+1} - theta_t|| after each inner step."""
        self.path += step_norm

    def bounds(self, eta_now=None):
        """eta_now: current per-coordinate LR vector (for the η-drift term).
        Falling back to a float (legacy) treats it as eta_max with Δη = 0 —
        valid ONLY when λ is frozen between probes."""
        if self.rho0 is None:
            raise RuntimeError("DriftHold.bounds before first probe")
        dH = self.M_H * self.path + self.nu_H
        if self.eta0 is not None and eta_now is not None \
                and not isinstance(eta_now, float):
            d_eta = float((eta_now - self.eta0).abs().max())
            eta_max0 = self.eta_max0
            hbar = self.kappa0 / max(self.eta_min0, 1e-30) + dH
        else:
            d_eta = 0.0
            eta_max0 = float(eta_now) if eta_now is not None else self.eta_max0
            hbar = 0.0
        iota = d_eta * hbar + (eta_max0 or 0.0) * dH
        return self.rho0 + iota, self.kappa0 + iota


class DriftHoldFailClosed(DriftHold):
    """DriftHold + a FAIL-CLOSED monitor on the M_H prior (control study F).

    The A4' hold predicts, between two probes separated by path length D,
        |rho_probe - rho_prev|  ~<  eta_max0 * (M_H * D + nu_H),
    so the probe-to-probe OBSERVED Hessian drift rate is estimated by

        M_obs = |rho_probe - rho_prev| / (eta_max0 * D).

    Rules (all online-checkable, no oracle):
      * M_obs > 0.5 * M_H  -> the prior is close to being exhausted: schedule
        an IMMEDIATE re-probe at the next step (refreshes the certificate at
        the current theta instead of extrapolating with a stale bound).
      * M_obs > M_H        -> the prior is *violated*: the certificate between
        the last two probes cannot be trusted, so the gate is CLOSED for ALL
        coordinates until a probe measures M_obs <= M_H again; each such
        transition increments `failclosed_events`.

    A forced (unscheduled) re-probe never forces another one, so the probe
    budget is at most doubled.  Nothing here touches DriftHold's own bounds:
    the base recursion, its diagnostics, and every existing call site are
    untouched -- this subclass only adds a monitor and a gate override.
    """

    def __init__(self, M_H: float, nu_H: float = 0.0):
        super().__init__(M_H, nu_H)
        self.m_obs = []             # observed Hessian drift rate per probe
        self.failclosed_events = 0  # closed-gate transitions
        self.n_reprobe_req = 0      # immediate re-probes requested
        self.closed = False         # gate currently forced shut
        self.want_reprobe = False   # request an immediate re-probe

    def probe(self, rho: float, kappa: float, eta_vec=None):
        prev_rho = self._last_probe_rho
        prev_eta_max = self.eta_max0
        d = self.path
        super().probe(rho, kappa, eta_vec)
        self.want_reprobe = False
        if prev_rho is None or d <= 1e-12 or not prev_eta_max:
            return
        rate = abs(rho - prev_rho) / (prev_eta_max * d)
        self.m_obs.append(rate)
        if rate > self.M_H:
            if not self.closed:
                self.failclosed_events += 1
            self.closed = True
            self.want_reprobe = True
            self.n_reprobe_req += 1
        else:
            self.closed = False
            if rate > 0.5 * self.M_H:
                self.want_reprobe = True
                self.n_reprobe_req += 1


class DriftHoldAdaptiveMH(DriftHoldFailClosed):
    """Round-4 (E2/E1 `--adaptive-mh`): ONLINE-ENFORCEABLE drift envelope.

    Instead of asserting a fixed Hessian-Lipschitz prior M_H, the envelope in
    force is re-stated at every probe as

        M_H,t = max(M_H_floor, KAPPA * max_{s <= t} M_obs,s)

    where M_obs,s is the probe-to-probe observed drift rate already computed by
    DriftHoldFailClosed.  With KAPPA >= 1 the deployed envelope is BY
    CONSTRUCTION never below any diagnostic the run has observed, and the floor
    keeps it never below the paper's deployed prior, so the arm is never LESS
    conservative than the shipped certificate.

    The fail-closed monitor is evaluated against the envelope that was in force
    over the interval just traversed (i.e. BEFORE this probe's observation is
    folded in) -- that is the envelope under which the interval's extrapolated
    bound was certified.  Raising the envelope afterwards therefore does not
    retroactively excuse the interval, but it does make every SUBSEQUENT
    interval conservative w.r.t. everything seen so far.

    `probe_log` rows are [M_H_in_force_before, M_obs, M_H_after, closed_after].
    """

    def __init__(self, M_H_floor: float, kappa: float, nu_H: float = 0.0):
        super().__init__(M_H=float(M_H_floor), nu_H=nu_H)
        self.M_H_floor = float(M_H_floor)
        self.kappa = float(kappa)
        self.m_obs_max = 0.0
        self.probe_log = []      # [M_H_before, M_obs, M_H_after, closed]
        self.n_raises = 0

    def probe(self, rho: float, kappa: float, eta_vec=None):
        n_before = len(self.m_obs)
        mh_before = self.M_H
        super().probe(rho, kappa, eta_vec)
        if len(self.m_obs) > n_before:
            rate = self.m_obs[-1]
            if rate > self.m_obs_max:
                self.m_obs_max = rate
            new = max(self.M_H_floor, self.kappa * self.m_obs_max)
            if new > self.M_H:
                self.n_raises += 1
            self.M_H = new
            self.probe_log.append([mh_before, rate, new, int(self.closed)])


class BiasCertificate:
    """Scalar anytime recursion  e_{t+1} = rho_t e_t + eta_err_t  with e_0 = 0.

    Maintains e_t >= ||S_t - S_hat_t||_F (T1). The hypergradient bias bound is
    beta_t = e_t * ||grad_theta ell_t||_2.
    """

    def __init__(self):
        self.e = 0.0
        self.history = []

    def reset(self):
        self.e = 0.0
        self.history = []

    def step(self, rho: float, eta_err: float) -> float:
        self.e = rho * self.e + eta_err
        self.history.append(self.e)
        return self.e

    def beta(self, meta_grad_norm: float) -> float:
        return self.e * meta_grad_norm
