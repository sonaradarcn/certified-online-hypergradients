"""COHG estimator: exact aligned block + rank-r streaming residual sketch.

State:  S_hat_t = D_t + L_t R_t^T
    d (p,)    -- the aligned block D_t (column j supported on group j)
    L (p, r), R (m, r) -- residual sketch

Refresh step (every K-th step; costs m + r HVPs):
    M      = A_t S_hat_t + B_t          (exact, dense in columns)
    d+     = aligned part of M          (exact bookkeeping, no error)
    W      = M - aligned(d+)            (off-block residual)
    L+,R+  = rank-r SVD truncation of W
    eta_err_t = sqrt(sum of discarded singular values^2)   -- EXACT

Lazy step (no HVP at all):
    d+     = d + b_t   with  b_t = -eta * g_t   (B_t absorbed exactly)
    eta_err_t = kappa_t * ||S_hat_t||_F         -- certified staleness charge
    (since S_hat_{t+1} - (A_t S_hat_t + B_t) = (I - A_t) S_hat_t)

Certificate:  e_{t+1} = rho_t * e_t + eta_err_t  (see certificate.py).

Endpoints: r = 0 -> "generalized HD with certificate"; r >= m on refresh-every
step -> exact FMD (W has rank <= m, nothing discarded, e_t = 0).

M0 implementation is dense-friendly (materializes p x m intermediates on
refresh steps); the chunked-column scale path lands with milestone M6.
"""

from __future__ import annotations

import math

import torch

from .certificate import BiasCertificate
from .groups import GroupSpec
from .hvp import HVPOracle


class COHGEstimator:
    def __init__(self, spec: GroupSpec, rank: int, refresh_every: int = 1,
                 device="cpu", dtype=torch.float64, discount: float = 1.0,
                 n_dense: int = 0):
        """discount < 1: estimate the discounted sensitivity (see fmd.py).
        Certificate contraction becomes gamma*rho_t: bounded whenever
        gamma < 1/rho_t even in non-contractive (rho_t > 1) phases.

        n_dense: number of GLOBAL hyperparameters (e.g. EWC strength) whose
        direct term spans all parameters. Their sensitivity columns X are
        maintained exactly-dense (no truncation error on refresh; lazy steps
        charge kappa_eff * ||X[:, j]|| like the aligned block). Hypergradient
        and e_col get m + n_dense entries (dense columns appended last)."""
        self.spec = spec
        self.rank = rank
        self.K = refresh_every
        self.gamma = discount
        self.n_dense = n_dense
        self.X = torch.zeros(spec.p, n_dense, device=device, dtype=dtype)
        self.d = torch.zeros(spec.p, device=device, dtype=dtype)
        self.L = torch.zeros(spec.p, 0, device=device, dtype=dtype)
        self.R = torch.zeros(spec.m, 0, device=device, dtype=dtype)
        self.cert = BiasCertificate()
        # column-wise certificate: e_col[j] >= ||(S~ - S_hat)[:, j]||_2.
        # Same induction per column (columns of A E + Delta); strictly more
        # informative than the Frobenius scalar for per-coordinate gating.
        self.e_col = torch.zeros(spec.m + n_dense, dtype=torch.float64)
        self.t = 0
        self.hvp_count = 0

    def reset_state(self):
        """Zero the estimator and certificate (used on divergence recovery)."""
        self.d.zero_()
        self.L = self.L[:, :0]
        self.R = self.R[:, :0]
        self.X.zero_()
        self.cert.reset()
        self.e_col = torch.zeros(self.spec.m + self.n_dense,
                                 dtype=torch.float64)

    # ---- exact Frobenius norm of S_hat (O(pr + mr^2)) ----
    def shat_fro(self) -> float:
        d, L, R, g = self.d, self.L, self.R, self.spec.g
        fro2 = float(d @ d)
        if L.shape[1] > 0:
            cross = float((d * (L * R[g]).sum(dim=1)).sum())
            gram = (L.T @ L) @ (R.T @ R)
            fro2 += 2.0 * cross + float(torch.trace(gram))
        return max(fro2, 0.0) ** 0.5

    def dense_shat(self) -> torch.Tensor:
        """Materialize S_hat (testing/small scale only)."""
        M = self.spec.aligned_to_matrix(self.d)
        if self.L.shape[1] > 0:
            M = M + self.L @ self.R.T
        if self.n_dense > 0:
            M = torch.cat([M, self.X], dim=1)
        return M

    def hypergrad(self, meta_grad: torch.Tensor) -> torch.Tensor:
        """S_hat^T v in O(p(1 + r + n_dense)); (m + n_dense,)."""
        v = meta_grad.to(self.d.dtype)
        out = self.spec.aligned_T_matvec(self.d, v)
        if self.L.shape[1] > 0:
            out = out + self.R @ (self.L.T @ v)
        if self.n_dense > 0:
            out = torch.cat([out, self.X.T @ v])
        return out

    def step(self, oracle: HVPOracle | None, lam: torch.Tensor,
             grad: torch.Tensor, rho: float, kappa: float,
             b_dense: torch.Tensor | None = None,
             eta_vec: torch.Tensor | None = None) -> None:
        """Advance S_hat_t -> S_hat_{t+1}; charge the certificate.

        oracle: HVP oracle at (theta_t, batch_t); may be None on lazy steps.
        grad: g_t (needed for B_t on every step).
        rho, kappa: certified spectral bounds for this step's A_t.
        b_dense: (p, n_dense) direct term of the global hyperparameters.
        eta_vec: per-coordinate LR override (defaults to spec.eta_vec(lam[:m]);
            pass explicitly when lam carries extra global coordinates).
        """
        spec = self.spec
        if eta_vec is None:
            eta_vec = spec.eta_vec(lam[: spec.m])
        eta = eta_vec.to(self.d.dtype)
        b = -(eta * grad.to(self.d.dtype))
        is_refresh = (self.t % self.K == 0)

        if is_refresh:
            if oracle is None:
                raise ValueError("refresh step requires an HVP oracle")
            work_dtype = oracle.grad.dtype
            large = spec.p * spec.m > 2 * 10 ** 8
            if not large:
                # dense-friendly path (small/medium models)
                D = spec.aligned_to_matrix(self.d)
                HD = oracle.hvp_mat(D.to(work_dtype)).to(self.d.dtype)
                M = D - eta.unsqueeze(1) * HD
                if self.L.shape[1] > 0:
                    HL = oracle.hvp_mat(self.L.to(work_dtype)).to(self.d.dtype)
                    AL = self.L - eta.unsqueeze(1) * HL
                    M = M + AL @ self.R.T
                    self.hvp_count += self.L.shape[1]
                self.hvp_count += spec.m
                W = self.gamma * M + spec.aligned_to_matrix(b)
                del M, D, HD
            else:
                # scale path: single p x m buffer, column-chunked A-application
                W = torch.empty(spec.p, spec.m, device=self.d.device,
                                dtype=self.d.dtype)
                chunk = max(1, oracle.hvp_chunk or 2)
                for j0 in range(0, spec.m, chunk):
                    J = list(range(j0, min(j0 + chunk, spec.m)))
                    DJ = torch.zeros(spec.p, len(J), device=self.d.device,
                                     dtype=self.d.dtype)
                    for i, j in enumerate(J):
                        mask = spec.column_mask(j)
                        DJ[mask, i] = self.d[mask]
                    HDJ = oracle.hvp_mat(DJ.to(work_dtype)).to(self.d.dtype)
                    W[:, J] = DJ - eta.unsqueeze(1) * HDJ
                    del DJ, HDJ
                self.hvp_count += spec.m
                if self.L.shape[1] > 0:
                    HL = oracle.hvp_mat(self.L.to(work_dtype)).to(self.d.dtype)
                    HL.mul_(-eta.unsqueeze(1)).add_(self.L)   # HL <- A L in place
                    W.addmm_(HL, self.R.T)
                    self.hvp_count += self.L.shape[1]
                    self.L = self.L[:, :0]                     # release old L early
                    del HL
                # all HVPs done for this refresh: drop the autograd graph
                # before the big matmuls below (saves ~4GB peak on GPT-2)
                oracle.release()
                W.mul_(self.gamma)
                W[self.spec._arange, self.spec.g] += b
            # exact split: aligned block absorbs its part losslessly
            d_new = spec.matrix_aligned_part(W).clone()
            W[self.spec._arange, self.spec.g] = 0.0
            # rank-r truncation via Gram eigendecomposition (exact SVD for
            # tall-skinny W; avoids cusolver SVD limits at p ~ 1e8)
            # fp64 ACCUMULATION without materializing an fp64 copy of W
            # (p x m fp64 + its square transiently cost +12GB on GPT-2)
            col_norm2 = (torch.linalg.vector_norm(
                W, dim=0, dtype=torch.float64) ** 2).cpu()
            evals = V = None
            if self.rank > 0 and bool(torch.isfinite(col_norm2).all()) \
                    and float(col_norm2.sum()) > 0:
                try:
                    if spec.p * spec.m > 2 * 10 ** 8:
                        # row-chunked fp64 Gram (exact accumulation, O(chunk*m))
                        G = torch.zeros(spec.m, spec.m, device=W.device,
                                        dtype=torch.float64)
                        step_rows = max(1, (8 * 10 ** 6))
                        for i0 in range(0, spec.p, step_rows):
                            Wc = W[i0:i0 + step_rows].double()
                            G += Wc.T @ Wc
                            del Wc
                    else:
                        G = (W.T @ W).double()
                    evals, V = torch.linalg.eigh(G)
                except Exception:
                    # ill-conditioned Gram (divergence regimes): fall through
                    # to the no-sketch branch — ALL of W is discarded mass,
                    # conservative and certificate-valid
                    evals = V = None
            k = 0
            if evals is not None:
                idx = torch.argsort(evals, descending=True)
                evals, V = evals[idx].clamp_min(0), V[:, idx]
                k = min(self.rank, int((evals > 1e-30).sum()))
            if k > 0:
                Vr = V[:, :k].to(W.dtype)
                self.L = W @ Vr                     # = U_r * sigma_r
                self.R = Vr.contiguous()
                # projected energy per column: sum_k (sigma_k V[j,k])^2
                proj2 = ((evals[:k].unsqueeze(0)
                          * V[:, :k].double() ** 2).sum(dim=1)).cpu()
                eta_err = float(evals[k:].sum().clamp_min(0).sqrt())
            else:
                self.L = W.new_zeros(spec.p, 0)
                self.R = W.new_zeros(spec.m, 0)
                proj2 = torch.zeros(spec.m, dtype=torch.float64)
                total = float(col_norm2.sum())
                eta_err = total ** 0.5 if math.isfinite(total) else float("inf")
            eta_err_col = (col_norm2 - proj2).clamp_min(0).sqrt()
            del W
            self.d = d_new
            # dense global columns: exact propagation, zero truncation error
            if self.n_dense > 0:
                HX = oracle.hvp_mat(self.X.to(work_dtype)).to(self.d.dtype)
                self.hvp_count += self.n_dense
                self.X = self.gamma * (self.X - eta.unsqueeze(1) * HX)
                if b_dense is not None:
                    self.X = self.X + b_dense.to(self.X.dtype)
                eta_err_col = torch.cat(
                    [eta_err_col, torch.zeros(self.n_dense, dtype=torch.float64)])
        else:
            # lazy: absorb B_t exactly, charge staleness (I - gamma*A_t) S_hat_t
            kappa_eff = self.gamma * kappa + (1.0 - self.gamma)
            eta_err = kappa_eff * self.shat_fro()
            eta_err_col = kappa_eff * self._shat_col_norms()
            self.d = self.d + b
            if self.n_dense > 0:
                x_norms = torch.linalg.vector_norm(
                    self.X, dim=0).double().cpu()
                eta_err_col = torch.cat([eta_err_col, kappa_eff * x_norms])
                eta_err = (eta_err ** 2 + float(
                    (kappa_eff * x_norms).pow(2).sum())) ** 0.5
                if b_dense is not None:
                    self.X = self.X + b_dense.to(self.X.dtype)

        self.cert.step(self.gamma * rho, eta_err)
        self.e_col = self.gamma * rho * self.e_col + eta_err_col
        self.t += 1

    def _shat_col_norms(self) -> torch.Tensor:
        """Upper bound on ||S_hat[:, j]||_2 per column (m,), float64 cpu.
        Triangle inequality: ||D[:,j]|| + ||L R[j,:]^T||, both exact."""
        d2 = torch.zeros(self.spec.m, dtype=torch.float64)
        d2.scatter_add_(0, self.spec.g.cpu(), (self.d.double() ** 2).cpu())
        d_part = d2.sqrt()
        if self.L.shape[1] > 0:
            G = (self.L.T @ self.L).double().cpu()
            Rc = self.R.double().cpu()
            lr_part = torch.einsum("jr,rs,js->j", Rc, G, Rc).clamp_min(0).sqrt()
            return d_part + lr_part
        return d_part

    @property
    def e(self) -> float:
        return self.cert.e

    def beta(self, meta_grad_norm: float) -> float:
        return self.cert.beta(meta_grad_norm)

    def beta_col(self, meta_grad_norm: float) -> torch.Tensor:
        """Per-coordinate bias bounds: |ghat_j - g_j| <= e_col[j]*||grad ell||."""
        return self.e_col * meta_grad_norm
