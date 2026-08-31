"""Hyperparameter group structure.

Each of the p flat parameter coordinates belongs to exactly one of m
hyperparameter groups (e.g. one per layer). The inner update is

    theta_{t+1} = theta_t - eta(lambda) * grad_t,   eta_i = exp(lambda_{g(i)})

so the sensitivity recursion S_{t+1} = A_t S_t + B_t has a direct term B_t
whose column j is supported exactly on group j:

    B_t[i, j] = -1{g(i)=j} * eta_i * grad_{t,i}.

A "group-aligned" p x m matrix (column j supported on group j) is fully
described by a single p-vector; GroupSpec provides the packing/unpacking.
"""

from __future__ import annotations

import torch


class GroupSpec:
    def __init__(self, group_index: torch.Tensor, m: int):
        if group_index.dtype != torch.long:
            raise TypeError("group_index must be int64")
        if int(group_index.max()) >= m or int(group_index.min()) < 0:
            raise ValueError("group indices out of range")
        self.g = group_index
        self.m = m
        self.p = group_index.numel()
        self._arange = torch.arange(self.p, device=group_index.device)

    @staticmethod
    def from_sizes(sizes, device="cpu") -> "GroupSpec":
        """Groups given as contiguous block sizes in flat parameter order."""
        idx = torch.cat([
            torch.full((s,), j, dtype=torch.long) for j, s in enumerate(sizes)
        ]).to(device)
        return GroupSpec(idx, len(sizes))

    @staticmethod
    def from_named_parameters(named_params, group_of_name) -> "GroupSpec":
        """group_of_name: name -> group id (contiguous 0..m-1)."""
        idx_parts, m = [], 0
        for name, param in named_params:
            j = group_of_name(name)
            m = max(m, j + 1)
            idx_parts.append(torch.full((param.numel(),), j, dtype=torch.long))
        return GroupSpec(torch.cat(idx_parts), m)

    def to(self, device) -> "GroupSpec":
        return GroupSpec(self.g.to(device), self.m)

    # ---- eta(lambda) ----
    def eta_vec(self, lam: torch.Tensor) -> torch.Tensor:
        """Per-coordinate learning rate from per-group log-LR lambda (m,)."""
        return torch.exp(lam)[self.g]

    # ---- aligned (block) matrices as p-vectors ----
    def aligned_to_matrix(self, d: torch.Tensor) -> torch.Tensor:
        """p-vector -> dense p x m with M[i, g(i)] = d_i (zeros elsewhere)."""
        M = d.new_zeros(self.p, self.m)
        M[self._arange, self.g] = d
        return M

    def matrix_aligned_part(self, M: torch.Tensor) -> torch.Tensor:
        """Extract the aligned entries M[i, g(i)] as a p-vector."""
        return M[self._arange, self.g]

    def column_mask(self, j: int) -> torch.Tensor:
        return (self.g == j)

    def aligned_column(self, d: torch.Tensor, j: int) -> torch.Tensor:
        """Column j of the aligned matrix described by d (p-vector, masked)."""
        out = torch.zeros_like(d)
        mask = self.column_mask(j)
        out[mask] = d[mask]
        return out

    def group_sums(self, x: torch.Tensor) -> torch.Tensor:
        """(m,) vector with sum of x over each group; = D^T 1 pattern helper."""
        out = x.new_zeros(self.m)
        out.scatter_add_(0, self.g, x)
        return out

    def aligned_T_matvec(self, d: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """D^T v for the aligned matrix D described by d: (m,)."""
        return self.group_sums(d * v)
