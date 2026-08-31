"""Flat-parameter functional wrapper for nn.Modules.

Bridges real models to the flat-theta closures that HVPOracle/ExactFMD expect:

    fm = FlatModule(module, loss)          # snapshots current params as theta0
    theta = fm.flat_params()
    loss = fm.loss_fn(theta, (x, y))       # differentiable in theta

Requirements for double-backward safety (E0+ experiments):
- no cuDNN RNN (implement recurrent cells manually)
- no flash/efficient SDPA (implement attention manually)
- no BatchNorm running-stat mutation (use GroupNorm/LayerNorm)
"""

from __future__ import annotations

import torch
from torch.func import functional_call

from .groups import GroupSpec


class FlatModule:
    def __init__(self, module: torch.nn.Module, loss_reduction):
        """loss_reduction: callable(outputs, targets) -> scalar."""
        self.module = module
        self.names = [n for n, _ in module.named_parameters()]
        params = [p for _, p in module.named_parameters()]
        self.shapes = [p.shape for p in params]
        self.sizes = [p.numel() for p in params]
        self.p = sum(self.sizes)
        self.loss_reduction = loss_reduction
        self.device = params[0].device
        self.dtype = params[0].dtype

    def flat_params(self) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1)
                          for _, p in self.module.named_parameters()])

    def unflatten(self, theta: torch.Tensor) -> dict:
        out, off = {}, 0
        for name, shape, size in zip(self.names, self.shapes, self.sizes):
            out[name] = theta[off:off + size].view(*shape)
            off += size
        return out

    def loss_fn(self, theta: torch.Tensor, batch) -> torch.Tensor:
        x, y = batch
        out = functional_call(self.module, self.unflatten(theta), (x,))
        return self.loss_reduction(out, y)

    def group_spec(self, group_of_name=None) -> GroupSpec:
        """Default: one group per parameter tensor."""
        if group_of_name is None:
            uniq = {n: i for i, n in enumerate(self.names)}
            group_of_name = lambda n: uniq[n]
        seen: dict[int, int] = {}
        idx_parts = []
        for name, size in zip(self.names, self.sizes):
            j_raw = group_of_name(name)
            if j_raw not in seen:
                seen[j_raw] = len(seen)
            idx_parts.append(torch.full((size,), seen[j_raw], dtype=torch.long))
        return GroupSpec(torch.cat(idx_parts).to(self.device), len(seen))
