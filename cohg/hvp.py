"""Hessian-vector products on flat parameter vectors via double backward.

The loss is a closure loss_fn(theta_flat, batch) -> scalar differentiable in
theta_flat. One make_hvp call performs one forward+backward with create_graph;
each subsequent hvp(v) costs one extra backward.
"""

from __future__ import annotations

import torch


class HVPOracle:
    """Holds the double-backward graph at (theta_t, batch_t).

    Attributes:
        grad: detached gradient g_t at theta_t (p,)
        loss: detached scalar loss
        n_hvp: number of hvp calls performed (cost accounting)
    """

    def __init__(self, loss_fn, theta: torch.Tensor, batch,
                 hvp_chunk: int | None = None):
        """hvp_chunk: max columns per batched double-backward in hvp_mat
        (vmap multiplies activation-grad memory by the chunk size; large
        models need small chunks, e.g. 2-4 for GPT-2 on 20GB)."""
        self._theta = theta.detach().clone().requires_grad_(True)
        loss = loss_fn(self._theta, batch)
        self._g = torch.autograd.grad(loss, self._theta, create_graph=True)[0]
        self.grad = self._g.detach()
        self.loss = loss.detach()
        self.n_hvp = 0
        self.hvp_chunk = hvp_chunk

    def hvp(self, v: torch.Tensor) -> torch.Tensor:
        self.n_hvp += 1
        (Hv,) = torch.autograd.grad(
            self._g, self._theta, grad_outputs=v.to(self._g.dtype),
            retain_graph=True,
        )
        return Hv.detach()

    def hvp_mat(self, V: torch.Tensor) -> torch.Tensor:
        """H @ V for V of shape (p, k): batched double-backward
        (autograd vmap via is_grads_batched), in memory-bounded chunks;
        falls back to a column loop on failure."""
        k = V.shape[1]
        if k == 0:
            return V.clone()
        chunk = self.hvp_chunk or k
        outs = []
        for i in range(0, k, chunk):
            Vc = V[:, i:i + chunk]
            kc = Vc.shape[1]
            self.n_hvp += kc
            try:
                (HV,) = torch.autograd.grad(
                    self._g, self._theta,
                    grad_outputs=Vc.T.contiguous().to(self._g.dtype),
                    retain_graph=True, is_grads_batched=True)
                outs.append(HV.T.detach())
            except RuntimeError:
                self.n_hvp -= kc
                cols = [self.hvp(Vc[:, j]) for j in range(kc)]
                outs.append(torch.stack(cols, dim=1))
        return torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]

    def release(self):
        """Drop the graph explicitly (helps peak memory in tight loops)."""
        self._g = None
        self._theta = None
