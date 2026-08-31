"""E0/E2/E3 model zoo — all double-backward-safe by construction.

- ManualGRU: hand-rolled GRU cell (cuDNN RNN kernels lack double backward)
- CharTransformer: manual attention (flash/efficient SDPA lacks double backward)
- SmallResNet: GroupNorm instead of BatchNorm (no buffer mutation under
  functional_call, deterministic loss for HVP)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- MLP
class DeepMLP(nn.Module):
    def __init__(self, widths=(784, 256, 256, 128, 128, 64, 10)):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(widths[:-1], widths[1:]))

    def forward(self, x):
        x = x.flatten(1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.tanh(x)
        return x


# ---------------------------------------------------------------- GRU
class ManualGRU(nn.Module):
    """Batch-first manual GRU + linear readout of final hidden state."""

    def __init__(self, n_in=1, n_hidden=128, n_out=1, n_layers=2):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.w_ih = nn.ParameterList()
        self.w_hh = nn.ParameterList()
        self.b_ih = nn.ParameterList()
        self.b_hh = nn.ParameterList()
        for l in range(n_layers):
            d_in = n_in if l == 0 else n_hidden
            k = 1.0 / math.sqrt(n_hidden)
            self.w_ih.append(nn.Parameter(torch.empty(3 * n_hidden, d_in).uniform_(-k, k)))
            self.w_hh.append(nn.Parameter(torch.empty(3 * n_hidden, n_hidden).uniform_(-k, k)))
            self.b_ih.append(nn.Parameter(torch.zeros(3 * n_hidden)))
            self.b_hh.append(nn.Parameter(torch.zeros(3 * n_hidden)))
        self.readout = nn.Linear(n_hidden, n_out)

    def forward(self, x):  # x: (B, T, n_in)
        B, T, _ = x.shape
        h = [x.new_zeros(B, self.n_hidden) for _ in range(self.n_layers)]
        for t in range(T):
            inp = x[:, t]
            for l in range(self.n_layers):
                gi = inp @ self.w_ih[l].T + self.b_ih[l]
                gh = h[l] @ self.w_hh[l].T + self.b_hh[l]
                i_r, i_z, i_n = gi.chunk(3, dim=1)
                h_r, h_z, h_n = gh.chunk(3, dim=1)
                r = torch.sigmoid(i_r + h_r)
                z = torch.sigmoid(i_z + h_z)
                n = torch.tanh(i_n + r * h_n)
                h[l] = (1 - z) * n + z * h[l]
                inp = h[l]
        return self.readout(h[-1])


# ---------------------------------------------------------------- Transformer
class ManualAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

    def forward(self, x):  # (B, T, D), causal
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        att = att.masked_fill(mask, float("-inf")).softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = ManualAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                nn.Linear(d_ff, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class CharTransformer(nn.Module):
    def __init__(self, vocab, d_model=128, n_heads=4, d_ff=256, n_layers=4,
                 max_len=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(0.01 * torch.randn(max_len, d_model))
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, d_ff) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, idx):  # (B, T) int64
        x = self.emb(idx) + self.pos[: idx.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))  # (B, T, vocab)


# ---------------------------------------------------------------- ResNet (GN)
class GNBasicBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.gn1 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.gn2 = nn.GroupNorm(8, c_out)
        self.short = None
        if stride != 1 or c_in != c_out:
            self.short = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                nn.GroupNorm(8, c_out))

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        sc = x if self.short is None else self.short(x)
        return F.relu(out + sc)


class ResNet18GN(nn.Module):
    """ResNet-18 topology with GroupNorm (double-backward-safe, no buffer
    mutation under functional_call); ~11.2M params for 32x32 inputs.
    GN instead of BN is standard practice when per-sample/functional grads
    are needed; stated explicitly in the paper's setup."""

    def __init__(self, n_classes=100, widths=(64, 128, 256, 512)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, 1, 1, bias=False),
            nn.GroupNorm(8, widths[0]), nn.ReLU())
        stages = []
        c_in = widths[0]
        for i, w in enumerate(widths):
            stride = 1 if i == 0 else 2
            stages.append(GNBasicBlock(c_in, w, stride=stride))
            stages.append(GNBasicBlock(w, w))
            c_in = w
        self.stages = nn.ModuleList(stages)
        self.head = nn.Linear(widths[-1], n_classes)

    def forward(self, x):
        x = self.stem(x)
        for blk in self.stages:
            x = blk(x)
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


class SmallResNet(nn.Module):
    """ResNet-8 style: stem + 3 stages x 1 block, GroupNorm, ~180K params."""

    def __init__(self, n_classes=10, widths=(32, 64, 128)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, 1, 1, bias=False),
            nn.GroupNorm(8, widths[0]), nn.ReLU())
        self.stage1 = GNBasicBlock(widths[0], widths[0])
        self.stage2 = GNBasicBlock(widths[0], widths[1], stride=2)
        self.stage3 = GNBasicBlock(widths[1], widths[2], stride=2)
        self.head = nn.Linear(widths[2], n_classes)

    def forward(self, x):
        x = self.stage3(self.stage2(self.stage1(self.stem(x))))
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))
