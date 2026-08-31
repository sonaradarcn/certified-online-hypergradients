"""Deterministic data streams for E0/E2 experiments."""

from __future__ import annotations

import os

import numpy as np
import torch

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


def mackey_glass(n: int, tau: int = 17, seed: int = 0, warmup: int = 500):
    """Standard Mackey-Glass delay DE, Euler-discretized (dt=1)."""
    rng = np.random.default_rng(seed)
    hist = 1.2 + 0.1 * rng.standard_normal(tau + 1)
    x = np.empty(n + warmup)
    x[: tau + 1] = hist
    for t in range(tau, n + warmup - 1):
        x[t + 1] = x[t] + 0.2 * x[t - tau] / (1 + x[t - tau] ** 10) - 0.1 * x[t]
    s = x[warmup:]
    return (s - s.mean()) / (s.std() + 1e-12)


def _lorenz_raw(n: int, seed: int = 0, rho: float = 28.0, dt: float = 0.01,
                subsample: int = 2):
    """x-component of Lorenz-63 (RK4), unstandardized."""
    rng = np.random.default_rng(seed)
    s = np.array([1.0, 1.0, 1.0]) + 0.1 * rng.standard_normal(3)

    def f(v):
        x, y, z = v
        return np.array([10.0 * (y - x), x * (rho - z) - y,
                         x * y - 8.0 / 3.0 * z])

    total = n * subsample + 2000
    out = np.empty(total)
    for i in range(total):
        k1 = f(s); k2 = f(s + 0.5 * dt * k1)
        k3 = f(s + 0.5 * dt * k2); k4 = f(s + dt * k3)
        s = s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i] = s[0]
    return out[2000::subsample][:n]


def lorenz_x(n: int, seed: int = 0, dt: float = 0.01, subsample: int = 2):
    """x-component of Lorenz-63 (RK4), standardized."""
    x = _lorenz_raw(n, seed=seed, dt=dt, subsample=subsample)
    return (x - x.mean()) / (x.std() + 1e-12)


def mackey_glass_drift(n: int, seed: int = 0):
    """Nonstationary Mackey-Glass: tau regime switches at thirds
    (17 -> 25 -> 12). Each segment standardized by GLOBAL stats so the
    switch is a genuine dynamics change, not a scale artifact."""
    seg = n // 3
    parts = [mackey_glass(seg, tau=17, seed=seed),
             mackey_glass(seg, tau=25, seed=seed + 1),
             mackey_glass(n - 2 * seg, tau=12, seed=seed + 2)]
    s = np.concatenate(parts)
    return (s - s.mean()) / (s.std() + 1e-12)


def apply_scale_shift(series, factor: float):
    """Round-4 amplitude shift: multiply the MIDDLE third of a series by
    `factor`, leaving the outer thirds untouched (schedule 1 -> F -> 1).

    The thirds are the SAME boundaries `mackey_glass_drift` / `lorenz_x_drift`
    use for their dynamics switch (`seg = n // 3`, `2 * seg`), so on the
    `OrderedWindowStream` the amplitude shift lands at exactly the same stream
    steps as the tau/rho regime switch (4004 and 8007 for n=25000, T=12000).

    Both the window inputs and the next-step target are drawn from this same
    array, so both are scaled.  `factor == 1` (or None) returns the input
    unchanged, i.e. the default path is untouched.
    """
    if factor is None or factor == 1.0:
        return series
    n = len(series)
    seg = n // 3
    out = series.copy()
    out[seg:2 * seg] = out[seg:2 * seg] * float(factor)
    return out


def lorenz_x_drift(n: int, seed: int = 0):
    """Nonstationary Lorenz: rho switches 28 -> 40 -> 22 at thirds."""
    seg = n // 3
    parts = [_lorenz_raw(seg, seed=seed, rho=28.0),
             _lorenz_raw(seg, seed=seed + 1, rho=40.0),
             _lorenz_raw(n - 2 * seg, seed=seed + 2, rho=22.0)]
    s = np.concatenate(parts)
    return (s - s.mean()) / (s.std() + 1e-12)


def _download_text(url: str) -> str:
    import ssl
    import urllib.request
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(url, timeout=60, context=ctx) as r:
        return r.read().decode("utf-8", errors="replace")


def sunspot_series(n: int | None = None):
    """Monthly total sunspot number (SIDC SN_m_tot_V2.0), standardized.
    Natural length ~3300 points; n is a cap, not a target."""
    cache = os.path.join(DATA_ROOT, "sunspot.npy")
    if os.path.exists(cache):
        s = np.load(cache)
    else:
        txt = _download_text("https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv")
        vals = []
        for line in txt.strip().splitlines():
            parts = line.split(";")
            if len(parts) >= 4:
                v = float(parts[3])
                if v >= 0:  # -1 marks missing
                    vals.append(v)
        s = np.array(vals)
        os.makedirs(DATA_ROOT, exist_ok=True)
        np.save(cache, s)
    if n:
        s = s[:n]
    return (s - s.mean()) / (s.std() + 1e-12)


def _download_bytes(url: str) -> bytes:
    import ssl
    import urllib.request
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(url, timeout=120, context=ctx) as r:
        return r.read()


def santa_fe_laser(n: int | None = None):
    """Santa Fe competition laser intensity (extended series, 10093 points,
    as shipped by reservoirpy; fallback: original 1000-point laser.txt)."""
    cache = os.path.join(DATA_ROOT, "santafe.npy")
    if os.path.exists(cache):
        s = np.load(cache)
    else:
        import io
        os.makedirs(DATA_ROOT, exist_ok=True)
        try:
            blob = _download_bytes(
                "https://raw.githubusercontent.com/reservoirpy/reservoirpy/"
                "master/reservoirpy/datasets/santafe_laser.npy")
            s = np.load(io.BytesIO(blob)).astype(np.float64).ravel()
        except Exception:
            txt = _download_text(
                "https://raw.githubusercontent.com/MaterialMan/CHARC/master/"
                "Support%20files/other/Datasets/laser.txt")
            s = np.array([float(x) for x in txt.split()])
        np.save(cache, s)
    if n:
        s = s[:n]
    return (s - s.mean()) / (s.std() + 1e-12)


class WindowStream:
    """Sliding-window next-step prediction batches from a 1-D series."""

    def __init__(self, series: np.ndarray, window: int = 20,
                 batch_size: int = 64, seed: int = 1, device="cuda"):
        self.x = torch.tensor(series, dtype=torch.float32)
        self.window = window
        self.bs = batch_size
        self.gen = torch.Generator().manual_seed(seed)
        self.device = device
        self.n = len(series) - window - 1

    def next(self):
        idx = torch.randint(0, self.n, (self.bs,), generator=self.gen)
        xb = torch.stack([self.x[i:i + self.window] for i in idx])
        yb = self.x[idx + self.window].unsqueeze(1)
        return xb.unsqueeze(-1).to(self.device), yb.to(self.device)


class OrderedWindowStream:
    """Time-ordered streaming batches: step t draws windows from a small
    recency band around position t * n / T_total — preserves nonstationarity
    (regime switches arrive in order), unlike the stationary WindowStream."""

    def __init__(self, series: np.ndarray, total_steps: int, window: int = 20,
                 batch_size: int = 64, band: int = 500, seed: int = 1,
                 device="cuda"):
        self.x = torch.tensor(series, dtype=torch.float32)
        self.window = window
        self.bs = batch_size
        self.band = band
        self.gen = torch.Generator().manual_seed(seed)
        self.device = device
        self.n = len(series) - window - 1
        self.T = total_steps
        self.t = 0

    def next(self):
        center = int(self.t / max(self.T - 1, 1) * (self.n - 1))
        lo = max(0, center - self.band)
        hi = min(self.n, center + 1)
        idx = lo + torch.randint(0, max(hi - lo, 1), (self.bs,),
                                 generator=self.gen)
        xb = torch.stack([self.x[i:i + self.window] for i in idx])
        yb = self.x[idx + self.window].unsqueeze(1)
        self.t += 1
        return xb.unsqueeze(-1).to(self.device), yb.to(self.device)


class ImageStream:
    def __init__(self, dataset_name: str, batch_size: int, seed: int = 1,
                 device="cuda"):
        import torchvision
        import torchvision.transforms as T
        os.makedirs(DATA_ROOT, exist_ok=True)
        if dataset_name == "mnist":
            ds = torchvision.datasets.MNIST(
                DATA_ROOT, train=True, download=True,
                transform=T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))]))
        elif dataset_name == "cifar10":
            ds = torchvision.datasets.CIFAR10(
                DATA_ROOT, train=True, download=True,
                transform=T.Compose([T.ToTensor(), T.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]))
        else:
            raise ValueError(dataset_name)
        self.xs = torch.stack([ds[i][0] for i in range(len(ds))])
        self.ys = torch.tensor([ds[i][1] for i in range(len(ds))])
        self.bs = batch_size
        self.gen = torch.Generator().manual_seed(seed)
        self.device = device

    def next(self):
        idx = torch.randint(0, len(self.ys), (self.bs,), generator=self.gen)
        return self.xs[idx].to(self.device), self.ys[idx].to(self.device)


PTB_URL = ("https://raw.githubusercontent.com/wojzaremba/lstm/master/data/"
           "ptb.train.txt")


class CharStream:
    """Char-level LM batches from Penn Treebank (direct download)."""

    def __init__(self, seq_len: int = 64, batch_size: int = 16, seed: int = 1,
                 device="cuda", max_chars: int = 2_000_000):
        cache = os.path.join(DATA_ROOT, "ptb_char.pt")
        if os.path.exists(cache):
            blob = torch.load(cache, weights_only=False)
            self.data, self.stoi = blob["data"], blob["stoi"]
        else:
            import urllib.request
            with urllib.request.urlopen(PTB_URL, timeout=60) as r:
                text = r.read().decode("utf-8")[:max_chars]
            chars = sorted(set(text))
            self.stoi = {c: i for i, c in enumerate(chars)}
            self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
            os.makedirs(DATA_ROOT, exist_ok=True)
            torch.save({"data": self.data, "stoi": self.stoi}, cache)
        self.vocab = len(self.stoi)
        self.seq_len = seq_len
        self.bs = batch_size
        self.gen = torch.Generator().manual_seed(seed)
        self.device = device

    def next(self):
        idx = torch.randint(0, len(self.data) - self.seq_len - 1, (self.bs,),
                            generator=self.gen)
        x = torch.stack([self.data[i:i + self.seq_len] for i in idx])
        y = torch.stack([self.data[i + 1:i + self.seq_len + 1] for i in idx])
        return x.to(self.device), y.to(self.device)
