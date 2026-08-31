"""COHG: Self-Certifying Online Hypergradients.

Core components:
- GroupSpec: hyperparameter-to-parameter-group structure (per-layer LR etc.)
- ExactFMD: reference O(pm) forward-mode sensitivity recursion (ground truth)
- COHGEstimator: exact aligned block + rank-r streaming residual sketch,
  with exact truncation accounting feeding an anytime bias certificate
- Certificate spectral tiers: oracle (analytic) / prior smoothness / (KW92 in M2)
- GatedController: certificate-gated meta-updates with trust region
"""

from .groups import GroupSpec
from .fmd import ExactFMD
from .estimator import COHGEstimator
from .certificate import SpectralOracle, SpectralPrior
from .controller import CoordGatedController, GatedController, T6ClipController

__all__ = [
    "GroupSpec", "ExactFMD", "COHGEstimator",
    "SpectralOracle", "SpectralPrior", "GatedController",
    "CoordGatedController", "T6ClipController",
]
