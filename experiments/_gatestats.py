"""Round-4 additive instrumentation: empirical |ghat_j| and certificate-scaled
threshold (c * beta_col_j) distributions.

Used by e2_timeseries.py / e3_continual.py / e4_gpt2_tta.py behind the
--log-gate-stats flag ONLY.  When the flag is off the collector is inert (add()
returns immediately, summary() returns None) and no key is written to the
result JSON, so every default code path stays byte-identical to the shipped
runs.

Purpose: the absgate transfer study needs, per domain,

  * where the |ghat_j| distribution actually sits (median / p90 / p99 / max),
  * where COHG's realized certificate threshold c * beta_col_j sits,
  * the scale mismatch factor between the two, and
  * what fraction of coordinate-steps a TRANSFERRED CONSTANT threshold would
    open, versus what the certificate gate actually opens.

All quantities are pooled over every (step, coordinate) pair of the run.
"""

from __future__ import annotations

import math

# the frozen mackey_drift-calibrated absgate constant (results/e2_controls/
# absgate_threshold.json).  Recorded here only so every driver reports the
# would-be open rate of the TRANSFERRED constant, even on runs that do not use
# it as their gate.
FROZEN_ABSGATE_T = 0.05806520209


def _q(sorted_v, p):
    if not sorted_v:
        return None
    n = len(sorted_v)
    return sorted_v[min(n - 1, max(0, int(p * n)))]


def _stats(v):
    v = [x for x in v if x is not None and math.isfinite(x)]
    if not v:
        return None
    s = sorted(v)
    n = len(s)
    mean = sum(s) / n
    return {
        "n": n, "min": s[0], "p10": _q(s, 0.10), "median": s[n // 2],
        "p90": _q(s, 0.90), "p99": _q(s, 0.99), "max": s[-1], "mean": mean,
        "geomean": (math.exp(sum(math.log(x) for x in s if x > 0)
                             / max(sum(1 for x in s if x > 0), 1))
                    if any(x > 0 for x in s) else 0.0),
        "n_zero": sum(1 for x in s if x == 0.0),
    }


class GateStats:
    """Pools |ghat_j| and c * beta_col_j over all (step, coordinate) pairs."""

    def __init__(self, enabled: bool, gate_factor: float = 2.0,
                 const_threshold: float = FROZEN_ABSGATE_T):
        self.on = bool(enabled)
        self.c = float(gate_factor)
        self.T = float(const_threshold)
        self.g = []          # |ghat_j|
        self.b = []          # c * beta_col_j  (certificate-scaled threshold)
        self.r = []          # |ghat_j| / (c * beta_col_j)
        self.n_pair = 0
        self.n_open_const = 0     # |ghat_j| >  T          (transferred const)
        self.n_open_cert = 0      # |ghat_j| >  c*beta_j   (certificate gate)
        self.step_max_g = []      # per-step max |ghat_j| (light trace)
        self.step_med_b = []      # per-step median c*beta_j (light trace)
        self._t = 0

    def add(self, ghat, beta_col=None, every: int = 1):
        """ghat: (m,) tensor.  beta_col: (m,) tensor or None (no certificate)."""
        if not self.on:
            return
        t = self._t
        self._t += 1
        if every > 1 and (t % every):
            return
        ga = [abs(float(x)) for x in ghat.detach().double().cpu().tolist()]
        self.g.extend(ga)
        self.n_pair += len(ga)
        self.n_open_const += sum(1 for x in ga if x > self.T)
        self.step_max_g.append(max(ga) if ga else 0.0)
        if beta_col is not None:
            bb = [self.c * float(x)
                  for x in beta_col.detach().double().cpu().tolist()]
            self.b.extend(bb)
            sb = sorted(bb)
            self.step_med_b.append(sb[len(sb) // 2] if sb else 0.0)
            for x, y in zip(ga, bb):
                if x > y:
                    self.n_open_cert += 1
                self.r.append(x / y if y > 0 else float("inf"))

    def summary(self):
        if not self.on or not self.g:
            return None
        out = {
            "const_threshold": self.T,
            "gate_factor": self.c,
            "n_coord_steps": self.n_pair,
            "ghat_abs": _stats(self.g),
            "cbeta": _stats(self.b) if self.b else None,
            "ratio_ghat_over_cbeta": _stats([x for x in self.r
                                             if math.isfinite(x)])
            if self.r else None,
            "ratio_n_inf": sum(1 for x in self.r if not math.isfinite(x)),
            "frac_open_const": self.n_open_const / max(self.n_pair, 1),
            "frac_open_cert": (self.n_open_cert / max(self.n_pair, 1)
                               if self.b else None),
        }
        gs, bs = out["ghat_abs"], out["cbeta"]
        if gs and bs and bs["median"] and bs["median"] > 0:
            out["scale_mismatch_median"] = self.T / bs["median"]
        else:
            out["scale_mismatch_median"] = None
        if gs and gs["median"] and gs["median"] > 0:
            out["const_over_ghat_median"] = self.T / gs["median"]
        else:
            out["const_over_ghat_median"] = None
        # light per-step traces, downsampled to <= 600 points each
        def thin(v, k=600):
            if len(v) <= k:
                return v
            st = len(v) / float(k)
            return [v[min(len(v) - 1, int(i * st))] for i in range(k)]
        out["step_max_ghat_trace"] = thin(self.step_max_g)
        out["step_med_cbeta_trace"] = thin(self.step_med_b) if self.b else None
        return out
