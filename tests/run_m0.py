"""M0 sanity suite (tracker R001/R002).

R001a: ExactFMD == independent dense recursion on quadratic (cross-impl check)
R001b: ExactFMD == central finite differences (quadratic + nonconvex MLP)
R002 : COHG certificate validity on quadratic: e_t >= ||S_t - S_hat_t||_F for
       every t, across r in {0, 2, m} x K in {1, 5}; exactness at r=m, K=1.
Plus:  gated controller smoke test.

Run:  python code/tests/run_m0.py   (fp64, CPU — exactness over speed)
"""

from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cohg import COHGEstimator, ExactFMD, GatedController, GroupSpec
from cohg.certificate import SpectralOracle
from cohg.hvp import HVPOracle
from cohg.problems import StochasticQuadratic, TinyMLP

torch.set_default_dtype(torch.float64)
RESULTS = {}
FAILURES = []


def check(name, ok, detail):
    RESULTS[name] = {"pass": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def run_train(problem, spec, theta0, lam, batches, fmd=None, true_S_dense=False):
    """SGD loop; optionally advances ExactFMD and/or the dense true recursion."""
    theta = theta0.clone()
    S_dense = torch.zeros(spec.p, spec.m) if true_S_dense else None
    for batch in batches:
        oracle = HVPOracle(problem.loss_fn, theta, batch)
        if fmd is not None:
            fmd.step(oracle, lam)
        if S_dense is not None:
            eta = spec.eta_vec(lam)
            S_dense = problem.true_S_step(S_dense, eta, oracle.grad, spec)
        theta = theta - spec.eta_vec(lam) * oracle.grad
        oracle.release()
    return theta, S_dense


def finite_diff_S(problem, spec, theta0, lam, batches, eps):
    cols = []
    for j in range(spec.m):
        outs = []
        for sign in (+1.0, -1.0):
            lam_p = lam.clone()
            lam_p[j] += sign * eps
            th, _ = run_train(problem, spec, theta0, lam_p, batches)
            outs.append(th)
        cols.append((outs[0] - outs[1]) / (2 * eps))
    return torch.stack(cols, dim=1)


# ---------------- R001a: FMD vs dense recursion (quadratic) ----------------
prob = StochasticQuadratic(p=60, seed=0, noise=0.1, cond=20.0)
spec = GroupSpec.from_sizes([20, 15, 15, 10])
lam = torch.full((spec.m,), float(torch.log(torch.tensor(0.5))))
theta0 = torch.randn(prob.p, generator=torch.Generator().manual_seed(3))
batches = prob.batches(T=300, seed=1)

fmd = ExactFMD(spec)
_, S_dense = run_train(prob, spec, theta0, lam, batches, fmd=fmd, true_S_dense=True)
err = float((fmd.S - S_dense).abs().max())
check("R001a_fmd_vs_dense_quadratic", err < 1e-10, f"max abs diff = {err:.3e}")

# ---------------- R001b: FMD vs finite differences ----------------
short = prob.batches(T=40, seed=1)
fmd_q = ExactFMD(spec)
run_train(prob, spec, theta0, lam, short, fmd=fmd_q)
S_fd = finite_diff_S(prob, spec, theta0, lam, short, eps=1e-6)
rel = float((fmd_q.S - S_fd).abs().max() / (S_fd.abs().max() + 1e-30))
check("R001b_fmd_vs_fd_quadratic", rel < 1e-6, f"max rel err = {rel:.3e}")

mlp = TinyMLP(seed=0)
spec_m = GroupSpec.from_sizes(mlp.sizes)
lam_m = torch.full((spec_m.m,), float(torch.log(torch.tensor(0.05))))
theta0_m = mlp.init_theta(seed=2)
batches_m = mlp.batches(T=30, seed=1)
fmd_m = ExactFMD(spec_m)
run_train(mlp, spec_m, theta0_m, lam_m, batches_m, fmd=fmd_m)
S_fd_m = finite_diff_S(mlp, spec_m, theta0_m, lam_m, batches_m, eps=1e-5)
rel_m = float((fmd_m.S - S_fd_m).abs().max() / (S_fd_m.abs().max() + 1e-30))
check("R001b_fmd_vs_fd_mlp", rel_m < 1e-5, f"max rel err = {rel_m:.3e} (nonconvex)")

# ---------------- R002: certificate validity (quadratic) ----------------
sp_oracle = SpectralOracle(prob.Q)
T_cert = 300
cert_summary = {}
for r in (0, 2, spec.m):
    for K in (1, 5):
        est = COHGEstimator(spec, rank=r, refresh_every=K)
        theta = theta0.clone()
        S_true = torch.zeros(spec.p, spec.m)
        n_valid, tight = 0, []
        for t, batch in enumerate(prob.batches(T=T_cert, seed=1)):
            eta = spec.eta_vec(lam)
            rho, kappa = sp_oracle.bounds(eta)
            need_oracle = (t % K == 0)
            oracle = HVPOracle(prob.loss_fn, theta, batch)
            est.step(oracle if need_oracle else None, lam, oracle.grad, rho, kappa)
            S_true = prob.true_S_step(S_true, eta, oracle.grad, spec)
            true_err = float(torch.linalg.matrix_norm(
                S_true - est.dense_shat(), ord="fro"))
            if est.e >= true_err * (1 - 1e-9) - 1e-12:
                n_valid += 1
            if true_err > 1e-12:
                tight.append(est.e / true_err)
            theta = theta - eta * oracle.grad
            oracle.release()
        valid_rate = n_valid / T_cert
        med_tight = float(torch.median(torch.tensor(tight))) if tight else float("nan")
        cert_summary[f"r{r}_K{K}"] = {
            "valid_rate": valid_rate, "median_tightness": med_tight,
            "final_e": est.e, "hvp_count": est.hvp_count,
        }
        check(f"R002_cert_valid_r{r}_K{K}", valid_rate == 1.0,
              f"valid {n_valid}/{T_cert}, median tightness "
              f"{med_tight:.2f}x, final e_t={est.e:.3e}, HVPs={est.hvp_count}")

exact_e = cert_summary[f"r{spec.m}_K1"]["final_e"]
check("R002_exact_endpoint_r=m_K=1", exact_e < 1e-10,
      f"r=m,K=1 must equal exact FMD: e_T = {exact_e:.3e}")

# ---------------- controller smoke test ----------------
ctrl = GatedController(meta_lr=0.1, gate_factor=2.0)
lam_c = lam.clone()
g_small = torch.full((spec.m,), 0.01)
_, moved_when_uncertain = ctrl.maybe_update(lam_c, g_small, beta=1.0)
lam_c2, moved_when_certain = ctrl.maybe_update(lam_c, torch.full((spec.m,), 1.0), beta=0.01)
check("controller_gate", (not moved_when_uncertain) and moved_when_certain
      and not torch.equal(lam_c, lam_c2),
      f"frozen under high beta, updated under low beta "
      f"(gate open frac = {ctrl.gate_open_fraction:.2f})")

# ---------------- write results ----------------
out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "m0_sanity.json"), "w") as f:
    json.dump({"results": RESULTS, "certificate_summary": cert_summary}, f, indent=2)

print(f"\n{'=' * 60}")
print(f"M0 sanity: {len(RESULTS) - len(FAILURES)}/{len(RESULTS)} passed")
if FAILURES:
    print("FAILED:", ", ".join(FAILURES))
    sys.exit(1)
