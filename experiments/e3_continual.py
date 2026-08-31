"""E3: continual learning with online-adaptive EWC strength + per-stage LRs
(tracker R040/R041, paper Table 3 + Fig. 5).

Protocol (online CL, single pass):
- Split-CIFAR-100: 10 tasks x 10 classes, tasks arrive sequentially as a
  minibatch stream (default 2 passes, batch 32 -> ~300 steps/task).
  TASK-INCREMENTAL protocol: eval and meta-objective mask logits to each
  task's own classes (the standard setting where EWC has real leverage —
  class-IL with a single head collapses regardless of lambda_EWC, leaving
  the hyperparameter nothing to trade off).
- Inner learner: SGD with per-group LR eta_j = exp(lambda_j) on the loss
      L_t(theta) = CE(theta; batch) + (s/2) * sum_i Fbar_i (theta_i - theta*_i)^2,
  s = exp(lambda_ewc)  (online EWC, accumulated Fisher Fbar, anchor theta*).
- lambda = [per-stage log-LRs (m groups); log EWC strength (1 global/dense)].
- Meta-objective ell_t = CE(current batch) + CE(holdout buffer of past tasks)
  (128 held-out samples per seen task, never trained on).
  With --no-holdout the past-task term is dropped and the meta-objective is
  purely prequential: ell_t = CE(current batch) evaluated before the update
  (see the boundary comment at the meta-gradient site).
- After each task: update Fbar (diag Fisher on 512 samples), reset anchor.

Methods: fixed | hd | tfmd | cohg | cohg_r0 | cohg_nogate | absgate.

`absgate` (round-4 threshold-transfer control) is COHG's estimator and COHG's
pure-sign step of size alpha (--meta-lr), with the certificate gate replaced by
a CONSTANT threshold: coordinate j opens iff |ghat_j| > --absgate-threshold.
It reads no certificate, so it pays no spectral probe -- its cost is
cohg_nogate's.  The constant is the one calibrated OFFLINE on E2 mackey_drift
(results/e2_controls/absgate_threshold.json) and is transferred here AS IS.
Metrics: final average accuracy, BWT (forgetting), instability events,
gate stats, lambda trajectories (EWC strength across task boundaries!).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cohg import COHGEstimator, CoordGatedController, GroupSpec
from cohg.baselines import HDBaseline
from cohg.certificate import DriftHold, SpectralKW
from cohg.fmd import ExactFMD
from cohg.functional import FlatModule
from cohg.hvp import HVPOracle

import models as M
import _gatestats as GS

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
LAM_MIN, LAM_MAX = math.log(1e-5), math.log(1.0)
EWC_MIN, EWC_MAX = math.log(1e-3), math.log(1e4)


# ---------------------------------------------------------------- data
def load_cifar100(device):
    import torchvision
    import torchvision.transforms as T
    tf = T.Compose([T.ToTensor(), T.Normalize(
        (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
    tr = torchvision.datasets.CIFAR100(DATA_ROOT, train=True, download=True,
                                       transform=tf)
    te = torchvision.datasets.CIFAR100(DATA_ROOT, train=False, download=True,
                                       transform=tf)
    def tensors(ds):
        xs = torch.stack([ds[i][0] for i in range(len(ds))])
        ys = torch.tensor([ds[i][1] for i in range(len(ds))])
        return xs, ys
    return tensors(tr), tensors(te)


class TaskStream:
    """Single-pass batch stream over one task's training samples, carving out
    a holdout buffer for the meta-objective."""

    def __init__(self, xs, ys, classes, batch_size, holdout, seed, device):
        mask = torch.isin(ys, torch.tensor(classes))
        idx = torch.nonzero(mask).squeeze(1)
        perm = idx[torch.randperm(len(idx),
                                  generator=torch.Generator().manual_seed(seed))]
        self.hold_x = xs[perm[:holdout]].to(device)
        self.hold_y = ys[perm[:holdout]].to(device)
        self.train_idx = perm[holdout:]
        self.xs, self.ys = xs, ys
        self.bs = batch_size
        self.pos = 0
        self.device = device

    def __iter__(self):
        while self.pos + self.bs <= len(self.train_idx):
            sel = self.train_idx[self.pos:self.pos + self.bs]
            self.pos += self.bs
            yield self.xs[sel].to(self.device), self.ys[sel].to(self.device)

    def epochs(self, n):
        for _ in range(n):
            self.pos = 0
            yield from iter(self)


# ---------------------------------------------------------------- EWC state
class EWCState:
    def __init__(self, p, device, dtype):
        self.fisher = torch.zeros(p, device=device, dtype=dtype)
        self.anchor = torch.zeros(p, device=device, dtype=dtype)
        self.active = False

    def update(self, fm, theta, batches, n_batches):
        """Accumulate diagonal Fisher (empirical, squared grads) and re-anchor."""
        fsum = torch.zeros_like(self.fisher)
        for i, (x, y) in enumerate(batches):
            if i >= n_batches:
                break
            th = theta.detach().requires_grad_(True)
            out = fm.loss_fn_ce(th, (x, y))
            (g,) = torch.autograd.grad(out, th)
            fsum += g.detach() ** 2
        self.fisher = self.fisher + fsum / max(n_batches, 1)
        self.anchor = theta.detach().clone()
        self.active = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["fixed", "hd", "tfmd", "cohg", "cohg_r0",
                             "cohg_nogate", "absgate"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--ewc0", type=float, default=10.0)
    ap.add_argument("--meta-lr", type=float, default=0.1)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--kw-eps", type=float, default=0.1)
    ap.add_argument("--kw-delta", type=float, default=0.01)
    ap.add_argument("--probe-every", type=int, default=20)
    ap.add_argument("--M-H", type=float, default=20.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--log-losses", action="store_true",
                    help="store the per-step training loss list in the output "
                         "JSON under the key 'losses'")
    ap.add_argument("--absgate-threshold", type=float, default=None,
                    help="round-4 transfer control: coordinate j opens iff "
                         "|ghat_j| > this CONSTANT (no certificate, no "
                         "spectral probe).  Required by --method absgate.")
    ap.add_argument("--log-gate-stats", action="store_true",
                    help="additive instrumentation: pool |ghat_j| and the "
                         "certificate-scaled threshold c*beta_col_j over every "
                         "(step, coordinate) pair and write their summary "
                         "under the key 'gate_stats'.  Read-only: it never "
                         "touches the run.")
    ap.add_argument("--no-holdout", action="store_true",
                    help="meta-objective uses ONLY the prequential loss of the "
                         "incoming batch (no retained past-task examples)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    (tr_x, tr_y), (te_x, te_y) = load_cifar100(device="cpu")
    class_order = torch.randperm(
        100, generator=torch.Generator().manual_seed(args.seed)).tolist()
    tasks = [class_order[i * 10:(i + 1) * 10] for i in range(args.n_tasks)]

    model = M.ResNet18GN(n_classes=100).to(device)

    def masked_ce(out, y):
        """Task-IL: restrict logits to the classes of each sample's task."""
        mask = torch.full_like(out, float("-inf"))
        for cls_list in tasks:
            cls = torch.tensor(cls_list, device=out.device)
            rows = torch.isin(y, cls)
            if rows.any():
                mask[rows.nonzero(as_tuple=True)[0][:, None], cls] = 0.0
        return F.cross_entropy(out + mask, y)

    fm = FlatModule(model, masked_ce)

    def group_fn(name):  # stem / stages.0-1 / 2-3 / 4-5 / 6-7 / head -> 6 groups
        if name.startswith("stem"):
            return "stem"
        if name.startswith("stages."):
            blk = int(name.split(".")[1])
            return f"stage{blk // 2}"
        return "head"

    spec = fm.group_spec(group_fn)
    p, m = spec.p, spec.m
    theta = fm.flat_params()
    ewc = EWCState(p, device, theta.dtype)

    # loss closures: CE only (for Fisher/meta) and CE + EWC penalty (training)
    fm.loss_fn_ce = fm.loss_fn
    s_holder = {"s": args.ewc0}

    def loss_fn_train(th, batch):
        base = fm.loss_fn_ce(th, batch)
        if ewc.active:
            base = base + 0.5 * s_holder["s"] * (
                ewc.fisher * (th - ewc.anchor) ** 2).sum()
        return base

    # lambda layout: [m log-LRs; log-EWC-strength]
    lam = torch.cat([torch.full((m,), math.log(args.lr), device=device),
                     torch.tensor([math.log(args.ewc0)], device=device)])
    lam_init = lam.clone()

    if args.method == "absgate" and args.absgate_threshold is None:
        raise ValueError("--method absgate requires --absgate-threshold")
    gstats = GS.GateStats(args.log_gate_stats, gate_factor=2.0)

    est = kw = dh = ctrl = hd = tfmd_fmd = None
    if args.method in ("cohg", "cohg_r0", "cohg_nogate", "absgate"):
        r = 0 if args.method == "cohg_r0" else args.rank
        est = COHGEstimator(spec, rank=r, refresh_every=args.K, device=device,
                            dtype=torch.float32, discount=args.gamma,
                            n_dense=1)
        kw = SpectralKW(p, eps=args.kw_eps, delta=args.kw_delta,
                        seed=args.seed + 7)
        dh = DriftHold(M_H=args.M_H)
        ctrl = CoordGatedController(args.meta_lr, gate_factor=2.0, trust=1.0)
    elif args.method == "hd":
        hd = HDBaseline(spec, args.meta_lr)  # LR part; EWC coord via manual 1-step
        hd_prev = {"g": None, "bd": None}
    elif args.method == "tfmd":
        tfmd_fmd = ExactFMD(spec, device=device, dtype=torch.float32,
                            discount=args.gamma)
        # NOTE: ExactFMD covers only the m LR columns; EWC column handled via
        # a dense exact column, same recursion (maintained here inline)
        x_col = torch.zeros(p, device=device, dtype=torch.float32)
        tf_t = 0

    def clamp_lam(v):
        lr_part = v[:m].clamp(LAM_MIN, LAM_MAX)
        ewc_part = v[m:].clamp(EWC_MIN, EWC_MAX)
        return torch.cat([lr_part, ewc_part])

    acc_matrix = []  # acc_matrix[k][j] = acc on task j after training task k
    events = 0
    gate_opens = gate_total = 0
    ctl_open = ctl_total = 0     # per-coordinate accounting for absgate
    lam_hist = []
    losses = []          # per-step training loss, only filled if --log-losses
    hvp_total = 0
    holdouts_x, holdouts_y = [], []
    t_global = 0
    t0 = time.time()

    for k, classes in enumerate(tasks):
        stream = TaskStream(tr_x, tr_y, classes, args.batch, holdout=128,
                            seed=args.seed + k, device=device)
        for x, y in stream.epochs(args.epochs):
            batch = (x, y)
            s_holder["s"] = float(torch.exp(lam[m]))
            eta = spec.eta_vec(lam[:m])

            # meta gradient (prequential + past-task holdouts)
            #
            # --no-holdout boundary (review item "no-retained-holdout"):
            # the flag removes retained past-task data from the META-OBJECTIVE
            # only.  ell_t degenerates to the prequential loss of the incoming
            # batch, evaluated at the current theta *before* the update, so no
            # example is ever revisited by the hyperparameter controller.
            # EWC keeps its anchor theta* and its accumulated Fisher Fbar: those
            # live inside loss_fn_train and are part of the LEARNER (the object
            # whose hyperparameters we tune), not part of the meta-objective.
            # Ablating them would change the learner, not the meta-objective,
            # and would confound this control.  The 128-example holdout is still
            # carved out of each task's training stream (TaskStream below) so
            # that the training data seen by the learner is bit-identical to the
            # default condition; under --no-holdout those samples are simply
            # never looked at again.
            th = theta.detach().requires_grad_(True)
            meta_loss = fm.loss_fn_ce(th, batch)
            if holdouts_x and not args.no_holdout:
                hx = torch.cat(holdouts_x)
                hy = torch.cat(holdouts_y)
                meta_loss = meta_loss + fm.loss_fn_ce(th, (hx, hy))
            (meta_g,) = torch.autograd.grad(meta_loss, th)
            meta_g = meta_g.detach()

            need_graph = (est is not None and
                          (t_global % args.K == 0 or
                           t_global % args.probe_every == 0)) or \
                         (tfmd_fmd is not None)
            if need_graph:
                oracle = HVPOracle(loss_fn_train, theta, batch)
                loss, g = oracle.loss, oracle.grad
            else:
                oracle = None
                th2 = theta.detach().requires_grad_(True)
                lo = loss_fn_train(th2, batch)
                (g,) = torch.autograd.grad(lo, th2)
                loss, g = lo.detach(), g.detach()

            # dense direct term for the EWC coordinate
            if ewc.active:
                b_dense = (-(eta * (s_holder["s"] * ewc.fisher *
                                    (theta - ewc.anchor)))).unsqueeze(1)
            else:
                b_dense = torch.zeros(p, 1, device=device)

            if est is not None:
                if args.method in ("cohg_nogate", "absgate"):
                    # no certificate is ever read -> the spectral probe is
                    # skipped entirely (nominal rho/kappa keep the estimator
                    # recursion identical; e_col becomes unused)
                    rho, kappa = 1.0, 0.0
                elif t_global % args.probe_every == 0:
                    pre = oracle.n_hvp
                    rho, kappa = kw.bounds(oracle.hvp, eta)
                    hvp_total += oracle.n_hvp - pre
                    dh.probe(rho, kappa, eta_vec=eta)
                    rho, kappa = dh.bounds(eta)
                else:
                    rho, kappa = dh.bounds(eta)
                ghat = est.hypergrad(meta_g)
                beta_col = est.beta_col(float(torch.linalg.vector_norm(meta_g)))
                gstats.add(ghat,
                           None if args.method in ("cohg_nogate", "absgate")
                           else beta_col)
                if args.method == "cohg_nogate":
                    lam = clamp_lam(lam - args.meta_lr * torch.sign(ghat))
                    gate_opens += 1
                elif args.method == "absgate":
                    # constant-threshold gate, transferred as is: SAME
                    # estimator and SAME pure-sign step of size alpha as
                    # cohg_nogate -- only the gate rule differs.
                    open_mask = ghat.abs() > args.absgate_threshold
                    ctl_total += int(open_mask.numel())
                    ctl_open += int(open_mask.sum())
                    if bool(open_mask.any()):
                        step = args.meta_lr * torch.sign(ghat)
                        lam = clamp_lam(lam - torch.where(
                            open_mask, step, torch.zeros_like(step)))
                        gate_opens += 1
                else:
                    lam_new, opened = ctrl.maybe_update(lam, ghat, beta_col)
                    lam = clamp_lam(lam_new)
                    gate_opens += int(opened)
                gate_total += 1
                pre = oracle.n_hvp if oracle is not None else 0
                est.step(oracle if t_global % args.K == 0 else None,
                         lam, g, rho, kappa, b_dense=b_dense,
                         eta_vec=spec.eta_vec(lam[:m]))
                if oracle is not None:
                    hvp_total += oracle.n_hvp - pre
            elif hd is not None:
                if hd_prev["g"] is not None:
                    h_lr = -spec.group_sums(meta_g * eta * hd_prev["g"])
                    h_ewc = float(hd_prev["bd"].squeeze(1) @ meta_g)
                    h = torch.cat([h_lr, torch.tensor([h_ewc], device=device)])
                    lam = clamp_lam(lam - args.meta_lr * h)
                hd_prev["g"] = g.clone()
                hd_prev["bd"] = b_dense.clone()
            elif tfmd_fmd is not None:
                if tf_t % 50 == 0:  # truncation window
                    tfmd_fmd.reset()
                    x_col.zero_()
                ghat = torch.cat([tfmd_fmd.hypergrad(meta_g).to(lam.dtype),
                                  (x_col @ meta_g).view(1)])
                lam = clamp_lam(lam - args.meta_lr * ghat)
                tfmd_fmd.step(oracle, lam[:m])
                Hx = oracle.hvp(x_col)
                x_col = args.gamma * (x_col - eta * Hx) + b_dense.squeeze(1)
                hvp_total += oracle.n_hvp
                tf_t += 1

            eta = spec.eta_vec(lam[:m])
            s_holder["s"] = float(torch.exp(lam[m]))
            theta = theta - eta * g
            if dh is not None:
                dh.step(float(torch.linalg.vector_norm(eta * g)))
            if oracle is not None:
                oracle.release()

            if args.log_losses:
                losses.append(float(loss))
            if not math.isfinite(float(loss)):
                events += 1
                # catastrophe backoff (F15, uniform across adaptive methods):
                # halve LR scale AND EWC strength instead of resetting to the
                # possibly-catastrophic init
                lam = clamp_lam(lam - math.log(2.0))
                if est is not None:
                    est.reset_state()
            if t_global % 25 == 0:
                lam_hist.append([t_global] + lam.tolist())
            t_global += 1

        # ---- task end: holdout, Fisher, anchor, eval ----
        holdouts_x.append(stream.hold_x)
        holdouts_y.append(stream.hold_y)
        fisher_stream = TaskStream(tr_x, tr_y, classes, args.batch, 0,
                                   seed=args.seed + 1000 + k, device=device)
        ewc.update(fm, theta, iter(fisher_stream), n_batches=4)

        accs = []
        with torch.no_grad():
            params = fm.unflatten(theta)
            from torch.func import functional_call
            for j in range(k + 1):
                mask = torch.isin(te_y, torch.tensor(tasks[j]))
                xs = te_x[mask].to(device)
                ys = te_y[mask].to(device)
                cls = torch.tensor(tasks[j], device=device)
                preds = []
                for i in range(0, len(ys), 512):
                    out = functional_call(fm.module, params, (xs[i:i + 512],))
                    preds.append(cls[out[:, cls].argmax(1)])  # task-IL argmax
                accs.append(float((torch.cat(preds) == ys).float().mean()))
        acc_matrix.append(accs)
        print(f"[task {k}] accs={['%.3f' % a for a in accs]} "
              f"lam_lr={[round(float(v), 2) for v in lam[:m]]} "
              f"lam_ewc={float(lam[m]):.2f} ({time.time() - t0:.0f}s)",
              flush=True)

    final_accs = acc_matrix[-1]
    avg_acc = sum(final_accs) / len(final_accs)
    bwt = sum(acc_matrix[-1][j] - acc_matrix[j][j]
              for j in range(len(tasks) - 1)) / max(len(tasks) - 1, 1)
    out = {
        "method": args.method, "seed": args.seed, "lr0": args.lr,
        "ewc0": args.ewc0, "meta_lr": args.meta_lr, "rank": args.rank,
        "K": args.K, "gamma": args.gamma,
        "avg_acc": avg_acc, "bwt": bwt, "events": events,
        "gate_open_frac": (gate_opens / gate_total) if gate_total else None,
        "coord_open_frac": ((ctl_open / ctl_total) if ctl_total
                            else (ctrl.gate_open_fraction if ctrl else None)),
        "hvp_total": hvp_total, "wall_s": time.time() - t0,
        "acc_matrix": acc_matrix, "lam_hist": lam_hist,
    }
    # additive keys: present only when the corresponding flag is set, so the
    # default-path JSON stays byte-identical to the pre-flag runs in results/e3
    if args.no_holdout:
        out["no_holdout"] = True
    if args.log_losses:
        out["losses"] = losses
    if args.method == "absgate":
        out["absgate_threshold"] = args.absgate_threshold
    _gsum = gstats.summary()
    if _gsum is not None:
        out["gate_stats"] = _gsum
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(__file__), "..", "results", "e3")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"cifar100_{args.method}_lr{args.lr:g}_ewc{args.ewc0:g}_s{args.seed}"
    with open(os.path.join(out_dir, tag + ".json"), "w") as f:
        json.dump(out, f)
    print(f"[{tag}] avg_acc={avg_acc:.4f} bwt={bwt:+.4f} events={events} "
          f"wall={out['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
