"""E0: structural premise study (tracker R003-R006, paper Fig. 1).

Maintains the EXACT sensitivity S_t = d theta_t / d lambda along a real SGD
trajectory (m HVPs per step) and measures the structure of the cross-group
residual  Res_t = S_t - D_t  (D_t = group-aligned part):

- singular value spectrum of Res_t (rank <= m)
- residual share ||Res||_F / ||S||_F
- best-rank-r capture ||Res_r||_F / ||Res||_F for r = 1..m
- stable rank of Res_t

Decision gate (plan M1): rank-ceil(m/4) capture >= 0.80 on >= 3/4 archs
(absolute rank-8 capture also recorded for the proposal's original phrasing).

Usage:
    python e0_structure.py --arch gru --steps 1200 --snap-every 25
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cohg.functional import FlatModule
from cohg.hvp import HVPOracle

import data as D
import models as M


def strip_wb(name: str) -> str:
    for suf in (".weight", ".bias"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def build(arch: str, device: str):
    if arch == "mlp":
        model = M.DeepMLP().to(device)
        stream = D.ImageStream("mnist", batch_size=128, device=device)
        loss = lambda out, y: torch.nn.functional.cross_entropy(out, y)
        group_fn, lr = None, 0.05          # per parameter tensor
    elif arch == "resnet":
        model = M.SmallResNet().to(device)
        stream = D.ImageStream("cifar10", batch_size=64, device=device)
        loss = lambda out, y: torch.nn.functional.cross_entropy(out, y)
        group_fn, lr = strip_wb, 0.05      # per submodule (merge w/b)
    elif arch == "transformer":
        stream = D.CharStream(seq_len=64, batch_size=16, device=device)
        model = M.CharTransformer(vocab=stream.vocab).to(device)
        loss = lambda out, y: torch.nn.functional.cross_entropy(
            out.reshape(-1, out.shape[-1]), y.reshape(-1))
        group_fn, lr = strip_wb, 0.05
    elif arch == "gru":
        series = D.mackey_glass(30_000, seed=0)
        stream = D.WindowStream(series, window=20, batch_size=64, device=device)
        model = M.ManualGRU().to(device)
        loss = lambda out, y: 0.5 * ((out - y) ** 2).mean()
        group_fn, lr = None, 0.05          # per parameter tensor
    else:
        raise ValueError(arch)
    return model, stream, loss, group_fn, lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True,
                    choices=["mlp", "resnet", "transformer", "gru"])
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--snap-every", type=int, default=25)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model, stream, loss, group_fn, lr = build(args.arch, args.device)
    fm = FlatModule(model, loss)
    spec = fm.group_spec(group_fn)
    p, m = spec.p, spec.m
    print(f"[{args.arch}] p={p:,}  m={m}  steps={args.steps}  lr={lr}")

    theta = fm.flat_params()
    lam = torch.full((m,), math.log(lr), device=args.device)
    S = torch.zeros(p, m, device=args.device, dtype=torch.float32)

    snaps, losses = [], []
    t0 = time.time()
    for t in range(args.steps):
        batch = stream.next()
        oracle = HVPOracle(fm.loss_fn, theta, batch)
        eta = spec.eta_vec(lam)
        HS = oracle.hvp_mat(S)
        B = spec.aligned_to_matrix(-(eta * oracle.grad))
        S = S - eta.unsqueeze(1) * HS + B
        losses.append(float(oracle.loss))

        if t % args.snap_every == 0 or t == args.steps - 1:
            d_aligned = spec.matrix_aligned_part(S)
            Res = S - spec.aligned_to_matrix(d_aligned)
            sig = torch.linalg.svdvals(Res.double())
            s_fro = float(torch.linalg.matrix_norm(S.double(), ord="fro"))
            r_fro = float(sig.pow(2).sum().sqrt())
            energy = sig.pow(2)
            cum = energy.cumsum(0) / energy.sum().clamp_min(1e-300)
            capture = cum.sqrt()  # ||Res_r||_F / ||Res||_F
            stable_rank = float(energy.sum() / energy[0].clamp_min(1e-300))
            snaps.append({
                "t": t,
                "loss": losses[-1],
                "res_share": r_fro / max(s_fro, 1e-30),
                "stable_rank": stable_rank,
                "sigma": sig.tolist(),
                "capture": capture.tolist(),
            })
            if t % (args.snap_every * 8) == 0:
                print(f"  t={t:5d} loss={losses[-1]:.4f} "
                      f"res_share={snaps[-1]['res_share']:.3f} "
                      f"srank={stable_rank:.2f} "
                      f"cap[r={max(1, m // 4)}]={capture[max(1, m // 4) - 1]:.3f}")

        theta = theta - eta * oracle.grad
        oracle.release()

    # ---- summary over second half of training (settled regime) ----
    half = [s for s in snaps if s["t"] >= args.steps // 2]
    r_rel = max(1, m // 4)
    cap_rel = sum(s["capture"][r_rel - 1] for s in half) / len(half)
    cap_abs8 = sum(s["capture"][min(8, m) - 1] for s in half) / len(half)
    res_share = sum(s["res_share"] for s in half) / len(half)
    summary = {
        "arch": args.arch, "p": p, "m": m, "steps": args.steps,
        "lr": lr, "seed": args.seed,
        "mean_res_share_2nd_half": res_share,
        "mean_capture_rank_rel": cap_rel, "rank_rel": r_rel,
        "mean_capture_rank8": cap_abs8,
        "mean_stable_rank_2nd_half":
            sum(s["stable_rank"] for s in half) / len(half),
        "final_loss": sum(losses[-50:]) / 50,
        "wall_s": time.time() - t0,
    }
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "e0")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{args.arch}_seed{args.seed}.json"), "w") as f:
        json.dump({"summary": summary, "snapshots": snaps}, f)
    print(f"[{args.arch}] DONE in {summary['wall_s']:.0f}s | "
          f"res_share={res_share:.3f} | cap[r={r_rel}]={cap_rel:.3f} | "
          f"cap[r=8]={cap_abs8:.3f} | srank={summary['mean_stable_rank_2nd_half']:.2f}")


if __name__ == "__main__":
    main()
