"""E4: GPT-2 124M streaming test-time adaptation under domain shift
(tracker R050, paper Fig. 7 + Table 5).

Stream: wiki -> news -> code token streams (e4_prepare_data.py), sequential
chunks (batch x seq_len), prequential protocol: online CE on each incoming
batch at current theta IS the reported metric (online PPL); then one SGD
step with per-block LRs eta_j = exp(lambda_j).

lambda: 12 transformer blocks + embeddings + ln_f/head = 14 log-LRs.
Methods: fixed | hd | cohg | cohg_r0 | cohg_nogate | cohg_ogd | absgate.

`absgate` (round-4 threshold-transfer control) is COHG's estimator and COHG's
pure-sign step of size alpha (--meta-lr) with the certificate gate replaced by
a CONSTANT threshold: coordinate j opens iff |ghat_j| > --absgate-threshold.
No certificate is read, so no spectral probe is paid (cost = cohg_nogate's for
the same --rank).  The constant is the one calibrated OFFLINE on E2
mackey_drift (results/e2_controls/absgate_threshold.json), transferred AS IS.

Engineering: fp32 throughout (v1); HVP chunking (hvp_chunk) bounds the vmap
activation memory; eager attention for double-backward safety; peak VRAM and
wall-clock overhead logged (paper Table 5 columns).
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


def _safe_exp(x):
    """exp that saturates instead of raising OverflowError (loss can be
    finite but astronomically large under diverged hyperparameters)."""
    return math.exp(x) if x < 700.0 else float("inf")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cohg import COHGEstimator, CoordGatedController
from cohg.baselines import HDBaseline
from cohg.certificate import DriftHold, SpectralKW
from cohg.functional import FlatModule
from cohg.hvp import HVPOracle

import _gatestats as GS

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
LAM_MIN, LAM_MAX = math.log(1e-6), math.log(0.1)


def load_gpt2(device):
    from modelscope import snapshot_download
    from transformers import GPT2LMHeadModel
    try:
        # Prefer the pinned local cache.  The hub API has been answering 500s;
        # a long detached queue must not die on that.  Same snapshot either
        # way, so the loaded weights (and therefore the run) are unchanged.
        path = snapshot_download("AI-ModelScope/gpt2", local_files_only=True)
    except Exception:                                         # noqa: BLE001
        path = snapshot_download("AI-ModelScope/gpt2")
    model = GPT2LMHeadModel.from_pretrained(
        path, attn_implementation="eager").to(device)
    model.config.use_cache = False
    return model


DOMAINS = ["wiki", "news", "code"]


def parse_domain_order(spec: str):
    order = [s.strip() for s in spec.split(",") if s.strip()]
    if sorted(order) != sorted(DOMAINS):
        raise ValueError(
            f"--domain-order must be a permutation of {DOMAINS}, got {order}")
    return order


def build_stream(seq_len, batch, seed, tokens_per_domain=0,
                 domain_order=None):
    doms = []
    for name in (domain_order or DOMAINS):
        ids = torch.load(os.path.join(DATA, f"e4_stream_{name}.pt"))
        if tokens_per_domain > 0:
            ids = ids[:tokens_per_domain]
        doms.append(ids)
    tokens = torch.cat(doms)
    boundaries = [len(doms[0]), len(doms[0]) + len(doms[1])]
    per_step = seq_len * batch
    n_steps = len(tokens) // per_step
    gen = torch.Generator().manual_seed(seed)
    offset = int(torch.randint(0, seq_len, (1,), generator=gen))
    def batches():
        for t in range(n_steps - 1):
            s = offset + t * per_step
            chunk = tokens[s:s + per_step + 1]
            if len(chunk) < per_step + 1:
                return
            x = chunk[:-1].view(batch, seq_len)
            y = chunk[1:].view(batch, seq_len)
            yield x, y
    bstep = [b // per_step for b in boundaries]
    return batches, n_steps - 1, bstep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["fixed", "hd", "cohg", "cohg_r0", "cohg_nogate",
                             "cohg_ogd", "absgate"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--meta-lr", type=float, default=0.4)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--kw-eps", type=float, default=0.1)
    ap.add_argument("--kw-delta", type=float, default=0.01)
    ap.add_argument("--probe-every", type=int, default=40)
    ap.add_argument("--M-H", type=float, default=50.0)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--hvp-chunk", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = full stream")
    ap.add_argument("--tokens-per-domain", type=int, default=0,
                    help="truncate each domain's token stream to this many "
                    "tokens before concatenation (0 = full domains)")
    ap.add_argument("--domain-order", default="wiki,news,code",
                    help="comma-separated permutation of wiki,news,code "
                    "giving the order in which the domain token streams are "
                    "concatenated (default: wiki,news,code)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--legacy-hold", action="store_true",
                    help="restore the pre-fix scalar drift-hold interface "
                         "(dh.probe without eta_vec, dh.bounds(float(eta.max())) "
                         "-> Delta eta_t == 0 and eta_max read at the CURRENT "
                         "step).  Default (off) uses the full vector-valued "
                         "held bound of Proposition 10, identical to the "
                         "E2/E3 code path: iota_t = Delta eta_t * Hbar_t + "
                         "eta_max,t0 * (M_H P_t + nu_H).")
    ap.add_argument("--absgate-threshold", type=float, default=None,
                    help="round-4 transfer control: coordinate j opens iff "
                         "|ghat_j| > this CONSTANT (no certificate, no "
                         "spectral probe).  Required by --method absgate.")
    ap.add_argument("--log-gate-stats", action="store_true",
                    help="additive instrumentation: pool |ghat_j| and the "
                         "certificate-scaled threshold c*beta_col_j over every "
                         "(step, coordinate) pair and write their summary "
                         "under the key 'gate_stats'.  Read-only.")
    args = ap.parse_args()
    if args.method == "absgate" and args.absgate_threshold is None:
        raise ValueError("--method absgate requires --absgate-threshold")
    domain_order = parse_domain_order(args.domain_order)

    torch.manual_seed(args.seed)
    device = args.device
    model = load_gpt2(device)

    def lm_loss(out, y):
        return F.cross_entropy(out.reshape(-1, out.shape[-1]), y.reshape(-1))

    class LMWrap(torch.nn.Module):
        def __init__(self, gpt):
            super().__init__()
            self.gpt = gpt
        def forward(self, x):
            return self.gpt(input_ids=x).logits

    fm = FlatModule(LMWrap(model), lm_loss)

    def group_fn(name):
        # coarse grouping (m=6) keeps the p x m refresh buffer ~3GB fp32
        if ".h." in name:
            blk = int(name.split(".h.")[1].split(".")[0])
            return f"blocks{blk // 3}"          # 4 groups of 3 blocks
        if "wte" in name or "wpe" in name:
            return "emb"
        return "final"

    spec = fm.group_spec(group_fn)
    p, m = spec.p, spec.m
    print(f"p={p:,} m={m}", flush=True)
    theta = fm.flat_params()
    lam = torch.full((m,), math.log(args.lr), device=device)
    lam_init = lam.clone()

    batches, n_steps, drift_steps = build_stream(args.seq_len, args.batch,
                                                 args.seed,
                                                 args.tokens_per_domain,
                                                 domain_order)
    if args.max_steps:
        n_steps = min(n_steps, args.max_steps)
    print(f"stream: {n_steps} steps, drift at {drift_steps} "
          f"order={','.join(domain_order)}", flush=True)

    gstats = GS.GateStats(args.log_gate_stats, gate_factor=2.0)

    est = kw = dh = ctrl = hd = None
    if args.method.startswith("cohg") or args.method == "absgate":
        r = 0 if args.method == "cohg_r0" else args.rank
        est = COHGEstimator(spec, rank=r, refresh_every=args.K,
                            device=device, dtype=torch.float32,
                            discount=args.gamma)
        kw = SpectralKW(p, eps=args.kw_eps, delta=args.kw_delta,
                        seed=args.seed + 7)
        dh = DriftHold(M_H=args.M_H)
        ctrl = CoordGatedController(
            args.meta_lr, gate_factor=2.0, lam_min=LAM_MIN, lam_max=LAM_MAX,
            mode="ogd" if args.method == "cohg_ogd" else "sign")
    elif args.method == "hd":
        hd = HDBaseline(spec, args.meta_lr, lam_min=LAM_MIN, lam_max=LAM_MAX)

    losses, lam_hist, gate_hist = [], [], []
    gate_open_steps = []
    ctl_open = ctl_total = 0     # per-coordinate accounting for absgate
    events = 0
    hvp_total = 0
    ckpt = theta.clone()
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats(device)

    for t, (x, y) in enumerate(batches()):
        if args.max_steps and t >= n_steps:
            break
        batch = (x.to(device), y.to(device))
        need_graph = est is not None and \
            args.method not in ("cohg_nogate", "absgate") and \
            (t % args.K == 0 or t % args.probe_every == 0)
        need_graph = need_graph or (est is not None and t % args.K == 0)
        if need_graph:
            oracle = HVPOracle(fm.loss_fn, theta, batch,
                               hvp_chunk=args.hvp_chunk)
            loss, g = oracle.loss, oracle.grad
        else:
            oracle = None
            th = theta.detach().requires_grad_(True)
            lo = fm.loss_fn(th, batch)
            (g,) = torch.autograd.grad(lo, th)
            loss, g = lo.detach(), g.detach()
        loss_val = float(loss)

        if est is not None:
            eta = spec.eta_vec(lam)
            if args.method in ("cohg_nogate", "absgate"):
                # no certificate is read -> no spectral probe is paid for
                rho, kappa = 1.0, 0.0
            elif t % args.probe_every == 0:
                pre = oracle.n_hvp
                rho, kappa = kw.bounds(oracle.hvp, eta)
                hvp_total += oracle.n_hvp - pre
                if args.legacy_hold:
                    dh.probe(rho, kappa)
                    rho, kappa = dh.bounds(float(eta.max()))
                else:
                    dh.probe(rho, kappa, eta_vec=eta)
                    rho, kappa = dh.bounds(eta)
            else:
                rho, kappa = (dh.bounds(float(eta.max())) if args.legacy_hold
                              else dh.bounds(eta))
            ghat = est.hypergrad(g)
            beta_col = est.beta_col(float(torch.linalg.vector_norm(g)))
            gstats.add(ghat,
                       None if args.method in ("cohg_nogate", "absgate")
                       else beta_col)
            if args.method == "cohg_nogate":
                lam = (lam - args.meta_lr * torch.sign(ghat)
                       ).clamp(LAM_MIN, LAM_MAX)
                opened = True
            elif args.method == "absgate":
                # constant-threshold gate, transferred as is: SAME estimator
                # and SAME pure-sign step of size alpha as cohg_nogate --
                # only the gate rule differs.
                open_mask = ghat.abs() > args.absgate_threshold
                ctl_total += int(open_mask.numel())
                ctl_open += int(open_mask.sum())
                opened = bool(open_mask.any())
                if opened:
                    step = args.meta_lr * torch.sign(ghat)
                    lam = (lam - torch.where(open_mask, step,
                                             torch.zeros_like(step))
                           ).clamp(LAM_MIN, LAM_MAX)
            else:
                lam, opened = ctrl.maybe_update(lam, ghat, beta_col)
            gate_hist.append(int(opened))
            if opened:
                gate_open_steps.append(t)
            pre = oracle.n_hvp if oracle is not None else 0
            est.step(oracle if t % args.K == 0 else None, lam, g, rho, kappa)
            if oracle is not None:
                hvp_total += oracle.n_hvp - pre
        elif hd is not None:
            lam = hd.update(lam, g)

        eta = spec.eta_vec(lam)
        step_vec = eta * g
        theta = theta - step_vec
        if dh is not None:
            dh.step(float(torch.linalg.vector_norm(step_vec)))
        if oracle is not None:
            oracle.release()

        if not math.isfinite(loss_val):
            events += 1
            theta = ckpt.clone()
            lam = lam_init.clone()
            if est is not None:
                est.reset_state()
        else:
            if t % 100 == 0:
                ckpt = theta.clone()
        losses.append(loss_val)
        if t % 20 == 0:
            lam_hist.append([t] + lam.tolist())
        if t % 500 == 0:
            fin = [x for x in losses[-500:] if math.isfinite(x)]
            print(f"t={t} ppl={_safe_exp(sum(fin)/len(fin)):.2f} "
                  f"lam[0]={float(lam[0]):.2f} "
                  f"peakGB={torch.cuda.max_memory_allocated(device)/2**30:.1f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    fin = [x for x in losses if math.isfinite(x)]
    mean_ll = sum(fin) / len(fin)
    out = {
        "method": args.method, "seed": args.seed, "lr0": args.lr,
        "meta_lr": args.meta_lr, "rank": args.rank, "K": args.K,
        "gamma": args.gamma, "steps": len(losses),
        "drift_steps": drift_steps,
        "domain_order": ",".join(domain_order),
        "tokens_per_domain": args.tokens_per_domain,
        "online_ppl": _safe_exp(mean_ll),
        "mean_logloss": mean_ll,
        "events": events,
        "gate_open_frac": (sum(gate_hist) / len(gate_hist)) if gate_hist else None,
        "coord_open_frac": ((ctl_open / ctl_total) if ctl_total
                            else (ctrl.gate_open_fraction if ctrl else None)),
        "hvp_total": hvp_total,
        "peak_mem_gb": torch.cuda.max_memory_allocated(device) / 2 ** 30,
        "wall_s": time.time() - t0,
        "losses": losses, "lam_hist": lam_hist,
        "mh_observed": (dh.mh_observed[-50:] if dh else None),
        "legacy_hold": bool(args.legacy_hold),
        "held_bound": ("scalar_legacy" if args.legacy_hold else "vector_prop10"),
        "gate_open_steps": (gate_open_steps if est is not None else None),
    }
    if args.method == "absgate":
        out["absgate_threshold"] = args.absgate_threshold
    _gsum = gstats.summary()
    if _gsum is not None:
        out["gate_stats"] = _gsum
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(__file__), "..", "results", "e4")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"gpt2_{args.method}_lr{args.lr:g}_s{args.seed}"
    with open(os.path.join(out_dir, tag + ".json"), "w") as f:
        json.dump(out, f)
    print(f"[{tag}] ppl={out['online_ppl']:.2f} events={events} "
          f"peak={out['peak_mem_gb']:.1f}GB wall={out['wall_s']:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
