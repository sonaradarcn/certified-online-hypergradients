"""E4 v2 launcher: GPT-2 124M streaming TTA, compressed 3-domain stream.

Fix for the v1 defect: v1 ran --max-steps 3000 over the FULL concatenated
stream whose domain boundaries sit at steps 7812/13671, so the runs never
left wiki. v2 truncates each domain to 512k tokens (--tokens-per-domain
512000); with batch=2 x seq=256 (512 tokens/step) the boundaries land at
steps 1000 and 2000. Note: 3 x 512000 tokens yield 2999 usable steps (each
step consumes per_step+1 tokens for the shifted labels), so the stream log
reads "stream: 2999 steps, drift at [1000, 2000]" — boundaries exact.

Matrix (seeds 0-2, output results/e4_v2):
  fixed lr in {1e-3, 3e-3}
  hd @ lr0=1e-3, meta-lr in {0.2, 2}
  cohg_r0 / cohg_nogate / cohg (r=4) @ lr0=1e-3, meta-lr 0.4

Queue order: cheapest first (fixed, hd ~0.5h), then cohg_r0 (~3.5h),
cohg_nogate (~5h), and the three cohg r=4 runs LAST (~6-13h each).

Cross-process safety: campaign_fix1.py joins this queue as a second worker
on GPU1 once its Task-B grid drains, so configs are claimed via atomic
.claim files (stale claims from dead PIDs are reaped). Per-run child logs
go to results/logs/e4v2/; each run uses a private tmp out-dir because
parallel hd runs with different meta-lr share the same default filename.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "e4_v2")
LOGDIR = os.path.join(HERE, "..", "results", "logs", "e4v2")
PY = sys.executable
SEEDS = [0, 1, 2]

COMMON = ["--tokens-per-domain", "512000", "--max-steps", "3000",
          "--batch", "2", "--seq-len", "256", "--probe-every", "100",
          "--kw-eps", "0.15", "--device", "cuda:0"]


def make_configs():
    """(method, seed, lr, meta_lr, tag) — cheapest first, cohg r=4 last."""
    cfgs = []
    for s in SEEDS:
        for lr in [1e-3, 3e-3]:
            cfgs.append(("fixed", s, lr, None, ""))
    for s in SEEDS:
        for ml in [0.2, 2.0]:
            cfgs.append(("hd", s, 1e-3, ml, f"_ml{ml:g}"))
    for meth in ["cohg_r0", "cohg_nogate", "cohg"]:
        for s in SEEDS:
            cfgs.append((meth, s, 1e-3, 0.4, ""))
    return cfgs


def result_path(meth, seed, lr, tag):
    return os.path.join(OUT, f"gpt2_{meth}_lr{lr:g}{tag}_s{seed}.json")


def pid_alive(pid: int) -> bool:
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not h:
        return False
    code = ctypes.c_ulong()
    ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return bool(ok) and code.value == STILL_ACTIVE


def try_claim(res: str) -> bool:
    """Atomically claim a config; reap claims left by dead launchers."""
    claim = res + ".claim"
    for _ in range(2):
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                pid = int(open(claim).read().strip() or 0)
            except (OSError, ValueError):
                pid = 0
            if pid and pid_alive(pid):
                return False
            try:
                os.remove(claim)
            except OSError:
                return False
    return False


def run_one(cfg, gpu):
    meth, seed, lr, ml, tag = cfg
    final = result_path(meth, seed, lr, tag)
    tmp = final + ".tmpdir"
    os.makedirs(tmp, exist_ok=True)
    cmd = [PY, os.path.join(HERE, "e4_gpt2_tta.py"), "--method", meth,
           "--seed", str(seed), "--lr", str(lr)] + COMMON + \
          ["--out-dir", tmp]
    if ml is not None:
        cmd += ["--meta-lr", str(ml)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    log = os.path.join(LOGDIR, f"gpt2_{meth}{tag}_lr{lr:g}_s{seed}.log")
    t0 = time.time()
    with open(log, "a") as lf:
        lf.write(f"\n=== {time.ctime()} gpu{gpu} {' '.join(cmd)}\n")
        lf.flush()
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                              env=env, cwd=HERE)
    src = os.path.join(tmp, f"gpt2_{meth}_lr{lr:g}_s{seed}.json")
    ok = proc.returncode == 0 and os.path.exists(src)
    if ok:
        os.replace(src, final)
    shutil.rmtree(tmp, ignore_errors=True)
    return ok, time.time() - t0, log


def worker(gpu):
    """Single pass over the config list, claim-checked; safe to run from
    several processes/threads concurrently."""
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(LOGDIR, exist_ok=True)
    n_ok = n_fail = 0
    for cfg in make_configs():
        meth, seed, lr, ml, tag = cfg
        final = result_path(meth, seed, lr, tag)
        if os.path.exists(final):
            continue
        if not try_claim(final):
            continue
        name = f"{meth}{tag}/lr{lr:g}/s{seed}"
        print(f"[e4v2 gpu{gpu}] start {name}", flush=True)
        try:
            ok, dt, log = run_one(cfg, gpu)
        finally:
            try:
                os.remove(final + ".claim")
            except OSError:
                pass
        if ok:
            n_ok += 1
            print(f"[e4v2 gpu{gpu}] done {name} ({dt:.0f}s)", flush=True)
        else:
            n_fail += 1
            print(f"[e4v2 gpu{gpu}] FAIL {name} (see {log})", flush=True)
    print(f"[e4v2 gpu{gpu}] worker done ok={n_ok} fail={n_fail}", flush=True)
    return n_fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cfgs = [c for c in make_configs()
            if not os.path.exists(result_path(c[0], c[1], c[2], c[4]))]
    print(f"E4v2: {len(cfgs)} configs pending on GPUs {args.gpus}",
          flush=True)
    if args.dry_run:
        for c in cfgs:
            print("  ", c)
        return
    threads = [threading.Thread(target=worker, args=(g,))
               for g in args.gpus]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print("E4v2 launcher exit", flush=True)


if __name__ == "__main__":
    main()
