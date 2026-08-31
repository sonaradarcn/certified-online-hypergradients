"""E4 production launcher (tracker R050): GPT-2 124M streaming TTA.

Matrix (local 20GB GPUs only):
  fixed lr in {3e-4, 1e-3, 3e-3}          x 3 seeds
  cohg / cohg_r0 / cohg_nogate @ lr0=1e-3 x 3 seeds
  hd @ lr0=1e-3, meta-lr in {0.2, 2, 20}  x 3 seeds  (per-domain tuning
      reported honestly — F19: raw-scale meta-LR does not transfer)

Config: probe=100, kw-eps 0.15, 3000 steps, batch 2, seq 256, fp32.
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "e4")
PY = sys.executable
SEEDS = [0, 1, 2]


def make_configs():
    cfgs = []
    for s in SEEDS:
        for lr in [3e-4, 1e-3, 3e-3]:
            cfgs.append(("fixed", s, lr, None, ""))
        for meth in ["cohg", "cohg_r0", "cohg_nogate"]:
            cfgs.append((meth, s, 1e-3, 0.4, ""))
        for ml in [0.2, 2.0, 20.0]:
            cfgs.append(("hd", s, 1e-3, ml, f"_ml{ml:g}"))
    return cfgs


def result_path(meth, seed, lr, tag):
    return os.path.join(OUT, f"gpt2_{meth}_lr{lr:g}{tag}_s{seed}.json")


def worker(gpu, q, failures):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    while True:
        try:
            meth, seed, lr, ml, tag = q.get_nowait()
        except queue.Empty:
            return
        cmd = [PY, os.path.join(HERE, "e4_gpt2_tta.py"), "--method", meth,
               "--seed", str(seed), "--lr", str(lr), "--max-steps", "3000",
               "--batch", "2", "--seq-len", "256", "--probe-every", "100",
               "--kw-eps", "0.15", "--device", "cuda:0", "--out-dir", OUT]
        if ml is not None:
            cmd += ["--meta-lr", str(ml)]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                              cwd=HERE)
        # e4 writes gpt2_{meth}_lr{lr}_s{seed}.json; add the ml tag if needed
        src = os.path.join(OUT, f"gpt2_{meth}_lr{lr:g}_s{seed}.json")
        dst = result_path(meth, seed, lr, tag)
        if tag and os.path.exists(src):
            os.replace(src, dst)
        name = f"{meth}{tag}/lr{lr:g}/s{seed}"
        if proc.returncode != 0:
            failures.append(name)
            print(f"[gpu{gpu}] FAIL {name}\n{proc.stderr[-300:]}", flush=True)
        else:
            print(f"[gpu{gpu}] done {name} ({time.time() - t0:.0f}s) "
                  f"| left ~{q.qsize()}", flush=True)
        q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cfgs = [c for c in make_configs()
            if not os.path.exists(result_path(c[0], c[1], c[2], c[4]))]
    print(f"E4: {len(cfgs)} configs on GPUs {args.gpus}")
    if args.dry_run:
        for c in cfgs[:8]:
            print("  ", c)
        return
    q = queue.Queue()
    for c in cfgs:
        q.put(c)
    failures = []
    threads = [threading.Thread(target=worker, args=(g, q, failures))
               for g in args.gpus]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print(f"E4 DONE | failures: {len(failures)}")
    for f in failures:
        print("  FAIL", f)


if __name__ == "__main__":
    main()
