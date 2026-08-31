"""E4 expansion launcher: extra seeds + alternate domain order (GPT-2 124M).

Answers the review critique "3 seeds / one domain order is too weak" by
extending the e4_v2 compressed 3-domain stream (--tokens-per-domain 512000,
boundaries at steps 1000/2000) along two axes:

  A. more seeds on the standard order wiki->news->code (seeds 3..7) for
     fixed lr1e-3, hd ml2, cohg_r0, cohg_nogate   -> results/e4_v2
  B. the reversed order code->news->wiki (seeds 0..2) for the same four
     methods                                       -> results/e4_orders
     (filenames carry an order tag: gpt2order_cnw_<method>...)

File naming follows launch_e4v2.py exactly, including the hd "_ml<meta_lr>"
tag, so the new JSONs drop straight into the existing analysis.

GPU coordination: a second agent runs many small E2 jobs (1-2 GB) on both
cards. This runner therefore gates every launch on *currently free* VRAM
measured with nvidia-smi, per method:

    fixed        6.05 GiB peak ->  7200 MiB free required
    hd           6.51 GiB peak ->  7800 MiB free required
    cohg_r0     16.96 GiB peak -> 18600 MiB free required
    cohg_nogate 18.80 GiB peak -> 19000 MiB free required

The two heavy thresholds are calibrated so that an *idle* card passes (the
display GPU idles at ~19270 MiB free, and e4_v2's nogate runs completed on
exactly that card at 18.8 GiB peak) while a single 1-2 GB E2 job blocks
them. Heavy arms therefore only ever start on a card nobody else is using.

A worker owns one physical GPU and runs one job at a time. It always picks
the first *pending* config that fits in the free memory it just measured, so
light arms keep flowing while the heavy arms wait for a card to drain; if
nothing fits it sleeps POLL_S and re-measures. It never touches another
process.

Robustness: skip-existing, atomic .claim files reaped from dead PIDs (same
scheme as launch_e4v2.py), private tmp out-dir per run then atomic rename,
per-run child logs under results/logs/e4_expand/, up to MAX_TRIES attempts
per config. Designed to be started detached and outlive its launcher.
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
ROOT = os.path.join(HERE, "..")
OUT_STD = os.path.join(ROOT, "results", "e4_v2")
OUT_ORD = os.path.join(ROOT, "results", "e4_orders")
LOGDIR = os.path.join(ROOT, "results", "logs", "e4_expand")
PY = sys.executable

POLL_S = 300          # re-measure free VRAM this often when nothing fits
COOLDOWN_S = 30       # let VRAM settle after a child exits
MAX_TRIES = 3         # attempts per config before giving up

COMMON = ["--tokens-per-domain", "512000", "--max-steps", "3000",
          "--batch", "2", "--seq-len", "256", "--probe-every", "100",
          "--kw-eps", "0.15", "--device", "cuda:0"]

# method -> (meta_lr, required free MiB, rough hours)
METHOD = {
    "fixed":       (None, 7200,  0.4),
    "hd":          (2.0,  7800,  0.6),
    "cohg_r0":     (0.4, 18600,  4.3),
    "cohg_nogate": (0.4, 19000,  5.4),
}

ALT_ORDER = "code,news,wiki"
ALT_PREFIX = "gpt2order_cnw_"


def cfg(method, seed, order):
    """One queue entry."""
    ml, need, hours = METHOD[method]
    tag = f"_ml{ml:g}" if method == "hd" else ""
    if order is None:
        out, name = OUT_STD, f"gpt2_{method}_lr0.001{tag}_s{seed}.json"
    else:
        out, name = OUT_ORD, f"{ALT_PREFIX}{method}_lr0.001{tag}_s{seed}.json"
    return {"method": method, "seed": seed, "meta_lr": ml, "order": order,
            "out_dir": out, "result": os.path.join(out, name),
            "need_mib": need, "hours": hours,
            "key": os.path.splitext(name)[0]}


def make_configs():
    """Light arms first; heavy arms interleave alternate-order with extra
    seeds so both review answers make progress even if compute runs out."""
    cfgs = []
    # ---- phase 1: light -------------------------------------------------
    for s in range(3, 8):
        cfgs.append(cfg("hd", s, None))
    for s in range(3, 8):
        cfgs.append(cfg("fixed", s, None))
    for s in range(3):
        cfgs.append(cfg("fixed", s, ALT_ORDER))
    for s in range(3):
        cfgs.append(cfg("hd", s, ALT_ORDER))
    # ---- phase 2: heavy -------------------------------------------------
    for i in range(3):                       # alt seeds 0-2 || std seeds 3-5
        cfgs.append(cfg("cohg_r0", i, ALT_ORDER))
        cfgs.append(cfg("cohg_r0", i + 3, None))
        cfgs.append(cfg("cohg_nogate", i, ALT_ORDER))
        cfgs.append(cfg("cohg_nogate", i + 3, None))
    for s in [6, 7]:                         # remaining standard seeds
        cfgs.append(cfg("cohg_r0", s, None))
        cfgs.append(cfg("cohg_nogate", s, None))
    return cfgs


# --------------------------------------------------------------------------
# claim files (cross-process safe, reaps claims of dead PIDs)
# --------------------------------------------------------------------------
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


def claimed(res: str) -> bool:
    claim = res + ".claim"
    try:
        pid = int(open(claim).read().strip() or 0)
    except (OSError, ValueError):
        return os.path.exists(claim)
    return bool(pid) and pid_alive(pid)


def try_claim(res: str) -> bool:
    claim = res + ".claim"
    for _ in range(2):
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            if claimed(res):
                return False
            try:
                os.remove(claim)
            except OSError:
                return False
    return False


# --------------------------------------------------------------------------
def free_mib(gpu: int) -> int:
    try:
        o = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits", "-i", str(gpu)],
            capture_output=True, text=True, timeout=60)
        return int(o.stdout.strip().splitlines()[0])
    except Exception as e:                                # noqa: BLE001
        print(f"[e4x gpu{gpu}] nvidia-smi failed: {e}", flush=True)
        return 0


def run_one(c, gpu):
    tmp = c["result"] + ".tmpdir"
    os.makedirs(tmp, exist_ok=True)
    cmd = [PY, os.path.join(HERE, "e4_gpt2_tta.py"),
           "--method", c["method"], "--seed", str(c["seed"]),
           "--lr", "0.001"] + COMMON + ["--out-dir", tmp]
    if c["meta_lr"] is not None:
        cmd += ["--meta-lr", str(c["meta_lr"])]
    if c["order"] is not None:
        cmd += ["--domain-order", c["order"]]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    log = os.path.join(LOGDIR, c["key"] + ".log")
    t0 = time.time()
    with open(log, "a") as lf:
        lf.write(f"\n=== {time.ctime()} gpu{gpu} {' '.join(cmd)}\n")
        lf.flush()
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                              env=env, cwd=HERE)
    src = os.path.join(
        tmp, f"gpt2_{c['method']}_lr0.001_s{c['seed']}.json")
    ok = proc.returncode == 0 and os.path.exists(src)
    if ok:
        os.replace(src, c["result"])
    shutil.rmtree(tmp, ignore_errors=True)
    return ok, time.time() - t0, log


def worker(gpu: int):
    tries: dict[str, int] = {}
    n_ok = n_fail = 0
    while True:
        pend = [c for c in make_configs()
                if not os.path.exists(c["result"])
                and tries.get(c["key"], 0) < MAX_TRIES
                and not claimed(c["result"])]
        if not pend:
            others = [c for c in make_configs()
                      if not os.path.exists(c["result"])
                      and tries.get(c["key"], 0) < MAX_TRIES]
            if not others:
                break                       # everything done / exhausted
            time.sleep(POLL_S)              # peer worker is on the rest
            continue
        avail = free_mib(gpu)
        pick = next((c for c in pend if c["need_mib"] <= avail), None)
        if pick is None:
            print(f"[e4x gpu{gpu}] {avail} MiB free, cheapest pending needs "
                  f"{min(c['need_mib'] for c in pend)} MiB; wait {POLL_S}s",
                  flush=True)
            time.sleep(POLL_S)
            continue
        if not try_claim(pick["result"]):
            time.sleep(5)                   # lost the race; re-scan
            continue
        tries[pick["key"]] = tries.get(pick["key"], 0) + 1
        print(f"[e4x gpu{gpu}] start {pick['key']} "
              f"(need {pick['need_mib']} MiB, free {avail} MiB, "
              f"~{pick['hours']:.1f}h) {time.ctime()}", flush=True)
        try:
            ok, dt, log = run_one(pick, gpu)
        finally:
            try:
                os.remove(pick["result"] + ".claim")
            except OSError:
                pass
        if ok:
            n_ok += 1
            print(f"[e4x gpu{gpu}] done {pick['key']} ({dt/3600:.2f}h)",
                  flush=True)
        else:
            n_fail += 1
            print(f"[e4x gpu{gpu}] FAIL {pick['key']} "
                  f"(try {tries[pick['key']]}/{MAX_TRIES}, see {log})",
                  flush=True)
            time.sleep(60)
        time.sleep(COOLDOWN_S)
    print(f"[e4x gpu{gpu}] worker exit ok={n_ok} fail={n_fail}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for d in (OUT_STD, OUT_ORD, LOGDIR):
        os.makedirs(d, exist_ok=True)
    cfgs = make_configs()
    pend = [c for c in cfgs if not os.path.exists(c["result"])]
    print(f"E4-expand: {len(pend)}/{len(cfgs)} configs pending, "
          f"{sum(c['hours'] for c in pend):.1f} GPU-hours, "
          f"GPUs {args.gpus}, pid {os.getpid()} @ {time.ctime()}", flush=True)
    if args.dry_run:
        for c in pend:
            print(f"   {c['key']:<44} need={c['need_mib']:>5} MiB  "
                  f"~{c['hours']:.1f}h  -> {c['result']}")
        return
    ths = [threading.Thread(target=worker, args=(g,)) for g in args.gpus]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    print(f"E4-expand launcher exit @ {time.ctime()}", flush=True)


if __name__ == "__main__":
    main()
