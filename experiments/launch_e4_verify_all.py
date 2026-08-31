"""Review P4 at FULL scope: rerun every legacy gated E4 run with the corrected
vector-valued Proposition-10 held bound, same seed, same flags.

`results/e4_fix` verified the corrected bound against the shipped runs on 3
seeds only (cohg_r0, standard order, seeds 0/1/2).  The reviewer requires the
same-seed check on EVERY legacy gated E4 run that remains a reported result.
This launcher queues exactly those, into

    results/e4_verify_all/<same basename as the legacy JSON>

Queue (11 runs; legacy provenance = the JSON lacks `legacy_hold` /
`held_bound` / `gate_open_steps`, the fields the fix added):

    results/e4_v2      cohg_r0 (rank 0) seeds 3,4,5,6,7   order wiki,news,code
    results/e4_orders  cohg_r0 (rank 0) seeds 0,1,2       order code,news,wiki
    results/e4_v2      cohg    (rank 4) seeds 0,1,2       order wiki,news,code

cohg_r0 seeds 0/1/2 on the standard order are NOT requeued -- they are exactly
the `results/e4_fix` runs and are folded into the comparison from there.

EXCLUDED BY CONSTRUCTION (noted in COMPARE_ALL.md, not rerun): `fixed`, `hd`
and `cohg_nogate`.  The held bound `(rho, kappa)` reaches the parameter update
only through `CoordGatedController.maybe_update` / `est.step`; `fixed` builds no
certificate (`est is None`), `hd` updates lambda with `HDBaseline.update`, and
`cohg_nogate` sets `rho, kappa = 1.0, 0.0` before the drift hold is consulted
and takes an ungated sign step.  Neither code path can change those runs.

Flags are read back from the shipped JSONs and are the e4_v2 / e4_expand /
r3_chain constants: --tokens-per-domain 512000 --max-steps 3000 --batch 2
--seq-len 256 --probe-every 100 --kw-eps 0.15 --lr 0.001 --meta-lr 0.4,
K=20 gamma=0.9 M_H=50 (driver defaults), rank 4 for `cohg`, forced to 0 for
`cohg_r0`.  The ONLY difference from the shipped runs is the default (fixed)
held-bound path; `--legacy-hold` reproduces the old scalar one.

Runner mechanics are lifted from launch_r3_chain.py: skip-existing, atomic
.claim files reaped from dead PIDs, private tmp out-dir per run then atomic
rename, per-run child logs under results/logs/e4_verify_all/, free-VRAM gating
via nvidia-smi with launch reservations, MAX_TRIES attempts per config.  One
E4 job per card (each needs 17-20 GB), r0 runs first, r4 runs last.

Usage (detached):
    python -u launch_e4_verify_all.py --gpus 0 1
    python launch_e4_verify_all.py --dry-run
    python launch_e4_verify_all.py --status
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
V2 = os.path.join(RES, "e4_v2")
ORD = os.path.join(RES, "e4_orders")
OUT = os.path.join(RES, "e4_verify_all")
LOGDIR = os.path.join(RES, "logs", "e4_verify_all")
E4 = os.path.join(HERE, "e4_gpt2_tta.py")
PY = sys.executable

POLL_S = 120
COOLDOWN_S = 45
MAX_TRIES = 3
WARMUP_S = 900          # a launch reserves need_mib for this long
HEARTBEAT_S = 1800

# e4_v2 / launch_e4_expand / launch_r3_chain constants (identical in all three)
E4_COMMON = ["--tokens-per-domain", "512000", "--max-steps", "3000",
             "--batch", "2", "--seq-len", "256", "--probe-every", "100",
             "--kw-eps", "0.15", "--device", "cuda:0"]

STD_ORDER = "wiki,news,code"
ALT_ORDER = "code,news,wiki"
ALT_PREFIX = "gpt2order_cnw_"

# arm -> (required free MiB, rough hours).  Thresholds from the peaks MEASURED
# under the CORRECTED path (the vector bound costs ~0.45 GiB more than the
# scalar one): cohg_r0 17.43 GiB in results/e4_fix (shipped scalar path 16.96);
# cohg r=4 19.281 GiB in a 45-step calibration run (shipped scalar path 18.82).
# 19.281 GiB = 19748 MiB allocated + ~350 MiB CUDA context, so the r=4 arm only
# fits on a card with <=~0.5 GB of other usage -- on this box that is GPU1
# (99 MiB), not the display card GPU0 (~850 MiB).  The gate below encodes that;
# it is a fit criterion, not a preference.
NEED_R0 = 18600
NEED_R4 = 19900
HOURS_R0 = 3.3          # results/e4_fix wall clock: 2.94 / 3.23 / 3.74 h
HOURS_R4 = 5.5          # r4/r0 wall ratio 1.66 in e4_v2, applied to 3.3 h


def cfg(method, seed, legacy_dir, order, prefix=""):
    """One queue entry, keyed by the legacy JSON's basename."""
    src = "gpt2_%s_lr0.001_s%d.json" % (method, seed)   # e4_gpt2_tta.py name
    name = (prefix + src[len("gpt2_"):]) if prefix else src
    cmd = ["--method", method, "--seed", str(seed), "--lr", "0.001",
           "--meta-lr", "0.4", "--domain-order", order] + E4_COMMON
    r4 = (method == "cohg")
    return {"method": method, "seed": seed, "order": order, "cmd": cmd,
            "src": src, "legacy": os.path.join(legacy_dir, name),
            "result": os.path.join(OUT, name), "heavy": r4,
            "need_mib": NEED_R4 if r4 else NEED_R0,
            "hours": HOURS_R4 if r4 else HOURS_R0,
            "key": os.path.splitext(name)[0]}


def make_configs():
    """r0 first (cheaper, ~3.3 h), r4 last (~5.5 h)."""
    c = []
    for s in (3, 4, 5, 6, 7):                       # standard order, e4_v2
        c.append(cfg("cohg_r0", s, V2, STD_ORDER))
    for s in (0, 1, 2):                             # reverse order, e4_orders
        c.append(cfg("cohg_r0", s, ORD, ALT_ORDER, ALT_PREFIX))
    for s in (0, 1, 2):                             # rank-4 arm, e4_v2
        c.append(cfg("cohg", s, V2, STD_ORDER))
    return c


CFGS = make_configs()


# --------------------------------------------------------------------------
# sanity check on a finished artifact
# --------------------------------------------------------------------------
def check(c):
    d = json.load(open(c["result"]))
    need = ["method", "seed", "lr0", "steps", "drift_steps", "online_ppl",
            "mean_logloss", "gate_open_frac", "coord_open_frac", "losses",
            "lam_hist", "held_bound", "legacy_hold", "gate_open_steps"]
    miss = [k for k in need if k not in d]
    if miss:
        return False, "missing keys %s" % miss
    if d["method"] != c["method"] or d["seed"] != c["seed"]:
        return False, "method/seed mismatch %s/%s" % (d["method"], d["seed"])
    if not 2900 <= d["steps"] <= 3000:
        return False, "steps=%s" % d["steps"]
    if list(d["drift_steps"]) != [1000, 2000]:
        return False, "drift_steps=%s != [1000, 2000]" % d["drift_steps"]
    if d.get("domain_order") != c["order"]:
        return False, "domain_order=%s != %s" % (d.get("domain_order"),
                                                 c["order"])
    if d["held_bound"] != "vector_prop10" or d["legacy_hold"]:
        return False, "held_bound=%s legacy_hold=%s" % (d["held_bound"],
                                                        d["legacy_hold"])
    if len(d["losses"]) != d["steps"]:
        return False, "losses length != steps"
    if d["gate_open_frac"] is None or d["gate_open_steps"] is None:
        return False, "certificate arm with gate_open_frac/steps = None"
    return True, ("steps=%d ppl=%.4f gate_open_frac=%.6g gate_open_steps=%s "
                  "peak=%.2fGB" % (d["steps"], d["online_ppl"],
                                   d["gate_open_frac"],
                                   d["gate_open_steps"][:5],
                                   d.get("peak_mem_gb", float("nan"))))


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
# VRAM accounting
# --------------------------------------------------------------------------
_RES_LOCK = threading.Lock()
_RESERVE = {}
STATE_LOCK = threading.RLock()
TRIES = {}


def reserve(gpu, mib):
    with _RES_LOCK:
        tok = [time.time(), mib]
        _RESERVE.setdefault(gpu, []).append(tok)
        return tok


def unreserve(gpu, tok):
    with _RES_LOCK:
        try:
            _RESERVE[gpu].remove(tok)
        except (KeyError, ValueError):
            pass


def reserved_mib(gpu):
    now = time.time()
    with _RES_LOCK:
        return sum(m for t, m in _RESERVE.get(gpu, []) if now - t < WARMUP_S)


def free_mib(gpu: int) -> int:
    try:
        o = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits", "-i", str(gpu)],
            capture_output=True, text=True, timeout=60)
        return int(o.stdout.strip().splitlines()[0])
    except Exception as e:                                  # noqa: BLE001
        print("[vfy gpu%d] nvidia-smi failed: %r" % (gpu, e), flush=True)
        return 0


def effective_free(gpu: int) -> int:
    return free_mib(gpu) - reserved_mib(gpu)


# --------------------------------------------------------------------------
def run_one(c, gpu):
    tmp = c["result"] + ".tmpdir"
    os.makedirs(tmp, exist_ok=True)
    cmd = [PY, "-u", E4] + c["cmd"] + ["--out-dir", tmp]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    log = os.path.join(LOGDIR, c["key"] + ".log")
    t0 = time.time()
    with open(log, "a") as lf:
        lf.write("\n=== %s gpu%d %s\n" % (time.ctime(), gpu, " ".join(cmd)))
        lf.flush()
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                              env=env, cwd=HERE)
    src = os.path.join(tmp, c["src"])
    ok = proc.returncode == 0 and os.path.exists(src)
    if ok:
        os.replace(src, c["result"])
    shutil.rmtree(tmp, ignore_errors=True)
    return ok, time.time() - t0, log


def pending_cfgs():
    with STATE_LOCK:
        return [c for c in CFGS
                if not os.path.exists(c["result"])
                and TRIES.get(c["key"], 0) < MAX_TRIES
                and not claimed(c["result"])]


def anything_left():
    return any(not os.path.exists(c["result"])
               and (TRIES.get(c["key"], 0) < MAX_TRIES
                    or claimed(c["result"]))
               for c in CFGS)


def choose(pend, avail, n_workers):
    """Pick the next config for a card with `avail` MiB usable.

    List order is r0-first, so the cheap ~3.3 h verifications land first.  The
    one exception keeps the queue from ending in a long single-card tail: only
    a card with ~0.5 GB of other usage can host the r=4 arm at all (19.281 GiB
    measured peak), so while there is still plenty of r0 work for the other
    card, the card that CAN take an r4 job takes it rather than competing for
    r0 and leaving all three 5.5 h runs to be serialised at the end.  Once the
    r0 backlog no longer covers every worker, that card falls back to r0 too.
    """
    fits = [c for c in pend if c["need_mib"] <= avail]
    if not fits:
        return None
    heavy = [c for c in fits if c["heavy"]]
    light = [c for c in fits if not c["heavy"]]
    light_pending = [c for c in pend if not c["heavy"]]
    if heavy and len(light_pending) >= n_workers:
        return heavy[0]
    return (light or heavy)[0]


def worker(gpu: int, n_workers: int):
    tag = "[vfy gpu%d]" % gpu
    n_ok = n_fail = 0
    while anything_left():
        pend = pending_cfgs()
        if not pend:
            time.sleep(POLL_S)          # only peers' running jobs remain
            continue
        avail = effective_free(gpu)
        pick = choose(pend, avail, n_workers)
        if pick is None:
            print("%s %d MiB usable, cheapest pending needs %d MiB; wait %ds"
                  % (tag, avail, min(c["need_mib"] for c in pend), POLL_S),
                  flush=True)
            time.sleep(POLL_S)
            continue
        if not try_claim(pick["result"]):
            time.sleep(5)
            continue
        with STATE_LOCK:
            TRIES[pick["key"]] = TRIES.get(pick["key"], 0) + 1
            ntry = TRIES[pick["key"]]
        tok = reserve(gpu, pick["need_mib"])
        print("%s start %s (need %d MiB, usable %d MiB, ~%.1fh, try %d/%d) %s"
              % (tag, pick["key"], pick["need_mib"], avail, pick["hours"],
                 ntry, MAX_TRIES, time.ctime()), flush=True)
        try:
            ok, dt, log = run_one(pick, gpu)
        finally:
            unreserve(gpu, tok)
            try:
                os.remove(pick["result"] + ".claim")
            except OSError:
                pass
        if ok:
            try:
                good, msg = check(pick)
            except Exception as e:                          # noqa: BLE001
                good, msg = False, "check raised %r" % (e,)
            n_ok += 1
            print("%s done %s (%.2fh) CHECK=%s %s"
                  % (tag, pick["key"], dt / 3600, "OK" if good else "FAIL",
                     msg), flush=True)
        else:
            n_fail += 1
            print("%s FAIL %s (try %d/%d, %.2fh, see %s)"
                  % (tag, pick["key"], ntry, MAX_TRIES, dt / 3600, log),
                  flush=True)
            time.sleep(60)
        time.sleep(COOLDOWN_S)
    print("%s worker exit ok=%d fail=%d" % (tag, n_ok, n_fail), flush=True)


def counts():
    have = sum(1 for c in CFGS if os.path.exists(c["result"]))
    run = sum(1 for c in CFGS
              if not os.path.exists(c["result"]) and claimed(c["result"]))
    return "%d/%d done, %d running" % (have, len(CFGS), run)


def heartbeat(stop):
    while not stop.is_set():
        stop.wait(HEARTBEAT_S)
        if stop.is_set():
            break
        print("[vfy heartbeat] %s %s free=[%d,%d] MiB"
              % (time.ctime(), counts(), free_mib(0), free_mib(1)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    for d in (OUT, LOGDIR):
        os.makedirs(d, exist_ok=True)

    if args.status:
        print("e4_verify_all @ %s: %s" % (time.ctime(), counts()))
        for c in CFGS:
            print("  %-40s %s" % (c["key"],
                                  "DONE" if os.path.exists(c["result"])
                                  else ("RUNNING" if claimed(c["result"])
                                        else "pending")))
        return

    pend = [c for c in CFGS if not os.path.exists(c["result"])]
    gh = sum(c["hours"] for c in pend)
    print("e4_verify_all: %d/%d configs pending, %.1f GPU-hours "
          "(~%.1f h wall on %d cards), pid %d @ %s"
          % (len(pend), len(CFGS), gh, gh / max(len(args.gpus), 1),
             len(args.gpus), os.getpid(), time.ctime()), flush=True)
    for c in CFGS:
        mark = "DONE" if os.path.exists(c["result"]) else "    "
        print("   %s %-40s %-8s s%d order=%-14s need=%5d ~%.1fh  legacy=%s"
              % (mark, c["key"], c["method"], c["seed"], c["order"],
                 c["need_mib"], c["hours"],
                 os.path.relpath(c["legacy"], ROOT)), flush=True)
    if args.dry_run:
        print("\ncommand template:")
        print("  %s -u %s %s --out-dir <tmp>"
              % (os.path.basename(PY), os.path.basename(E4),
                 " ".join(CFGS[0]["cmd"])))
        return

    stop = threading.Event()
    hb = threading.Thread(target=heartbeat, args=(stop,), daemon=True)
    hb.start()
    ths = [threading.Thread(target=worker, args=(g, len(args.gpus)))
           for g in args.gpus]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    stop.set()
    print("e4_verify_all exit @ %s: %s" % (time.ctime(), counts()), flush=True)


if __name__ == "__main__":
    main()
