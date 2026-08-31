"""Round-3 review chain launcher: ONE detached, phase-ordered GPU queue.

Answers three review items in strict order on the local 2x RTX 3080 (20 GB):

  PHASE 1  p1_e4fix    review P4 -- E4 held-bound implementation fix.
      e4_gpt2_tta.py now takes the FULL vector-valued drift-hold interface
      (dh.probe(..., eta_vec=eta); dh.bounds(eta)), i.e. Proposition 10's
          iota_t = Delta eta_t * Hbar_t + eta_max,t0 * (M_H P_t + nu_H)
      with Delta eta_t = ||eta_t - eta_t0||_inf live and eta_max taken at the
      LAST PROBE -- exactly what e2_timeseries.py / e3_continual.py do.  The
      old scalar path (Delta eta == 0, eta_max at the current step) survives
      behind --legacy-hold.  This phase reruns cohg_r0 seeds 0,1,2 with the
      corrected bound and identical e4_v2 flags       -> results/e4_fix
      The comparison target is results/e4_v2/gpt2_cohg_r0_lr0.001_s{0,1,2}.

  PHASE 2  p2_misset   review P7 -- "adaptation when it is needed".
      2a: mis-set initialisation eta0 = 1e-4 (10x too small, mirroring the E2
          design) on the standard stream: fixed / cohg_r0 / cohg_nogate,
          seeds 0-2, 9 runs                            -> results/e4_misset
      2b: reverse-order cohg_r0 seeds 3-7, 5 runs      -> results/e4_orders
          (new files only; the existing s0-2 are never touched)

  PHASE 3  p3_e3traced review P5/P9 -- E3 principal arms WITH loss traces.
      The retained-holdout condition (E3 default, i.e. NO --no-holdout) with
      --log-losses, at both EWC operating points:
          {cohg, cohg_nogate, hd, fixed} x {ewc10, ewc1000} x seeds 0-9
      80 runs                                          -> results/e3_traced
      results/e3 is never touched; e3_traced becomes the traced canonical set.

Runner mechanics follow launch_e4_expand.py / launch_e3_wave2.py:
skip-existing, atomic .claim files reaped from dead PIDs, private tmp out-dir
per run then atomic rename, per-run child logs under results/logs/r3_chain/,
free-VRAM gating via nvidia-smi, MAX_TRIES attempts per config.  Two extra
mechanisms:

  * VRAM RESERVATIONS.  SLOTS_PER_GPU worker threads share a card, so a
    freshly launched child's memory is not yet visible to nvidia-smi.  Each
    launch reserves need_mib for WARMUP_S seconds and every free-memory
    measurement subtracts the live reservations of that card.  With the
    7000 MiB E3 gate two jobs fit per card; with the 18.6/19.0 GB GPT-2 gates
    exactly one does.

  * CANARY GATING.  Each phase names one config as its canary.  No other
    config of that phase starts until the canary's JSON exists AND passes a
    phase-specific sanity check (keys, step count, drift boundaries, ...).
    A failed canary aborts that phase (and any phase declaring it as a
    dependency) instead of burning days of GPU time on a broken flag.

Usage (detached):
    python -u launch_r3_chain.py --gpus 0 1
    python launch_r3_chain.py --dry-run        # show the queue, run nothing
    python launch_r3_chain.py --status         # artifact counts per phase
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
OUT_FIX = os.path.join(RES, "e4_fix")
OUT_MIS = os.path.join(RES, "e4_misset")
OUT_ORD = os.path.join(RES, "e4_orders")
OUT_E3T = os.path.join(RES, "e3_traced")
LOGROOT = os.path.join(RES, "logs")
LOGDIR = os.path.join(LOGROOT, "r3_chain")
PY = sys.executable

POLL_S = 120           # re-measure free VRAM this often when nothing fits
COOLDOWN_S = 30        # let VRAM settle after a child exits
MAX_TRIES = 3          # attempts per config before giving up
SLOTS_PER_GPU = 2      # worker threads per physical card
WARMUP_S = 600         # a launch reserves need_mib for this long
HEARTBEAT_S = 1800     # queue-wide status line

E4 = os.path.join(HERE, "e4_gpt2_tta.py")
E3 = os.path.join(HERE, "e3_continual.py")

# e4_v2 standard flags (Phase 1 and 2 share them; only --lr / order differ)
E4_COMMON = ["--tokens-per-domain", "512000", "--max-steps", "3000",
             "--batch", "2", "--seq-len", "256", "--probe-every", "100",
             "--kw-eps", "0.15", "--device", "cuda:0"]

ALT_ORDER = "code,news,wiki"
ALT_PREFIX = "gpt2order_cnw_"

# GPT-2 arm -> (meta_lr, required free MiB, rough hours); thresholds are the
# calibrated e4_v2 peaks (fixed 6.05, cohg_r0 16.96, cohg_nogate 18.80 GiB).
E4_ARM = {
    "fixed":       (None, 7200,  0.4),
    "cohg_r0":     (0.4, 18600,  4.4),
    "cohg_nogate": (0.4, 19000,  5.4),
}
# E3 arm -> (meta_lr, required free MiB, rough hours) from results/e3 walltimes
E3_ARM = {
    "cohg":        (0.4,  7000, 1.75),
    "cohg_nogate": (0.4,  7000, 0.45),
    "hd":          (0.02, 7000, 0.25),
    "fixed":       (None, 7000, 0.25),
}


# --------------------------------------------------------------------------
# config builders
# --------------------------------------------------------------------------
def e4_cfg(phase, method, seed, lr, out_dir, order=None, prefix=""):
    ml, need, hours = E4_ARM[method]
    src = "gpt2_%s_lr%g_s%d.json" % (method, lr, seed)   # e4_gpt2_tta.py name
    # order-tagged runs follow launch_e4v2/launch_e4_expand: the "gpt2_" stem
    # is REPLACED by the tag, e.g. gpt2order_cnw_cohg_r0_lr0.001_s3.json
    name = (prefix + src[len("gpt2_"):]) if prefix else src
    cmd = ["--method", method, "--seed", str(seed), "--lr", "%g" % lr]
    cmd += E4_COMMON
    if ml is not None:
        cmd += ["--meta-lr", str(ml)]
    if order is not None:
        cmd += ["--domain-order", order]
    return {"phase": phase, "script": E4, "cmd": cmd, "src": src,
            "result": os.path.join(out_dir, name), "need_mib": need,
            "hours": hours, "key": os.path.splitext(name)[0], "check": "e4",
            "expect_order": (order or "wiki,news,code")}


def e3_cfg(phase, method, ewc, seed):
    ml, need, hours = E3_ARM[method]
    name = "cifar100_%s_lr0.05_ewc%g_s%d.json" % (method, ewc, seed)
    cmd = ["--method", method, "--seed", str(seed), "--lr", "0.05",
           "--ewc0", "%g" % ewc, "--log-losses", "--device", "cuda:0"]
    if ml is not None:
        cmd += ["--meta-lr", str(ml)]
    return {"phase": phase, "script": E3, "cmd": cmd, "src": name,
            "result": os.path.join(OUT_E3T, name), "need_mib": need,
            "hours": hours, "key": os.path.splitext(name)[0], "check": "e3"}


def build_phases():
    ph = []

    # ---- PHASE 1: E4 held-bound fix, corrected bound, seeds 0,1,2 --------
    p1 = [e4_cfg("p1_e4fix", "cohg_r0", s, 1e-3, OUT_FIX) for s in (0, 1, 2)]
    ph.append({"name": "p1_e4fix", "cfgs": p1,
               "canary": p1[0]["key"], "depends": None,
               # every phase-1 config is the SAME command up to --seed and the
               # corrected code path was already validated end-to-end by a
               # 25-step smoke run, so waiting 4.4 h for the canary's JSON
               # would idle the second card for nothing.  Release the bulk
               # once the canary's child log proves the stream is the intended
               # one and the run is 1/6 of the way in; the full JSON check
               # still runs when the artifact lands and still aborts whatever
               # is left pending if it fails.
               "progress": ["stream: 2999 steps, drift at [1000, 2000]",
                            "t=500 "],
               "targets": [(OUT_FIX, 3)]})

    # ---- PHASE 2: mis-set eta0, then reverse-order seeds 3-7 -------------
    p2 = []
    p2 += [e4_cfg("p2_misset", "fixed", s, 1e-4, OUT_MIS) for s in range(3)]
    p2 += [e4_cfg("p2_misset", "cohg_r0", s, 1e-4, OUT_MIS) for s in range(3)]
    p2 += [e4_cfg("p2_misset", "cohg_nogate", s, 1e-4, OUT_MIS)
           for s in range(3)]
    p2 += [e4_cfg("p2_misset", "cohg_r0", s, 1e-3, OUT_ORD,
                  order=ALT_ORDER, prefix=ALT_PREFIX) for s in range(3, 8)]
    ph.append({"name": "p2_misset", "cfgs": p2,
               "canary": p2[0]["key"],           # fixed lr1e-4 s0, ~0.4 h
               "depends": "p1_e4fix",
               "targets": [(OUT_MIS, 9), (OUT_ORD, 17)]})

    # ---- PHASE 3: E3 principal arms with loss traces ---------------------
    p3 = [e3_cfg("p3_e3traced", "fixed", 10, 0)]      # canary, ~0.25 h
    p3 += [e3_cfg("p3_e3traced", m, 10, 0)
           for m in ("cohg", "cohg_nogate", "hd")]
    p3 += [e3_cfg("p3_e3traced", m, 1000, 0)
           for m in ("cohg", "cohg_nogate", "hd", "fixed")]
    for m in ("cohg", "cohg_nogate", "hd", "fixed"):
        for e in (10, 1000):
            p3 += [e3_cfg("p3_e3traced", m, e, s) for s in range(1, 10)]
    ph.append({"name": "p3_e3traced", "cfgs": p3,
               "canary": p3[0]["key"], "depends": None,
               "targets": [(OUT_E3T, 80)]})
    return ph


PHASES = build_phases()
BY_KEY = {c["key"]: c for p in PHASES for c in p["cfgs"]}


# --------------------------------------------------------------------------
# canary sanity checks
# --------------------------------------------------------------------------
def check_e4(c):
    d = json.load(open(c["result"]))
    # gate_open_frac / gate_open_steps are legitimately None for the arms that
    # never build a certificate (fixed, hd) -- presence is checked, not value.
    need = ["method", "seed", "lr0", "steps", "drift_steps", "online_ppl",
            "mean_logloss", "gate_open_frac", "losses", "lam_hist",
            "held_bound", "gate_open_steps"]
    miss = [k for k in need if k not in d]
    if miss:
        return False, "missing keys %s" % miss
    if not 2900 <= d["steps"] <= 3000:
        return False, "steps=%s not in [2900,3000]" % d["steps"]
    if list(d["drift_steps"]) != [1000, 2000]:
        return False, "drift_steps=%s != [1000, 2000]" % d["drift_steps"]
    if d.get("domain_order") != c["expect_order"]:
        return False, "domain_order=%s" % d.get("domain_order")
    if d["held_bound"] != "vector_prop10":
        return False, "held_bound=%s (expected vector_prop10)" % d["held_bound"]
    if not (0 < d["online_ppl"] < 1e4):
        return False, "online_ppl=%s" % d["online_ppl"]
    if len(d["losses"]) != d["steps"]:
        return False, "losses length != steps"
    gof = d["gate_open_frac"]
    gos = d["gate_open_steps"]
    if d["method"].startswith("cohg") and (gof is None or gos is None):
        return False, "certificate arm with gate_open_frac/steps = None"
    return True, ("steps=%d drift=%s ppl=%.4f gate_open_frac=%s "
                  "gate_open_steps=%s peak=%.2fGB"
                  % (d["steps"], d["drift_steps"], d["online_ppl"],
                     ("%.6g" % gof) if gof is not None else "n/a",
                     (gos[:5] if gos is not None else "n/a"),
                     d.get("peak_mem_gb", float("nan"))))


def check_e3(c):
    d = json.load(open(c["result"]))
    need = ["method", "seed", "lr0", "ewc0", "avg_acc", "bwt", "acc_matrix",
            "lam_hist", "losses"]
    miss = [k for k in need if k not in d]
    if miss:
        return False, "missing keys %s" % miss
    if len(d["acc_matrix"]) != 10:
        return False, "acc_matrix len %d != 10" % len(d["acc_matrix"])
    if not d["losses"]:
        return False, "empty losses trace (--log-losses not honoured)"
    if not (0.0 < d["avg_acc"] < 1.0):
        return False, "avg_acc=%s" % d["avg_acc"]
    return True, ("avg_acc=%.4f bwt=%.4f losses=%d tasks=%d"
                  % (d["avg_acc"], d["bwt"], len(d["losses"]),
                     len(d["acc_matrix"])))


CHECKS = {"e4": check_e4, "e3": check_e3}


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
_RESERVE = {}                      # gpu -> [[t_launch, mib], ...]
STATE_LOCK = threading.RLock()
TRIES = {}
ABORTED = set()
EXHAUSTED = set()
CANARY_OK = set()


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
    except Exception as e:                                # noqa: BLE001
        print("[r3 gpu%d] nvidia-smi failed: %r" % (gpu, e), flush=True)
        return 0


def effective_free(gpu: int) -> int:
    return free_mib(gpu) - reserved_mib(gpu)


# --------------------------------------------------------------------------
def run_one(c, gpu):
    tmp = c["result"] + ".tmpdir"
    os.makedirs(tmp, exist_ok=True)
    cmd = [PY, "-u", c["script"]] + c["cmd"] + ["--out-dir", tmp]
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


def incomplete(ph):
    return [c for c in ph["cfgs"] if not os.path.exists(c["result"])]


def phase_blocked(ph):
    """True when no remaining config of the phase can still be attempted."""
    inc = incomplete(ph)
    if not inc:
        return False
    with STATE_LOCK:
        live = [c for c in inc if TRIES.get(c["key"], 0) < MAX_TRIES]
    if live:
        return False
    return not any(claimed(c["result"]) for c in inc)


PROGRESS_OK = set()


def canary_progress_ok(ph, can):
    """Optional early release: all `progress` markers present in the canary's
    child log.  Only relaxes WHEN the bulk may start; the full JSON check on
    the finished canary still runs and can still abort the phase."""
    pats = ph.get("progress")
    if not pats:
        return False
    if ph["name"] in PROGRESS_OK:
        return True
    log = os.path.join(LOGDIR, can["key"] + ".log")
    try:
        with open(log, "r", errors="ignore") as f:
            txt = f.read()
    except OSError:
        return False
    if all(p in txt for p in pats):
        with STATE_LOCK:
            if ph["name"] not in PROGRESS_OK:
                PROGRESS_OK.add(ph["name"])
                print("[r3] canary PROGRESS OK %s/%s (markers %s seen) -- "
                      "releasing the rest of the phase"
                      % (ph["name"], can["key"], pats), flush=True)
        return True
    return False


def phase_tail_only(ph):
    """True when every config still missing from the phase is ALREADY running.

    Nothing a worker could start remains here, so the next phase may begin and
    fill the idle card -- the tail of a phase is often a single long run that
    cannot be split across GPUs.  Configs that are merely gated (the canary has
    not been validated yet) are NOT running, so this stays False and the canary
    rule keeps its teeth."""
    inc = incomplete(ph)
    return bool(inc) and all(claimed(c["result"]) for c in inc)


def canary_gate(ph):
    """Configs a worker may pick from, honouring this phase's canary rule."""
    inc = incomplete(ph)
    with STATE_LOCK:
        pend = [c for c in inc if TRIES.get(c["key"], 0) < MAX_TRIES]
    pend = [c for c in pend if not claimed(c["result"])]
    if ph["name"] in CANARY_OK:
        return pend, inc
    can = BY_KEY[ph["canary"]]
    if not os.path.exists(can["result"]):
        if canary_progress_ok(ph, can):
            return pend, inc
        return [c for c in pend if c["key"] == ph["canary"]], inc
    with STATE_LOCK:
        if ph["name"] in ABORTED:
            return [], inc
        if ph["name"] not in CANARY_OK:
            try:
                ok, msg = CHECKS[can["check"]](can)
            except Exception as e:                        # noqa: BLE001
                ok, msg = False, "check raised %r" % (e,)
            if ok:
                CANARY_OK.add(ph["name"])
                print("[r3] CANARY OK  %s/%s: %s"
                      % (ph["name"], can["key"], msg), flush=True)
            else:
                ABORTED.add(ph["name"])
                print("[r3] *** CANARY FAILED %s/%s: %s -- phase ABORTED, "
                      "bulk NOT started ***" % (ph["name"], can["key"], msg),
                      flush=True)
                return [], inc
    return pend, inc


def active_phase():
    """First phase, in declaration order, that can still make progress."""
    for ph in PHASES:
        name = ph["name"]
        if name in ABORTED or name in EXHAUSTED:
            continue
        dep = ph.get("depends")
        if dep and dep in ABORTED:
            with STATE_LOCK:
                if name not in ABORTED:
                    ABORTED.add(name)
                    print("[r3] phase %s ABORTED (depends on aborted %s)"
                          % (name, dep), flush=True)
            continue
        inc = incomplete(ph)
        if not inc:
            continue                       # phase complete -> next phase
        if phase_blocked(ph):
            with STATE_LOCK:
                if name not in EXHAUSTED:
                    EXHAUSTED.add(name)
                    print("[r3] phase %s EXHAUSTED (%d configs hit MAX_TRIES);"
                          " moving on" % (name, len(inc)), flush=True)
            continue
        if phase_tail_only(ph):
            continue       # only running configs left -> overlap the next phase
        return ph
    return None


def worker(gpu: int, slot: int):
    tag = "[r3 gpu%d.%d]" % (gpu, slot)
    n_ok = n_fail = 0
    while True:
        ph = active_phase()
        if ph is None:
            break
        pend, inc = canary_gate(ph)
        if not pend:
            if not inc:
                continue
            time.sleep(POLL_S)             # peers own the rest / canary runs
            continue
        avail = effective_free(gpu)
        pick = next((c for c in pend if c["need_mib"] <= avail), None)
        if pick is None:
            print("%s %s: %d MiB usable, cheapest pending needs %d MiB; "
                  "wait %ds" % (tag, ph["name"], avail,
                                min(c["need_mib"] for c in pend), POLL_S),
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
        print("%s start %s (phase %s, need %d MiB, usable %d MiB, ~%.2fh) %s"
              % (tag, pick["key"], ph["name"], pick["need_mib"], avail,
                 pick["hours"], time.ctime()), flush=True)
        try:
            ok, dt, log = run_one(pick, gpu)
        finally:
            unreserve(gpu, tok)
            try:
                os.remove(pick["result"] + ".claim")
            except OSError:
                pass
        if ok:
            n_ok += 1
            print("%s done %s (%.2fh)" % (tag, pick["key"], dt / 3600),
                  flush=True)
        else:
            n_fail += 1
            print("%s FAIL %s (try %d/%d, see %s)"
                  % (tag, pick["key"], ntry, MAX_TRIES, log), flush=True)
            time.sleep(60)
        time.sleep(COOLDOWN_S)
    print("%s worker exit ok=%d fail=%d" % (tag, n_ok, n_fail), flush=True)


def counts():
    out = []
    for ph in PHASES:
        have = sum(1 for c in ph["cfgs"] if os.path.exists(c["result"]))
        flag = (" ABORTED" if ph["name"] in ABORTED else
                (" EXHAUSTED" if ph["name"] in EXHAUSTED else ""))
        out.append("%s %d/%d%s" % (ph["name"], have, len(ph["cfgs"]), flag))
    return " | ".join(out)


def heartbeat(stop):
    while not stop.is_set():
        stop.wait(HEARTBEAT_S)
        if stop.is_set():
            break
        print("[r3 heartbeat] %s %s free=[%d,%d] MiB"
              % (time.ctime(), counts(), free_mib(0), free_mib(1)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--slots", type=int, default=SLOTS_PER_GPU)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    for d in (OUT_FIX, OUT_MIS, OUT_ORD, OUT_E3T, LOGROOT, LOGDIR):
        os.makedirs(d, exist_ok=True)

    if args.status:
        print("R3-chain status @ %s: %s" % (time.ctime(), counts()))
        for ph in PHASES:
            for d, t in ph["targets"]:
                have = len([f for f in os.listdir(d)
                            if f.endswith(".json")]) if os.path.isdir(d) else 0
                print("   %-12s %d/%d json" % (os.path.basename(d), have, t))
        return

    tot = sum(len(p["cfgs"]) for p in PHASES)
    pend = [c for p in PHASES for c in p["cfgs"]
            if not os.path.exists(c["result"])]
    print("R3-chain: %d/%d configs pending, %.1f GPU-hours, GPUs %s x%d "
          "slots, pid %d @ %s"
          % (len(pend), tot, sum(c["hours"] for c in pend), args.gpus,
             args.slots, os.getpid(), time.ctime()), flush=True)
    for ph in PHASES:
        p = [c for c in ph["cfgs"] if not os.path.exists(c["result"])]
        print("   %-12s %3d pending  %6.1f GPU-h  canary=%s"
              % (ph["name"], len(p), sum(c["hours"] for c in p),
                 ph["canary"]), flush=True)
    if args.dry_run:
        for ph in PHASES:
            print("--- %s ---" % ph["name"])
            for c in ph["cfgs"]:
                mark = "DONE" if os.path.exists(c["result"]) else "    "
                print(" %s %-46s need=%5d ~%.2fh -> %s"
                      % (mark, c["key"], c["need_mib"], c["hours"],
                         c["result"]))
                if c["key"] == ph["canary"]:
                    print("        ^ CANARY   %s %s"
                          % (os.path.basename(c["script"]), " ".join(c["cmd"])))
        return

    stop = threading.Event()
    hb = threading.Thread(target=heartbeat, args=(stop,), daemon=True)
    hb.start()
    ths = [threading.Thread(target=worker, args=(g, s))
           for g in args.gpus for s in range(args.slots)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    stop.set()
    print("R3-chain exit @ %s: %s" % (time.ctime(), counts()), flush=True)


if __name__ == "__main__":
    main()
