"""Round-4 threshold-transfer launcher: ONE detached, phase-ordered GPU queue.

Reviewer question: the E2 control `absgate` (gate on |ghat_j| > T with the
CONSTANT T = 0.05806520209 calibrated offline on mackey_drift to match COHG's
open rate) matches COHG on its calibration stream at 1 HVP/step.  Does that
static threshold TRANSFER, unchanged, to other domains?

  PHASE A  a_e4   GPT-2 124M streaming TTA (results/e4_absgate)
      a0: cohg_r0 seed 0, e4_v2 flags + --log-gate-stats.  Two jobs in one:
          (i) reproduction check of the PATCHED e4_gpt2_tta.py against the
              stored results/e4_v2/gpt2_cohg_r0_lr0.001_s0.json, and
          (ii) the source of COHG's REALIZED certificate threshold
              distribution (c * beta_col_j) on this domain.
      a1: absgate seeds 0,1,2, --rank 0 (bit-for-bit the cohg_r0 estimator),
          T transferred as is, no certificate, no spectral probe.
      One job per card (17-19 GiB peak on a 20 GiB 3080).

  PHASE B  b_e3   Split-CIFAR-100 continual learning (results/e3_absgate)
      b0: cohg lr0.05 ewc{10,1000} seed 0 + --log-gate-stats.  Same double
          duty: reproduction check against results/e3_traced, and the source
          of COHG's realized c * beta_col_j on this domain.
      b1: absgate ewc{10,1000} x seeds 0-9, alpha 0.4, T transferred as is.
      Two jobs per card.

Runner mechanics are launch_r3_chain.py's: skip-existing, atomic .claim files
reaped from dead PIDs, private tmp out-dir per run then atomic rename, per-run
child logs under results/logs/r4_absgate/, free-VRAM gating via nvidia-smi with
warm-up reservations, MAX_TRIES attempts per config, phase ordering that lets
the tail of a phase overlap the next one.

Usage (detached):
    python -u launch_r4_absgate.py --gpus 0 1
    python launch_r4_absgate.py --dry-run
    python launch_r4_absgate.py --status
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
OUT_E4 = os.path.join(RES, "e4_absgate")
OUT_E4R = os.path.join(OUT_E4, "_ref")
OUT_E3 = os.path.join(RES, "e3_absgate")
OUT_E3R = os.path.join(OUT_E3, "_ref")
LOGDIR = os.path.join(RES, "logs", "r4_absgate")
PY = sys.executable

# frozen mackey_drift-calibrated constant (results/e2_controls/
# absgate_threshold.json) -- transferred to the other domains AS IS.
T_CONST = "0.05806520209"

POLL_S = 120
COOLDOWN_S = 30
MAX_TRIES = 3
SLOTS_PER_GPU = 2
WARMUP_S = 900
HEARTBEAT_S = 1800

E4 = os.path.join(HERE, "e4_gpt2_tta.py")
E3 = os.path.join(HERE, "e3_continual.py")

# e4_v2 / launch_e4v2 / launch_r3_chain constants (identical in all three)
E4_COMMON = ["--tokens-per-domain", "512000", "--max-steps", "3000",
             "--batch", "2", "--seq-len", "256", "--probe-every", "100",
             "--kw-eps", "0.15", "--device", "cuda:0"]


def e4_cfg(phase, method, seed, out_dir, extra=(), need=18600, hours=4.0):
    name = "gpt2_%s_lr0.001_s%d.json" % (method, seed)
    cmd = ["--method", method, "--seed", str(seed), "--lr", "0.001"]
    cmd += E4_COMMON + ["--meta-lr", "0.4", "--log-gate-stats"] + list(extra)
    return {"phase": phase, "script": E4, "cmd": cmd, "src": name,
            "result": os.path.join(out_dir, name), "need_mib": need,
            "hours": hours, "key": "e4_" + os.path.splitext(name)[0]
            + ("_ref" if out_dir.endswith("_ref") else ""), "check": "e4"}


def e3_cfg(phase, method, ewc, seed, out_dir, extra=(), need=7000, hours=1.0):
    name = "cifar100_%s_lr0.05_ewc%g_s%d.json" % (method, ewc, seed)
    cmd = ["--method", method, "--seed", str(seed), "--lr", "0.05",
           "--ewc0", "%g" % ewc, "--meta-lr", "0.4", "--log-losses",
           "--log-gate-stats", "--device", "cuda:0"] + list(extra)
    return {"phase": phase, "script": E3, "cmd": cmd, "src": name,
            "result": os.path.join(out_dir, name), "need_mib": need,
            "hours": hours, "key": "e3_" + os.path.splitext(name)[0]
            + ("_ref" if out_dir.endswith("_ref") else ""), "check": "e3"}


def build_phases():
    ph = []
    ABS = ["--absgate-threshold", T_CONST, "--rank", "0"]

    a = [e4_cfg("a_e4", "cohg_r0", 0, OUT_E4R, hours=4.0)]
    a += [e4_cfg("a_e4", "absgate", s, OUT_E4, extra=ABS, hours=4.0)
          for s in (0, 1, 2)]
    ph.append({"name": "a_e4", "cfgs": a, "canary": a[1]["key"],
               "depends": None,
               "progress": ["stream: 2999 steps, drift at [1000, 2000]",
                            "t=0 "],
               "targets": [(OUT_E4R, 1), (OUT_E4, 3)]})

    ABS3 = ["--absgate-threshold", T_CONST]
    b = [e3_cfg("b_e3", "absgate", 10, 0, OUT_E3, extra=ABS3, hours=1.0)]
    b += [e3_cfg("b_e3", "cohg", e, 0, OUT_E3R, hours=2.8) for e in (10, 1000)]
    b += [e3_cfg("b_e3", "absgate", e, s, OUT_E3, extra=ABS3, hours=1.0)
          for e in (10, 1000) for s in range(10)
          if not (e == 10 and s == 0)]
    ph.append({"name": "b_e3", "cfgs": b, "canary": b[0]["key"],
               "depends": None, "progress": ["[task 0] accs="],
               "targets": [(OUT_E3R, 2), (OUT_E3, 20)]})
    return ph


PHASES = build_phases()
BY_KEY = {c["key"]: c for p in PHASES for c in p["cfgs"]}


# --------------------------------------------------------------- checks
def check_e4(c):
    d = json.load(open(c["result"]))
    need = ["method", "seed", "lr0", "steps", "drift_steps", "online_ppl",
            "mean_logloss", "gate_open_frac", "losses", "lam_hist",
            "held_bound", "gate_open_steps"]
    miss = [k for k in need if k not in d]
    if miss:
        return False, "missing keys %s" % miss
    if not 2900 <= d["steps"] <= 3000:
        return False, "steps=%s" % d["steps"]
    if list(d["drift_steps"]) != [1000, 2000]:
        return False, "drift_steps=%s" % d["drift_steps"]
    if d.get("domain_order") != "wiki,news,code":
        return False, "domain_order=%s" % d.get("domain_order")
    if not (0 < d["online_ppl"] < 1e6):
        return False, "online_ppl=%s" % d["online_ppl"]
    if d["method"] == "absgate" and d.get("absgate_threshold") is None:
        return False, "absgate run without absgate_threshold"
    if "gate_stats" not in d:
        return False, "--log-gate-stats not honoured"
    return True, ("steps=%d ppl=%.4f coord_open=%.3e peak=%.2fGB"
                  % (d["steps"], d["online_ppl"],
                     d["coord_open_frac"] or 0.0,
                     d.get("peak_mem_gb", float("nan"))))


def check_e3(c):
    d = json.load(open(c["result"]))
    need = ["method", "seed", "lr0", "ewc0", "avg_acc", "bwt", "acc_matrix",
            "lam_hist", "losses", "gate_stats"]
    miss = [k for k in need if k not in d]
    if miss:
        return False, "missing keys %s" % miss
    if len(d["acc_matrix"]) != 10:
        return False, "acc_matrix len %d" % len(d["acc_matrix"])
    if not d["losses"]:
        return False, "empty losses trace"
    if not (0.0 < d["avg_acc"] < 1.0):
        return False, "avg_acc=%s" % d["avg_acc"]
    if d["method"] == "absgate" and d.get("absgate_threshold") is None:
        return False, "absgate run without absgate_threshold"
    return True, ("avg_acc=%.4f bwt=%.4f coord_open=%.3e losses=%d"
                  % (d["avg_acc"], d["bwt"], d["coord_open_frac"] or 0.0,
                     len(d["losses"])))


CHECKS = {"e4": check_e4, "e3": check_e3}


# --------------------------------------------------------------- claims
def pid_alive(pid: int) -> bool:
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, int(pid))
    if not h:
        return False
    code = ctypes.c_ulong()
    ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return bool(ok) and code.value == 259


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


# --------------------------------------------------------------- VRAM
_RES_LOCK = threading.Lock()
_RESERVE = {}
STATE_LOCK = threading.RLock()
TRIES = {}
ABORTED = set()
EXHAUSTED = set()
CANARY_OK = set()
PROGRESS_OK = set()


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
    except Exception as e:                                    # noqa: BLE001
        print("[r4 gpu%d] nvidia-smi failed: %r" % (gpu, e), flush=True)
        return 0


def effective_free(gpu: int) -> int:
    return free_mib(gpu) - reserved_mib(gpu)


# --------------------------------------------------------------- runner
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
    inc = incomplete(ph)
    if not inc:
        return False
    with STATE_LOCK:
        live = [c for c in inc if TRIES.get(c["key"], 0) < MAX_TRIES]
    if live:
        return False
    return not any(claimed(c["result"]) for c in inc)


def canary_progress_ok(ph, can):
    pats = ph.get("progress")
    if not pats:
        return False
    if ph["name"] in PROGRESS_OK:
        return True
    try:
        with open(os.path.join(LOGDIR, can["key"] + ".log"),
                  "r", errors="ignore") as f:
            txt = f.read()
    except OSError:
        return False
    if all(p in txt for p in pats):
        with STATE_LOCK:
            if ph["name"] not in PROGRESS_OK:
                PROGRESS_OK.add(ph["name"])
                print("[r4] canary PROGRESS OK %s/%s -- releasing the phase"
                      % (ph["name"], can["key"]), flush=True)
        return True
    return False


def phase_tail_only(ph):
    inc = incomplete(ph)
    return bool(inc) and all(claimed(c["result"]) for c in inc)


def canary_gate(ph):
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
            except Exception as e:                            # noqa: BLE001
                ok, msg = False, "check raised %r" % (e,)
            if ok:
                CANARY_OK.add(ph["name"])
                print("[r4] CANARY OK  %s/%s: %s"
                      % (ph["name"], can["key"], msg), flush=True)
            else:
                ABORTED.add(ph["name"])
                print("[r4] *** CANARY FAILED %s/%s: %s -- phase ABORTED ***"
                      % (ph["name"], can["key"], msg), flush=True)
                return [], inc
    return pend, inc


def active_phase():
    for ph in PHASES:
        name = ph["name"]
        if name in ABORTED or name in EXHAUSTED:
            continue
        dep = ph.get("depends")
        if dep and dep in ABORTED:
            with STATE_LOCK:
                if name not in ABORTED:
                    ABORTED.add(name)
            continue
        inc = incomplete(ph)
        if not inc:
            continue
        if phase_blocked(ph):
            with STATE_LOCK:
                if name not in EXHAUSTED:
                    EXHAUSTED.add(name)
                    print("[r4] phase %s EXHAUSTED (%d configs)"
                          % (name, len(inc)), flush=True)
            continue
        if phase_tail_only(ph):
            continue
        return ph
    return None


def worker(gpu: int, slot: int):
    tag = "[r4 gpu%d.%d]" % (gpu, slot)
    n_ok = n_fail = 0
    while True:
        ph = active_phase()
        if ph is None:
            break
        pend, inc = canary_gate(ph)
        if not pend:
            if not inc:
                continue
            time.sleep(POLL_S)
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
        print("%s start %s (phase %s, need %d MiB, usable %d, ~%.2fh) %s"
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
        print("[r4 heartbeat] %s %s free=[%d,%d] MiB"
              % (time.ctime(), counts(), free_mib(0), free_mib(1)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    for d in (OUT_E4, OUT_E4R, OUT_E3, OUT_E3R, LOGDIR):
        os.makedirs(d, exist_ok=True)

    if args.status:
        print(counts())
        return
    if args.dry_run:
        for ph in PHASES:
            print("== phase", ph["name"], len(ph["cfgs"]), "configs, canary",
                  ph["canary"])
            for c in ph["cfgs"]:
                mark = "HAVE" if os.path.exists(c["result"]) else "todo"
                print("  [%s] %-42s %5d MiB  %s"
                      % (mark, c["key"], c["need_mib"],
                         " ".join(c["cmd"])))
        return

    print("[r4] %s | %d configs" % (counts(),
                                    sum(len(p["cfgs"]) for p in PHASES)),
          flush=True)
    stop = threading.Event()
    hb = threading.Thread(target=heartbeat, args=(stop,), daemon=True)
    hb.start()
    ts = []
    for g in args.gpus:
        for s in range(SLOTS_PER_GPU):
            t = threading.Thread(target=worker, args=(g, s), daemon=False)
            t.start()
            ts.append(t)
            time.sleep(3)
    for t in ts:
        t.join()
    stop.set()
    print("[r4] ALL DONE %s | %s" % (time.ctime(), counts()), flush=True)


if __name__ == "__main__":
    main()
