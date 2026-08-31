"""E3 wave-2 launcher: the *no-retained-holdout* continual-learning control.

Review item answered
--------------------
E3's meta-objective is  ell_t = CE(incoming batch) + CE(holdout buffer), where
the buffer keeps 128 held-out examples per seen task.  A reviewer can object
that retaining past-task data is exactly what a continual learner is not
supposed to need, so the hypergradient signal is "cheating".  This wave reruns
the E3 arms with `--no-holdout`, i.e. a purely prequential meta-objective

    ell_t = CE(theta_t; incoming batch)          (eval-before-train, no replay)

Everything else is unchanged.  EWC keeps its anchor and Fisher: those belong to
the LEARNER whose hyperparameters are being tuned, not to the meta-objective
(see the boundary comment in e3_continual.py).  Runs also carry `--log-losses`
so the prequential loss trace is available for the analysis.

Grid (40 runs)
--------------
  methods {cohg, cohg_nogate, hd, fixed} x seeds 0..9
  lr0 = 0.05, ewc0 = 10  (the E3 "well-set" operating point)
  meta_lr from the frozen tables: cohg 0.4, cohg_nogate 0.4, hd 0.02, fixed n/a
  -> results/e3_noholdout/cifar100_<method>_lr0.05_ewc10_s<seed>.json

Chaining
--------
This machine is busy with the E4 GPT-2 expansion queue (launch_e4_expand.py).
`--watch` polls that queue every WATCH_POLL_S seconds and only then starts the
wave-2 workers, so the two campaigns never contend for VRAM:

    python launch_e3_wave2.py --watch --gpus 0 1

The watcher writes its poll lines to results/logs/e3_wave2_watcher.log and
spawns the queue itself with stdout -> results/logs/e3_wave2.log.

Runner mechanics follow launch_e4_expand.py: one worker thread per physical
GPU, free-VRAM gating via nvidia-smi, skip-existing, atomic .claim files reaped
from dead PIDs, private tmp out-dir then atomic rename, per-run child logs,
MAX_TRIES attempts per config.  Designed to be started detached and outlive its
launcher.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "results", "e3_noholdout")
LOGROOT = os.path.join(ROOT, "results", "logs")
LOGDIR = os.path.join(LOGROOT, "e3_wave2")
QUEUE_LOG = os.path.join(LOGROOT, "e3_wave2.log")
PY = sys.executable

POLL_S = 300           # re-measure free VRAM this often when nothing fits
COOLDOWN_S = 30        # let VRAM settle after a child exits
MAX_TRIES = 3          # attempts per config before giving up

WATCH_POLL_S = 600     # E4-done poll interval (10 min)
WATCH_SETTLE_S = 120   # let the last E4 child's VRAM drain before arming

# ---- the E4 queue we are chained behind -----------------------------------
E4_LAUNCHER = os.path.join(HERE, "launch_e4_expand.py")
E4_TARGETS = [(os.path.join(ROOT, "results", "e4_v2"), 47),
              (os.path.join(ROOT, "results", "e4_orders"), 12)]

LR0, EWC0 = 0.05, 10.0
COMMON = ["--lr", "0.05", "--ewc0", "10",
          "--log-losses", "--no-holdout", "--device", "cuda:0"]

# method -> (meta_lr, required free MiB, rough hours)
# hours are the measured wall-times of the corresponding results/e3 runs; the
# no-holdout arms are if anything cheaper (one fewer forward/backward on the
# growing holdout buffer per step).  VRAM needs are set generously: E3 is a
# ResNet18-GN on 32x32 inputs, far below the E4 GPT-2 arms, and the queue only
# ever runs on cards the E4 campaign has already released.
METHOD = {
    "cohg":        (0.4,  7000, 1.75),
    "cohg_nogate": (0.4,  7000, 0.45),
    "hd":          (0.02, 4500, 0.25),
    "fixed":       (None, 4500, 0.25),
}


def cfg(method, seed):
    """One queue entry."""
    ml, need, hours = METHOD[method]
    name = f"cifar100_{method}_lr{LR0:g}_ewc{EWC0:g}_s{seed}.json"
    return {"method": method, "seed": seed, "meta_lr": ml,
            "result": os.path.join(OUT, name),
            "need_mib": need, "hours": hours,
            "key": os.path.splitext(name)[0]}


def make_configs():
    """Seed 0 of every method first (early sanity across all four arms), then
    longest-arm-first so the 1.75 h cohg runs are never the straggler."""
    cfgs = [cfg(m, 0) for m in ("cohg", "cohg_nogate", "hd", "fixed")]
    for m in ("cohg", "cohg_nogate", "hd", "fixed"):
        cfgs += [cfg(m, s) for s in range(1, 10)]
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
        print(f"[e3w2 gpu{gpu}] nvidia-smi failed: {e}", flush=True)
        return 0


def run_one(c, gpu):
    tmp = c["result"] + ".tmpdir"
    os.makedirs(tmp, exist_ok=True)
    cmd = [PY, os.path.join(HERE, "e3_continual.py"),
           "--method", c["method"], "--seed", str(c["seed"])] + COMMON + \
          ["--out-dir", tmp]
    if c["meta_lr"] is not None:
        cmd += ["--meta-lr", str(c["meta_lr"])]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    log = os.path.join(LOGDIR, c["key"] + ".log")
    t0 = time.time()
    with open(log, "a") as lf:
        lf.write(f"\n=== {time.ctime()} gpu{gpu} {' '.join(cmd)}\n")
        lf.flush()
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                              env=env, cwd=HERE)
    # e3_continual.py names its own file exactly as c["key"]
    src = os.path.join(tmp, c["key"] + ".json")
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
            print(f"[e3w2 gpu{gpu}] {avail} MiB free, cheapest pending needs "
                  f"{min(c['need_mib'] for c in pend)} MiB; wait {POLL_S}s",
                  flush=True)
            time.sleep(POLL_S)
            continue
        if not try_claim(pick["result"]):
            time.sleep(5)                   # lost the race; re-scan
            continue
        tries[pick["key"]] = tries.get(pick["key"], 0) + 1
        print(f"[e3w2 gpu{gpu}] start {pick['key']} "
              f"(need {pick['need_mib']} MiB, free {avail} MiB, "
              f"~{pick['hours']:.2f}h) {time.ctime()}", flush=True)
        try:
            ok, dt, log = run_one(pick, gpu)
        finally:
            try:
                os.remove(pick["result"] + ".claim")
            except OSError:
                pass
        if ok:
            n_ok += 1
            print(f"[e3w2 gpu{gpu}] done {pick['key']} ({dt/3600:.2f}h)",
                  flush=True)
        else:
            n_fail += 1
            print(f"[e3w2 gpu{gpu}] FAIL {pick['key']} "
                  f"(try {tries[pick['key']]}/{MAX_TRIES}, see {log})",
                  flush=True)
            time.sleep(60)
        time.sleep(COOLDOWN_S)
    print(f"[e3w2 gpu{gpu}] worker exit ok={n_ok} fail={n_fail}", flush=True)


# --------------------------------------------------------------------------
# chaining: wait for the E4 expansion queue
# --------------------------------------------------------------------------
def e4_pending():
    """Configs still pending in the E4 expansion queue, or None if unknown.

    Primary signal: `launch_e4_expand.py --dry-run` prints
        "E4-expand: <pending>/<total> configs pending, ..."
    Fallback (if that call fails for any reason): the artifact counts.
    """
    try:
        o = subprocess.run([PY, E4_LAUNCHER, "--dry-run"],
                           capture_output=True, text=True, timeout=300,
                           cwd=HERE)
        mo = re.search(r"E4-expand:\s*(\d+)\s*/\s*(\d+)\s*configs pending",
                       o.stdout)
        if mo:
            return int(mo.group(1))
        print(f"[e3w2 watch] could not parse dry-run output:\n"
              f"{o.stdout[-400:]}\n{o.stderr[-400:]}", flush=True)
    except Exception as e:                                # noqa: BLE001
        print(f"[e3w2 watch] dry-run failed: {e}", flush=True)
    missing = 0
    for d, target in E4_TARGETS:
        have = len([f for f in os.listdir(d)
                    if f.endswith(".json")]) if os.path.isdir(d) else 0
        missing += max(target - have, 0)
    print(f"[e3w2 watch] fallback artifact count -> {missing} missing",
          flush=True)
    return missing


def e4_procs_alive():
    """(launcher_alive, n_children) for the E4 expansion campaign.

    Safety net: if launch_e4_expand.py gives up (MAX_TRIES exhausted on some
    config) it exits with pending > 0, and a watcher keyed only on the pending
    count would poll forever.  When the launcher is gone the GPUs are free
    whatever the count says, so we arm anyway.
    """
    ps = ("Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
          "ForEach-Object { $_.CommandLine }")
    try:
        o = subprocess.run(["powershell", "-NoProfile", "-NonInteractive",
                            "-Command", ps],
                           capture_output=True, text=True, timeout=120)
        lines = [l for l in o.stdout.splitlines() if l.strip()]
    except Exception as e:                                # noqa: BLE001
        print(f"[e3w2 watch] process probe failed: {e}", flush=True)
        return True, -1          # unknown -> assume alive, keep waiting
    return (any("launch_e4_expand.py" in l for l in lines),
            sum("e4_gpt2_tta.py" in l for l in lines))


def artifact_counts():
    return [(os.path.basename(d), target,
             len([f for f in os.listdir(d) if f.endswith(".json")])
             if os.path.isdir(d) else 0)
            for d, target in E4_TARGETS]


def watch(gpus):
    print(f"[e3w2 watch] armed, pid {os.getpid()}, poll every "
          f"{WATCH_POLL_S}s, gpus {gpus} @ {time.ctime()}", flush=True)
    gone = 0                      # consecutive polls with no E4 launcher
    while True:
        pend = e4_pending()
        counts = ", ".join(f"{n}={have}/{target}"
                           for n, target, have in artifact_counts())
        alive, kids = e4_procs_alive()
        gone = 0 if alive else gone + 1
        print(f"[e3w2 watch] {time.ctime()} E4 pending={pend} ({counts}) "
              f"launcher_alive={alive} children={kids}", flush=True)
        if pend == 0:
            print("[e3w2 watch] E4 queue reports 0 pending.", flush=True)
            break
        if gone >= 2:
            print(f"[e3w2 watch] E4 launcher gone for {gone} polls with "
                  f"{pend} still pending (queue gave up or was stopped); "
                  f"arming wave-2 anyway.", flush=True)
            break
        time.sleep(WATCH_POLL_S)
    print(f"[e3w2 watch] E4 queue drained; settling {WATCH_SETTLE_S}s before "
          f"starting wave-2 @ {time.ctime()}", flush=True)
    time.sleep(WATCH_SETTLE_S)
    cmd = [PY, "-u", os.path.abspath(__file__), "--gpus"] + [str(g) for g in gpus]
    print(f"[e3w2 watch] launching: {' '.join(cmd)} -> {QUEUE_LOG}", flush=True)
    with open(QUEUE_LOG, "a") as lf:
        lf.write(f"\n=== e3 wave-2 queue start {time.ctime()} "
                 f"({' '.join(cmd)})\n")
        lf.flush()
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                            cwd=HERE).returncode
    print(f"[e3w2 watch] wave-2 queue exited rc={rc} @ {time.ctime()}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--watch", action="store_true",
                    help="poll the E4 expansion queue and start wave-2 once "
                         "it has drained")
    args = ap.parse_args()
    for d in (OUT, LOGROOT, LOGDIR):
        os.makedirs(d, exist_ok=True)
    if args.watch:
        watch(args.gpus)
        return
    cfgs = make_configs()
    pend = [c for c in cfgs if not os.path.exists(c["result"])]
    print(f"E3-wave2(no-holdout): {len(pend)}/{len(cfgs)} configs pending, "
          f"{sum(c['hours'] for c in pend):.1f} GPU-hours, "
          f"GPUs {args.gpus}, pid {os.getpid()} @ {time.ctime()}", flush=True)
    if args.dry_run:
        for c in pend:
            print(f"   {c['key']:<44} need={c['need_mib']:>5} MiB  "
                  f"~{c['hours']:.2f}h  -> {c['result']}")
        return
    ths = [threading.Thread(target=worker, args=(g,)) for g in args.gpus]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    print(f"E3-wave2 launcher exit @ {time.ctime()}", flush=True)


if __name__ == "__main__":
    main()
