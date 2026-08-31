"""Round-4 threshold-transfer, E2 Lorenz arm: detached CPU queue.

Transfers the mackey_drift-calibrated absgate constant T = 0.05806520209
(results/e2_controls/absgate_threshold.json) to the OTHER E2 drift stream,
lorenz_drift, with NO recalibration.

Config (the lorenz_drift analogue of the e2_controls mackey_drift study --
identical in every flag except the dataset): mis-set init lr0 = 0.003 (10x too
low, the same mis-set-low operating point launch_e2.py uses for lorenz_drift),
12000 steps, alpha (--meta-lr) 0.4, gamma 0.9, kw-eps 0.1, probe-every 20,
K 10, rank 4, gate factor c = 2, M_H 5 (default), seeds 0-9, device CPU.

Arms (all into results/e2_lorenz_absgate):
  absgate      T transferred as is, no certificate, no spectral probe
  cohg         certificate gate (the reference)         + --log-gate-stats
  cohg_nogate  ungated pure-sign step, same alpha       + --log-gate-stats

Plus ONE reproduction check into results/e2_lorenz_absgate/_verify:
the e2_controls mackey_drift absgate seed 0 config re-run on the PATCHED
e2_timeseries.py, to be diffed against the stored
results/e2_controls/mackey_drift_absgate_lr0.003_a0.4_s0.json.

Usage (detached):
    python -u launch_r4_e2cpu.py            # 6 workers (R4Q_WORKERS to change)
    python launch_r4_e2cpu.py --dry-run
    python launch_r4_e2cpu.py --status
"""

from __future__ import annotations

import argparse
import ctypes
import os
import queue
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "e2_lorenz_absgate")
OUT_V = os.path.join(OUT, "_verify")
OUT_C = os.path.join(OUT, "_calib_ref")
LOGDIR = os.path.join(RES, "logs", "r4_e2cpu")
PY = sys.executable
E2 = os.path.join(HERE, "e2_timeseries.py")

T_CONST = "0.05806520209"
DATASET = "lorenz_drift"
LR0 = "0.003"
STEPS = "12000"
ALPHA = "0.4"

# keep well under the box: another agent is draining a ~12-way CPU queue
WORKERS = int(os.environ.get("R4Q_WORKERS", "6"))
MAX_TRIES = 2

ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
           OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
           CUDA_VISIBLE_DEVICES="")

COMMON = ["--steps", STEPS, "--meta-lr", ALPHA, "--gamma", "0.9",
          "--kw-eps", "0.1", "--probe-every", "20", "--device", "cpu"]


def cfg(method, seed, dataset=DATASET, out_dir=OUT, tag="", extra=(),
        gate_stats=True, hours=0.2):
    name = "%s_%s_lr%s%s_s%d.json" % (dataset, method, "0.003", tag, seed)
    cmd = ["--method", method, "--dataset", dataset, "--seed", str(seed),
           "--lr", LR0] + COMMON + list(extra)
    if tag:
        cmd += ["--tag", tag]
    if gate_stats:
        cmd += ["--log-gate-stats"]
    return {"cmd": cmd, "src": name, "result": os.path.join(out_dir, name),
            "key": os.path.splitext(name)[0]
                   + ("_verify" if out_dir is OUT_V else "")
                   + ("_calibref" if out_dir is OUT_C else ""),
            "hours": hours}


def build(part="main"):
    ABS = ["--absgate-threshold", T_CONST]
    jobs = []
    if part == "calib":
        # COHG on the CALIBRATION stream itself, with gate stats: the anchor
        # of the cross-domain scale-mismatch table (where |ghat| and c*beta
        # sit on the stream T was fitted to).
        return [cfg("cohg", s, dataset="mackey_drift", out_dir=OUT_C,
                    hours=1.5) for s in range(3)]
    # reproduction check FIRST (cheap, and it validates the patched script);
    # no --log-gate-stats so the JSON is directly comparable key-for-key.
    jobs.append(cfg("absgate", 0, dataset="mackey_drift", out_dir=OUT_V,
                    tag="_a0.4", extra=ABS, gate_stats=False, hours=0.2))
    # the expensive arm first so it is not left as a long tail
    jobs += [cfg("cohg", s, hours=1.5) for s in range(10)]
    jobs += [cfg("absgate", s, extra=ABS, hours=0.2) for s in range(10)]
    jobs += [cfg("cohg_nogate", s, hours=0.2) for s in range(10)]
    return jobs


JOBS = build(os.environ.get("R4Q_PART", "main"))


def pid_alive(pid: int) -> bool:
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, int(pid))
    if not h:
        return False
    code = ctypes.c_ulong()
    ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return bool(ok) and code.value == 259


def claimed(res):
    claim = res + ".claim"
    try:
        pid = int(open(claim).read().strip() or 0)
    except (OSError, ValueError):
        return os.path.exists(claim)
    return bool(pid) and pid_alive(pid)


def try_claim(res):
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


def run_one(c):
    tmp = c["result"] + ".tmpdir"
    os.makedirs(tmp, exist_ok=True)
    cmd = [PY, "-u", E2] + c["cmd"] + ["--out-dir", tmp]
    log = os.path.join(LOGDIR, c["key"] + ".log")
    t0 = time.time()
    with open(log, "a") as lf:
        lf.write("\n=== %s %s\n" % (time.ctime(), " ".join(cmd)))
        lf.flush()
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                              env=ENV, cwd=HERE, stdin=subprocess.DEVNULL)
    src = os.path.join(tmp, c["src"])
    ok = proc.returncode == 0 and os.path.exists(src)
    if ok:
        os.replace(src, c["result"])
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    return ok, time.time() - t0, log


def counts():
    have = sum(1 for c in JOBS if os.path.exists(c["result"]))
    return "%d/%d" % (have, len(JOBS))


def worker(wid, q):
    n_ok = n_fail = 0
    while True:
        try:
            c = q.get_nowait()
        except queue.Empty:
            break
        if os.path.exists(c["result"]) or claimed(c["result"]):
            q.task_done()
            continue
        if not try_claim(c["result"]):
            q.task_done()
            continue
        print("[r4cpu w%d] start %s (~%.1fh) %s"
              % (wid, c["key"], c["hours"], time.ctime()), flush=True)
        try:
            ok, dt, log = run_one(c)
        finally:
            try:
                os.remove(c["result"] + ".claim")
            except OSError:
                pass
        if ok:
            n_ok += 1
            print("[r4cpu w%d] done %s (%.2fh) | %s"
                  % (wid, c["key"], dt / 3600, counts()), flush=True)
        else:
            n_fail += 1
            print("[r4cpu w%d] FAIL %s (see %s)" % (wid, c["key"], log),
                  flush=True)
        q.task_done()
    print("[r4cpu w%d] exit ok=%d fail=%d" % (wid, n_ok, n_fail), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    for d in (OUT, OUT_V, OUT_C, LOGDIR):
        os.makedirs(d, exist_ok=True)
    if args.status:
        print(counts())
        return
    if args.dry_run:
        for c in JOBS:
            print("[%s] %-46s %s"
                  % ("HAVE" if os.path.exists(c["result"]) else "todo",
                     c["key"], " ".join(c["cmd"])))
        return
    todo = [c for c in JOBS if not os.path.exists(c["result"])]
    print("[r4cpu] %s | %d todo | %d workers"
          % (counts(), len(todo), WORKERS), flush=True)
    q = queue.Queue()
    for c in todo:
        q.put(c)
    ts = []
    for w in range(WORKERS):
        t = threading.Thread(target=worker, args=(w, q), daemon=False)
        t.start()
        ts.append(t)
        time.sleep(2)
    for t in ts:
        t.join()
    print("[r4cpu] ALL DONE %s | %s" % (time.ctime(), counts()), flush=True)


if __name__ == "__main__":
    main()
