"""Detached CPU queue for the --probe-dense-until study (launch_r4_denseprobe.make_jobs).

Same contract as r4_cpu_queue.py: idempotent (skips jobs whose output JSON
already exists), single-thread BLAS, CUDA hidden, one retry per job.

Env: RDQ_WORKERS (default 32), RDQ_PART (default "all"), RDQ_MAX_HOURS (12).
"""
from __future__ import annotations

import os
import sys
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import launch_r4_denseprobe as L  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, ".."))
LOGDIR = os.path.join(ROOT, "results", "logs")
os.makedirs(LOGDIR, exist_ok=True)
STATUS = os.path.join(LOGDIR, "r4_denseprobe_status.txt")

WORKERS = int(os.environ.get("RDQ_WORKERS", "32"))
PART = os.environ.get("RDQ_PART", "all")
MAX_HOURS = float(os.environ.get("RDQ_MAX_HOURS", "12"))
POLL = 30.0

ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
           OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
           CUDA_VISIBLE_DEVICES="")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    jobs = L.make_jobs(PART)
    for _n, p, _c in jobs:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    log(f"expected jobs: {len(jobs)} (part={PART}, workers={WORKERS}); "
        f"already landed {sum(1 for _n, p, _c in jobs if os.path.exists(p))}")
    running, attempts = {}, {}
    launched = done = failed = 0
    t_start = time.time()
    while True:
        if time.time() - t_start > MAX_HOURS * 3600:
            log("max wall time reached")
            break
        for name, (proc, path, t0) in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                ok = (rc == 0 and os.path.exists(path))
                done += int(ok)
                failed += int(not ok)
                log(f"{'done' if ok else f'FAIL rc={rc}'} {name} "
                    f"({time.time() - t0:.0f}s)")
                running.pop(name)
        pending = [j for j in jobs if not os.path.exists(j[1])
                   and j[0] not in running and attempts.get(j[0], 0) < 2]
        n_left = sum(1 for j in jobs if not os.path.exists(j[1]))
        with open(STATUS, "w") as f:
            f.write(f"ts={time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"total={len(jobs)} remaining={n_left} "
                    f"running={len(running)} pending={len(pending)} "
                    f"launched={launched} done={done} failed={failed}\n")
        if not pending and not running:
            log("all artifacts present -- exiting")
            break
        while pending and len(running) < WORKERS:
            name, path, cmd = pending.pop(0)
            if os.path.exists(path):
                continue
            proc = subprocess.Popen(cmd, env=ENV, cwd=HERE,
                                    stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.STDOUT)
            running[name] = (proc, path, time.time())
            attempts[name] = attempts.get(name, 0) + 1
            launched += 1
            log(f"launch {name} pid={proc.pid} ({launched} launched, "
                f"{n_left} left)")
        time.sleep(POLL)
    for name, (proc, path, _t) in list(running.items()):
        proc.wait()
        log(f"exited {name} rc={proc.returncode}")
    left = [n for n, p, _c in jobs if not os.path.exists(p)]
    with open(STATUS, "w") as f:
        f.write(f"ts={time.strftime('%Y-%m-%d %H:%M:%S')} FINISHED "
                f"total={len(jobs)} remaining={len(left)} "
                f"launched={launched}\n")
    log(f"QUEUE FINISHED | launched {launched} | missing {len(left)}")
    for n in left:
        log(f"  MISSING {n}")


if __name__ == "__main__":
    main()
