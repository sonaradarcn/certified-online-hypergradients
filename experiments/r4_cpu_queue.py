"""Round-4 CPU completion queue (single detached runner).

Same pattern as r3_cpu_queue.py: build the full expected job list from
``launch_round4.make_jobs``, drop everything whose output JSON already exists,
and dispatch the rest on N CPU workers with CUDA_VISIBLE_DEVICES="" (the GPUs
are reserved for another agent).  Idempotent and restart-safe: it polls, so a
job whose process died is retried once.

Env:
    R4Q_WORKERS     concurrent processes (default 16 of the box's 80 cores)
    R4Q_PART        launch_round4 part (default "all")
    R4Q_F           space-separated scale-shift factors (default from module)
    R4Q_MAX_HOURS   dispatch deadline (default 24)
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import launch_round4 as L  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, ".."))
LOGDIR = os.path.join(ROOT, "results", "logs")
os.makedirs(LOGDIR, exist_ok=True)

WORKERS = int(os.environ.get("R4Q_WORKERS", "16"))
PART = os.environ.get("R4Q_PART", "all")
FLIST = ([float(x) for x in os.environ["R4Q_F"].split()]
         if os.environ.get("R4Q_F") else None)
POLL = 30.0
MAX_HOURS = float(os.environ.get("R4Q_MAX_HOURS", "24"))
STATUS = os.path.join(LOGDIR, "r4_queue_status.txt")

ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
           OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
           CUDA_VISIBLE_DEVICES="")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


PS = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
      "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")
TOK = re.compile(r'(--[A-Za-z0-9\-]+)\s+("(?:[^"]*)"|\S+)')


def _key(pairs, is_e1):
    d = {k: v.strip('"') for k, v in pairs}
    if is_e1:
        return ("e1", d.get("--tag", ""), d.get("--seed-offset", "0"),
                os.path.basename(d.get("--out-dir", "").rstrip("\/")))
    return ("e2", d.get("--dataset", ""), d.get("--method", ""),
            d.get("--seed", ""), d.get("--tag", ""), d.get("--lr", ""),
            d.get("--scale-shift", ""),
            os.path.basename(d.get("--out-dir", "").rstrip("\/")))


def key_of_cmd(cmd):
    is_e1 = "e1_certificate.py" in " ".join(cmd)
    pairs = [(a, cmd[i + 1]) for i, a in enumerate(cmd)
             if a.startswith("--") and i + 1 < len(cmd)
             and not cmd[i + 1].startswith("--")]
    return _key(pairs, is_e1)


def key_of_cmdline(s):
    if "e1_certificate.py" in s:
        is_e1 = True
    elif "e2_timeseries.py" in s:
        is_e1 = False
    else:
        return None
    return _key(TOK.findall(s), is_e1)


def external_inflight(my_pids):
    """Job keys currently being executed by SOME OTHER python.exe -- e.g. the
    orphaned children of a previous runner that was restarted to change the
    worker count.  Never dispatch those again."""
    try:
        p = subprocess.run(["powershell", "-NoProfile", "-NonInteractive",
                            "-Command", PS], capture_output=True, text=True,
                           timeout=120)
        out = p.stdout
    except Exception as e:  # pragma: no cover
        log(f"WARN process scan failed: {e}")
        return {}
    seen = {}
    for line in out.splitlines():
        if "	" not in line:
            continue
        pid_s, cmdline = line.split("	", 1)
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if pid in my_pids:
            continue
        k = key_of_cmdline(cmdline)
        if k is not None:
            seen[k] = pid
    return seen


def main():
    jobs = L.make_jobs(PART, FLIST)
    for _n, path, _c in jobs:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    log(f"expected round-4 jobs: {len(jobs)} (part={PART}, workers={WORKERS})")
    log(f"already landed: {sum(1 for _n, p, _c in jobs if os.path.exists(p))}")

    running = {}          # name -> (proc, path, t0)
    attempts = {}
    MAX_TRIES = 2
    t_start = time.time()
    launched = done = failed = 0
    while True:
        if time.time() - t_start > MAX_HOURS * 3600:
            log("max wall time reached, stopping dispatch")
            break
        for name, (proc, path, t0) in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                ok = (rc == 0 and os.path.exists(path))
                done += int(ok)
                failed += int(not ok)
                log(f"{'done' if ok else 'FAIL rc=%s' % rc} {name} "
                    f"({time.time() - t0:.0f}s)")
                running.pop(name)
        pending = [j for j in jobs if not os.path.exists(j[1])
                   and j[0] not in running
                   and attempts.get(j[0], 0) < MAX_TRIES]
        n_left = sum(1 for j in jobs if not os.path.exists(j[1]))
        with open(STATUS, "w") as f:
            f.write(f"ts={time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"total={len(jobs)} remaining={n_left} "
                    f"running={len(running)} pending={len(pending)} "
                    f"launched={launched} done={done} failed={failed}\n")
        if not pending and not running:
            log("all expected artifacts present -- exiting")
            break
        ext = {}
        if pending and len(running) < WORKERS:
            my_pids = {p.pid for p, _pa, _t in running.values()} | {os.getpid()}
            ext = external_inflight(my_pids)
            claimed = sum(1 for j in jobs if key_of_cmd(j[2]) in ext)
            if claimed and claimed != main._last_claimed:
                main._last_claimed = claimed
                log(f"{claimed} job(s) in flight in another process "
                    f"-- excluded and counted against the worker budget")
        budget = WORKERS - sum(1 for j in jobs if key_of_cmd(j[2]) in ext)
        while pending and len(running) < budget:
            name, path, cmd = pending.pop(0)
            if os.path.exists(path) or key_of_cmd(cmd) in ext:
                continue
            proc = subprocess.Popen(cmd, env=ENV, cwd=HERE,
                                    stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.STDOUT)
            running[name] = (proc, path, time.time())
            attempts[name] = attempts.get(name, 0) + 1
            launched += 1
            log(f"launch {name} pid={proc.pid} "
                f"({launched} launched, {n_left} left)")
        time.sleep(POLL)
    for name, (proc, path, _t0) in list(running.items()):
        proc.wait()
        log(f"exited {name} rc={proc.returncode}")
    left = [n for n, p, _c in jobs if not os.path.exists(p)]
    with open(STATUS, "w") as f:
        f.write(f"ts={time.strftime('%Y-%m-%d %H:%M:%S')} FINISHED "
                f"total={len(jobs)} remaining={len(left)} launched={launched}\n")
    log(f"QUEUE FINISHED | launched {launched} | still missing {len(left)}")
    for n in left:
        log(f"  MISSING {n}")


main._last_claimed = None

if __name__ == "__main__":
    main()
