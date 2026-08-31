"""Round-3 CPU completion queue (single detached runner).

Builds the FULL expected round-3 CPU job list by importing
``launch_round3.make_jobs("all")`` (b1 madgate, b2 ogd_doubling, b2ref
cohg_ogd, b3 gamma sweep, b4 E1 misspecification, verify), then repeatedly:

  * drops every job whose output JSON already exists (skip-existing),
  * drops every job that is currently being executed by SOME OTHER python
    process on this box (in-flight exclusion, via Win32_Process command
    lines) -- so the still-alive launch_round3.py pools are never duplicated,
  * dispatches whatever is left on N CPU workers with CUDA_VISIBLE_DEVICES="".

It keeps polling so that any job an external pool never gets to (because its
launcher was killed) is eventually picked up here.  Idempotent: safe to run
next to the other launchers.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import launch_round3 as L  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, ".."))
LOGDIR = os.path.join(ROOT, "results", "logs")
os.makedirs(LOGDIR, exist_ok=True)

WORKERS = int(os.environ.get("R3Q_WORKERS", "8"))
POLL = 45.0
MAX_HOURS = float(os.environ.get("R3Q_MAX_HOURS", "24"))

ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
           OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
           CUDA_VISIBLE_DEVICES="")

TOK = re.compile(r'(--[A-Za-z0-9\-]+)\s+("(?:[^"]*)"|\S+)')


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def key_from_pairs(pairs, is_e1):
    d = {k: v.strip('"') for k, v in pairs}
    if is_e1:
        return ("e1", d.get("--tag", ""), d.get("--seed-offset", "0"))
    return ("e2", d.get("--method", ""), d.get("--seed", ""),
            d.get("--tag", ""), d.get("--gamma", ""),
            os.path.basename(d.get("--out-dir", "").rstrip("\\/")))


def key_of_cmd(cmd):
    s = " ".join(cmd)
    is_e1 = "e1_certificate.py" in s
    pairs = []
    for i, a in enumerate(cmd):
        if a.startswith("--") and i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
            pairs.append((a, cmd[i + 1]))
    return key_from_pairs(pairs, is_e1)


def key_of_cmdline(s):
    if "e1_certificate.py" in s:
        is_e1 = True
    elif "e2_timeseries.py" in s:
        is_e1 = False
    else:
        return None
    return key_from_pairs(TOK.findall(s), is_e1)


PS = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
      "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")


_POOL_CACHE = {}


def pool_jobs(cmdline):
    """Job keys owned by a still-alive launch_round3.py pool process."""
    if cmdline in _POOL_CACHE:
        return _POOL_CACHE[cmdline]
    toks = [t.strip('"') for t in re.findall(r'"[^"]*"|\S+', cmdline)]
    part, gammas, seeds, cur = "all", None, None, None
    for t in toks:
        if t == "--part":
            cur = "part"
        elif t == "--gammas":
            cur, gammas = "g", []
        elif t == "--seed-list":
            cur, seeds = "s", []
        elif t.startswith("--"):
            cur = None
        elif cur == "part":
            part, cur = t, None
        elif cur == "g":
            gammas.append(float(t))
        elif cur == "s":
            seeds.append(int(t))
    try:
        buf, sys.stdout = sys.stdout, open(os.devnull, "w")
        try:
            ks = {key_of_cmd(c) for _n, _p, c in
                  L.make_jobs(part, gammas, seeds)}
        finally:
            sys.stdout.close()
            sys.stdout = buf
    except Exception as e:
        log(f"WARN could not parse pool '{cmdline[:120]}': {e}")
        ks = set()
    _POOL_CACHE[cmdline] = ks
    return ks


def external_inflight(my_pids):
    """Keys claimed by other processes: either a python.exe actually running
    that config right now, or a still-alive launch_round3.py worker pool that
    has that config in its own queue."""
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
        if "\t" not in line:
            continue
        pid_s, cmdline = line.split("\t", 1)
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if pid in my_pids:
            continue
        if "launch_round3.py" in cmdline and "r3_cpu_queue" not in cmdline:
            for k in pool_jobs(cmdline):
                seen.setdefault(k, pid)
            continue
        k = key_of_cmdline(cmdline)
        if k is not None:
            seen[k] = pid
    return seen


def main():
    jobs = L.make_jobs("all")
    for _, path, _c in jobs:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    keyed = [(name, path, cmd, key_of_cmd(cmd)) for name, path, cmd in jobs]
    log(f"expected round-3 CPU jobs: {len(keyed)}")
    done0 = sum(1 for _n, p, _c, _k in keyed if os.path.exists(p))
    log(f"already landed: {done0}")

    running = {}  # key -> (proc, name, t0)
    attempts = {}  # key -> number of dispatches by this runner
    MAX_TRIES = 2
    t_start = time.time()
    launched = 0
    while True:
        if time.time() - t_start > MAX_HOURS * 3600:
            log("max wall time reached, stopping dispatch")
            break
        # reap
        for k, (proc, name, t0) in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                log(f"{'done' if rc == 0 else 'FAIL rc=%d' % rc} {name} "
                    f"({time.time() - t0:.0f}s)")
                running.pop(k)
        pending = [j for j in keyed if not os.path.exists(j[1])
                   and j[3] not in running
                   and attempts.get(j[3], 0) < MAX_TRIES]
        if not pending and not running:
            log("all expected artifacts present -- exiting")
            break
        if pending and len(running) < WORKERS:
            my_pids = {p.pid for p, _n, _t in running.values()} | {os.getpid()}
            ext = external_inflight(my_pids)
            free = [j for j in pending if j[3] not in ext]
            blocked = len(pending) - len(free)
            status = (len(pending), blocked, len(free), len(running))
            if status != main._last_status:
                main._last_status = status
                log(f"pending {len(pending)} | in-flight elsewhere {blocked} "
                    f"| dispatchable {len(free)} | mine {len(running)}")
            for name, path, cmd, k in free:
                if len(running) >= WORKERS:
                    break
                if os.path.exists(path):
                    continue
                proc = subprocess.Popen(cmd, env=ENV, cwd=HERE,
                                        stdin=subprocess.DEVNULL,
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.STDOUT)
                running[k] = (proc, name, time.time())
                attempts[k] = attempts.get(k, 0) + 1
                launched += 1
                log(f"launch {name} pid={proc.pid} ({launched} launched)")
        time.sleep(POLL)
    for _k, (proc, name, _t0) in list(running.items()):
        proc.wait()
        log(f"exited {name} rc={proc.returncode}")
    left = [n for n, p, _c, _k in keyed if not os.path.exists(p)]
    log(f"QUEUE FINISHED | launched {launched} | still missing {len(left)}")
    for n in left:
        log(f"  MISSING {n}")


main._last_status = None

if __name__ == "__main__":
    main()
