"""Adopt a queue child whose launcher is being restarted underneath it.

launch_r3_chain.py's worker owns three things for each run: the child process,
the atomic rename of <tmpdir>/<src> -> <result>, and the .claim file.  Killing
the launcher to pick up a code change therefore orphans any child that is
mid-run: the child keeps going (it inherited the log handle and has no parent
dependency) but nobody will ever promote its JSON, and the stale claim -- which
holds the DEAD launcher's pid -- gets reaped by the next launcher, which then
starts a duplicate of a run that is still going.

This adopter takes over those three duties for exactly one run:

  1. rewrite the .claim with ITS OWN (live) pid, so the incoming launcher's
     `claimed()` sees a live owner and never re-queues the run;
  2. wait for the orphaned child to exit;
  3. promote <tmpdir>/<src> -> <result> (validating that it parses as JSON),
     then remove the tmpdir;
  4. only AFTER the result is in place, drop the claim.

Doing (4) last is what closes the race: from the launcher's point of view the
run is continuously either claimed-by-a-live-pid or finished, never both
unclaimed and missing.  If the old launcher won the race and promoted the JSON
itself, this exits cleanly without touching anything.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import time


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="final .json path")
    ap.add_argument("--child-pid", type=int, required=True)
    ap.add_argument("--src", default=None,
                    help="basename inside <result>.tmpdir (default: basename "
                         "of --result)")
    ap.add_argument("--poll-s", type=float, default=5.0)
    a = ap.parse_args()

    result = os.path.abspath(a.result)
    tmp = result + ".tmpdir"
    claim = result + ".claim"
    src = os.path.join(tmp, a.src or os.path.basename(result))

    with open(claim, "w") as f:                     # step 1: take ownership
        f.write(str(os.getpid()))
    print("[adopt] pid %d took the claim for %s (child pid %d) @ %s"
          % (os.getpid(), os.path.basename(result), a.child_pid, time.ctime()),
          flush=True)

    while pid_alive(a.child_pid):                   # step 2: wait it out
        time.sleep(a.poll_s)
    print("[adopt] child %d exited @ %s" % (a.child_pid, time.ctime()),
          flush=True)
    time.sleep(2)                                   # let the last write land

    if os.path.exists(result):                      # old launcher won the race
        print("[adopt] %s already promoted; nothing to do" % result, flush=True)
    elif os.path.exists(src):                       # step 3: promote
        try:
            with open(src) as f:
                d = json.load(f)
            print("[adopt] %s parses (steps=%s, online_ppl=%s)"
                  % (os.path.basename(src), d.get("steps"),
                     d.get("online_ppl")), flush=True)
            os.replace(src, result)
            print("[adopt] promoted -> %s" % result, flush=True)
        except Exception as e:                                # noqa: BLE001
            print("[adopt] REFUSING to promote %s: %r" % (src, e), flush=True)
            return
    else:
        print("[adopt] child left no %s -- run did not finish; leaving the "
              "config unclaimed so the queue retries it" % src, flush=True)

    shutil.rmtree(tmp, ignore_errors=True)
    try:
        os.remove(claim)                            # step 4: release, last
    except OSError:
        pass
    print("[adopt] claim released @ %s" % time.ctime(), flush=True)


if __name__ == "__main__":
    main()
