"""Detach the E2-controls launcher(s) from the calling shell.

Same pattern as resume_local.py: CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS so
the queue survives the session that started it. Logs land in results/logs/.

    python run_e2_controls.py calib
    python run_e2_controls.py main [--absgate-threshold X] [--workers N]
"""

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXPERIMENTS = os.path.join(ROOT, "code", "experiments")
LOGS = os.path.join(ROOT, "results", "logs")
PY = sys.executable


def running(pattern: str) -> bool:
    ps = ("(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          f"Where-Object {{ $_.CommandLine -match '{pattern}' }}).Count")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, text=True)
    try:
        return int(r.stdout.strip() or 0) > 0
    except ValueError:
        return False


def launch(part, extra):
    os.makedirs(LOGS, exist_ok=True)
    log = os.path.join(LOGS, f"e2_controls_{part}.log")
    args = [PY, os.path.join(EXPERIMENTS, "launch_e2_controls.py"),
            "--part", part] + extra
    subprocess.Popen(args, cwd=EXPERIMENTS,
                     stdout=open(log, "a"), stderr=subprocess.STDOUT,
                     creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                     | subprocess.DETACHED_PROCESS)
    print(f"launched: launch_e2_controls.py --part {part} {' '.join(extra)}")
    print(f"   log: {log}")


if __name__ == "__main__":
    part = sys.argv[1]
    extra = sys.argv[2:]
    pat = f"launch_e2_controls.py --part {part}"
    if running(pat):
        print(f"already running: {pat}")
    else:
        launch(part, extra)
