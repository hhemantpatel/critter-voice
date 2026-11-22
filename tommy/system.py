"""Thin wrappers around the machine. Every external effect goes through here, always as an argv list
(never a shell string), so model-provided text can never become a command."""
import os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(ROOT / ".venv" / "bin" / "python")
PREFIX = "tommy-"        # every timer unit we create starts with this, so we only ever touch our own


def run(argv, timeout=10):
    """Run argv, return stripped stdout. Never raises on a non-zero exit; returns '' instead."""
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def qs_ipc(target, fn, *args):
    return run(["qs", "-c", "caelestia", "ipc", "call", target, fn, *map(str, args)])


def say(text):
    """Show text in Sprout's speech bubble."""
    qs_ipc("buddy", "say", text)


def notify(title, body=""):
    # -a Sprout: Buddy.qml:200 skips its "notice" startle for notifications from this app name
    subprocess.Popen(["notify-send", "-a", "Sprout", "-i", "appointment-soon", title, body])


def schedule(kind, seconds, argv, description):
    """Run argv after `seconds` using a transient systemd user timer. Survives a daemon crash."""
    unit = f"{PREFIX}{kind}-{int(time.time())}{os.urandom(2).hex()}"
    description = f"{description} |due={int(time.time() + seconds)}"      # systemd's own elapse time is boot-relative
    out = subprocess.run(
        ["systemd-run", "--user", f"--unit={unit}", f"--on-active={int(seconds)}s", "--timer-property=AccuracySec=1s",
         f"--description={description}", f"--working-directory={ROOT}", "--collect", "--quiet", *argv],
        capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or "systemd-run failed")
    return unit


def fire_cmd(*args):
    """argv that runs tommy.fire (what a timer executes when it goes off)."""
    return [PYTHON, "-m", "tommy.fire", *args]


def scheduled():
    """[(unit, description, seconds_left)] for our own pending timers."""
    rows = []
    for line in run(["systemctl", "--user", "list-units", f"{PREFIX}*.timer", "--plain", "--no-legend", "--all"]).splitlines():
        unit = line.split()[0] if line.split() else ""
        if not unit.startswith(PREFIX):
            continue
        desc = run(["systemctl", "--user", "show", unit, "-p", "Description", "--value"])
        text, _, due = desc.partition(" |due=")
        left = max(0, int(due) - time.time()) if due.isdigit() else 0
        rows.append((unit, text, left))
    return rows


def cancel(unit):
    if not unit.startswith(PREFIX):                 # never stop anything that isn't ours
        raise ValueError("not a tommy timer")
    run(["systemctl", "--user", "stop", unit if unit.endswith(".timer") else unit + ".timer"])


def is_locked():
    """UNVERIFIED while actually locked (see plan): Caelestia's lock may read false. Used only as a guard."""
    return qs_ipc("lock", "isLocked").lower() == "true"


def launch(argv):
    """Start a program and let go of it (it keeps running after Tommy). argv is always a list."""
    subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
