from typing import Literal
from ..registry import tool
from .. import system
from ..durations import parse_span, human, is_now


def _need_span(text):
    if is_now(text):
        return 1.0                      # "now" = a 1 second delay (also gives the user a moment to see it)
    secs = parse_span(text)
    if secs is None:
        raise ValueError(f"could not read a duration from {text!r}")
    return secs


@tool(duration="the time span exactly as spoken, e.g. '25 minutes' or 'pachees minute'", label="what the timer is for")
def start_timer(duration: str, label: str = "") -> str:
    """Start a countdown timer that alerts when it ends."""
    secs = _need_span(duration)
    system.schedule("timer", secs, system.fire_cmd("timer", f"{label or 'Timer'} is done"), f"tommy timer {label}".strip())
    return f"Timer set for {human(secs)}."


@tool(risk="confirm", action="what to do: sleep, poweroff or lock",
      after="how long to wait, exactly as spoken, e.g. '25 minutes' or 'pachees minute baad'")
def schedule_power(action: Literal["sleep", "poweroff", "lock"], after: str) -> str:
    """Put the PC to sleep, shut it down or lock it after a delay."""
    secs = _need_span(after)
    argv = {"sleep": ["systemctl", "suspend"], "poweroff": ["systemctl", "poweroff"],
            "lock": ["loginctl", "lock-session"]}[action]
    system.schedule(action, secs, argv, f"tommy {action}")
    return f"OK, {action} now." if is_now(after) else f"OK, {action} in {human(secs)}."


@tool()
def list_scheduled() -> str:
    """List pending timers, reminders and scheduled power actions."""
    rows = system.scheduled()
    if not rows:
        return "Nothing scheduled."
    return "; ".join(f"{d.replace('tommy ', '')} in {human(left)}" for _, d, left in rows)


@tool(what="which to cancel: 'all', or a word like 'sleep', 'timer', 'reminder'")
def cancel_scheduled(what: str = "all") -> str:
    """Cancel pending timers, reminders or scheduled power actions."""
    rows = [r for r in system.scheduled() if what.lower() in ("all", "") or what.lower() in r[1].lower()]
    for unit, _, _ in rows:
        system.cancel(unit)
    return f"Cancelled {len(rows)}." if rows else "Nothing to cancel."
