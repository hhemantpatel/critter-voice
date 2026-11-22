from typing import Literal
from ..registry import tool
from .. import system
from ..durations import parse_span


@tool(duration="optional focus length as spoken, e.g. '25 minutes'; leave empty for the default 25")
def focus_start(duration: str = "") -> str:
    """Start a focus (Pomodoro) session."""
    mins = int(round(parse_span(duration) / 60)) if duration and parse_span(duration) else 0
    left = system.qs_ipc("focus", "start", mins)
    return f"Focus started, {left} left." if left else "Couldn't reach the shell."


@tool()
def focus_stop() -> str:
    """Stop the focus session."""
    system.qs_ipc("focus", "stop")
    return "Focus stopped."


@tool()
def focus_pause() -> str:
    """Pause or resume the focus session."""
    return f"Focus {system.qs_ipc('focus', 'pause') or 'toggled'}."


@tool()
def focus_status() -> str:
    """Say how the focus session and today's count are going."""
    return system.qs_ipc("focus", "status") or "Couldn't reach the shell."


@tool(action="start, pause, reset or status")
def stopwatch(action: Literal["start", "pause", "reset", "status"]) -> str:
    """Control the stopwatch."""
    status = system.qs_ipc("stopwatch", "status")
    running = status.startswith("running")
    if action == "reset":
        system.qs_ipc("stopwatch", "reset"); return "Stopwatch reset."
    if action == "start" and not running:
        system.qs_ipc("stopwatch", "toggle"); return "Stopwatch running."
    if action == "pause" and running:
        system.qs_ipc("stopwatch", "toggle"); return "Stopwatch paused."
    return status or "Couldn't reach the shell."
