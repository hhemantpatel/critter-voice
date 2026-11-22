from datetime import datetime
from ..registry import tool
from .. import system
from ..durations import parse_span, parse_clock, human


@tool(text="what to remind about", when="when, exactly as spoken: a delay ('in 10 minutes', '10 minute baad') or a time ('5 pm', 'shaam 5 baje')")
def set_reminder(text: str, when: str) -> str:
    """Remind me about something after a delay or at a clock time."""
    secs = parse_span(when)
    if secs is None:
        at = parse_clock(when)
        if at is None:
            raise ValueError(f"could not read a time from {when!r}")
        secs = (at - datetime.now()).total_seconds()
    system.schedule("reminder", secs, system.fire_cmd("reminder", f"Reminder: {text}"), f"tommy reminder: {text}")
    return f"I'll remind you in {human(secs)}."
