"""Run one model-proposed call safely: validate, apply the risk tier, execute, return text for the bubble."""
from .registry import check_args, load_all
from . import system
from .durations import human, parse_span


def run_call(name, args, confirm=None):
    """confirm(prompt) -> bool is asked for 'confirm'-tier tools; None means refuse them."""
    load_all()
    try:
        tool, clean = check_args(name, args)
    except ValueError as e:
        return f"I didn't understand that ({e})."
    if tool.risk == "confirm":
        if system.is_locked():
            return "The screen is locked, so I won't do that."
        shown = ", ".join(f"{k}={v}" + (f" (= {human(parse_span(v))})" if k in ("after", "duration") and parse_span(v) else "") for k, v in clean.items())
        if confirm is None or not confirm(f"{name}({shown})?"):
            return "Cancelled."
    try:
        return tool.fn(**clean)
    except (ValueError, RuntimeError) as e:
        return f"Couldn't do that: {e}"
