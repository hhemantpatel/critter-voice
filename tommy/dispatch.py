"""Run one model-proposed call safely: validate, apply the risk tier, execute, return a Reply."""
from .registry import Reply, check_args, load_all
from . import system
from .durations import human, is_now, parse_span


def describe(name, a):
    """The question asked before a risky call, in plain words."""
    if name == "schedule_power":
        what = {"sleep": "Put the PC to sleep", "poweroff": "Shut the PC down", "lock": "Lock the screen"}[a["action"]]
        secs = parse_span(a["after"])
        return f"{what} now?" if is_now(a["after"]) or not secs else f"{what} in {human(secs)}?"
    if name == "cancel_scheduled":
        return "Cancel everything?" if a.get("what", "all") in ("all", "") else f"Cancel the {a['what']}?"
    return f"Turn the {name} {a.get('state', 'on')}?" if "state" in a else f"{name.replace('_', ' ').capitalize()}?"


def run_call(name, args, confirm=None):
    """confirm(question, line) -> bool is asked for 'confirm'-tier tools; None means refuse them."""
    load_all()
    try:
        tool, clean = check_args(name, args)
    except ValueError as e:
        print(f"  (rejected: {e})", flush=True)
        return Reply("I didn't get that, sir.", "not_understood")
    if tool.risk == "confirm":
        if system.is_locked():
            return Reply("The screen is locked, so I won't do that.", "locked")
        if confirm is None or not confirm(describe(name, clean), f"confirm_{clean.get('action', name)}"):
            return Reply("Okay, I won't do that.", "declined")
    try:
        out = tool.fn(**clean)
    except (ValueError, RuntimeError) as e:
        return Reply(f"I couldn't do that: {e}", "failed")
    return out if isinstance(out, Reply) else Reply(out, tool.says)
