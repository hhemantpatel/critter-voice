from pathlib import Path
from typing import Literal
from ..registry import tool
from .. import system
from ..durations import parse_span


@tool(state="on or off", duration="optional, how long to keep it on, as spoken")
def do_not_disturb(state: Literal["on", "off"], duration: str = "") -> str:
    """Turn do-not-disturb on or off, optionally for a while."""
    system.qs_ipc("notifs", "enableDnd" if state == "on" else "disableDnd")
    secs = parse_span(duration) if duration else None
    if state == "on" and secs:
        system.schedule("dnd", secs, ["qs", "-c", "caelestia", "ipc", "call", "notifs", "disableDnd"], "tommy dnd off")
        return f"Do not disturb on for {int(secs // 60)} min."
    return f"Do not disturb {state}."


@tool(risk="confirm", state="on or off")
def hotspot(state: Literal["on", "off"]) -> str:
    """Turn the Wi-Fi hotspot on or off."""
    verb = "up" if state == "on" else "down"
    system.run(["nmcli", "--wait", "30", "connection", verb, "Hotspot"], timeout=40)
    return f"Hotspot {state}."


@tool()
def network_status() -> str:
    """Say whether the network is healthy, from the last connection check."""
    hist = Path("/var/log/net-diag/history.log")
    if not hist.exists():
        return "No network checks recorded."
    last = hist.read_text().splitlines()[-1].split(None, 2)
    return f"Last check {last[0]} {last[1]}: {last[2]}" if len(last) == 3 else "No network checks recorded."
