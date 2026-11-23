"""Every kind of answer Jarvis can give has a recorded line, and failures are told apart. Nothing real runs."""
import sys, os, re; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
from tommy import dispatch, registry, speak, system
from tommy.registry import Reply
src = "".join(p.read_text() for p in (Path(__file__).parent.parent / "tommy").rglob("*.py"))
kinds = {"done", "cancelled", "declined", "not_understood", "no_app", "not_open", "no_time", "failed", "locked"}
used = set(re.findall(r'Reply\([^\n]*?,\s*"(\w+)"\)', src)) | set(re.findall(r'says="(\w+)"', src))
assert used <= kinds | {"info"}, used - kinds
lines = kinds | {"greet", "what_time", "remind_what", "note_what", "confirm", "did_you_mean"} | {f"confirm_{x}" for x in ("sleep", "poweroff", "lock", "hotspot", "cancel_scheduled", "close_app", "close_all")}
assert speak.CLIPS.exists() and not [k for k in lines if not speak.pick_clip(k)], [k for k in lines if not speak.pick_clip(k)]
for _ in range(30): assert re.fullmatch(r"confirm_\d+\.wav", speak.pick_clip("confirm").name)      # never "shut the PC down?" by accident

registry.load_all(); real = dict(registry.TOOLS); system.is_locked = lambda: False
def fake(fn, **k):
    t = real["hotspot"]; registry.TOOLS["hotspot"] = registry.Tool(t.name, fn, k.get("risk", t.risk), t.description, t.params, t.required, k.get("says", "done"))
run = lambda confirm: dispatch.run_call("hotspot", {"state": "on"}, confirm)
fake(lambda state: "Hotspot on.")
assert run(lambda *_: True) == Reply("Hotspot on.", "done")
assert run(lambda *_: False).kind == "declined" and run(None).kind == "declined"            # no way to ask = refuse
fake(lambda state: (_ for _ in ()).throw(RuntimeError("boom"))); assert run(lambda *_: True).kind == "failed"
system.is_locked = lambda: True; assert run(lambda *_: True).kind == "locked"
assert dispatch.run_call("rm_rf", {}).kind == "not_understood" and dispatch.run_call("stopwatch", {"action": "explode"}).kind == "not_understood"
system.ok = lambda *a, **k: False; system.is_locked = lambda: False; registry.TOOLS.update(real)
assert dispatch.run_call("hotspot", {"state": "on"}, lambda *_: True).kind == "failed"      # it used to claim success without looking
print("replies: all passed")
