"""App matching and the ask-before-opening rule, with made-up applications and nothing launched."""
import sys, os, tempfile; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
from tommy import apps, agent as A
from tommy.tools import media

# --- reading .desktop files: hidden, broken and power entries must never be reachable ---
d = Path(tempfile.mkdtemp())
def entry(f, **k): (d / f).write_text("[Desktop Entry]\nType=Application\n" + "".join(f"{a}={b}\n" for a, b in k.items()))
entry("a.desktop", Name="YouTube", Exec="browser --app-id=x")
entry("b.desktop", Name="Google Classroom", Exec="browser --app-id=y")
entry("c.desktop", Name="Instagram", Exec="browser --app-id=z")
entry("d.desktop", Name="Htop", Exec="htop", Terminal="true")
entry("e.desktop", Name="GTK Settings", Exec="gtk-settings")
entry("f.desktop", Name="Qt6 Settings", Exec="qt6ct")
entry("j.desktop", Name="WhatsApp Web", Exec="browser --app-id=w")
entry("g.desktop", Name="Shutdown", Exec="systemctl poweroff")
entry("h.desktop", Name="Secret", Exec="secret", NoDisplay="true")
(d / "i.desktop").write_text("not a desktop file")
apps.DIRS, apps._cache = [d], None
names = {a.name for a in apps.apps()}
assert names == {"YouTube", "Google Classroom", "Instagram", "Htop", "GTK Settings", "Qt6 Settings", "WhatsApp Web"}, names

# --- matching ---
for said, want, sure in [("youtube", "YouTube", True), ("you 2", "YouTube", True), ("your tube", "YouTube", True), ("classroom", "Google Classroom", True),
                         ("class room", "Google Classroom", True), ("insta", "Instagram", True), ("settings", "GTK Settings", False),
                         ("shutdown", None, False), ("secret", None, False), ("banana split", None, False), ("", None, False), ("gmail", "gmail", True),
                         ("whatsapp", "WhatsApp Web", True), ("whatapp", "WhatsApp Web", True)]:      # misspelt: still the installed one, not the site
    app, s = apps.resolve(said)
    assert (app.name if app else None, s) == (want, sure), (said, app, s)

# --- a malicious "name" only ever selects an entry: it is never part of a command ---
for bad in ("; rm -rf ~", "youtube; reboot", "$(reboot)", "../../etc/passwd", "--app=http://evil"):
    app, _ = apps.resolve(bad)
    assert app is None or all(bad not in part for part in apps.command(app)), bad
assert apps.command(next(a for a in apps.apps() if a.name == "Htop"))[:2] == ["kitty", "--"]

# --- the agent asks when unsure, and does nothing when the answer is no ---
launched = []
media.system.launch = lambda argv: launched.append(argv)
class Fake(A.Agent):
    def __init__(self, call): A.load_all(); self.call = call
    def propose(self, text): return self.call
def go(name, yes):
    launched.clear(); asked = []
    reply, _ = Fake(("open_app", {"name": name})).ask(f"open {name}", lambda q, *_: (asked.append(q), yes)[1])
    return reply.text, asked, len(launched)
assert go("youtube", True) == ("Opening YouTube.", [], 1)
assert go("settings", False)[1:] == (["Did you mean GTK Settings?"], 0)
assert go("settings", True) == ("Opening GTK Settings.", ["Did you mean GTK Settings?"], 1)
r = go("banana split", True); assert r == ("There's no app called banana split, sir.", [], 0), r
print("apps: all passed")

# --- "open X" works even when the model does not understand it (or is an older one that only knew sites) ---
for said, call, want in [("Open classroom.", ("clarify", {"question": "?"}), "Opening Google Classroom."),
                         ("Sir open you 2.", ("open_site", {"site": "youtube"}), "Opening YouTube."),
                         ("instagram kholo", None, "Opening Instagram."),
                         ("Sir, open classroom", ("open_site", {"site": "github"}), "Opening Google Classroom."),
                         ("what is the weather", ("clarify", {"question": "Say again?"}), "Say again?"),
                         ("open the door", ("clarify", {"question": "Say again?"}), "There's no app called the door, sir.")]:
    launched.clear()
    reply, _ = Fake(call).ask(said, lambda q, *_: False)
    assert reply.text == want, (said, reply, want)
print("open fallback: all passed")

# --- closing: the rule, the windows it picks, and what it asks (no window is touched) ---
entry("j.desktop", Name="Brave", Exec="/usr/bin/brave %U", StartupWMClass="brave-browser")
entry("k.desktop", Name="WhatsApp Web", Exec="/usr/bin/brave --app=https://w", StartupWMClass="brave-web.whatsapp.com__-Default")
entry("l.desktop", Name="kitty", Exec="kitty")
apps._cache = None
OPEN, FOCUS, SPECIAL = [], [], []
def opened(*classes): OPEN[:] = [{"address": f"0x{i:x}", "class": c.lower()} for i, c in enumerate(classes, 10)]
closed = []
apps.windows = lambda: list(OPEN)
def fake_hypr(*args):
    if args[0] == "monitors":
        import json; return json.dumps([   # the shape hyprctl really prints: an unused special workspace has no id
            {"focused": False, "activeWorkspace": {"id": 9, "name": "9"}, "specialWorkspace": {"address": "", "type": "", "name": ""}},
            {"focused": True, "activeWorkspace": {"id": 1, "name": "1"}, "specialWorkspace": {"id": -98, "name": "special:x"} if SPECIAL else {"address": "", "type": "", "name": ""}}])
    if args[0] == "activewindow" and FOCUS:
        import json; return json.dumps({"address": FOCUS[0], "class": "x"})
    if args[0] == "activewindow":
        import json; return json.dumps(OPEN[0]) if OPEN else "{}"
    closed.append(args[1])
    addr = args[1].split("address:")[1].split('"')[0]
    if "stuck" not in sys.modules: OPEN[:] = [w for w in OPEN if w["address"] != addr]
    return "ok"
apps._hypr = fake_hypr
A.system.is_locked = lambda: False
def shut(said, yes, call=("cancel_scheduled", {"what": "all"})):         # the model's (wrong) guess must never matter
    closed.clear(); asked = []
    r, c = Fake(call).ask(said, lambda q, line="": (asked.append(q), yes)[1])
    return r.text, asked, len(closed)
WA, BR = "brave-web.whatsapp.com__-Default", "brave-browser"
opened(WA, BR, BR); assert shut("Sir, close WhatsApp.", True) == ("Closed WhatsApp Web.", [], 1) and len(OPEN) == 2      # Brave untouched
opened(WA, BR, BR); assert shut("whatsapp band kar do", True)[0] == "Closed WhatsApp Web."
opened(WA, BR); assert shut("close brave", True) == ("Closed Brave.", [], 1) and OPEN[0]["class"] == WA.lower()           # the web app untouched
opened(WA, BR, BR, BR); assert shut("close all brave windows", False) == ("Okay, I won't do that.", ["Close all 3 Brave windows?"], 0)
assert shut("close brave", True) == ("Closed 3 Brave windows.", ["Close all 3 Brave windows?"], 3) and len(OPEN) == 1
opened("kitty"); assert shut("close terminal", False) == ("Okay, I won't do that.", ["Close kitty?"], 0)                  # a terminal is always asked
opened(BR, WA); assert shut("close this window", False) == ("Okay, I won't do that.", ["Close Brave?"], 0)
opened(BR); assert shut("close whatsapp", True) == ("WhatsApp Web isn't open, sir.", [], 0)
assert shut("close banana", True) == ("There's no app called banana open, sir.", [], 0)
for said in ("close the timer", "timer band karo", "laptop band kar do", "hotspot band karo", "focus band karo", "remind me to close whatsapp in 10 minutes",
             "note this down close whatsapp", "open whatsapp", "cancel all timers", "darwaza band kar do", "notifications band karo"):
    assert A.close_request(said) is None, said
opened(BR); A.system.is_locked = lambda: True; assert shut("close brave", True)[0].startswith("The screen is locked") and len(OPEN) == 1
A.system.is_locked = lambda: False; sys.modules["stuck"] = sys; assert "still open" in shut("close brave", True)[0]        # an app asking to save
del sys.modules["stuck"]
# several windows: the one on the screen in use (workspace 1), not the ones on another workspace or monitor (9)
def spread(*ws, focus=None):
    OPEN[:] = [{"address": f"0x{i:x}", "class": BR, "workspace": w} for i, w in enumerate(ws, 10)]; FOCUS[:] = [focus] if focus else ["0xfff"]
spread(9, 1, 4); assert shut("close brave", False) == ("Closed Brave. 2 more still open.", [], 1) and [w["workspace"] for w in OPEN] == [9, 4]
spread(9, 1, 1, focus="0xc"); assert shut("close brave", False) == ("Closed Brave. 2 more still open.", [], 1) and "0xc" not in [w["address"] for w in OPEN]
spread(9, 1, 1); assert shut("close brave", False) == ("Okay, I won't do that.", ["Close 2 Brave windows on this screen?"], 0)
spread(9, 4, 5); assert shut("close brave", False) == ("Okay, I won't do that.", ["Close all 3 Brave windows?"], 0)          # none here: ask, never guess
spread(9, 1, 4); assert shut("close all brave windows", True) == ("Closed 3 Brave windows.", ["Close all 3 Brave windows?"], 3)
spread(9, 1, 4); assert shut("close brave on this screen", False)[2] == 1 and A.close_request("close brave on the current monitor") == "brave"
SPECIAL.append(1); spread(9, -98, 4); assert shut("close brave", False) == ("Closed Brave. 2 more still open.", [], 1); SPECIAL.clear()
FOCUS.clear()
print("closing: all passed")
