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
entry("g.desktop", Name="Shutdown", Exec="systemctl poweroff")
entry("h.desktop", Name="Secret", Exec="secret", NoDisplay="true")
(d / "i.desktop").write_text("not a desktop file")
apps.DIRS, apps._cache = [d], None
names = {a.name for a in apps.apps()}
assert names == {"YouTube", "Google Classroom", "Instagram", "Htop", "GTK Settings", "Qt6 Settings"}, names

# --- matching ---
for said, want, sure in [("youtube", "YouTube", True), ("you 2", "YouTube", True), ("your tube", "YouTube", True), ("classroom", "Google Classroom", True),
                         ("class room", "Google Classroom", True), ("insta", "Instagram", True), ("settings", "GTK Settings", False),
                         ("shutdown", None, False), ("secret", None, False), ("banana split", None, False), ("", None, False), ("gmail", "gmail", True)]:
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
