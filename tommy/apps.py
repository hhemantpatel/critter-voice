"""Find and open an installed app (or a site you installed as an app, e.g. from Brave) by the name you said.

The model only copies the name; matching is here, against this PC's own .desktop entries, so a new app works the moment it is
installed and nothing has to be added. Nothing the model says is ever run: it only selects one of the installed entries.

    .venv/bin/python -m tommy.apps "you 2"      # show the best matches
"""
import configparser, difflib, os, re, shlex
from dataclasses import dataclass
from pathlib import Path

DIRS = [Path.home() / ".local/share/applications", Path("/usr/local/share/applications"), Path("/usr/share/applications"),
        Path.home() / ".local/share/flatpak/exports/share/applications", Path("/var/lib/flatpak/exports/share/applications")]
SURE, ASK = 0.80, 0.60            # at or above SURE: open it; between: "Did you mean ...?"; below: not found
# never reachable by voice through this path: power and session commands have their own confirmed tools
BLOCKED = ("systemctl", "poweroff", "reboot", "shutdown", "loginctl", "hyprctl dispatch exit")
# spoken names that share no letters with the entry's own name
ALIASES = {"vs code": "Code - OSS", "vscode": "Code - OSS", "visual studio code": "Code - OSS", "code editor": "Code - OSS",
           "terminal": "kitty", "browser": "Brave", "my browser": "Brave", "web browser": "Brave", "file manager": "Files"}
# websites that are not installed as apps; installed ones always win
SITES = {"gmail": "https://mail.google.com", "mail": "https://mail.google.com", "github": "https://github.com",
         "google maps": "https://www.google.com/maps", "maps": "https://www.google.com/maps", "google drive": "https://drive.google.com",
         "google": "https://www.google.com", "chatgpt": "https://chatgpt.com", "youtube": "https://www.youtube.com",
         "whatsapp": "https://web.whatsapp.com", "linkedin": "https://www.linkedin.com", "reddit": "https://www.reddit.com",
         "netflix": "https://www.netflix.com", "amazon": "https://www.amazon.in", "wikipedia": "https://www.wikipedia.org"}
FILLER = {"the", "my", "app", "application", "please", "plz", "kholo", "khol", "do", "kar", "karo", "chalu", "open", "launch", "start", "sir", "a"}
DIGITS = {"0": "o", "1": "one", "2": "tu", "3": "three", "4": "for", "5": "five", "6": "six", "7": "seven", "8": "ate", "9": "nine"}


@dataclass
class App:
    name: str
    path: str = ""                # .desktop file, or "" for a website
    url: str = ""
    exec: str = ""
    terminal: bool = False
    extra: tuple = ()             # generic name, keywords, program name: weaker ways to name it


_cache = None


def apps():
    """Installed, visible applications (re-read when an applications folder changes)."""
    global _cache
    stamp = tuple((str(d), d.stat().st_mtime) for d in DIRS if d.exists())
    if _cache and _cache[0] == stamp:
        return _cache[1]
    found = {}
    for d in reversed(DIRS):                          # user entries override system ones of the same id
        for f in sorted(d.rglob("*.desktop")) if d.exists() else []:
            c = configparser.ConfigParser(interpolation=None, strict=False)
            try:
                c.read(f, encoding="utf-8")
                e = c["Desktop Entry"]
            except (configparser.Error, KeyError, UnicodeDecodeError, OSError):
                continue
            ex, name = e.get("Exec", ""), e.get("Name", "").strip()
            if e.get("Type", "Application") != "Application" or e.get("NoDisplay", "").lower() == "true" or \
                    e.get("Hidden", "").lower() == "true" or not ex or not name or any(b in ex.lower() for b in BLOCKED):
                found.pop(f.name, None)
                continue
            prog = os.path.basename(shlex.split(ex)[0]) if ex else ""
            extra = tuple(x for x in [e.get("GenericName", ""), prog, *e.get("Keywords", "").split(";")] if x)
            found[f.name] = App(name, str(f), "", ex, e.get("Terminal", "").lower() == "true", extra)
    _cache = (stamp, list(found.values()))
    return _cache[1]


def _words(text):
    text = re.sub(r"\d", lambda m: DIGITS[m.group(0)], text.lower())
    return [w for w in re.findall(r"[a-z]+", text)]


def _squash(text):
    """Letters only, with sounds-alike spellings folded together."""
    s = "".join(_words(text))
    for a, b in (("ph", "f"), ("ck", "k"), ("c", "k"), ("q", "k"), ("z", "s"), ("w", "v"), ("ee", "i"), ("oo", "u"), ("ou", "u")):
        s = s.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", s)


def _score(query, name):
    q, n = [w for w in _words(query) if w not in FILLER] or _words(query), _words(name)
    if not q or not n:
        return 0.0
    qs, ns = "".join(q), "".join(n)
    if qs == ns:
        return 1.0
    best = max(difflib.SequenceMatcher(None, qs, ns).ratio(),
               0.98 * difflib.SequenceMatcher(None, _squash(query), _squash(name)).ratio())
    if len(qs) >= 4 and ns.startswith(qs):                           # "insta" for Instagram
        best = max(best, 0.86)
    if len(qs) >= 5 and qs in ns:                                    # "class room" in "Google Classroom"
        best = max(best, 0.88)
    if set(q) <= set(n):                                             # "classroom" in "Google Classroom"
        best = max(best, 0.90 + 0.08 * len(q) / len(n))
    elif len(q) > 1 and any(len(w) > 3 and difflib.SequenceMatcher(None, w, m).ratio() >= 0.85 for w in q for m in n):
        best = max(best, 0.70)                                       # one clear word in common: worth asking about
    return best


def find(query, limit=5):
    """[(score, App)] best first."""
    query = query.strip(" .,!?")
    alias = ALIASES.get(" ".join(w for w in _words(query) if w not in FILLER - {"my"}), None) or ALIASES.get(query.lower())
    scored = []
    for a in apps():
        s = 1.0 if alias and a.name == alias else _score(query, a.name)
        s = max(s, max((0.9 * _score(query, x) for x in a.extra), default=0.0))
        scored.append((round(s, 3), a))
    for name, url in SITES.items():
        scored.append((round(_score(query, name) - 0.01, 3), App(name, "", url)))      # an installed app of the same name wins
    if any(s >= SURE and not a.url and s >= 0.85 for s, a in scored):
        scored = [t for t in scored if not t[1].url]                 # something installed fits: never the website instead
    scored.sort(key=lambda t: (-t[0], len(t[1].name)))
    return [t for t in scored if t[0] >= ASK][:limit]


def resolve(query):
    """(app, sure): the best match, and whether it is clear enough to open without asking. (None, False) if nothing fits."""
    hits = find(query)
    if not hits:
        return None, False
    score, app = hits[0]
    tie = len(hits) > 1 and score < 1.0 and hits[1][0] >= score - 0.01 and hits[1][1].name != app.name
    return app, score >= SURE and not tie


def command(app):
    """argv that opens it."""
    if app.url:
        return ["xdg-open", app.url]
    if app.terminal:
        return ["kitty", "--", *[w for w in shlex.split(app.exec) if not w.startswith("%")]]
    return ["gio", "launch", app.path]


if __name__ == "__main__":
    import sys
    for q in sys.argv[1:]:
        print(f"{q!r}: " + ", ".join(f"{a.name} {s:.2f}{' (site)' if a.url else ''}" for s, a in find(q)) or "nothing")
