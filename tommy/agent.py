"""Utterance -> safe action. The fine-tuned model proposes one call; Python checks it before anything runs.

Checks, in order:
  1. the reply parses and names a real tool                      (else: ask)
  2. every time phrase the model copied reads the same as the time in what you actually said
     (a dropped word like "paune do ghante" -> "do ghante" changes 1 h 45 into 2 h)   (else: ask)
  3. the dispatcher's own checks: argument types, confirm tier, locked screen
"""
import json, re, urllib.error, urllib.request
from . import apps, system
from .dispatch import run_call
from .durations import parse_span, parse_clock, human, is_now
from .model import Server
from .registry import Reply, TOOLS, load_all
from .system import ROOT

GGUF = ROOT / "models/tommy/qwen3-0.6b-usage-q8.gguf"         # retrained on your reviewed use (learn.py), if it exists
if not GGUF.exists():
    GGUF = ROOT / "models/tommy/qwen3-0.6b-lora_f-q8.gguf"
SYSTEM = "You control a desktop assistant. Reply with exactly one function call."   # must match train/sft.py
TIME_KEYS = {"duration", "after", "when"}
_CALL = re.compile(r"<tool_call>\s*(\{.*\})\s*(?:</tool_call>|$)", re.S)


_OPEN = (re.compile(r"\b(?:open|launch)\s+(?:up\s+)?(.+)", re.I), re.compile(r"(.+?)\s+(?:kholo|khol\s+do|kholna|open\s+kar(?:o|\s+do))\b", re.I))


def open_request(text):
    """The name in "open X" / "X kholo", or None. Used when the model did not see it as one."""
    for pat in _OPEN:
        m = pat.search(text)
        if m:
            return m.group(1).strip(" .,!?") or None
    return None


_CLOSE = re.compile(r"^\W*(?:(?:please|can|could|will|would|you|sir|jarvis|hey|ok|okay|now|just|so|and|haan|then|i want to)\W+)*"
                    r"(?:close|quit|exit)\s+(?:down\s+)?(.+)", re.I)           # at the start only: not "remind me to close the door"
_BAND = re.compile(r"(.+?)\s+(?:band|bandh|close)\s*(?:kar\s*d[oe]|kardo|krdo|karo|kro|kar)\b", re.I)
# "close/stop" these is something else: the model's own tools (timers, power, focus, hotspot ...)
_NOT_APPS = {"timer", "timers", "alarm", "alarms", "reminder", "reminders", "stopwatch", "countdown", "focus", "pomodoro", "session", "hotspot", "dnd",
             "disturb", "notification", "notifications", "everything", "scheduled", "pc", "laptop", "computer", "system", "screen", "music", "song",
             "gaana", "gana", "internet", "wifi", "sleep", "jarvis"}


def close_request(text):
    """The app name in "close X" / "X band karo", or None. Read by rule, before the model: it knows no such tool and would guess another."""
    m = _CLOSE.search(text)
    hindi = None if m else _BAND.search(text)
    if not (m or hindi):
        return None
    name = re.sub(r"\b(please|plz|sir|app|application|windows?|all|every|both|sab|saare|sare|dono|the|of|my|on|this screen|current screen|this monitor|current monitor|here)\b|\bjarvis\b", " ", (m or hindi).group(1), flags=re.I)
    name = " ".join(name.split()).strip(" .,!?")
    if not name or set(re.findall(r"[a-z]+", name.lower())) & _NOT_APPS:
        return None
    if hindi and not apps.to_close(name)[2] and not apps.resolve(name)[1]:       # "band karo" ends many sentences: only if it names an app
        return None
    return name


def build_prompt(text):
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")


def _meaning(phrase):
    secs = parse_span(phrase)
    return ("span", secs) if secs is not None else ("clock", parse_clock(phrase))


class Agent:
    def __init__(self, gguf=GGUF):
        load_all()
        from .system import write_state
        write_state()                    # the shell's floating clock watches this file, so make sure it exists
        self.server = Server(gguf)

    def start(self):
        return self.server.start()

    def stop(self):
        self.server.stop()

    def propose(self, text):
        body = {"prompt": build_prompt(text), "n_predict": 96, "temperature": 0, "stop": ["</tool_call>"], "cache_prompt": True}
        req = urllib.request.Request(f"http://127.0.0.1:{self.server.port}/completion", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            out = json.load(urllib.request.urlopen(req, timeout=20))["content"]
        except (urllib.error.URLError, ConnectionError):           # the model server died: bring it back once
            self.server.start()
            out = json.load(urllib.request.urlopen(req, timeout=20))["content"]
        out += "</tool_call>"
        m = _CALL.search(out)
        try:
            d = json.loads(m.group(1)); return d["name"], dict(d["arguments"])
        except (AttributeError, ValueError, KeyError, TypeError):
            return None

    def _close(self, name, everywhere, confirm):
        app, wins, sure, more = apps.to_close(name, everywhere)
        if not wins:
            known, certain = apps.resolve(name)
            return Reply(f"{known.name} isn't open, sir.", "not_open") if certain else Reply(f"There's no app called {name} open, sir.", "no_app")
        if system.is_locked():
            return Reply("The screen is locked, so I won't do that.", "locked")
        n, terminal = len(wins), any(w["class"] in apps.TERMINALS for w in wins)
        if n > 1 or not sure or terminal:
            q = f"Close {n} {app.name} windows on this screen?" if n > 1 and more else f"Close all {n} {app.name} windows?" if n > 1 else f"Close {app.name}?"
            if not (confirm and confirm(q, "confirm_close_all" if n > 1 else "confirm_close_app")):
                return Reply("Okay, I won't do that.", "declined")
        left = apps.close(wins)
        if left:
            return Reply(f"{app.name} is still open; it may be asking you to save.", "failed")
        rest = f" {more} more still open." if more else ""
        return Reply((f"Closed {app.name}." if n == 1 else f"Closed {n} {app.name} windows.") + rest)

    def ask(self, text, confirm=None, ask_back=None, call=...):
        """Returns (Reply, call) for one utterance; `call` may be a proposal made earlier for the same text.
        confirm(question, line) -> bool  asks yes/no.   ask_back(kind) -> str | None  asks for something missing (kind "time" ...)."""
        name = close_request(text)
        if name:
            every = bool(re.search(r"\b(all|every|sab|saare|sare|dono|both)\b", text, re.I))
            return self._close(name, every, confirm), ("close_app", {"name": name})
        call = self.propose(text) if call is ... else call
        # "open X" is regular enough to recognise without the model: use it when the model gave up, or (an older model) named a site
        if call is None or call[0] in ("clarify", "open_site") or call[0] not in TOOLS:
            name = open_request(text) or (call[1].get("site") if call and call[0] == "open_site" else None)
            if name:
                call = ("open_app", {"name": name})
        if call is None or call[0] not in TOOLS:
            return Reply("I didn't get that, sir.", "not_understood"), call
        name, args = call
        tool = TOOLS[name]
        for key in TIME_KEYS & tool.params.keys():
            val = args.get(key)
            if key not in tool.required and val in (None, "", "none"):
                continue                                         # optional time, not given
            if is_now(text) or is_now(str(val or "")):
                args[key] = "now"
                continue
            heard = _meaning(text)                               # what the words you said hold
            got = _meaning(str(val)) if val not in (None, "") else (None, None)
            if heard[1] is None:                                 # you gave no time: whatever the model wrote is invented
                ans = ask_back("time") if ask_back else None
                if ans is None:
                    return Reply("Never mind, sir.", "declined"), call
                if is_now(ans):
                    args[key] = "now"
                elif _meaning(ans)[1] is None:
                    return Reply("I couldn't work out the time, sir.", "no_time"), call
                else:
                    args[key] = ans
            elif got[1] is None:                                 # you gave a time but the model failed to copy it
                args[key] = text
            elif got != heard:                                   # the model's copy disagrees with what you said
                said = human(heard[1]) if heard[0] == "span" else "that time"
                if confirm and confirm(f"Did you mean {said}?", "did_you_mean"):
                    args[key] = text
                else:
                    return Reply("Okay, I won't do that.", "declined"), call
        if name == "open_app" and args.get("name"):
            app, sure = apps.resolve(str(args["name"]))
            if app is None:
                return Reply(f"There's no app called {args['name']}, sir.", "no_app"), call
            if not sure and not (confirm and confirm(f"Did you mean {app.name}?", "did_you_mean")):
                return Reply("Okay, I won't do that.", "declined"), call
            args["name"] = app.name
        for key in tool.required:                                # e.g. "set a reminder": about what?
            if key in TIME_KEYS or args.get(key) not in (None, ""):
                continue
            ans = ask_back(f"{name}.{key}") if ask_back else None
            if not ans:
                return Reply("Never mind, sir.", "declined"), call
            args[key] = ans
        return run_call(name, args, confirm), call
