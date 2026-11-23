"""Utterance -> safe action. The fine-tuned model proposes one call; Python checks it before anything runs.

Checks, in order:
  1. the reply parses and names a real tool                      (else: ask)
  2. every time phrase the model copied reads the same as the time in what you actually said
     (a dropped word like "paune do ghante" -> "do ghante" changes 1 h 45 into 2 h)   (else: ask)
  3. the dispatcher's own checks: argument types, confirm tier, locked screen
"""
import json, re, urllib.error, urllib.request
from . import apps
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

    def ask(self, text, confirm=None, ask_back=None, call=...):
        """Returns (Reply, call) for one utterance; `call` may be a proposal made earlier for the same text.
        confirm(question, line) -> bool  asks yes/no.   ask_back(kind) -> str | None  asks for something missing (kind "time" ...)."""
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
