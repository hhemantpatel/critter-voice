"""Utterance -> safe action. The fine-tuned model proposes one call; Python checks it before anything runs.

Checks, in order:
  1. the reply parses and names a real tool                      (else: ask)
  2. every time phrase the model copied reads the same as the time in what you actually said
     (a dropped word like "paune do ghante" -> "do ghante" changes 1 h 45 into 2 h)   (else: ask)
  3. the dispatcher's own checks: argument types, confirm tier, locked screen
"""
import json, re, urllib.request
from .dispatch import run_call
from .durations import parse_span, parse_clock, human, is_now
from .model import Server
from .registry import TOOLS, load_all
from .system import ROOT

GGUF = ROOT / "models/tommy/qwen3-0.6b-lora_b-q8.gguf"
SYSTEM = "You control a desktop assistant. Reply with exactly one function call."   # must match train/sft.py
TIME_KEYS = {"duration", "after", "when"}
_CALL = re.compile(r"<tool_call>\s*(\{.*\})\s*(?:</tool_call>|$)", re.S)


def build_prompt(text):
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")


def _meaning(phrase):
    secs = parse_span(phrase)
    return ("span", secs) if secs is not None else ("clock", parse_clock(phrase))


class Agent:
    def __init__(self, gguf=GGUF):
        load_all()
        self.server = Server(gguf)

    def start(self):
        return self.server.start()

    def stop(self):
        self.server.stop()

    def propose(self, text):
        body = {"prompt": build_prompt(text), "n_predict": 96, "temperature": 0, "stop": ["</tool_call>"], "cache_prompt": True}
        req = urllib.request.Request(f"http://127.0.0.1:{self.server.port}/completion", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        out = json.load(urllib.request.urlopen(req, timeout=60))["content"] + "</tool_call>"
        m = _CALL.search(out)
        try:
            d = json.loads(m.group(1)); return d["name"], dict(d["arguments"])
        except (AttributeError, ValueError, KeyError, TypeError):
            return None

    def ask(self, text, confirm=None, ask_back=None):
        """Returns (reply_text, debug) for one utterance.
        confirm(question) -> bool  asks yes/no.   ask_back(kind) -> str | None  asks the user to say something (kind "time")."""
        call = self.propose(text)
        if call is None or call[0] not in TOOLS:
            return "Sorry, I didn't get that.", call
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
                    return "Never mind, sir.", call
                if is_now(ans):
                    args[key] = "now"
                elif _meaning(ans)[1] is None:
                    return "Sorry sir, I couldn't read that time.", call
                else:
                    args[key] = ans
            elif got[1] is None:                                 # you gave a time but the model failed to copy it
                args[key] = text
            elif got != heard:                                   # the model's copy disagrees with what you said
                said = human(heard[1]) if heard[0] == "span" else "that time"
                if confirm and confirm(f"Did you mean {said}?"):
                    args[key] = text
                else:
                    return "Cancelled.", call
        return run_call(name, args, confirm), call
