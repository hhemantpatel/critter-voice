"""Jarvis: wait for the wake word, record one command, act, answer.

    .venv-train/bin/python -m tommy.voice               # always listening for "Jarvis" (normally run by jarvis.service)
    .venv-train/bin/python -m tommy.voice --no-wake     # press Enter to talk instead

mic.py (sound in) -> ears.py (wake word, end of speech, speech-to-text) -> agent.py (what to do, safety) -> speak.py (answer)
Audio stays in memory, except the wake clips kept for learn.py (--no-learn turns that off).
"""
import argparse, collections, json, os, re, signal, time, wave
from pathlib import Path
import numpy as np
from . import speak, system
from .agent import Agent
from .dispatch import is_yes
from .ears import Ears
from .mic import FRAME_S, Mic, RATE
from .system import ROOT

FALSE_WAKE = {"wrong", "false alarm", "not you", "galti se", "galti", "kuch nahi", "nothing", "ignore", "never mind", "galat", "not for you"}
QUESTIONS = {"time": ("what_time", "What time, sir?"), "set_reminder.text": ("remind_what", "What should I remind you about, sir?"),
             "add_note.text": ("note_what", "What should I note down, sir?")}
USAGE = ROOT / "data" / "usage"          # local only (git-ignored); delete the folder to forget everything
PAUSED = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "jarvis-paused"      # holds the time until which to ignore the wake word


def is_false_wake(text):
    """You said "wrong" / "galti se" right after it woke: you did not call it."""
    t = re.sub(r"[^a-z ]", "", text.lower()).strip()
    return t in FALSE_WAKE or any(t.startswith(p) for p in ("false alarm", "galti se", "not you", "wrong wake"))


def log_event(event, audio):
    """Keep what happened, so it can be reviewed and learned from later (learn.py)."""
    (USAGE / "wake").mkdir(parents=True, exist_ok=True)
    event["wake_audio"] = f"{event['ts']}.wav"
    with wave.open(str(USAGE / "wake" / event["wake_audio"]), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE); w.writeframes(audio.tobytes())
    with (USAGE / "events.jsonl").open("a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    if event["outcome"] == "false":                                     # labelled by you: goes into the next wake-word retrain
        with (USAGE / "reviewed.jsonl").open("a") as f:
            f.write(json.dumps({"ts": event["ts"], "wake_audio": event["wake_audio"], "wake": False}) + "\n")


def paused():
    try:
        return time.time() < float(PAUSED.read_text())
    except (OSError, ValueError):
        return False


class Loop:
    def __init__(self, learn=True):
        self.learn = learn
        self.ears, self.agent, self.mic = Ears(), Agent(), None
        self.mute_until = 0.0              # the wake word is ignored while our own spoken answer is playing
        self.event = {}

    # ---- answering ----
    def answer(self, reply):
        print(f"  {reply.kind}: {reply.text}", flush=True)
        system.say(reply.text)
        if reply.kind != "info":           # the answer to a question is the text itself
            self.mute_until = time.monotonic() + speak.say_async(reply.kind) + 0.3

    def question(self, line, text):
        """Ask out loud, listening from its last words on. Returns what was said, or None."""
        print(f"  asking: {text}", flush=True)
        system.say(text)
        got = self.ears.capture(self.mic, wait_s=8.0, skip=speak.ask(self.mic, line))
        said = self.ears.transcribe(got[0]) if got else None
        print(f'  answer: "{said}"', flush=True)
        return said or None

    def confirm(self, text, line="confirm"):
        self.asked = True
        return is_yes(self.question(line if speak.pick_clip(line) else "confirm", text))

    def ask_back(self, kind):
        self.asked = True
        line, text = QUESTIONS[kind] if kind in QUESTIONS else ("not_understood", "Could you say that again, sir?")
        return self.question(line, text) or self.question(line, text)

    # ---- one command ----
    def think(self, audio):
        """Runs in the background during the pause after your last word. No side effects: it only works out what to do."""
        t0 = time.perf_counter()
        text = self.ears.transcribe(audio)
        t1 = time.perf_counter()
        call = self.agent.propose(text) if text else None
        return text, call, t1 - t0, time.perf_counter() - t1

    def command(self, wait_s, skip=0):
        """Listen for one command and do it. None: nobody spoke. False: only the wake word was said. True: handled."""
        got = self.ears.capture(self.mic, wait_s, skip, self.think)
        if got is None:
            return None
        _, job, last_word = got
        stopped = time.monotonic()
        text, call, stt, model = job.result()
        if not text:
            return False
        print(f'  heard: "{text}" -> {call}', flush=True)
        if is_false_wake(text):
            self.event.update(heard=text, outcome="false")
            print("  marked: false wake", flush=True)
            self.mute_until = time.monotonic() + speak.say_async("declined") + 0.3
            return True
        self.asked = False
        reply, call = self.agent.ask(text, self.confirm, self.ask_back, call)
        self.event.update(heard=text, call=list(call) if call else None, reply=reply.text, outcome="command")
        if not self.asked:
            print(f"  timing: stopped listening {stopped - last_word:.1f}s after your last word | speech-to-text {stt:.2f} model {model:.2f} "
                  f"| done {time.monotonic() - last_word:.2f}s after your last word", flush=True)
        self.answer(reply)
        return True

    def after_wake(self):
        # you may already be speaking ("Jarvis, set a timer ..."): listen at once, ignoring the tail of "Jarvis" itself
        if not self.command(1.3, skip=3):
            self.command(6.0, skip=speak.ask(self.mic, "greet"))         # nothing yet: greet, then wait; silence stays silent

    # ---- forever ----
    def run(self, mic, wake=True):
        self.mic, last_wake, last_noise, pause_check, is_paused = mic, 0.0, 0.0, 0.0, False
        ring = collections.deque(maxlen=int(2.5 / FRAME_S))              # what the wake word heard
        while True:
            if not wake:
                input("press Enter, then speak ")
                mic.drain()
                self.event = {}
                self.command(6.0)
                continue
            frame = mic.read()
            ring.append(frame)
            woke, now = self.ears.woke(frame), time.monotonic()
            if now - pause_check > 1.0:
                pause_check, is_paused = now, paused()
            if is_paused or now < self.mute_until or now - last_wake < 1.0:
                continue
            if self.ears.noise and now - last_noise > 10:                # fired without a voice: noted for training, nothing is reset
                last_noise = now
                print(f"  (ignored: score {self.ears.score:.2f}, no voice)", flush=True)
                if self.learn:
                    log_event({"ts": time.strftime("%Y%m%d-%H%M%S"), "wake_score": round(self.ears.score, 3),
                               "voice": round(self.ears.voice, 2), "outcome": "ignored"}, np.concatenate(ring))
            if not woke:
                continue
            print(f"* woke (score {self.ears.score:.2f}, voice {self.ears.voice:.2f})", flush=True)
            self.event = {"ts": time.strftime("%Y%m%d-%H%M%S"), "wake_score": round(self.ears.score, 3),
                          "voice": round(self.ears.voice, 2), "outcome": "nothing"}
            self.after_wake()
            if self.learn:
                log_event(self.event, np.concatenate(ring))
            ring.clear()
            mic.drain()
            self.ears.forget()
            last_wake = time.monotonic()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wake", action="store_true", help="press Enter instead of saying Jarvis")
    ap.add_argument("--no-learn", action="store_true", help="keep no wake clips or requests in data/usage")
    a = ap.parse_args()

    def stop(*_):
        raise KeyboardInterrupt                     # systemctl stop -> same clean exit as Ctrl-C
    signal.signal(signal.SIGTERM, stop)
    t0 = time.monotonic()
    loop = Loop(learn=not a.no_learn)
    try:
        loop.agent.start()
        mic = Mic()
        print(f"[ready in {time.monotonic() - t0:.1f}s] learning {'off' if a.no_learn else 'on (data/usage, this PC only)'}", flush=True)
        loop.run(mic, wake=not a.no_wake)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        speak.show_listening(False)
        loop.agent.stop()


if __name__ == "__main__":
    main()
