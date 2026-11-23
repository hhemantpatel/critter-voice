"""The real loop with a fake microphone playing your recordings in real time. Agent actions, speakers and desktop are stubs.
Run: .venv-train/bin/python tests/test_voice_loop.py        (skipped if the recordings are not on this machine)"""
import sys, os, json, time, wave; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from tommy import ears as E, speak, system, voice as V
from tommy.mic import FRAME, FRAME_S, GAIN, RATE
from tommy.registry import Reply
from tommy.system import ROOT

lab = ROOT / "data/wake/labels.jsonl"
if not lab.exists():
    print("skipped: no recordings here"); raise SystemExit
def load(p):
    w = wave.open(str(p)); return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
rows = [json.loads(l) for l in lab.read_text().splitlines()]
one_breath = [(r["text"], load(ROOT / "data/wake" / r["file"])) for r in rows if r["kind"] == "JARVIS" and "," in r["text"]][:3]
bare = [load(ROOT / "data/wake" / r["file"]) for r in rows if r["kind"] == "JARVIS" and r["text"] in ("Jarvis", "Hey Jarvis")][:1]
cmds = [(json.loads(l)["said"], load(ROOT / "data/recordings" / json.loads(l)["file"])) for l in (ROOT / "data/recordings/labels.jsonl").read_text().splitlines()]
timer = next(c for c in cmds if c[0] == "set a timer for 10 minutes")
rng = np.random.default_rng(0)
room = lambda s: (rng.standard_normal(int(s * RATE)) * 4).astype(np.float32)
quiet = None
for l in (ROOT / "data/usage/events.jsonl").read_text().splitlines() if (ROOT / "data/usage/events.jsonl").exists() else []:
    e = json.loads(l); f = ROOT / "data/usage/wake" / str(e.get("wake_audio"))
    if e["outcome"] == "ignored" and f.exists(): quiet = load(f) / GAIN; break


class FakeMic:
    """Real time: frames become available every 80 ms whether or not anybody reads them, like the real one."""
    def __init__(self, parts):
        x = np.clip(np.concatenate(parts) * GAIN, -32768, 32767).astype(np.int16)
        self.f = [x[i:i + FRAME] for i in range(0, len(x) - FRAME, FRAME)]; self.i, self.t0 = 0, time.monotonic()
    def read(self):
        if self.i >= len(self.f): raise EOFError
        time.sleep(max(0.0, self.t0 + (self.i + 1) * FRAME_S - time.monotonic())); self.i += 1
        return self.f[self.i - 1]
    def drain(self): self.i = max(self.i, min(len(self.f), int((time.monotonic() - self.t0) / FRAME_S)))


speak.say_async = lambda k: spoken.append(k) or 0.2
speak.show_listening = E.show_listening = lambda on: None
system.say = lambda t: None
spoken, done = [], []
class FakeAgent:
    def propose(self, text): return REAL.propose(text)
    def ask(self, text, confirm=None, ask_back=None, call=None): done.append((text, call, time.monotonic())); return Reply("ok"), call
from tommy.agent import Agent
from tommy.model import Server
REAL = Agent(); REAL.server = Server(REAL.server.cmd[2], port=8190); REAL.start()
loop = V.Loop.__new__(V.Loop); loop.learn, loop.ears, loop.agent, loop.mute_until, loop.event = False, E.Ears(), FakeAgent(), 0.0, {}
timings = []
orig = print
def play(parts):
    done.clear(); spoken.clear(); loop.mute_until = 0.0; loop.ears.forget()
    try: loop.run(FakeMic(parts))
    except EOFError: pass
    return list(done)
try:
    print("--- one breath"); hits = 0
    for text, c in one_breath:
        r = play([room(2), c, room(5)]); hits += bool(r); print("   ", repr(text), "->", r[0][1] if r else "missed")
    assert hits >= 2, "one-breath commands are not reaching the agent"
    print("--- Jarvis, greeting, then the command")
    r = play([room(2), bare[0], room(2.6), timer[1], room(5)]); assert r and r[0][1] and r[0][1][0] == "start_timer", r
    assert "greet" in spoken
    print("--- a pause of 1 s inside the sentence must not cut it")
    half = len(timer[1]) // 2; a = timer[1]; n = len(a) // 320 * 320; lv = a[:n].reshape(-1, 320).std(axis=1); mid = (n // 320) // 2
    cut = (mid - 25 + int(np.argmin(lv[mid - 25:mid + 25]))) * 320
    r = play([room(2), bare[0], room(2.6), a[:cut], room(1.0), a[cut:], room(5)]); assert len(r) == 1, r; print("    heard once:", r[0][0])
    if quiet is not None:
        print("--- room noise that used to wake it, then a real Jarvis at once")
        r = play([room(2), quiet, quiet, bare[0], room(2.6), timer[1], room(5)]); assert r and r[0][1][0] == "start_timer", r
    print("--- paused"); V.paused = lambda: True
    assert not play([room(2), *[c for _, c in one_breath[:1]], room(3)]), "answered while paused"
    V.paused = lambda: False
    print("--- speed: from the last word to the action")
    for said, c in cmds[:6]:
        n = len(c) // 320 * 320; f = c[:n].reshape(-1, 320).std(axis=1); last = (np.where(f > max(np.percentile(f, 20) * 4, f.max() * .08))[0][-1] + 1) * 320 / RATE
        mic = FakeMic([room(0.5), c, room(6)]); loop.mic, loop.asked, loop.event = mic, False, {}; done.clear()
        loop.command(4.0)
        timings.append(done[0][2] - mic.t0 - 0.5 - last if done else np.nan); print(f"    {said!r}: {timings[-1]:.2f}s")
    print(f"median {np.nanmedian(timings):.2f}s  max {np.nanmax(timings):.2f}s")
    assert np.nanmedian(timings) <= 2.0 and np.nanmax(timings) <= 2.6, timings
    print("voice loop: all passed")
finally:
    REAL.stop()
