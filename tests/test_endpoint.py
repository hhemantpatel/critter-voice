"""Does listening stop on time? Your recordings (data/recordings) followed by real sound of your room (wakes with no voice in them).
Run: .venv-train/bin/python tests/test_endpoint.py        (skipped if the recordings are not on this machine)"""
import sys, os, json, wave; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from tommy.wake import stub_sklearn; stub_sklearn()
from openwakeword.vad import VAD
from tommy.ears import Endpoint, END_S
from tommy.mic import FRAME, FRAME_S, GAIN
from tommy.system import ROOT

recs = sorted((ROOT / "data/recordings").glob("*.wav"))
if not recs:
    print("skipped: no recordings here"); raise SystemExit
def load(p):
    w = wave.open(str(p)); return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
vad = VAD()
def probs(x):
    vad.reset_states(); return np.array([vad.predict(x[i:i + FRAME], frame_size=320) for i in range(0, len(x) - FRAME + 1, FRAME)])
room = []
for l in (ROOT / "data/usage/events.jsonl").read_text().splitlines() if (ROOT / "data/usage/events.jsonl").exists() else []:
    f = ROOT / "data/usage/wake" / str(json.loads(l).get("wake_audio"))
    if f.exists():
        a = load(f)
        if len(a) > 20000 and probs(a.astype(np.int16)).max() < 0.15: room.append(a)
rng = np.random.default_rng(0)
if not room: room = [rng.standard_normal(32000).astype(np.float32) * 60]
def last_word(c):                                # loudness only, independent of the voice detector
    n = len(c) // 320 * 320; f = c[:n].reshape(-1, 320).std(axis=1)
    return (np.where(f > max(np.percentile(f, 20) * 4, f.max() * 0.08))[0][-1] + 1) * 320 / 16000
data = []
for p in recs:
    c = load(p); x = np.clip(np.concatenate([c * GAIN] + [room[rng.integers(len(room))] for _ in range(6)]), -32768, 32767).astype(np.int16)
    data.append((p.name, x, probs(x), last_word(c)))
def run(start):
    waits, never, early, missed = [], [], [], []
    for name, x, ps, last in data:
        e = Endpoint(); e.START = start; t = None
        for i, p in enumerate(ps):
            e.feed(float(p), float(x[i * FRAME:(i + 1) * FRAME].astype(float).std()))
            if e.started and e.quiet * FRAME_S >= END_S: t = (i + 1) * FRAME_S; break
        if not e.started: missed.append(name)
        elif t is None: never.append(name)
        else: waits.append(t - last); (early.append(name) if t - last < 0.5 else None)
    return np.array(waits), never, early, missed
for s in (0.2, 0.3, 0.4):
    w, never, early, missed = run(s)
    print(f"start {s}: ended {len(w)}/{len(data)}  never {never}  not heard {missed}  cut early {early}  wait median {np.median(w):.2f} max {w.max():.2f}")
# and the room itself must not start a command
for s in (0.2, 0.3, 0.4):
    print(f"start {s}: room clips that would start a command: {sum(probs(a.astype(np.int16)).max() > s for a in room)}/{len(room)}")
w, never, early, missed = run(Endpoint.START)
assert len(w) >= 45 and not never and not early and np.median(w) <= 1.6, "end-of-speech rule got worse"
print("ok")
