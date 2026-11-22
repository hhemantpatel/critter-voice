"""Speech-to-text benchmark on your own recordings.
Metrics: word error rate (lower is better), and "time understood": for phrases that contain a time, does the parser
read the transcript the same as what you really said? (That is what a sleep timer depends on.) Also seconds per clip.
usage: python bench/asr.py [--gain 1.0] [--dump]"""
import argparse, json, re, sys, time, wave
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tommy.durations import parse_span, parse_clock
from transformers import pipeline

ROOT = Path(__file__).resolve().parent.parent
MODELS = [("whisper-base (control, en)", "whisper-base", "en"),
          ("whisper-tiny-hinglish", "whisper-tiny-hinglish", None),
          ("whisper-base-hinglish", "whisper-base-hinglish", None),
          ("Hindi2Hinglish-Swift", "Whisper-Hindi2Hinglish-Swift", None)]


def norm(s):
    s = re.sub(r"[^a-z0-9 :]", " ", s.lower().replace("-", " "))
    return re.sub(r"\s+", " ", s).strip().split()


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (r[i - 1] != h[j - 1]))
    return d[len(h)], len(r)


def meaning(t):
    s = parse_span(t)
    if s:
        return ("span", s)
    c = parse_clock(t)
    return ("clock", c.strftime("%H:%M")) if c else None


ap = argparse.ArgumentParser()
ap.add_argument("--gain", type=float, default=1.0)
ap.add_argument("--dump", action="store_true")
a = ap.parse_args()

rows = [json.loads(l) for l in (ROOT / "data/recordings/labels.jsonl").read_text().splitlines()]
clips = []
for r in rows:
    w = wave.open(str(ROOT / "data/recordings" / r["file"]))
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768 * a.gain
    clips.append((r["said"], np.clip(x, -1, 1)))

dev = 0 if torch.cuda.is_available() else -1
for label, folder, lang in MODELS:
    pipe = pipeline("automatic-speech-recognition", model=str(ROOT / "models/asr" / folder), device=dev,
                    dtype=torch.float16 if dev == 0 else torch.float32)
    kw = {"generate_kwargs": {"language": lang, "task": "transcribe"}} if lang else {}
    pipe({"raw": clips[0][1], "sampling_rate": 16000}, **kw)  # warm-up
    errs = words = tok = tok_ok = 0
    hyps = []
    t0 = time.time()
    for said, x in clips:
        hyp = pipe({"raw": x, "sampling_rate": 16000}, **kw)["text"].strip()
        hyps.append(hyp)
        e, n = wer(said, hyp)
        errs += e
        words += n
        m = meaning(said)
        if m:
            tok += 1
            tok_ok += meaning(hyp) == m
    print(f"{label:30} WER {100 * errs / words:5.1f}%   time understood {tok_ok}/{tok}   {(time.time() - t0) / len(clips):.2f}s/clip", flush=True)
    if a.dump:
        for (said, _), h in list(zip(clips, hyps))[:10]:
            print(f"     said: {said!r}\n      got: {h!r}")
    del pipe
    torch.cuda.empty_cache()
